"""Isolated gate tests; synthetic rankings NEVER enter research result files.

Fixtures live only in TemporaryDirectory. Original recovered qrels and query
identities are copied as read-only fixture inputs; rankings/reference metrics
are synthetic test data. The production corpus-count constant is patched to a
small fixture count so tests need no million-row cache and no retrieval.
"""
from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ranking_input_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fixture_ndcg(documents: list[str], judgments: dict[str, int]) -> float:
    """Fixture reference formula independent of the gate's scoring function."""
    gains = [2 ** judgments.get(doc, 0) - 1 for doc in documents[:10]]
    ideal = sorted((2 ** grade - 1 for grade in judgments.values() if grade > 0), reverse=True)[:10]
    numerator = math.fsum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    denominator = math.fsum(gain / math.log2(position + 2) for position, gain in enumerate(ideal))
    return numerator / denominator if denominator else 0.0


class RankingGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        recovery = json.loads((ROOT / "sigir_analysis/qrels_recovery.json").read_text())
        qrels_bytes = (ROOT / recovery["local_path"]).read_bytes()
        original = {}
        for line in qrels_bytes.decode("utf-8").splitlines():
            qid, _, doc, grade = line.split()
            original.setdefault(qid, {})[doc] = int(grade)
        with (ROOT / gate.BASE / "results/queries.csv").open(encoding="utf-8-sig", newline="") as stream:
            queries = [row for row in csv.DictReader(stream) if row["source"] == gate.SOURCE]
        queries.sort(key=lambda row: row["query_key"])
        corpus = sorted({doc for judgments in original.values() for doc in judgments})
        cls.corpus_count = len(corpus)
        exact_rows, hnsw_rows, references, sweep = [], [], [], []
        for query in queries:
            key, qid = query["query_key"], query["raw_query_id"]
            judgments = original[qid]
            exact = sorted((doc for doc in judgments if judgments[doc] > 0), key=lambda doc: (-judgments[doc], doc))[:10]
            if len(exact) < 10:
                exact += [doc for doc in corpus if doc not in exact][:10 - len(exact)]
            exact_score = fixture_ndcg(exact, judgments)
            positive = sum(grade > 0 for grade in judgments.values())
            references.append(dict(query_key=key, source=gate.SOURCE, **{"exact_nDCG@10": exact_score}, positive_qrel_count=positive))
            for rank, doc in enumerate(exact, 1):
                exact_rows.append(dict(query_key=key, query_id=qid, rank=rank, doc_id=doc, score=20.0-rank))
            for ef in gate.EFS:
                approximate = list(exact)
                if ef in (10, 16):
                    replacement = next(doc for doc in corpus if doc not in exact and judgments.get(doc, 0) == 0)
                    approximate[-1] = replacement
                score = fixture_ndcg(approximate, judgments)
                for rank, doc in enumerate(approximate, 1):
                    hnsw_rows.append(dict(query_key=key, query_id=qid, efSearch=ef, rank=rank, doc_id=doc, score=20.0-rank))
                sweep.append(dict(
                    task="TOP10_RANKING", query_key=key, source=gate.SOURCE,
                    search_k=10, efSearch=ef, positive_qrel_count=positive,
                    **{"ANN_Recall@10": len(set(exact) & set(approximate))/10,
                       "nDCG@10": score, "nDCG_loss": max(0.0, exact_score-score)},
                ))
        cls.baseline = {
            recovery["local_path"]: qrels_bytes,
            "sigir_analysis/qrels_recovery.json": json.dumps(recovery).encode(),
            (gate.BASE / "results/queries.csv").as_posix(): csv_bytes(queries),
            (gate.BASE / "research_grade_validation/exact_query_reference.csv").as_posix(): csv_bytes(references),
            (gate.BASE / "research_grade_validation/top10_query_budget_results.csv").as_posix(): csv_bytes(sweep),
            "exact.csv": csv_bytes(exact_rows),
            "hnsw.csv": csv_bytes(hnsw_rows),
            "docids.txt": ("\n".join(corpus)+"\n").encode(),
        }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cura_ranking_gate_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, data in self.baseline.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.count_patch = patch.object(gate, "CORPUS_DOCUMENT_COUNT", self.corpus_count)
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)
        self.manifest = dict(
            original_corpus_document_count=self.corpus_count,
            corpus_docids_path="docids.txt",
            corpus_docids_sha256=gate._sha256(self.root/"docids.txt"),
            exact_rankings_sha256=gate._sha256(self.root/"exact.csv"),
            hnsw_rankings_sha256=gate._sha256(self.root/"hnsw.csv"),
            original_run_provenance="SYNTHETIC UNIT TEST FIXTURE. This is not an original research run.",
            original_corpus_identity_confirmed=True,
        )
        self.write_manifest()
        self.refresh_protocol()

    def write_manifest(self) -> None:
        (self.root/"provenance.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def refresh_rank_hashes(self) -> None:
        for name in ("exact", "hnsw"):
            self.manifest[f"{name}_rankings_sha256"] = gate._sha256(self.root/f"{name}.csv")
        self.write_manifest()

    def refresh_protocol(self) -> None:
        paths = [gate.BASE/"results/queries.csv", gate.BASE/"research_grade_validation/exact_query_reference.csv", gate.BASE/"research_grade_validation/top10_query_budget_results.csv"]
        protocol = dict(input_hashes={path.as_posix(): gate._sha256(self.root/path) for path in paths})
        (self.root/"sigir_analysis/analysis_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    def read_rows(self, name: str) -> list[dict[str, str]]:
        with (self.root/name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def write_rows(self, name: str, rows: list[dict[str, object]]) -> None:
        (self.root/name).write_bytes(csv_bytes(rows))

    def load(self) -> dict:
        return gate.load_and_validate_rankings(self.root, Path("exact.csv"), Path("hnsw.csv"), Path("provenance.json"))

    def assert_blocked(self, code: str) -> gate.RankingValidationError:
        with self.assertRaises(gate.RankingValidationError) as raised:
            self.load()
        self.assertEqual(raised.exception.code, code)
        return raised.exception

    def test_valid_inputs_compare_every_original_query_budget_without_writes(self) -> None:
        before = {path.relative_to(self.root): gate._sha256(path) for path in self.root.rglob("*") if path.is_file()}
        result = self.load()
        after = {path.relative_to(self.root): gate._sha256(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(result["exact"]), 43)
        self.assertEqual(len(result["hnsw"]), 430)
        self.assertTrue(all(len(documents) == 10 for documents in result["exact"].values()))
        self.assertEqual(len(result["reproduction"]), 473)
        self.assertTrue(result["reproduction"].comparison_passed.all())
        self.assertFalse(result["validation"]["original_independent_corpus_checksum_available"])
        self.assertFalse(result["validation"]["experiment_results_written"])

    def test_exact_depth_above_10_is_valid_but_returns_only_top_10(self) -> None:
        rows = self.read_rows("exact.csv")
        corpus = (self.root/"docids.txt").read_text().splitlines()
        for key in sorted({row["query_key"] for row in rows}):
            group = [row for row in rows if row["query_key"] == key]
            extra = dict(group[0], rank="11", doc_id=next(doc for doc in corpus if doc not in {row["doc_id"] for row in group}), score="0")
            rows.append(extra)
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        result = self.load()
        self.assertEqual(result["validation"]["exact_depth"], 11)
        self.assertTrue(all(len(documents) == 10 for documents in result["exact"].values()))

    def test_missing_query_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows = [row for row in rows if row["query_key"] != rows[0]["query_key"]]
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("RANKING_COVERAGE")

    def test_missing_ef_query_group_is_blocked(self) -> None:
        rows = self.read_rows("hnsw.csv")
        rows = [row for row in rows if not (row["query_key"] == rows[0]["query_key"] and row["efSearch"] == "10")]
        self.write_rows("hnsw.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("RANKING_COVERAGE")

    def test_wrong_raw_query_id_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows[0]["query_id"] = "WRONG_QUERY"
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("QUERY_COVERAGE")

    def test_duplicate_rank_is_blocked(self) -> None:
        rows = self.read_rows("hnsw.csv")
        rows.append(dict(rows[0]))
        self.write_rows("hnsw.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("DUPLICATE_RANK")

    def test_duplicate_document_is_blocked(self) -> None:
        rows = self.read_rows("hnsw.csv")
        rows[1]["doc_id"] = rows[0]["doc_id"]
        self.write_rows("hnsw.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("DUPLICATE_DOCUMENT")

    def test_noncontiguous_rank_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows[9]["rank"] = "11"
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("INVALID_RANK")

    def test_nonuniform_exact_depth_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        corpus = (self.root/"docids.txt").read_text().splitlines()
        used = {row["doc_id"] for row in rows[:10]}
        rows.append(dict(rows[0], rank="11", doc_id=next(doc for doc in corpus if doc not in used), score="0"))
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("INVALID_DEPTH")

    def test_extra_schema_field_is_blocked(self) -> None:
        rows = [dict(row, extra="do not silently drop") for row in self.read_rows("exact.csv")]
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("INVALID_SCHEMA")

    def test_nonfinite_score_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows[0]["score"] = "NaN"
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("INVALID_NUMBER")

    def test_empty_document_padding_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows[0]["doc_id"] = ""
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("INVALID_ID")

    def test_document_absent_from_original_map_is_blocked(self) -> None:
        rows = self.read_rows("exact.csv")
        rows[0]["doc_id"] = "NOT_IN_ORIGINAL_CORPUS"
        self.write_rows("exact.csv", rows)
        self.refresh_rank_hashes()
        self.assert_blocked("CORPUS_NOT_VERIFIED")

    def test_missing_map_remains_blocked(self) -> None:
        (self.root/"docids.txt").unlink()
        self.assert_blocked("MISSING_INPUT")

    def test_false_corpus_identity_attestation_is_blocked(self) -> None:
        self.manifest["original_corpus_identity_confirmed"] = False
        self.write_manifest()
        self.assert_blocked("CORPUS_NOT_VERIFIED")

    def test_rankings_checksum_mismatch_is_blocked(self) -> None:
        with (self.root/"exact.csv").open("ab") as stream:
            stream.write(b"\n")
        self.assert_blocked("CHECKSUM_MISMATCH")

    def test_altered_original_qrels_are_blocked(self) -> None:
        metadata = json.loads((self.root/"sigir_analysis/qrels_recovery.json").read_text())
        with (self.root/metadata["local_path"]).open("ab") as stream:
            stream.write(b"\n")
        self.assert_blocked("QRELS_NOT_VERIFIED")

    def test_unpinned_saved_reference_changes_are_blocked(self) -> None:
        path = self.root/gate.BASE/"research_grade_validation/exact_query_reference.csv"
        with path.open("ab") as stream:
            stream.write(b"\n")
        self.assert_blocked("INVALID_REFERENCE")

    def test_single_query_exact_metric_mismatch_blocks_all_density_results(self) -> None:
        name = (gate.BASE/"research_grade_validation/exact_query_reference.csv").as_posix()
        rows = self.read_rows(name)
        rows[0]["exact_nDCG@10"] = str(float(rows[0]["exact_nDCG@10"]) + 0.01)
        self.write_rows(name, rows)
        self.refresh_protocol()
        error = self.assert_blocked("FULL_METRIC_MISMATCH")
        self.assertEqual(error.details["failed_rows"], 1)
        self.assertEqual(len(error.reproduction), 473)

    def test_single_budget_recall_mismatch_is_not_hidden_by_mean_match(self) -> None:
        name = (gate.BASE/"research_grade_validation/top10_query_budget_results.csv").as_posix()
        rows = self.read_rows(name)
        rows[0]["ANN_Recall@10"] = "0.8"
        self.write_rows(name, rows)
        self.refresh_protocol()
        error = self.assert_blocked("FULL_METRIC_MISMATCH")
        self.assertEqual(error.details["failed_rows"], 1)
        self.assertEqual(error.details["total_rows"], 473)

    def test_known_graded_ndcg_uses_all_positive_judgments_for_idcg(self) -> None:
        judgments = {"a": 3, "b": 2, "c": 1, "d": 0}
        expected = (3 + 7/math.log2(3)) / (7 + 3/math.log2(3) + 1/math.log2(4))
        self.assertAlmostEqual(gate.ndcg_at_10(["b", "a", "unjudged"], judgments), expected, places=15)
        self.assertEqual(gate.ndcg_at_10(["unjudged"], judgments), 0)
        self.assertEqual(gate.ndcg_at_10(["unjudged"], {}), 0)


if __name__ == "__main__":
    unittest.main()
