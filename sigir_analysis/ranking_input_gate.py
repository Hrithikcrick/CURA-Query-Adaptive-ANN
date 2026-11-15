"""Read-only gate for restored ORIGINAL TREC-2019 ranked-document outputs.

This module never retrieves documents, loads a model/index, trains a controller,
or writes experiment results. Missing or inconsistent inputs raise
``RankingValidationError``. Callers must not produce density-study observations
unless ``load_and_validate_rankings`` returns successfully.

Normalized CSV schemas (column order is immaterial; extra columns are rejected):

* Exact: query_key,query_id,rank,doc_id,score
* HNSW: query_key,query_id,efSearch,rank,doc_id,score

Ranks are 1-based integers. Exact depth must be uniform, between 10 and 100;
HNSW depth is exactly 10. All 43 original TREC query identities are required,
with HNSW at every ef in EFS. Scores must be finite. Raw document IDs are
strings, not embedding row numbers or IDs from another corpus.

Required JSON provenance manifest fields:

    original_corpus_document_count: 1000629
    corpus_docids_path: "path/to/restored/original/docids.txt"
    corpus_docids_sha256: "<SHA256 of original ordered document map>"
    exact_rankings_sha256: "<SHA256 of normalized exact CSV>"
    hnsw_rankings_sha256: "<SHA256 of normalized HNSW CSV>"
    original_run_provenance: "<restorer's concrete account of run origin>"
    original_corpus_identity_confirmed: true

Paths in the manifest are resolved relative to the supplied workspace root.
The document map is UTF-8, one nonempty raw document ID per line, in original
FAISS row order. Blank, duplicate, padded or whitespace-containing IDs fail.
All ranked IDs and original judged IDs must belong to this map.

The original independent corpus-map checksum is unavailable. Thus a supplied
checksum plus an explicit restorer attestation establishes internal consistency,
not independent cryptographic proof of the historical corpus. This limitation
is always recorded in returned validation metadata. Per-query FULL metrics
must additionally reproduce the archived original references before returning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256)
QUERY_COUNT = 43
CORPUS_DOCUMENT_COUNT = 1_000_629
IR_TOLERANCE = 0.025
NDCG_TOLERANCE = 1e-6
RECALL_TOLERANCE = 1e-9
ORIGINAL_QRELS_SHA256 = (
    "8a1f10d550732e4cd91d7fc49846a3784de4040972f583e69285a88f3c5fee92"
)
ORIGINAL_QRELS_ROWS = 9260
SOURCE = "trec-dl-2019"
BASE = Path("msmarco_large/msmarco_1000000_q1000")
EXACT_COLUMNS = frozenset(("query_key", "query_id", "rank", "doc_id", "score"))
HNSW_COLUMNS = EXACT_COLUMNS | {"efSearch"}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class RankingValidationError(ValueError):
    """A failed gate; no density observations may be produced after this error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "INPUT_VALIDATION_FAILED",
        details: dict[str, Any] | None = None,
        reproduction: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.reproduction = reproduction


def _require(condition: Any, message: str, code: str = "INPUT_VALIDATION_FAILED") -> None:
    if not condition:
        raise RankingValidationError(message, code=code)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"Required original input is missing: {path}", "MISSING_INPUT")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}", "INVALID_JSON")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required original input is missing: {path}", "MISSING_INPUT")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RankingValidationError(f"Invalid JSON at {path}: {error}", code="INVALID_JSON") from error
    _require(isinstance(value, dict), f"Expected a JSON object: {path}", "INVALID_JSON")
    return value


def _read_csv(path: Path, expected_columns: frozenset[str] | None = None) -> list[dict[str, str]]:
    _require(path.is_file(), f"Required original input is missing: {path}", "MISSING_INPUT")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            columns = next(reader, None)
            _require(columns is not None, f"Empty CSV: {path}", "INVALID_SCHEMA")
            _require(len(columns) == len(set(columns)), f"Duplicate CSV column: {path}", "INVALID_SCHEMA")
            if expected_columns is not None:
                _require(
                    set(columns) == expected_columns,
                    f"Invalid schema for {path.name}: expected {sorted(expected_columns)}, got {columns}",
                    "INVALID_SCHEMA",
                )
            records = []
            for line, fields in enumerate(reader, 2):
                _require(
                    len(fields) == len(columns),
                    f"CSV row {line} has {len(fields)} fields; expected {len(columns)}: {path}",
                    "INVALID_SCHEMA",
                )
                records.append(dict(zip(columns, fields)))
            _require(records, f"CSV has no observations: {path}", "EMPTY_RANKINGS")
            return records
    except (UnicodeError, csv.Error) as error:
        raise RankingValidationError(f"Invalid CSV at {path}: {error}", code="INVALID_SCHEMA") from error


def _id(value: str, label: str) -> str:
    _require(
        bool(value) and not any(char.isspace() for char in value) and "\x00" not in value,
        f"Empty, padded or invalid {label}: {value!r}",
        "INVALID_ID",
    )
    _require(value.casefold() not in {"nan", "none", "null"}, f"Placeholder {label}: {value!r}", "INVALID_ID")
    return value


def _integer(value: str, label: str) -> int:
    _require(bool(_INTEGER.fullmatch(value)), f"Expected canonical integer {label}, got {value!r}", "INVALID_INTEGER")
    return int(value)


def _finite(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (ValueError, TypeError) as error:
        raise RankingValidationError(f"Expected finite {label}, got {value!r}", code="INVALID_NUMBER") from error
    _require(math.isfinite(parsed), f"Expected finite {label}, got {value!r}", "INVALID_NUMBER")
    return parsed


def _queries(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    required = {"query_key", "raw_query_id", "source"}
    _require(required <= rows[0].keys(), f"Original query CSV lacks {sorted(required)}", "INVALID_SCHEMA")
    selected = [row for row in rows if row["source"] == SOURCE]
    _require(len(selected) == QUERY_COUNT, f"Expected exactly {QUERY_COUNT} original TREC queries, got {len(selected)}", "QUERY_COVERAGE")
    mapping: dict[str, str] = {}
    raw_ids: set[str] = set()
    for row in selected:
        key = _id(row["query_key"], "query_key")
        qid = _id(row["raw_query_id"], "raw_query_id")
        _require(key not in mapping and qid not in raw_ids, "Original TREC query mapping is not a bijection", "QUERY_COVERAGE")
        mapping[key] = qid
        raw_ids.add(qid)
    return dict(sorted(mapping.items()))


def _qrels(root: Path, recovery: dict[str, Any], mapping: dict[str, str]) -> tuple[Path, dict[str, dict[str, int]]]:
    _require(recovery.get("status") == "RECOVERED", "Original qrels have not been recovered", "QRELS_NOT_VERIFIED")
    _require(recovery.get("sha256") == ORIGINAL_QRELS_SHA256, "Qrels recovery metadata does not identify the original NIST file", "QRELS_NOT_VERIFIED")
    _require(recovery.get("original_query_identity_match") is True, "Original qrels/query identity has not been verified", "QRELS_NOT_VERIFIED")
    _require(isinstance(recovery.get("local_path"), str) and recovery["local_path"], "Qrels recovery metadata lacks local_path", "QRELS_NOT_VERIFIED")
    path = _resolve(root, recovery["local_path"])
    _require(_sha256(path) == ORIGINAL_QRELS_SHA256, "Original NIST qrels checksum mismatch", "QRELS_NOT_VERIFIED")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        _require(len(fields) == 4, f"Invalid qrels row {line_number}", "QRELS_NOT_VERIFIED")
        qid, _, doc, value = fields
        _id(qid, "qrels query_id")
        _id(doc, "qrels doc_id")
        grade = _integer(value, "qrels grade")
        _require(grade in (0, 1, 2, 3), f"Invalid qrels grade at row {line_number}", "QRELS_NOT_VERIFIED")
        _require(doc not in qrels[qid], f"Duplicate qrels pair at row {line_number}", "QRELS_NOT_VERIFIED")
        qrels[qid][doc] = grade
        count += 1
    _require(count == ORIGINAL_QRELS_ROWS, f"Expected {ORIGINAL_QRELS_ROWS} original judgments, got {count}", "QRELS_NOT_VERIFIED")
    _require(set(qrels) == set(mapping.values()), "Original qrels do not match exactly the 43 raw query IDs", "QUERY_COVERAGE")
    return path, dict(qrels)


def _provenance(root: Path, path: Path, exact_path: Path, hnsw_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = _read_json(path)
    required = {
        "original_corpus_document_count", "corpus_docids_path", "corpus_docids_sha256",
        "exact_rankings_sha256", "hnsw_rankings_sha256", "original_run_provenance",
        "original_corpus_identity_confirmed",
    }
    _require(required <= manifest.keys(), f"Provenance manifest lacks: {sorted(required - manifest.keys())}", "PROVENANCE_NOT_VERIFIED")
    _require(
        type(manifest["original_corpus_document_count"]) is int
        and manifest["original_corpus_document_count"] == CORPUS_DOCUMENT_COUNT,
        f"Provenance must identify the original {CORPUS_DOCUMENT_COUNT}-document controlled corpus",
        "CORPUS_NOT_VERIFIED",
    )
    _require(manifest["original_corpus_identity_confirmed"] is True, "Restorer must explicitly attest original_corpus_identity_confirmed=true", "CORPUS_NOT_VERIFIED")
    _require(
        isinstance(manifest["original_run_provenance"], str) and bool(manifest["original_run_provenance"].strip()),
        "A nonempty original_run_provenance account is required", "PROVENANCE_NOT_VERIFIED",
    )
    _require(isinstance(manifest["corpus_docids_path"], str) and bool(manifest["corpus_docids_path"].strip()), "Original ordered corpus map is required", "CORPUS_NOT_VERIFIED")
    for key in ("corpus_docids_sha256", "exact_rankings_sha256", "hnsw_rankings_sha256"):
        _require(isinstance(manifest[key], str) and bool(_SHA256.fullmatch(manifest[key])), f"Invalid manifest SHA256: {key}", "PROVENANCE_NOT_VERIFIED")
    corpus_path = _resolve(root, manifest["corpus_docids_path"])
    for input_path, hash_key in (
        (corpus_path, "corpus_docids_sha256"),
        (exact_path, "exact_rankings_sha256"),
        (hnsw_path, "hnsw_rankings_sha256"),
    ):
        _require(_sha256(input_path) == manifest[hash_key].lower(), f"Manifest checksum mismatch for {input_path}", "CHECKSUM_MISMATCH")
    return manifest, corpus_path


def _corpus_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                value = _id(line.rstrip("\r\n"), f"corpus doc_id at row {line_number}")
                _require(value not in ids, f"Duplicate original corpus document ID at row {line_number}: {value}", "CORPUS_NOT_VERIFIED")
                ids.add(value)
    except UnicodeError as error:
        raise RankingValidationError(f"Original corpus map is not UTF-8: {path}", code="CORPUS_NOT_VERIFIED") from error
    _require(len(ids) == CORPUS_DOCUMENT_COUNT, f"Original corpus map has {len(ids)} IDs; expected {CORPUS_DOCUMENT_COUNT}", "CORPUS_NOT_VERIFIED")
    return ids


def _rankings(
    path: Path,
    mapping: dict[str, str],
    corpus_ids: set[str],
    *,
    hnsw: bool,
) -> tuple[dict[Any, list[str]], int]:
    records = _read_csv(path, HNSW_COLUMNS if hnsw else EXACT_COLUMNS)
    groups: dict[Any, dict[int, str]] = defaultdict(dict)
    docs_per_group: dict[Any, set[str]] = defaultdict(set)
    for line, row in enumerate(records, 2):
        key = _id(row["query_key"], "query_key")
        qid = _id(row["query_id"], "query_id")
        _require(key in mapping and mapping[key] == qid, f"Query/raw-ID mismatch at {path.name}:{line}: {key}, {qid}", "QUERY_COVERAGE")
        doc = _id(row["doc_id"], "doc_id")
        _require(doc in corpus_ids, f"Ranked doc_id is absent from the original corpus map at {path.name}:{line}: {doc}", "CORPUS_NOT_VERIFIED")
        rank = _integer(row["rank"], "rank")
        _require(rank > 0, f"Ranks must start at 1: {path.name}:{line}", "INVALID_RANK")
        _finite(row["score"], "retrieval score")
        if hnsw:
            ef = _integer(row["efSearch"], "efSearch")
            _require(ef in EFS, f"Unexpected efSearch at {path.name}:{line}: {ef}", "EF_COVERAGE")
            group = (key, ef)
        else:
            group = key
        _require(rank not in groups[group], f"Duplicate query/ef/rank at {path.name}:{line}", "DUPLICATE_RANK")
        _require(doc not in docs_per_group[group], f"Duplicate document within ranking at {path.name}:{line}: {doc}", "DUPLICATE_DOCUMENT")
        groups[group][rank] = doc
        docs_per_group[group].add(doc)
    expected = {(key, ef) for key in mapping for ef in EFS} if hnsw else set(mapping)
    _require(set(groups) == expected, f"Incomplete {'HNSW query/ef' if hnsw else 'exact query'} ranking coverage: {len(groups)}/{len(expected)} groups", "RANKING_COVERAGE")
    depths = {len(group) for group in groups.values()}
    _require(len(depths) == 1, f"Nonuniform ranking depths: {sorted(depths)}", "INVALID_DEPTH")
    depth = next(iter(depths))
    _require(depth == 10 if hnsw else 10 <= depth <= 100, f"Invalid {'HNSW' if hnsw else 'exact'} ranking depth: {depth}", "INVALID_DEPTH")
    ordered = {}
    for group in sorted(groups):
        values = groups[group]
        _require(set(values) == set(range(1, depth + 1)), f"Noncontiguous 1-based ranks for {group}", "INVALID_RANK")
        ordered[group] = [values[rank] for rank in range(1, 11)]
    return ordered, depth


def ndcg_at_10(documents: list[str], judgments: dict[str, int]) -> float:
    """Original graded nDCG convention; unjudged=0 and IDCG uses all qrels."""
    dcg = sum((2 ** judgments.get(doc, 0) - 1) / math.log2(rank + 2) for rank, doc in enumerate(documents[:10]))
    ideal = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)[:10]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _saved_metrics(
    reference_path: Path,
    sweep_path: Path,
    mapping: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, int], dict[str, str]]]:
    reference_rows = _read_csv(reference_path)
    sweep_rows = _read_csv(sweep_path)
    reference_required = {"query_key", "source", "exact_nDCG@10", "positive_qrel_count"}
    sweep_required = {"query_key", "source", "search_k", "efSearch", "ANN_Recall@10", "nDCG@10", "nDCG_loss", "positive_qrel_count"}
    _require(reference_required <= reference_rows[0].keys(), "Saved exact reference has an invalid schema", "INVALID_REFERENCE")
    _require(sweep_required <= sweep_rows[0].keys(), "Saved HNSW sweep has an invalid schema", "INVALID_REFERENCE")
    reference = {}
    for row in reference_rows:
        if row["source"] != SOURCE:
            continue
        key = row["query_key"]
        _require(key in mapping and key not in reference, f"Unexpected or duplicate exact reference query: {key}", "INVALID_REFERENCE")
        positive = sum(grade > 0 for grade in qrels[mapping[key]].values())
        _require(_integer(row["positive_qrel_count"], "reference positive count") == positive, f"Saved exact positive count differs from original qrels: {key}", "INVALID_REFERENCE")
        _finite(row["exact_nDCG@10"], "saved exact nDCG")
        reference[key] = row
    _require(set(reference) == set(mapping), "Saved exact reference does not cover exactly the original 43 queries", "INVALID_REFERENCE")
    sweep = {}
    for row in sweep_rows:
        if row["source"] != SOURCE:
            continue
        key = row["query_key"]
        _require(key in mapping, f"Unexpected saved sweep query: {key}", "INVALID_REFERENCE")
        ef = _integer(row["efSearch"], "saved efSearch")
        group = (key, ef)
        _require(ef in EFS and group not in sweep, f"Unexpected or duplicate saved sweep group: {group}", "INVALID_REFERENCE")
        _require(_integer(row["search_k"], "saved search_k") == 10, f"Saved sweep search depth is not 10: {group}", "INVALID_REFERENCE")
        if "task" in row:
            _require(row["task"] == "TOP10_RANKING", f"Unexpected saved sweep task: {group}", "INVALID_REFERENCE")
        positive = sum(grade > 0 for grade in qrels[mapping[key]].values())
        _require(_integer(row["positive_qrel_count"], "sweep positive count") == positive, f"Saved sweep positive count differs from qrels: {group}", "INVALID_REFERENCE")
        for name in ("ANN_Recall@10", "nDCG@10", "nDCG_loss"):
            _finite(row[name], f"saved {name}")
        sweep[group] = row
    _require(set(sweep) == {(key, ef) for key in mapping for ef in EFS}, "Saved sweep does not contain exactly all 430 original query/budget pairs", "INVALID_REFERENCE")
    return reference, sweep


def _reproduce(
    mapping: dict[str, str],
    qrels: dict[str, dict[str, int]],
    exact: dict[str, list[str]],
    hnsw: dict[tuple[str, int], list[str]],
    reference: dict[str, dict[str, str]],
    sweep: dict[tuple[str, int], dict[str, str]],
) -> pd.DataFrame:
    rows = []
    exact_scores = {}
    for key, qid in mapping.items():
        score = ndcg_at_10(exact[key], qrels[qid])
        exact_scores[key] = score
        saved = float(reference[key]["exact_nDCG@10"])
        difference = abs(saved - score)
        rows.append(dict(
            method="Exact", query_key=key, query_id=qid, efSearch=None,
            archived_nDCG_at_10=saved, recomputed_nDCG_at_10=score,
            nDCG_absolute_difference=difference,
            archived_nDCG_loss=None, recomputed_nDCG_loss=None, loss_absolute_difference=None,
            archived_ANN_Recall_at_10=None, recomputed_ANN_Recall_at_10=None, recall_absolute_difference=None,
            archived_IR_violation=None, recomputed_IR_violation=None,
            comparison_completed=True, comparison_passed=difference <= NDCG_TOLERANCE,
        ))
    for key, qid in mapping.items():
        for ef in EFS:
            values = sweep[(key, ef)]
            score = ndcg_at_10(hnsw[(key, ef)], qrels[qid])
            loss = max(0.0, exact_scores[key] - score)
            recall = len(set(exact[key]) & set(hnsw[(key, ef)])) / 10.0
            saved_score = float(values["nDCG@10"])
            saved_loss = float(values["nDCG_loss"])
            saved_recall = float(values["ANN_Recall@10"])
            score_difference = abs(saved_score - score)
            loss_difference = abs(saved_loss - loss)
            recall_difference = abs(saved_recall - recall)
            violation = loss > IR_TOLERANCE
            saved_violation = saved_loss > IR_TOLERANCE
            passed = (
                score_difference <= NDCG_TOLERANCE
                and loss_difference <= NDCG_TOLERANCE
                and recall_difference <= RECALL_TOLERANCE
                and violation == saved_violation
            )
            rows.append(dict(
                method="HNSW", query_key=key, query_id=qid, efSearch=ef,
                archived_nDCG_at_10=saved_score, recomputed_nDCG_at_10=score,
                nDCG_absolute_difference=score_difference,
                archived_nDCG_loss=saved_loss, recomputed_nDCG_loss=loss, loss_absolute_difference=loss_difference,
                archived_ANN_Recall_at_10=saved_recall, recomputed_ANN_Recall_at_10=recall, recall_absolute_difference=recall_difference,
                archived_IR_violation=saved_violation, recomputed_IR_violation=violation,
                comparison_completed=True, comparison_passed=passed,
            ))
    reproduction = pd.DataFrame(rows)
    failures = reproduction[~reproduction.comparison_passed]
    if not failures.empty:
        raise RankingValidationError(
            f"FULL metric reproduction failed for {len(failures)} of {len(reproduction)} query/method/budget rows; density evaluation is blocked",
            code="FULL_METRIC_MISMATCH",
            details={"failed_rows": len(failures), "total_rows": len(reproduction), "examples": failures.head(8).to_dict("records")},
            reproduction=reproduction,
        )
    return reproduction


def load_and_validate_rankings(
    root: Path,
    exact_path: Path,
    hnsw_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    """Validate all original IDs and FULL metrics, returning frozen top-10 lists.

    Return keys: ``exact`` (query_key -> 10 document strings), ``hnsw``
    ((query_key, efSearch) -> 10 document strings), ``query_mapping``, ``qrels``
    (raw query ID -> original grade mapping), ``ranking_hashes``,
    ``source_hashes``, ``reproduction`` (473-row DataFrame), and ``validation``.
    All file access is read-only. A missing corpus map is a blocking error.
    """
    root = Path(root).resolve()
    _require(root.is_dir(), f"Workspace root does not exist: {root}", "MISSING_INPUT")
    exact_path = _resolve(root, exact_path)
    hnsw_path = _resolve(root, hnsw_path)
    provenance_path = _resolve(root, provenance_path)
    query_path = root / BASE / "results/queries.csv"
    reference_path = root / BASE / "research_grade_validation/exact_query_reference.csv"
    sweep_path = root / BASE / "research_grade_validation/top10_query_budget_results.csv"
    recovery_path = root / "sigir_analysis/qrels_recovery.json"
    protocol_path = root / "sigir_analysis/analysis_protocol.json"
    recovery = _read_json(recovery_path)
    mapping = _queries(query_path)
    qrels_path, qrels = _qrels(root, recovery, mapping)
    manifest, corpus_path = _provenance(root, provenance_path, exact_path, hnsw_path)
    paths = [exact_path, hnsw_path, provenance_path, corpus_path, query_path, reference_path, sweep_path, recovery_path, qrels_path, protocol_path]
    hashes = {_label(root, path): _sha256(path) for path in paths}
    protocol = _read_json(protocol_path)
    original_hashes = protocol.get("input_hashes", {})
    _require(isinstance(original_hashes, dict), "Analysis protocol lacks original input hashes", "INVALID_REFERENCE")
    for path in (query_path, reference_path, sweep_path):
        label = _label(root, path)
        _require(original_hashes.get(label) == hashes[label], f"Original archived source checksum differs from analysis protocol: {label}", "INVALID_REFERENCE")
    corpus_ids = _corpus_ids(corpus_path)
    judged = {doc for judgments in qrels.values() for doc in judgments}
    missing_judged = judged - corpus_ids
    _require(not missing_judged, f"Original corpus map omits {len(missing_judged)} original TREC judged IDs, which the original construction forced into the corpus", "CORPUS_NOT_VERIFIED")
    exact, exact_depth = _rankings(exact_path, mapping, corpus_ids, hnsw=False)
    hnsw, hnsw_depth = _rankings(hnsw_path, mapping, corpus_ids, hnsw=True)
    reference, sweep = _saved_metrics(reference_path, sweep_path, mapping, qrels)
    reproduction = _reproduce(mapping, qrels, exact, hnsw, reference, sweep)
    for path in paths:
        _require(_sha256(path) == hashes[_label(root, path)], f"Input changed during validation: {path}", "INPUT_CHANGED")
    exact_serialized = [[key, exact[key]] for key in sorted(exact)]
    hnsw_serialized = [[key, ef, hnsw[(key, ef)]] for key, ef in sorted(hnsw)]
    ranking_hashes = dict(
        exact_file_sha256=hashes[_label(root, exact_path)],
        hnsw_file_sha256=hashes[_label(root, hnsw_path)],
        exact_top10_sha256=_canonical_sha256(exact_serialized),
        hnsw_top10_sha256=_canonical_sha256(hnsw_serialized),
        combined_top10_sha256=_canonical_sha256(dict(exact=exact_serialized, hnsw=hnsw_serialized)),
    )
    validation = dict(
        status="PASS", exact_query_count=len(exact), hnsw_query_budget_count=len(hnsw),
        queries_expected=QUERY_COUNT, queries_found=len(mapping),
        ef_values_expected=list(EFS), ef_values_found=list(EFS),
        exact_depth=exact_depth, hnsw_depth=hnsw_depth, evaluation_depth=10,
        raw_query_id_bijection_verified=True, duplicate_rankings_absent=True,
        duplicate_documents_absent=True, contiguous_ranks_verified=True,
        finite_scores_verified=True, document_id_membership_verified=True,
        corpus_document_count=len(corpus_ids), corpus_order_checksum_verified=True,
        corpus_identity_verified=True,
        corpus_identity_evidence="explicit restorer attestation, supplied original ordered-map checksum/count, original judged-ID membership and FULL saved-metric consistency",
        original_corpus_identity_confirmed_by_restorer=True,
        original_independent_corpus_checksum_available=False,
        corpus_identity_limitation="No independent historical corpus-map checksum is available. The supplied checksum/count and restorer attestation establish consistency, not independent proof of historical corpus identity.",
        original_run_provenance=manifest["original_run_provenance"],
        original_qrels_sha256=ORIGINAL_QRELS_SHA256,
        full_metric_reproduction_verified=True,
        full_reproduction_rows=len(reproduction),
        full_exact_comparisons=QUERY_COUNT, full_hnsw_comparisons=QUERY_COUNT * len(EFS),
        ndcg_and_loss_absolute_tolerance=NDCG_TOLERANCE,
        recall_absolute_tolerance=RECALL_TOLERANCE,
        violation_comparison="identical per query/ef; loss > 0.025",
        retrieval_executed=False, embeddings_regenerated=False,
        index_rebuilt=False, controller_retrained=False, experiment_results_written=False,
    )
    return dict(
        exact=exact, hnsw=hnsw, query_mapping=mapping, qrels=qrels,
        ranking_hashes=ranking_hashes, source_hashes=hashes,
        reproduction=reproduction, validation=validation,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--hnsw", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = load_and_validate_rankings(args.root, args.exact, args.hnsw, args.provenance)
    except RankingValidationError as error:
        print(json.dumps(dict(status="BLOCKED", code=error.code, reason=str(error)), indent=2))
        return 2
    print(json.dumps(dict(validation=result["validation"], ranking_hashes=result["ranking_hashes"], source_hashes=result["source_hashes"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
