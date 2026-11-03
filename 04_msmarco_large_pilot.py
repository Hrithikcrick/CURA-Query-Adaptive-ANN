# ============================================================
# 04_msmarco_large_pilot.py
#
# COL764 PROJECT
# LARGE-SCALE MS MARCO HNSW PILOT
#
# Goal:
# Validate whether the ANN-recall-vs-IR-utility gap observed
# on SciFact survives at a substantially larger HNSW scale.
#
# This script is a LARGE PILOT.
# It does NOT yet implement DARTH or Patience in Proximity.
# Those will be evaluated after this phenomenon is validated.
#
# ============================================================


# ============================================================
# 0. WINDOWS TEMPORARY-FILE FIX
#
# IMPORTANT:
# This must appear before "import ir_datasets".
# ============================================================

import os

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

IR_DATASETS_TMP_DIR = os.path.join(
    SCRIPT_DIR,
    "ir_datasets_tmp"
)

os.makedirs(
    IR_DATASETS_TMP_DIR,
    exist_ok=True
)

os.environ["IR_DATASETS_TMP"] = IR_DATASETS_TMP_DIR
os.environ["TEMP"] = IR_DATASETS_TMP_DIR
os.environ["TMP"] = IR_DATASETS_TMP_DIR


# ============================================================
# 1. IMPORTS
# ============================================================

import gc
import gzip
import json
import math
import time
import random
import hashlib
import argparse
import urllib.request

from collections import defaultdict

import numpy as np
import pandas as pd

import faiss
import torch
import ir_datasets

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Large-scale MS MARCO HNSW pilot"
)

parser.add_argument(
    "--n_docs",
    type=int,
    default=200_000,
    help="Approximate number of MS MARCO passages."
)

parser.add_argument(
    "--n_dev_queries",
    type=int,
    default=300,
    help="Number of MS MARCO dev/small queries."
)

parser.add_argument(
    "--repeats",
    type=int,
    default=1,
    help="Number of latency repetitions."
)

parser.add_argument(
    "--threads",
    type=int,
    default=4,
    help="CPU threads for building/reference search."
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)

CPU_THREADS = max(
    1,
    min(
        args.threads,
        os.cpu_count() or 4
    )
)

faiss.omp_set_num_threads(
    CPU_THREADS
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

TOTAL_MSMARCO_DOCS = 8_841_823

TARGET_DOCS = min(
    args.n_docs,
    TOTAL_MSMARCO_DOCS
)

N_DEV_QUERIES = args.n_dev_queries


MODEL_NAME = (
    "sentence-transformers/"
    "msmarco-MiniLM-L6-cos-v5"
)


ANN_K = 10

SEARCH_K = 100


# Since we retrieve top-100 documents,
# use efSearch >= 100.

EF_VALUES = [
    100,
    150,
    200,
    300,
    400,
    600,
    800
]


HNSW_M = 32

EF_CONSTRUCTION = 200


# ============================================================
# QUALITY TARGETS
# ============================================================

TARGET_ANN_RECALL = 0.90


# Exploratory Phase-1 quality tolerances.
# These will later be swept in proper ablations.

NDCG_LOSS_TOLERANCE = 0.05

RECALL100_LOSS_TOLERANCE = 0.10


MAX_FIXED_VIOLATION = 0.05


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 6. RUN DIRECTORY
# ============================================================

RUN_NAME = (
    f"msmarco_{TARGET_DOCS}"
    f"_q{N_DEV_QUERIES}"
)


ROOT_DIR = os.path.join(
    SCRIPT_DIR,
    "msmarco_large",
    RUN_NAME
)


CACHE_DIR = os.path.join(
    ROOT_DIR,
    "cache"
)


RESULTS_DIR = os.path.join(
    ROOT_DIR,
    "results"
)


LOCAL_TREC_DIR = os.path.join(
    SCRIPT_DIR,
    "local_data",
    "trec2019"
)


os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    LOCAL_TREC_DIR,
    exist_ok=True
)


# ============================================================
# 7. PRINT RUN CONFIGURATION
# ============================================================

print("=" * 100)

print(
    "MS MARCO LARGE HNSW PILOT"
)

print("=" * 100)

print(
    "Target passages:",
    TARGET_DOCS
)

print(
    "MS MARCO dev queries:",
    N_DEV_QUERIES
)

print(
    "Timing repeats:",
    args.repeats
)

print(
    "Build/reference threads:",
    CPU_THREADS
)

print(
    "Torch device:",
    DEVICE
)

print(
    "Output directory:",
    ROOT_DIR
)

print(
    "Temp directory:",
    IR_DATASETS_TMP_DIR
)


if DEVICE == "cpu":

    print(
        "\nNOTE: Encoder is running on CPU."
    )

    print(
        "Corpus encoding will be the slowest step."
    )


# ============================================================
# 8. LOCAL TREC-DL FILES
#
# Do not use ir_datasets to download these tiny files because
# Windows file locking caused repeated os.replace failures.
# ============================================================

TREC_QUERY_FILE = os.path.join(
    LOCAL_TREC_DIR,
    "msmarco-test2019-queries.tsv.gz"
)


TREC_QRELS_FILE = os.path.join(
    LOCAL_TREC_DIR,
    "2019qrels-pass.txt"
)


TREC_QUERY_URL = (
    "https://msmarco.z22.web.core.windows.net/"
    "msmarcoranking/"
    "msmarco-test2019-queries.tsv.gz"
)


TREC_QRELS_URL = (
    "https://trec.nist.gov/data/deep/"
    "2019qrels-pass.txt"
)


def ensure_local_file(
    path,
    url
):

    if os.path.exists(path):

        if os.path.getsize(path) > 0:

            print(
                "Using local file:",
                path
            )

            return


    print(
        "Downloading:",
        url
    )

    urllib.request.urlretrieve(
        url,
        path
    )


ensure_local_file(
    TREC_QUERY_FILE,
    TREC_QUERY_URL
)


ensure_local_file(
    TREC_QRELS_FILE,
    TREC_QRELS_URL
)


# ============================================================
# 9. LOAD MS MARCO DEV/SMALL
# ============================================================

print(
    "\nLoading MS MARCO dev/small..."
)


dev_dataset = ir_datasets.load(
    "msmarco-passage/dev/small"
)


dev_query_text = {}


for query in dev_dataset.queries_iter():

    dev_query_text[
        str(query.query_id)
    ] = query.text


dev_qrels_raw = defaultdict(
    dict
)


for qrel in dev_dataset.qrels_iter():

    dev_qrels_raw[
        str(qrel.query_id)
    ][
        str(qrel.doc_id)
    ] = int(
        qrel.relevance
    )


eligible_dev_qids = [

    qid

    for qid
    in dev_qrels_raw

    if qid
    in dev_query_text
]


rng = random.Random(
    SEED
)


rng.shuffle(
    eligible_dev_qids
)


selected_dev_qids = (

    eligible_dev_qids[
        :min(
            N_DEV_QUERIES,
            len(
                eligible_dev_qids
            )
        )
    ]
)


print(
    "Available judged dev queries:",
    len(
        eligible_dev_qids
    )
)


print(
    "Selected dev queries:",
    len(
        selected_dev_qids
    )
)


# ============================================================
# 10. LOAD LOCAL TREC-DL-2019 QUERIES
# ============================================================

print(
    "\nLoading local TREC-DL-2019..."
)


trec_query_text = {}


with gzip.open(
    TREC_QUERY_FILE,
    "rt",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.rstrip(
            "\n"
        )


        if not line:

            continue


        parts = line.split(
            "\t",
            1
        )


        if len(parts) != 2:

            continue


        qid = str(
            parts[0]
        )


        text = parts[1]


        trec_query_text[
            qid
        ] = text


# ============================================================
# 11. LOAD LOCAL TREC-DL-2019 QRELS
# ============================================================

trec_qrels_raw = defaultdict(
    dict
)


with open(
    TREC_QRELS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()


        if not line:

            continue


        parts = line.split()


        if len(parts) < 4:

            continue


        qid = str(
            parts[0]
        )


        doc_id = str(
            parts[2]
        )


        relevance = int(
            parts[3]
        )


        trec_qrels_raw[
            qid
        ][
            doc_id
        ] = relevance


selected_trec_qids = sorted([

    qid

    for qid
    in trec_qrels_raw.keys()

    if qid
    in trec_query_text
])


print(
    "Queries in TREC query file:",
    len(
        trec_query_text
    )
)


print(
    "Deeply judged TREC queries:",
    len(
        selected_trec_qids
    )
)


print(
    "TREC qrel entries:",
    sum(
        len(value)
        for value
        in trec_qrels_raw.values()
    )
)


# ============================================================
# 12. BUILD UNIFIED QUERY SET
# ============================================================

query_records = []

qrels = {}


# ------------------------------------------------------------
# MS MARCO dev
# ------------------------------------------------------------

for qid in selected_dev_qids:

    internal_qid = (
        "dev::"
        +
        qid
    )


    query_records.append({

        "query_key":
            internal_qid,

        "raw_query_id":
            qid,

        "source":
            "msmarco-dev-small",

        "query_text":
            dev_query_text[
                qid
            ]
    })


    qrels[
        internal_qid
    ] = dev_qrels_raw[
        qid
    ]


# ------------------------------------------------------------
# TREC DL 2019
# ------------------------------------------------------------

for qid in selected_trec_qids:

    internal_qid = (
        "trec19::"
        +
        qid
    )


    query_records.append({

        "query_key":
            internal_qid,

        "raw_query_id":
            qid,

        "source":
            "trec-dl-2019",

        "query_text":
            trec_query_text[
                qid
            ]
    })


    qrels[
        internal_qid
    ] = trec_qrels_raw[
        qid
    ]


query_df = pd.DataFrame(
    query_records
)


query_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "queries.csv"
    ),

    index=False
)


print(
    "\nTotal evaluation queries:",
    len(
        query_df
    )
)


print(
    "MS MARCO dev queries:",
    (
        query_df[
            "source"
        ]
        ==
        "msmarco-dev-small"
    ).sum()
)


print(
    "TREC-DL-2019 queries:",
    (
        query_df[
            "source"
        ]
        ==
        "trec-dl-2019"
    ).sum()
)


# ============================================================
# 13. FORCE ALL JUDGED DOCUMENTS INTO SUBSET
#
# This ensures evaluation documents are not accidentally
# removed by the controlled corpus subsampling.
# ============================================================

forced_doc_ids = set()


for query_key in qrels:

    forced_doc_ids.update(
        qrels[
            query_key
        ].keys()
    )


print(
    "\nJudged passages forced into corpus:",
    len(
        forced_doc_ids
    )
)


# ============================================================
# 14. DETERMINISTIC HASH SAMPLING
#
# This is preferable to simply taking the first N documents.
# Every repeated experiment receives the same corpus.
# ============================================================

def stable_fraction(
    value
):

    digest = hashlib.blake2b(

        str(value).encode(
            "utf-8"
        ),

        digest_size=8
    ).digest()


    integer_value = int.from_bytes(

        digest,

        byteorder="big",

        signed=False
    )


    return (

        integer_value

        /

        float(
            2 ** 64
        )
    )


if TARGET_DOCS >= TOTAL_MSMARCO_DOCS:

    SAMPLE_PROBABILITY = 1.0

else:

    number_background_required = max(

        0,

        TARGET_DOCS

        -

        len(
            forced_doc_ids
        )
    )


    number_background_available = max(

        1,

        TOTAL_MSMARCO_DOCS

        -

        len(
            forced_doc_ids
        )
    )


    SAMPLE_PROBABILITY = min(

        1.0,

        number_background_required

        /

        number_background_available
    )


print(
    "Background sampling probability:",
    round(
        SAMPLE_PROBABILITY,
        8
    )
)


# ============================================================
# 15. CORPUS CACHE PATHS
# ============================================================

CORPUS_TSV = os.path.join(
    CACHE_DIR,
    "selected_corpus.tsv"
)


DOCIDS_FILE = os.path.join(
    CACHE_DIR,
    "docids.txt"
)


CORPUS_META_FILE = os.path.join(
    CACHE_DIR,
    "corpus_meta.json"
)


# ============================================================
# 16. SELECT MS MARCO CORPUS
# ============================================================

def prepare_corpus():

    # --------------------------------------------------------
    # Reuse complete cached corpus
    # --------------------------------------------------------

    if (

        os.path.exists(
            CORPUS_TSV
        )

        and

        os.path.exists(
            DOCIDS_FILE
        )

        and

        os.path.exists(
            CORPUS_META_FILE
        )

    ):

        with open(
            CORPUS_META_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            meta = json.load(
                f
            )


        cached_count = int(
            meta[
                "selected_docs"
            ]
        )


        print(
            "\nUsing cached selected corpus."
        )


        print(
            "Cached passages:",
            cached_count
        )


        return cached_count


    # --------------------------------------------------------
    # Remove incomplete leftovers
    # --------------------------------------------------------

    for path in [
        CORPUS_TSV,
        DOCIDS_FILE
    ]:

        if os.path.exists(path):

            os.remove(
                path
            )


    print(
        "\nSelecting controlled MS MARCO corpus..."
    )


    print(
        "The full collection will be scanned once."
    )


    corpus_dataset = ir_datasets.load(
        "msmarco-passage"
    )


    selected_count = 0

    forced_found = set()


    with open(
        CORPUS_TSV,
        "w",
        encoding="utf-8"
    ) as corpus_output, open(
        DOCIDS_FILE,
        "w",
        encoding="utf-8"
    ) as id_output:


        iterator = tqdm(

            corpus_dataset.docs_iter(),

            total=TOTAL_MSMARCO_DOCS,

            desc="Scanning MS MARCO"
        )


        for document in iterator:

            doc_id = str(
                document.doc_id
            )


            is_forced = (
                doc_id
                in
                forced_doc_ids
            )


            selected = (

                is_forced

                or

                (
                    stable_fraction(
                        doc_id
                    )

                    <

                    SAMPLE_PROBABILITY
                )
            )


            if not selected:

                continue


            if is_forced:

                forced_found.add(
                    doc_id
                )


            text = (

                document.text

                if document.text is not None

                else ""
            )


            text = (

                text

                .replace(
                    "\t",
                    " "
                )

                .replace(
                    "\r",
                    " "
                )

                .replace(
                    "\n",
                    " "
                )
            )


            corpus_output.write(

                doc_id

                +

                "\t"

                +

                text

                +

                "\n"
            )


            id_output.write(

                doc_id

                +

                "\n"
            )


            selected_count += 1


    # --------------------------------------------------------
    # Check judged docs were included
    # --------------------------------------------------------

    missing_forced_docs = (

        forced_doc_ids

        -

        forced_found
    )


    if missing_forced_docs:

        raise RuntimeError(

            "Some judged documents were not found "
            "in the MS MARCO collection. "
            f"Missing count = "
            f"{len(missing_forced_docs)}"
        )


    metadata = {

        "requested_docs":
            TARGET_DOCS,

        "selected_docs":
            selected_count,

        "forced_docs":
            len(
                forced_doc_ids
            ),

        "sampling_probability":
            SAMPLE_PROBABILITY,

        "seed":
            SEED
    }


    with open(
        CORPUS_META_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


    print(
        "\nSelected passages:",
        selected_count
    )


    return selected_count


N_SELECTED_DOCS = prepare_corpus()


# ============================================================
# 17. LOAD ROW -> DOCUMENT ID MAP
# ============================================================

print(
    "\nLoading document-ID map..."
)


with open(
    DOCIDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    ROW_TO_DOCID = [

        line.rstrip(
            "\n"
        )

        for line
        in f
    ]


if (
    len(
        ROW_TO_DOCID
    )
    !=
    N_SELECTED_DOCS
):

    raise RuntimeError(
        "docids.txt size does not match selected corpus."
    )


# ============================================================
# 18. LOAD DENSE ENCODER
# ============================================================

print(
    "\nLoading dense encoder..."
)


model = SentenceTransformer(

    MODEL_NAME,

    device=DEVICE
)


DIMENSION = int(

    model.get_sentence_embedding_dimension()
)


print(
    "Embedding dimension:",
    DIMENSION
)


# ============================================================
# 19. QUERY EMBEDDINGS
# ============================================================

QUERY_EMBEDDINGS_FILE = os.path.join(
    CACHE_DIR,
    "query_embeddings.npy"
)


if os.path.exists(
    QUERY_EMBEDDINGS_FILE
):

    print(
        "\nLoading cached query embeddings..."
    )


    query_embeddings = np.load(
        QUERY_EMBEDDINGS_FILE
    )


else:

    print(
        "\nEncoding queries..."
    )


    query_embeddings = model.encode(

        query_df[
            "query_text"
        ].tolist(),

        batch_size=128,

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True
    )


    query_embeddings = np.asarray(

        query_embeddings,

        dtype=np.float32
    )


    np.save(

        QUERY_EMBEDDINGS_FILE,

        query_embeddings
    )


# ============================================================
# 20. CORPUS EMBEDDING CACHE
# ============================================================

CORPUS_EMBEDDINGS_FILE = os.path.join(
    CACHE_DIR,
    "corpus_embeddings.f32"
)


CORPUS_EMBEDDINGS_DONE = os.path.join(
    CACHE_DIR,
    "corpus_embeddings.done"
)


# ============================================================
# 21. ENCODE CORPUS
# ============================================================

def encode_corpus():

    expected_bytes = (

        N_SELECTED_DOCS

        *

        DIMENSION

        *

        4
    )


    if (

        os.path.exists(
            CORPUS_EMBEDDINGS_FILE
        )

        and

        os.path.exists(
            CORPUS_EMBEDDINGS_DONE
        )

        and

        os.path.getsize(
            CORPUS_EMBEDDINGS_FILE
        )
        ==
        expected_bytes

    ):

        print(
            "\nUsing cached corpus embeddings."
        )

        return


    # --------------------------------------------------------
    # Remove incomplete embedding file
    # --------------------------------------------------------

    if os.path.exists(
        CORPUS_EMBEDDINGS_FILE
    ):

        os.remove(
            CORPUS_EMBEDDINGS_FILE
        )


    if os.path.exists(
        CORPUS_EMBEDDINGS_DONE
    ):

        os.remove(
            CORPUS_EMBEDDINGS_DONE
        )


    print(
        "\nEncoding MS MARCO corpus..."
    )


    print(
        "This is the expensive step."
    )


    print(
        "It will be cached and reused on later runs."
    )


    mmap = np.memmap(

        CORPUS_EMBEDDINGS_FILE,

        dtype=np.float32,

        mode="w+",

        shape=(
            N_SELECTED_DOCS,
            DIMENSION
        )
    )


    if DEVICE == "cuda":

        ENCODER_BATCH = 256

        BUFFER_SIZE = 1024

    else:

        ENCODER_BATCH = 64

        BUFFER_SIZE = 256


    text_buffer = []

    current_row = 0


    def flush_buffer():

        nonlocal text_buffer
        nonlocal current_row


        if not text_buffer:

            return


        embeddings = model.encode(

            text_buffer,

            batch_size=ENCODER_BATCH,

            show_progress_bar=False,

            normalize_embeddings=True,

            convert_to_numpy=True
        )


        embeddings = np.asarray(

            embeddings,

            dtype=np.float32
        )


        next_row = (

            current_row

            +

            len(
                embeddings
            )
        )


        mmap[
            current_row:
            next_row
        ] = embeddings


        current_row = next_row


        text_buffer = []


    with open(
        CORPUS_TSV,
        "r",
        encoding="utf-8"
    ) as f:


        iterator = tqdm(

            f,

            total=N_SELECTED_DOCS,

            desc="Encoding passages"
        )


        for line in iterator:

            parts = (

                line.rstrip(
                    "\n"
                )

                .split(
                    "\t",
                    1
                )
            )


            if len(parts) == 2:

                text = parts[1]

            else:

                text = ""


            text_buffer.append(
                text
            )


            if (
                len(
                    text_buffer
                )
                >=
                BUFFER_SIZE
            ):

                flush_buffer()


        flush_buffer()


    mmap.flush()


    del mmap


    if current_row != N_SELECTED_DOCS:

        raise RuntimeError(

            f"Encoded {current_row} passages, "
            f"expected {N_SELECTED_DOCS}."
        )


    with open(
        CORPUS_EMBEDDINGS_DONE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "complete\n"
        )


    print(
        "Corpus encoding complete."
    )


encode_corpus()


# ============================================================
# 22. OPEN MEMORY-MAPPED CORPUS EMBEDDINGS
# ============================================================

corpus_embeddings = np.memmap(

    CORPUS_EMBEDDINGS_FILE,

    dtype=np.float32,

    mode="r",

    shape=(
        N_SELECTED_DOCS,
        DIMENSION
    )
)


# ============================================================
# 23. EXACT-SEARCH CACHE
# ============================================================

EXACT_INDICES_FILE = os.path.join(
    CACHE_DIR,
    "exact_top100_indices.npy"
)


EXACT_SCORES_FILE = os.path.join(
    CACHE_DIR,
    "exact_top100_scores.npy"
)


# ============================================================
# 24. EXACT DENSE SEARCH
# ============================================================

if (

    os.path.exists(
        EXACT_INDICES_FILE
    )

    and

    os.path.exists(
        EXACT_SCORES_FILE
    )

):

    print(
        "\nLoading cached exact-search results..."
    )


    exact_indices = np.load(
        EXACT_INDICES_FILE
    )


    exact_scores = np.load(
        EXACT_SCORES_FILE
    )


else:

    print(
        "\nBuilding exact FAISS reference index..."
    )


    faiss.omp_set_num_threads(
        CPU_THREADS
    )


    exact_index = faiss.IndexFlatIP(
        DIMENSION
    )


    ADD_CHUNK = 50_000


    for start in tqdm(

        range(
            0,
            N_SELECTED_DOCS,
            ADD_CHUNK
        ),

        desc="Adding exact vectors"
    ):

        end = min(

            start
            +
            ADD_CHUNK,

            N_SELECTED_DOCS
        )


        block = np.ascontiguousarray(

            corpus_embeddings[
                start:end
            ],

            dtype=np.float32
        )


        exact_index.add(
            block
        )


    print(
        "\nRunning exact top-100 search..."
    )


    exact_scores, exact_indices = exact_index.search(

        np.ascontiguousarray(
            query_embeddings,
            dtype=np.float32
        ),

        SEARCH_K
    )


    np.save(

        EXACT_INDICES_FILE,

        exact_indices
    )


    np.save(

        EXACT_SCORES_FILE,

        exact_scores
    )


    del exact_index


    gc.collect()


# ============================================================
# 25. HNSW INDEX CACHE
# ============================================================

HNSW_FILE = os.path.join(

    CACHE_DIR,

    (
        f"hnsw_M{HNSW_M}"
        f"_efC{EF_CONSTRUCTION}.faiss"
    )
)


# ============================================================
# 26. BUILD / LOAD HNSW INDEX
# ============================================================

if os.path.exists(
    HNSW_FILE
):

    print(
        "\nLoading cached HNSW index..."
    )


    hnsw_index = faiss.read_index(
        HNSW_FILE
    )


else:

    print(
        "\nBuilding HNSW index..."
    )


    faiss.omp_set_num_threads(
        CPU_THREADS
    )


    hnsw_index = faiss.IndexHNSWFlat(

        DIMENSION,

        HNSW_M,

        faiss.METRIC_INNER_PRODUCT
    )


    hnsw_index.hnsw.efConstruction = (
        EF_CONSTRUCTION
    )


    ADD_CHUNK = 25_000


    build_start = time.perf_counter()


    for start in tqdm(

        range(
            0,
            N_SELECTED_DOCS,
            ADD_CHUNK
        ),

        desc="Building HNSW"
    ):

        end = min(

            start
            +
            ADD_CHUNK,

            N_SELECTED_DOCS
        )


        block = np.ascontiguousarray(

            corpus_embeddings[
                start:end
            ],

            dtype=np.float32
        )


        hnsw_index.add(
            block
        )


    build_seconds = (

        time.perf_counter()

        -

        build_start
    )


    print(
        "HNSW build time:",
        round(
            build_seconds,
            2
        ),
        "seconds"
    )


    print(
        "Saving HNSW index..."
    )


    faiss.write_index(

        hnsw_index,

        HNSW_FILE
    )


# ============================================================
# 27. OPTIONAL FAISS HNSW SEARCH-WORK COUNTERS
# ============================================================

def hnsw_stats_available():

    try:

        stats = faiss.cvar.hnsw_stats

        _ = stats.ndis

        return True

    except Exception:

        return False


HAS_HNSW_STATS = hnsw_stats_available()


print(
    "\nFAISS HNSW work counters available:",
    HAS_HNSW_STATS
)


def reset_hnsw_stats():

    if not HAS_HNSW_STATS:

        return


    try:

        faiss.cvar.hnsw_stats.reset()

    except Exception:

        pass


def read_hnsw_stats():

    if not HAS_HNSW_STATS:

        return (
            np.nan,
            np.nan
        )


    try:

        stats = faiss.cvar.hnsw_stats


        ndis = float(
            getattr(
                stats,
                "ndis",
                np.nan
            )
        )


        nhops = float(
            getattr(
                stats,
                "nhops",
                np.nan
            )
        )


        return (
            ndis,
            nhops
        )


    except Exception:

        return (
            np.nan,
            np.nan
        )


# ============================================================
# 28. METRIC HELPERS
# ============================================================

def rows_to_docids(
    row_indices
):

    result = []


    for row in row_indices:

        row = int(
            row
        )


        if row < 0:

            continue


        result.append(

            ROW_TO_DOCID[
                row
            ]
        )


    return result


def ann_recall_at_k(
    exact_rows,
    approximate_rows,
    k=10
):

    exact_set = set(
        exact_rows[:k]
    )


    approximate_set = set(
        approximate_rows[:k]
    )


    return (

        len(
            exact_set
            &
            approximate_set
        )

        /

        k
    )


def rank_weighted_overlap_at_k(
    exact_rows,
    approximate_rows,
    k=10
):

    """
    Diagnostic only.

    This gives more weight to recovery of
    higher-ranked exact neighbours.

    It is NOT being claimed as an implementation
    of any specific published metric.
    """

    approximate_set = set(
        approximate_rows[:k]
    )


    numerator = 0.0

    denominator = 0.0


    for rank, row in enumerate(
        exact_rows[:k]
    ):

        weight = (

            1.0

            /

            math.log2(
                rank + 2
            )
        )


        denominator += weight


        if row in approximate_set:

            numerator += weight


    if denominator == 0:

        return 0.0


    return (

        numerator

        /

        denominator
    )


def ndcg_at_k(
    retrieved_doc_ids,
    query_key,
    k=10
):

    query_qrels = qrels.get(
        query_key,
        {}
    )


    dcg = 0.0


    for rank, doc_id in enumerate(

        retrieved_doc_ids[:k]
    ):

        relevance = int(

            query_qrels.get(
                doc_id,
                0
            )
        )


        gain = (

            2 ** relevance

            -

            1
        )


        dcg += (

            gain

            /

            math.log2(
                rank + 2
            )
        )


    ideal_relevances = sorted(

        [

            int(rel)

            for rel
            in query_qrels.values()

            if rel > 0
        ],

        reverse=True

    )[:k]


    idcg = 0.0


    for rank, relevance in enumerate(

        ideal_relevances
    ):

        gain = (

            2 ** relevance

            -

            1
        )


        idcg += (

            gain

            /

            math.log2(
                rank + 2
            )
        )


    if idcg == 0:

        return 0.0


    return (

        dcg

        /

        idcg
    )


def recall_at_k(
    retrieved_doc_ids,
    query_key,
    k=100
):

    query_qrels = qrels.get(
        query_key,
        {}
    )


    relevant_docs = {

        doc_id

        for doc_id, relevance
        in query_qrels.items()

        if relevance > 0
    }


    if not relevant_docs:

        return 0.0


    retrieved_set = set(

        retrieved_doc_ids[:k]
    )


    return (

        len(
            relevant_docs
            &
            retrieved_set
        )

        /

        len(
            relevant_docs
        )
    )


def mrr_at_k(
    retrieved_doc_ids,
    query_key,
    k=10
):

    query_qrels = qrels.get(
        query_key,
        {}
    )


    for rank, doc_id in enumerate(

        retrieved_doc_ids[:k]
    ):

        if (

            query_qrels.get(
                doc_id,
                0
            )

            >

            0
        ):

            return (

                1.0

                /

                (
                    rank + 1
                )
            )


    return 0.0


# ============================================================
# 29. EXACT RETRIEVAL QUALITY
# ============================================================

print(
    "\nComputing exact retrieval quality..."
)


exact_metric_rows = []


for i, query_row in query_df.iterrows():

    query_key = query_row[
        "query_key"
    ]


    exact_docids = rows_to_docids(

        exact_indices[
            i
        ]
    )


    exact_metric_rows.append({

        "query_key":
            query_key,

        "source":
            query_row[
                "source"
            ],

        "exact_nDCG@10":
            ndcg_at_k(
                exact_docids,
                query_key,
                10
            ),

        "exact_Recall@100":
            recall_at_k(
                exact_docids,
                query_key,
                100
            ),

        "exact_MRR@10":
            mrr_at_k(
                exact_docids,
                query_key,
                10
            )
    })


exact_metrics_df = pd.DataFrame(
    exact_metric_rows
)


exact_metrics_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "exact_query_metrics.csv"
    ),

    index=False
)


exact_metric_map = (

    exact_metrics_df

    .set_index(
        "query_key"
    )
)


# ============================================================
# 30. PRINT EXACT EFFECTIVENESS
# ============================================================

print(
    "\nExact retrieval effectiveness:"
)


for source_name, group in (

    exact_metrics_df
    .groupby(
        "source"
    )
):

    print(
        "\nSource:",
        source_name
    )


    print(
        "Queries:",
        len(group)
    )


    print(
        "Mean nDCG@10:",
        group[
            "exact_nDCG@10"
        ].mean()
    )


    print(
        "Mean Recall@100:",
        group[
            "exact_Recall@100"
        ].mean()
    )


    print(
        "Mean MRR@10:",
        group[
            "exact_MRR@10"
        ].mean()
    )


# ============================================================
# 31. HNSW SEARCH SWEEP
#
# Search uses one FAISS thread for stable per-query latency.
# ============================================================

print(
    "\nStarting HNSW efSearch sweep..."
)


faiss.omp_set_num_threads(
    1
)


N_QUERIES = len(
    query_df
)


all_query_budget_rows = []

summary_rows = []


for ef in EF_VALUES:

    print(
        "\n"
        +
        "=" * 80
    )


    print(
        "HNSW efSearch =",
        ef
    )


    print(
        "=" * 80
    )


    hnsw_index.hnsw.efSearch = int(
        ef
    )


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_queries = min(
        20,
        N_QUERIES
    )


    for i in range(
        warmup_queries
    ):

        _ = hnsw_index.search(

            query_embeddings[
                i:
                i + 1
            ],

            SEARCH_K
        )


    # --------------------------------------------------------
    # Timing matrix
    # --------------------------------------------------------

    latency_matrix = np.zeros(

        (
            args.repeats,
            N_QUERIES
        ),

        dtype=np.float64
    )


    first_indices = np.empty(

        (
            N_QUERIES,
            SEARCH_K
        ),

        dtype=np.int64
    )


    first_scores = np.empty(

        (
            N_QUERIES,
            SEARCH_K
        ),

        dtype=np.float32
    )


    first_ndis = np.full(

        N_QUERIES,

        np.nan,

        dtype=np.float64
    )


    first_nhops = np.full(

        N_QUERIES,

        np.nan,

        dtype=np.float64
    )


    # --------------------------------------------------------
    # Repeated randomized query-order timing
    # --------------------------------------------------------

    for repeat in range(
        args.repeats
    ):

        order = np.arange(
            N_QUERIES
        )


        repeat_rng = np.random.default_rng(

            SEED

            +

            ef

            +

            repeat
        )


        repeat_rng.shuffle(
            order
        )


        for query_index in tqdm(

            order,

            desc=(
                f"ef={ef}, "
                f"repeat={repeat + 1}"
            )
        ):

            query_vector = np.ascontiguousarray(

                query_embeddings[
                    query_index:
                    query_index + 1
                ],

                dtype=np.float32
            )


            if repeat == 0:

                reset_hnsw_stats()


            start_ns = (
                time.perf_counter_ns()
            )


            scores, indices = hnsw_index.search(

                query_vector,

                SEARCH_K
            )


            end_ns = (
                time.perf_counter_ns()
            )


            latency_matrix[
                repeat,
                query_index
            ] = (

                end_ns

                -

                start_ns

            ) / 1_000_000.0


            if repeat == 0:

                first_indices[
                    query_index
                ] = indices[
                    0
                ]


                first_scores[
                    query_index
                ] = scores[
                    0
                ]


                (
                    ndis,
                    nhops
                ) = read_hnsw_stats()


                first_ndis[
                    query_index
                ] = ndis


                first_nhops[
                    query_index
                ] = nhops


    # --------------------------------------------------------
    # Mean latency for each query across repeats
    # --------------------------------------------------------

    mean_query_latency = (

        latency_matrix.mean(
            axis=0
        )
    )


    # --------------------------------------------------------
    # Query-level effectiveness
    # --------------------------------------------------------

    ef_rows = []


    for i, query_row in query_df.iterrows():

        query_key = query_row[
            "query_key"
        ]


        source = query_row[
            "source"
        ]


        approximate_indices = (

            first_indices[
                i
            ]
        )


        exact_rows = (

            exact_indices[
                i
            ]
        )


        approximate_docids = rows_to_docids(

            approximate_indices
        )


        exact_record = (

            exact_metric_map.loc[
                query_key
            ]
        )


        ann_recall = ann_recall_at_k(

            exact_rows,

            approximate_indices,

            ANN_K
        )


        weighted_overlap = (

            rank_weighted_overlap_at_k(

                exact_rows,

                approximate_indices,

                ANN_K
            )
        )


        ndcg = ndcg_at_k(

            approximate_docids,

            query_key,

            10
        )


        recall100 = recall_at_k(

            approximate_docids,

            query_key,

            100
        )


        mrr10 = mrr_at_k(

            approximate_docids,

            query_key,

            10
        )


        exact_ndcg = float(

            exact_record[
                "exact_nDCG@10"
            ]
        )


        exact_recall100 = float(

            exact_record[
                "exact_Recall@100"
            ]
        )


        exact_mrr10 = float(

            exact_record[
                "exact_MRR@10"
            ]
        )


        ndcg_loss = max(

            0.0,

            exact_ndcg

            -

            ndcg
        )


        recall100_loss = max(

            0.0,

            exact_recall100

            -

            recall100
        )


        ann_safe = int(

            ann_recall

            >=

            TARGET_ANN_RECALL
        )


        ndcg_safe = int(

            ndcg_loss

            <=

            NDCG_LOSS_TOLERANCE
        )


        recall100_safe = int(

            recall100_loss

            <=

            RECALL100_LOSS_TOLERANCE
        )


        ir_safe = int(

            ndcg_safe == 1

            and

            recall100_safe == 1
        )


        record = {

            "query_key":
                query_key,

            "raw_query_id":
                query_row[
                    "raw_query_id"
                ],

            "source":
                source,

            "efSearch":
                int(
                    ef
                ),

            "ANN_Recall@10":
                ann_recall,

            "RankWeightedOverlap@10":
                weighted_overlap,

            "nDCG@10":
                ndcg,

            "Recall@100":
                recall100,

            "MRR@10":
                mrr10,

            "exact_nDCG@10":
                exact_ndcg,

            "exact_Recall@100":
                exact_recall100,

            "exact_MRR@10":
                exact_mrr10,

            "nDCG_loss":
                ndcg_loss,

            "Recall100_loss":
                recall100_loss,

            "ANN_safe":
                ann_safe,

            "nDCG_safe":
                ndcg_safe,

            "Recall100_safe":
                recall100_safe,

            "IR_safe":
                ir_safe,

            "ANN_unsafe_IR_safe":
                int(
                    ann_safe == 0
                    and
                    ir_safe == 1
                ),

            "ANN_safe_IR_unsafe":
                int(
                    ann_safe == 1
                    and
                    ir_safe == 0
                ),

            "mean_latency_ms":
                float(
                    mean_query_latency[
                        i
                    ]
                ),

            "repeat_p95_latency_ms":
                float(
                    np.percentile(

                        latency_matrix[
                            :,
                            i
                        ],

                        95
                    )
                ),

            "distance_computations":
                first_ndis[
                    i
                ],

            "hnsw_hops":
                first_nhops[
                    i
                ]
        }


        ef_rows.append(
            record
        )


        all_query_budget_rows.append(
            record
        )


    ef_df = pd.DataFrame(
        ef_rows
    )


    # --------------------------------------------------------
    # Aggregate ALL + each source separately
    # --------------------------------------------------------

    groups_to_summarize = [

        (
            "ALL",
            ef_df
        )
    ]


    for source_name in sorted(

        ef_df[
            "source"
        ].unique()
    ):

        groups_to_summarize.append(

            (

                source_name,

                ef_df[

                    ef_df[
                        "source"
                    ]
                    ==
                    source_name
                ]
            )
        )


    for group_name, group in groups_to_summarize:

        summary_rows.append({

            "group":
                group_name,

            "efSearch":
                int(
                    ef
                ),

            "queries":
                len(
                    group
                ),

            "mean_ANN_Recall@10":
                group[
                    "ANN_Recall@10"
                ].mean(),

            "mean_RankWeightedOverlap@10":
                group[
                    "RankWeightedOverlap@10"
                ].mean(),

            "mean_nDCG@10":
                group[
                    "nDCG@10"
                ].mean(),

            "mean_Recall@100":
                group[
                    "Recall@100"
                ].mean(),

            "mean_MRR@10":
                group[
                    "MRR@10"
                ].mean(),

            "ANN_violation_rate":
                (
                    1.0

                    -

                    group[
                        "ANN_safe"
                    ].mean()
                ),

            "IR_violation_rate":
                (
                    1.0

                    -

                    group[
                        "IR_safe"
                    ].mean()
                ),

            "ANN_unsafe_IR_safe_rate":
                group[
                    "ANN_unsafe_IR_safe"
                ].mean(),

            "ANN_safe_IR_unsafe_rate":
                group[
                    "ANN_safe_IR_unsafe"
                ].mean(),

            "mean_latency_ms":
                group[
                    "mean_latency_ms"
                ].mean(),

            "p50_latency_ms":
                group[
                    "mean_latency_ms"
                ].quantile(
                    0.50
                ),

            "p95_latency_ms":
                group[
                    "mean_latency_ms"
                ].quantile(
                    0.95
                ),

            "p99_latency_ms":
                group[
                    "mean_latency_ms"
                ].quantile(
                    0.99
                ),

            "mean_distance_computations":
                group[
                    "distance_computations"
                ].mean(),

            "mean_hnsw_hops":
                group[
                    "hnsw_hops"
                ].mean()
        })


# ============================================================
# 32. SAVE QUERY-BUDGET RESULTS
# ============================================================

query_budget_df = pd.DataFrame(

    all_query_budget_rows
)


summary_df = pd.DataFrame(

    summary_rows
)


query_budget_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "query_budget_results.csv"
    ),

    index=False
)


summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "efsearch_summary.csv"
    ),

    index=False
)


# ============================================================
# 33. ORACLE BUDGET ANALYSIS
# ============================================================

oracle_rows = []


for query_key, group in query_budget_df.groupby(

    "query_key"
):

    group = group.sort_values(
        "efSearch"
    )


    ann_safe_rows = group[

        group[
            "ANN_safe"
        ]
        ==
        1
    ]


    ir_safe_rows = group[

        group[
            "IR_safe"
        ]
        ==
        1
    ]


    ann_safe_found = (

        len(
            ann_safe_rows
        )

        >

        0
    )


    ir_safe_found = (

        len(
            ir_safe_rows
        )

        >

        0
    )


    if ann_safe_found:

        min_ann_safe_ef = int(

            ann_safe_rows.iloc[
                0
            ][
                "efSearch"
            ]
        )

    else:

        min_ann_safe_ef = int(
            EF_VALUES[-1]
        )


    if ir_safe_found:

        min_ir_safe_ef = int(

            ir_safe_rows.iloc[
                0
            ][
                "efSearch"
            ]
        )

    else:

        min_ir_safe_ef = int(
            EF_VALUES[-1]
        )


    first_row = group.iloc[
        0
    ]


    oracle_rows.append({

        "query_key":
            query_key,

        "source":
            first_row[
                "source"
            ],

        "min_ANN_safe_ef":
            min_ann_safe_ef,

        "min_IR_safe_ef":
            min_ir_safe_ef,

        "ANN_safe_found":
            int(
                ann_safe_found
            ),

        "IR_safe_found":
            int(
                ir_safe_found
            ),

        "IR_minus_ANN_budget":
            (
                min_ir_safe_ef

                -

                min_ann_safe_ef
            ),

        "same_budget":
            int(
                min_ir_safe_ef
                ==
                min_ann_safe_ef
            ),

        "IR_allows_lower":
            int(
                min_ir_safe_ef
                <
                min_ann_safe_ef
            ),

        "IR_requires_higher":
            int(
                min_ir_safe_ef
                >
                min_ann_safe_ef
            )
    })


oracle_df = pd.DataFrame(
    oracle_rows
)


oracle_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "oracle_budget_gap.csv"
    ),

    index=False
)


# ============================================================
# 34. ORACLE SUMMARY
# ============================================================

oracle_summary_rows = []


oracle_groups = [

    (
        "ALL",
        oracle_df
    )
]


for source_name in sorted(

    oracle_df[
        "source"
    ].unique()
):

    oracle_groups.append(

        (

            source_name,

            oracle_df[

                oracle_df[
                    "source"
                ]
                ==
                source_name
            ]
        )
    )


for group_name, group in oracle_groups:

    oracle_summary_rows.append({

        "group":
            group_name,

        "queries":
            len(
                group
            ),

        "mean_ANN_safe_ef":
            group[
                "min_ANN_safe_ef"
            ].mean(),

        "median_ANN_safe_ef":
            group[
                "min_ANN_safe_ef"
            ].median(),

        "mean_IR_safe_ef":
            group[
                "min_IR_safe_ef"
            ].mean(),

        "median_IR_safe_ef":
            group[
                "min_IR_safe_ef"
            ].median(),

        "same_budget_rate":
            group[
                "same_budget"
            ].mean(),

        "IR_allows_lower_rate":
            group[
                "IR_allows_lower"
            ].mean(),

        "IR_requires_higher_rate":
            group[
                "IR_requires_higher"
            ].mean()
    })


oracle_summary_df = pd.DataFrame(

    oracle_summary_rows
)


oracle_summary_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "oracle_summary.csv"
    ),

    index=False
)


# ============================================================
# 35. SELECT FIXED ANN / IR BASELINES
# ============================================================

all_summary = (

    summary_df[

        summary_df[
            "group"
        ]
        ==
        "ALL"
    ]

    .sort_values(
        "efSearch"
    )
)


fixed_ann_ef = int(
    EF_VALUES[-1]
)


fixed_ir_ef = int(
    EF_VALUES[-1]
)


for _, row in all_summary.iterrows():

    if (

        row[
            "ANN_violation_rate"
        ]

        <=

        MAX_FIXED_VIOLATION
    ):

        fixed_ann_ef = int(

            row[
                "efSearch"
            ]
        )

        break


for _, row in all_summary.iterrows():

    if (

        row[
            "IR_violation_rate"
        ]

        <=

        MAX_FIXED_VIOLATION
    ):

        fixed_ir_ef = int(

            row[
                "efSearch"
            ]
        )

        break


# ============================================================
# 36. PLOT: ANN QUALITY VS BUDGET
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "mean_ANN_Recall@10"
    ],

    marker="o",

    label="ANN Recall@10"
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "mean_RankWeightedOverlap@10"
    ],

    marker="o",

    label="Rank-weighted overlap@10"
)


plt.axhline(

    TARGET_ANN_RECALL,

    linestyle="--",

    label="ANN target"
)


plt.xlabel(
    "HNSW efSearch"
)


plt.ylabel(
    "Quality"
)


plt.title(
    "MS MARCO: ANN Quality vs Search Budget"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "quality_vs_budget.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 37. PLOT: ANN VS IR VIOLATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "ANN_violation_rate"
    ],

    marker="o",

    label="ANN violation"
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "IR_violation_rate"
    ],

    marker="o",

    label="IR-utility violation"
)


plt.axhline(

    MAX_FIXED_VIOLATION,

    linestyle="--",

    label="5% target"
)


plt.xlabel(
    "HNSW efSearch"
)


plt.ylabel(
    "Violation rate"
)


plt.title(
    "ANN vs IR-Utility Constraint Violations"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "ann_vs_ir_violation.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 38. PLOT: LATENCY
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "mean_latency_ms"
    ],

    marker="o",

    label="Mean latency"
)


plt.plot(

    all_summary[
        "efSearch"
    ],

    all_summary[
        "p95_latency_ms"
    ],

    marker="o",

    label="p95 latency"
)


plt.xlabel(
    "HNSW efSearch"
)


plt.ylabel(
    "Latency (ms/query)"
)


plt.title(
    "MS MARCO HNSW Search Latency"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "latency_vs_budget.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 39. PLOT: SEARCH WORK
# ============================================================

if HAS_HNSW_STATS:

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        all_summary[
            "efSearch"
        ],

        all_summary[
            "mean_distance_computations"
        ],

        marker="o"
    )


    plt.xlabel(
        "HNSW efSearch"
    )


    plt.ylabel(
        "Mean distance computations"
    )


    plt.title(
        "HNSW Search Work vs Search Budget"
    )


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            "search_work_vs_budget.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 40. PLOT: ORACLE BUDGET COMPARISON
# ============================================================

overall_oracle = (

    oracle_summary_df[

        oracle_summary_df[
            "group"
        ]
        ==
        "ALL"
    ]

    .iloc[
        0
    ]
)


plt.figure(
    figsize=(7, 5)
)


plt.bar(

    [
        "ANN-safe oracle",
        "IR-safe oracle"
    ],

    [
        overall_oracle[
            "mean_ANN_safe_ef"
        ],

        overall_oracle[
            "mean_IR_safe_ef"
        ]
    ]
)


plt.ylabel(
    "Mean minimum efSearch"
)


plt.title(
    "ANN-Safe vs IR-Safe Oracle Budget"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "oracle_budget_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 41. COMPUTE ORACLE BUDGET REDUCTION
# ============================================================

mean_ann_oracle = float(

    overall_oracle[
        "mean_ANN_safe_ef"
    ]
)


mean_ir_oracle = float(

    overall_oracle[
        "mean_IR_safe_ef"
    ]
)


if mean_ann_oracle > 0:

    oracle_budget_reduction = (

        (
            mean_ann_oracle

            -

            mean_ir_oracle
        )

        /

        mean_ann_oracle

        *

        100.0
    )

else:

    oracle_budget_reduction = 0.0


# ============================================================
# 42. SAVE TEXT SUMMARY
# ============================================================

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "large_pilot_summary.txt"
)


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MS MARCO LARGE HNSW PILOT\n"
    )


    f.write(
        "=" * 70
        +
        "\n"
    )


    f.write(
        f"Selected passages: "
        f"{N_SELECTED_DOCS}\n"
    )


    f.write(
        f"Evaluation queries: "
        f"{len(query_df)}\n"
    )


    f.write(
        f"MS MARCO dev queries: "
        f"{len(selected_dev_qids)}\n"
    )


    f.write(
        f"TREC DL 2019 queries: "
        f"{len(selected_trec_qids)}\n\n"
    )


    f.write(
        f"Fixed ANN-safe ef: "
        f"{fixed_ann_ef}\n"
    )


    f.write(
        f"Fixed IR-safe ef: "
        f"{fixed_ir_ef}\n\n"
    )


    f.write(
        f"Mean ANN-safe oracle ef: "
        f"{mean_ann_oracle:.4f}\n"
    )


    f.write(
        f"Mean IR-safe oracle ef: "
        f"{mean_ir_oracle:.4f}\n"
    )


    f.write(
        f"Oracle budget reduction: "
        f"{oracle_budget_reduction:.2f}%\n\n"
    )


    f.write(
        "Same ANN/IR oracle budget: "
        f"{overall_oracle['same_budget_rate']:.4f}\n"
    )


    f.write(
        "IR allows lower budget: "
        f"{overall_oracle['IR_allows_lower_rate']:.4f}\n"
    )


    f.write(
        "IR requires higher budget: "
        f"{overall_oracle['IR_requires_higher_rate']:.4f}\n"
    )


# ============================================================
# 43. FINAL TERMINAL REPORT
# ============================================================

print(
    "\n"
    +
    "=" * 100
)


print(
    "MS MARCO LARGE PILOT COMPLETED SUCCESSFULLY"
)


print(
    "=" * 100
)


print(
    "\nSelected passages =",
    N_SELECTED_DOCS
)


print(
    "Total evaluation queries =",
    len(
        query_df
    )
)


print(
    "MS MARCO dev queries =",
    len(
        selected_dev_qids
    )
)


print(
    "TREC-DL-2019 queries =",
    len(
        selected_trec_qids
    )
)


print(
    "\nFixed ANN-safe efSearch =",
    fixed_ann_ef
)


print(
    "Fixed IR-safe efSearch =",
    fixed_ir_ef
)


print(
    "\nMean ANN-safe oracle ef =",
    mean_ann_oracle
)


print(
    "Mean IR-safe oracle ef =",
    mean_ir_oracle
)


print(
    "Oracle budget reduction (%) =",
    oracle_budget_reduction
)


print(
    "\nSame ANN/IR oracle budget =",
    float(
        overall_oracle[
            "same_budget_rate"
        ]
    )
)


print(
    "IR allows LOWER budget =",
    float(
        overall_oracle[
            "IR_allows_lower_rate"
        ]
    )
)


print(
    "IR requires HIGHER budget =",
    float(
        overall_oracle[
            "IR_requires_higher_rate"
        ]
    )
)


print(
    "\n"
    +
    "=" * 100
)


print(
    "OVERALL efSearch SUMMARY"
)


print(
    "=" * 100
)


print(

    all_summary[

        [
            "efSearch",
            "mean_ANN_Recall@10",
            "mean_nDCG@10",
            "mean_Recall@100",
            "ANN_violation_rate",
            "IR_violation_rate",
            "mean_latency_ms",
            "p95_latency_ms",
            "mean_distance_computations"
        ]
    ]

    .to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 100
)


print(
    "ORACLE SUMMARY BY QUERY SET"
)


print(
    "=" * 100
)


print(

    oracle_summary_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)


print(
    RESULTS_DIR
)


print(
    "\nIMPORTANT:"
)


print(
    "This is a controlled MS MARCO subset experiment."
)


print(
    "Do NOT report its effectiveness numbers as "
    "official full-corpus MS MARCO results."
)


print(
    "\nDONE."
)