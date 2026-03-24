from fastapi import FastAPI, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from ingestion import (
    VectorStoreManager,
    DEFAULT_PASSAGES_SOURCE,
    DEFAULT_TEST_QA_SOURCE,
    load_qa_pairs_from_parquet,
)
from ailab.utils.azure import get_ailab_auth_status


app = FastAPI(title="Mini Wikipedia RAG API", version="0.1.0")

# Global vector store instance
_vector_store: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    """Get or initialize the global vector store manager."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    message: str


class VectorDBStatus(BaseModel):
    initialized: bool
    document_count: int
    index_info: str | None


class IngestRequest(BaseModel):
    source: str = DEFAULT_PASSAGES_SOURCE
    limit: int | None = 100


class IngestResponse(BaseModel):
    ingested_count: int
    source: str
    status: VectorDBStatus


class AuthStatusResponse(BaseModel):
    authenticated: bool
    auth_source: str | None
    scope: str
    error: str | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryEmbeddingResponse(BaseModel):
    query: str
    embedding: list[float]
    dimensions: int


class RetrievedDocument(BaseModel):
    text: str
    score: float | None = None
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    top_k: int
    embedding: list[float]
    results: list[RetrievedDocument]


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RagQueryResponse(BaseModel):
    query: str
    top_k: int
    embedding: list[float]
    results: list[RetrievedDocument]
    prompt: str
    answer: str


class RagEvaluationRequest(BaseModel):
    source: str = DEFAULT_TEST_QA_SOURCE
    limit: int = 5
    top_k: int = 5
    max_workers: int = 3


class RagEvaluationItem(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    exact_match: bool
    contains_expected: bool


class RagEvaluationSummary(BaseModel):
    total: int
    exact_match_rate: float
    contains_expected_rate: float


class RagEvaluationResponse(BaseModel):
    source: str
    limit: int
    top_k: int
    max_workers: int
    summary: RagEvaluationSummary
    items: list[RagEvaluationItem]


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def evaluate_generated_answer(expected_answer: str, generated_answer: str) -> tuple[bool, bool]:
    """Return exact-match and substring-style quality checks."""
    expected = _normalize_text(expected_answer)
    generated = _normalize_text(generated_answer)
    exact_match = generated == expected
    contains_expected = expected in generated if expected else False
    return exact_match, contains_expected


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoResponse)
def echo(payload: EchoRequest) -> EchoResponse:
    return EchoResponse(message=payload.message)


@app.get("/auth/status", response_model=AuthStatusResponse)
def auth_status() -> AuthStatusResponse:
    """Return whether the server currently has usable auth material available."""
    return AuthStatusResponse(**get_ailab_auth_status())


@app.get("/vectordb/status", response_model=VectorDBStatus)
def get_vectordb_status(store: VectorStoreManager = Depends(get_vector_store)) -> VectorDBStatus:
    """Get current vector database status."""
    status = store.get_status()
    return VectorDBStatus(**status)


@app.get("/vectordb/query")
def query_vectordb(
    q: str,
    top_k: int = 5,
    store: VectorStoreManager = Depends(get_vector_store),
) -> list[dict]:
    """Query the vector database for similar documents."""
    return store.query(q, top_k=top_k)


@app.post("/vectordb/query/embedding", response_model=QueryEmbeddingResponse)
def embed_query(
    payload: QueryRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> QueryEmbeddingResponse:
    """Generate an embedding for a user query."""
    embedding = store.generate_query_embedding(payload.query)
    return QueryEmbeddingResponse(
        query=payload.query,
        embedding=embedding,
        dimensions=len(embedding),
    )


@app.post("/vectordb/query", response_model=QueryResponse)
def query_vectordb_post(
    payload: QueryRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> QueryResponse:
    """Embed a query, run similarity search, and return retrieved documents."""
    embedding = store.generate_query_embedding(payload.query)
    results = store.retrieve(
        query_str=payload.query,
        top_k=payload.top_k,
        query_embedding=embedding,
    )
    return QueryResponse(
        query=payload.query,
        top_k=payload.top_k,
        embedding=embedding,
        results=[RetrievedDocument(**result) for result in results],
    )


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(
    payload: RagQueryRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> RagQueryResponse:
    """Run end-to-end retrieval-augmented generation for a user query."""
    rag_result = store.answer_query(query_str=payload.query, top_k=payload.top_k)
    return RagQueryResponse(
        query=rag_result["query"],
        top_k=rag_result["top_k"],
        embedding=rag_result["embedding"],
        results=[RetrievedDocument(**result) for result in rag_result["results"]],
        prompt=rag_result["prompt"],
        answer=rag_result["answer"],
    )


@app.post("/rag/evaluate", response_model=RagEvaluationResponse)
def rag_evaluate(
    payload: RagEvaluationRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> RagEvaluationResponse:
    """Evaluate RAG answers against a question/answer parquet dataset."""
    qa_pairs = load_qa_pairs_from_parquet(source=payload.source, limit=payload.limit)

    def _evaluate_item(qa: dict[str, str]) -> RagEvaluationItem:
        rag_result = store.answer_query(query_str=qa["question"], top_k=payload.top_k)
        exact_match, contains_expected = evaluate_generated_answer(
            expected_answer=qa["answer"],
            generated_answer=rag_result["answer"],
        )
        return RagEvaluationItem(
            question=qa["question"],
            expected_answer=qa["answer"],
            generated_answer=rag_result["answer"],
            exact_match=exact_match,
            contains_expected=contains_expected,
        )

    max_workers = min(max(payload.max_workers, 1), 8)
    if max_workers == 1:
        items = [_evaluate_item(qa) for qa in qa_pairs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            items = list(executor.map(_evaluate_item, qa_pairs))

    total = len(items)
    exact_match_rate = (sum(1 for item in items if item.exact_match) / total) if total else 0.0
    contains_expected_rate = (sum(1 for item in items if item.contains_expected) / total) if total else 0.0

    return RagEvaluationResponse(
        source=payload.source,
        limit=payload.limit,
        top_k=payload.top_k,
        max_workers=max_workers,
        summary=RagEvaluationSummary(
            total=total,
            exact_match_rate=exact_match_rate,
            contains_expected_rate=contains_expected_rate,
        ),
        items=items,
    )


@app.post("/vectordb/ingest", response_model=IngestResponse)
def ingest_vectordb(
    payload: IngestRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> IngestResponse:
    """Ingest documents from a source parquet file into vector DB."""
    ingested_count = store.ingest_from_source(source=payload.source, limit=payload.limit)
    status = VectorDBStatus(**store.get_status())
    return IngestResponse(
        ingested_count=ingested_count,
        source=payload.source,
        status=status,
    )

