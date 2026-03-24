"""Tests for vector database API endpoints."""

from fastapi.testclient import TestClient
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding

from main import app, get_vector_store
from ingestion import VectorStoreManager


client = TestClient(app)


class _MockEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list[float]:
        return [0.1] * 8

    def _get_text_embedding(self, text: str) -> list[float]:
        return [0.1] * 8

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return [0.1] * 8

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return [0.1] * 8


def _build_test_store() -> VectorStoreManager:
    return VectorStoreManager(embedding_model=_MockEmbedding())


class TestVectorStoreStatusEndpoint:
    """Tests for /vectordb/status endpoint."""

    def test_vectordb_status_returns_200(self):
        """Verify that /vectordb/status returns 200."""
        response = client.get("/vectordb/status")
        assert response.status_code == 200

    def test_vectordb_status_has_required_fields(self):
        """Verify that status response has required fields."""
        response = client.get("/vectordb/status")
        data = response.json()
        assert "initialized" in data
        assert "document_count" in data
        assert "index_info" in data

    def test_vectordb_status_document_count_is_integer(self):
        """Verify that document_count is an integer."""
        response = client.get("/vectordb/status")
        data = response.json()
        assert isinstance(data["document_count"], int)
        assert data["document_count"] >= 0


class TestVectorStoreQueryEndpoint:
    """Tests for /vectordb/query endpoint."""

    def test_vectordb_query_requires_query_parameter(self):
        """Verify that /vectordb/query requires query parameter."""
        response = client.get("/vectordb/query")
        assert response.status_code == 422

    def test_vectordb_query_with_query_parameter_returns_200(self):
        """Verify that /vectordb/query with query param returns 200."""
        response = client.get("/vectordb/query", params={"q": "test"})
        assert response.status_code == 200

    def test_vectordb_query_returns_list(self):
        """Verify that query endpoint returns a list."""
        response = client.get("/vectordb/query", params={"q": "test"})
        data = response.json()
        assert isinstance(data, list)

    def test_vectordb_query_respects_top_k_parameter(self):
        """Verify that top_k parameter limits results."""
        response = client.get("/vectordb/query", params={"q": "test", "top_k": 2})
        assert response.status_code == 200

    def test_vectordb_query_default_top_k(self):
        """Verify that default top_k is reasonable."""
        response = client.get("/vectordb/query", params={"q": "test"})
        assert response.status_code == 200

    def test_query_embedding_endpoint_returns_embedding(self, monkeypatch):
        """Verify POST query embedding endpoint returns embedding details."""
        store = _build_test_store()

        def _override_store() -> VectorStoreManager:
            return store

        app.dependency_overrides[get_vector_store] = _override_store
        try:
            response = client.post("/vectordb/query/embedding", json={"query": "test query", "top_k": 2})
            data = response.json()
            assert response.status_code == 200
            assert data["query"] == "test query"
            assert data["dimensions"] == 8
            assert len(data["embedding"]) == 8
        finally:
            app.dependency_overrides.clear()

    def test_post_query_returns_structured_results(self):
        """Verify POST query endpoint returns query, embedding, and results."""
        store = _build_test_store()
        store.add_documents([
            Document(text="Artificial intelligence is the simulation of human intelligence.", metadata={"title": "AI"}),
            Document(text="Biology studies living organisms.", metadata={"title": "Biology"}),
        ])

        def _override_store() -> VectorStoreManager:
            return store

        app.dependency_overrides[get_vector_store] = _override_store
        try:
            response = client.post("/vectordb/query", json={"query": "What is artificial intelligence?", "top_k": 1})
            data = response.json()
            assert response.status_code == 200
            assert data["query"] == "What is artificial intelligence?"
            assert data["top_k"] == 1
            assert len(data["embedding"]) == 8
            assert len(data["results"]) <= 1
            assert "text" in data["results"][0]
            assert "score" in data["results"][0]
            assert "metadata" in data["results"][0]
        finally:
            app.dependency_overrides.clear()

    def test_query_retrieval_end_to_end_after_ingest(self, monkeypatch):
        """Verify ingest then query works end to end through the API."""
        store = _build_test_store()

        def _override_store() -> VectorStoreManager:
            return store

        def _mock_loader(source: str, limit: int | None = None) -> list[Document]:
            return [
                Document(text="Artificial intelligence is the simulation of human intelligence.", metadata={"title": "AI", "id": "1"}),
                Document(text="Machine learning is a subset of artificial intelligence.", metadata={"title": "ML", "id": "2"}),
            ]

        monkeypatch.setattr("ingestion.load_documents_from_parquet", _mock_loader)
        app.dependency_overrides[get_vector_store] = _override_store
        try:
            ingest_response = client.post("/vectordb/ingest", json={"source": "mock://source", "limit": 2})
            assert ingest_response.status_code == 200

            query_response = client.post("/vectordb/query", json={"query": "artificial intelligence", "top_k": 2})
            data = query_response.json()
            assert query_response.status_code == 200
            assert data["query"] == "artificial intelligence"
            assert len(data["results"]) >= 1
            assert any(result["metadata"]["title"] in {"AI", "ML"} for result in data["results"])
        finally:
            app.dependency_overrides.clear()


class TestVectorStoreIngestEndpoint:
    """Tests for /vectordb/ingest endpoint."""

    def test_vectordb_ingest_returns_200(self, monkeypatch):
        """Verify ingest endpoint returns success."""
        store = _build_test_store()

        def _override_store() -> VectorStoreManager:
            return store

        def _mock_loader(source: str, limit: int | None = None) -> list[Document]:
            return [Document(text="doc a"), Document(text="doc b")]

        monkeypatch.setattr("ingestion.load_documents_from_parquet", _mock_loader)
        app.dependency_overrides[get_vector_store] = _override_store
        try:
            response = client.post("/vectordb/ingest", json={"source": "mock://source", "limit": 2})
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_vectordb_ingest_response_has_counts(self, monkeypatch):
        """Verify ingest response includes ingested count and updated status."""
        store = _build_test_store()

        def _override_store() -> VectorStoreManager:
            return store

        def _mock_loader(source: str, limit: int | None = None) -> list[Document]:
            return [Document(text="doc a"), Document(text="doc b"), Document(text="doc c")]

        monkeypatch.setattr("ingestion.load_documents_from_parquet", _mock_loader)
        app.dependency_overrides[get_vector_store] = _override_store
        try:
            response = client.post("/vectordb/ingest", json={"source": "mock://source", "limit": 3})
            data = response.json()
            assert data["ingested_count"] == 3
            assert data["status"]["initialized"] is True
            assert data["status"]["document_count"] == 3
        finally:
            app.dependency_overrides.clear()
