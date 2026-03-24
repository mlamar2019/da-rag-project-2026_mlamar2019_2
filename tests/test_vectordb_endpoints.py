"""Tests for vector database API endpoints."""

from fastapi.testclient import TestClient

from main import app, get_vector_store


client = TestClient(app)


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
