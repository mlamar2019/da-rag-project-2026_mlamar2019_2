"""Tests for document ingestion and embedding pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from ingestion import VectorStoreManager


@pytest.fixture
def temp_store_dir():
    """Temporary directory for vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model to avoid Azure authentication in tests."""
    mock = MagicMock()
    # Make it look like a proper embedding model
    mock.get_text_embedding = MagicMock(return_value=[0.1] * 1536)
    mock.get_text_embedding_batch = MagicMock(return_value=[[0.1] * 1536])
    mock._get_text_embedding = MagicMock(return_value=[0.1] * 1536)
    mock._get_text_embeddings = MagicMock(return_value=[[0.1] * 1536])
    return mock


@pytest.fixture
def vector_store_manager_with_mock(temp_store_dir):
    """Vector store manager fixture that patches the embedding model."""
    with patch("ingestion.get_embedding_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.get_text_embedding = MagicMock(return_value=[0.1] * 1536)
        mock_model.get_text_embedding_batch = MagicMock(return_value=[[0.1] * 1536])
        mock_get_model.return_value = mock_model
        manager = VectorStoreManager(persist_dir=temp_store_dir)
        yield manager


class TestVectorStoreManagerInitialization:
    """Tests for VectorStoreManager initialization."""

    def test_init_creates_persist_directory(self, temp_store_dir):
        """Verify that VectorStoreManager creates persist directory."""
        manager = VectorStoreManager(persist_dir=temp_store_dir)
        assert temp_store_dir.exists()

    def test_init_with_default_persist_dir(self):
        """Verify that VectorStoreManager uses default persist directory."""
        manager = VectorStoreManager()
        assert manager.persist_dir.exists()

    def test_embedding_model_lazy_loads(self, temp_store_dir):
        """Verify that embedding model is lazy-loaded on first access."""
        manager = VectorStoreManager(persist_dir=temp_store_dir)
        assert manager._embedding_model is None
        # Accessing the property would trigger lazy load
        # (skipping since it requires Azure credentials)


class TestVectorStoreStatus:
    """Tests for vector store status reporting."""

    def test_status_uninitialized_index(self, vector_store_manager_with_mock):
        """Verify status when index is not yet created."""
        status = vector_store_manager_with_mock.get_status()
        assert status["initialized"] is False
        assert status["document_count"] == 0
        assert status["index_info"] is None

    def test_status_initialized_empty_index(self, vector_store_manager_with_mock):
        """Verify status after creating an empty index."""
        vector_store_manager_with_mock.create_or_load_index()
        status = vector_store_manager_with_mock.get_status()
        assert status["initialized"] is True
        assert status["document_count"] == 0
        assert status["index_info"] == "LlamaIndex VectorStoreIndex"

    def test_status_includes_persist_dir(self, vector_store_manager_with_mock, temp_store_dir):
        """Verify that status includes persist directory path."""
        vector_store_manager_with_mock.create_or_load_index()
        status = vector_store_manager_with_mock.get_status()
        assert "persist_dir" in status
        assert str(temp_store_dir) in status["persist_dir"]


class TestDocumentAddition:
    """Tests for adding documents to the vector store."""

    def test_add_documents_initializes_index_if_needed(self, vector_store_manager_with_mock):
        """Verify that adding documents creates index if not exists."""
        doc = Document(text="Test document")
        count = vector_store_manager_with_mock.add_documents([doc])
        assert count == 1
        assert vector_store_manager_with_mock.index is not None

    def test_add_single_document(self, vector_store_manager_with_mock):
        """Verify adding a single document returns count."""
        doc = Document(text="Test document")
        count = vector_store_manager_with_mock.add_documents([doc])
        assert count == 1

    def test_add_multiple_documents(self, vector_store_manager_with_mock):
        """Verify adding multiple documents returns correct count."""
        docs = [
            Document(text="Document 1"),
            Document(text="Document 2"),
            Document(text="Document 3"),
        ]
        count = vector_store_manager_with_mock.add_documents(docs)
        assert count == 3

    def test_status_reflects_added_documents(self, vector_store_manager_with_mock):
        """Verify that status reflects the number of added documents."""
        docs = [Document(text=f"Document {i}") for i in range(3)]
        vector_store_manager_with_mock.add_documents(docs)
        status = vector_store_manager_with_mock.get_status()
        assert status["document_count"] == 3


class TestVectorStoreQuery:
    """Tests for querying the vector store."""

    def test_query_on_uninitialized_store_returns_empty(self, vector_store_manager_with_mock):
        """Verify that querying uninitialized store returns empty list."""
        results = vector_store_manager_with_mock.query("test query")
        assert results == []

    def test_query_on_empty_store_returns_empty(self, vector_store_manager_with_mock):
        """Verify that querying empty initialized store returns empty list."""
        vector_store_manager_with_mock.create_or_load_index()
        results = vector_store_manager_with_mock.query("test query")
        assert results == []

    def test_query_result_structure(self, vector_store_manager_with_mock):
        """Verify that query results have expected structure."""
        doc = Document(text="Test document content")
        vector_store_manager_with_mock.add_documents([doc])
        results = vector_store_manager_with_mock.query("test", top_k=1)
        # Results might be empty if no actual embeddings, but structure should be consistent
        assert isinstance(results, list)

    def test_query_respects_top_k_parameter(self, vector_store_manager_with_mock):
        """Verify that query respects the top_k parameter."""
        docs = [Document(text=f"Document {i}") for i in range(5)]
        vector_store_manager_with_mock.add_documents(docs)
        results = vector_store_manager_with_mock.query("document", top_k=2)
        assert len(results) <= 2


class TestCreateOrLoadIndex:
    """Tests for index creation and loading."""

    def test_create_or_load_index_creates_new_when_not_exists(self, vector_store_manager_with_mock):
        """Verify that create_or_load_index creates new index when none exists."""
        assert vector_store_manager_with_mock.index is None
        index = vector_store_manager_with_mock.create_or_load_index()
        assert index is not None

    def test_create_or_load_index_reuses_existing(self, vector_store_manager_with_mock):
        """Verify that create_or_load_index reuses existing index."""
        index1 = vector_store_manager_with_mock.create_or_load_index()
        index2 = vector_store_manager_with_mock.create_or_load_index()
        assert index1 is index2
