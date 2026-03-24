"""Document ingestion and embedding pipeline for RAG system.

This module handles:
- Loading documents from Hugging Face datasets
- Generating embeddings using Azure OpenAI models
- Storing embeddings in a local LlamaIndex vector store
- Querying the vector store
"""

from pathlib import Path
from threading import Lock
from typing import Any

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.embeddings import BaseEmbedding

from llamaindex_models import get_chat_model, get_embedding_model


DEFAULT_PASSAGES_SOURCE = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
DEFAULT_TEST_QA_SOURCE = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet"


def load_documents_from_parquet(source: str, limit: int | None = None) -> list[Document]:
    """Load documents from a parquet source.

    Args:
        source: Parquet source URI or path.
        limit: Optional number of rows to load.

    Returns:
        List of LlamaIndex Document objects.
    """
    import pandas as pd

    df = pd.read_parquet(source)
    if limit is not None:
        df = df.head(limit)

    text_column = None
    preferred_columns = ["text", "passage", "content", "body"]
    for col in preferred_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        string_columns = [c for c in df.columns if df[c].dtype == "object"]
        if not string_columns:
            raise ValueError("No text-like column found in source data")
        text_column = string_columns[0]

    documents: list[Document] = []
    for _, row in df.iterrows():
        text = str(row[text_column])
        metadata: dict[str, Any] = {}
        if "id" in df.columns:
            metadata["id"] = row["id"]
        if "title" in df.columns:
            metadata["title"] = row["title"]

        documents.append(Document(text=text, metadata=metadata))

    return documents


def load_qa_pairs_from_parquet(source: str = DEFAULT_TEST_QA_SOURCE, limit: int | None = None) -> list[dict[str, str]]:
    """Load question/answer rows from a parquet source for evaluation.

    Args:
        source: Parquet source URI or path.
        limit: Optional number of rows to load.

    Returns:
        List of dictionaries containing question and answer strings.
    """
    import pandas as pd

    df = pd.read_parquet(source)
    if limit is not None:
        df = df.head(limit)

    question_column = None
    answer_column = None

    question_candidates = ["question", "query", "q"]
    answer_candidates = ["answer", "expected_answer", "answers"]

    for col in question_candidates:
        if col in df.columns:
            question_column = col
            break

    for col in answer_candidates:
        if col in df.columns:
            answer_column = col
            break

    if question_column is None or answer_column is None:
        raise ValueError("No question/answer columns found in test dataset")

    qa_pairs: list[dict[str, str]] = []
    for _, row in df.iterrows():
        question_value = row[question_column]
        answer_value = row[answer_column]

        if isinstance(answer_value, list):
            answer_text = str(answer_value[0]) if answer_value else ""
        else:
            answer_text = str(answer_value)

        qa_pairs.append(
            {
                "question": str(question_value),
                "answer": answer_text,
            }
        )

    return qa_pairs


class VectorStoreManager:
    """Manages vector store lifecycle and operations."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: BaseEmbedding | None = None,
        chat_model: Any | None = None,
        context_char_limit: int = 600,
        answer_cache_size: int = 256,
    ):
        """Initialize vector store manager.
        
        Args:
            persist_dir: Optional directory to persist the vector store.
            embedding_model: Optional embedding model. If not provided, will load Azure model.
        """
        self.persist_dir = persist_dir or Path.cwd() / ".vector_store"
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.index: VectorStoreIndex | None = None
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self.context_char_limit = max(100, context_char_limit)
        self.answer_cache_size = max(1, answer_cache_size)
        self._answer_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._cache_lock = Lock()

    @property
    def embedding_model(self):
        """Lazy-load embedding model on first access."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def chat_model(self):
        """Lazy-load chat model on first access."""
        if self._chat_model is None:
            # Deterministic generation reduces response variance during evaluation.
            self._chat_model = get_chat_model(temperature=0)
        return self._chat_model

    @staticmethod
    def _normalize_query(query_str: str) -> str:
        """Normalize user query for consistent caching and embedding calls."""
        return " ".join(query_str.strip().split())

    def create_or_load_index(self) -> VectorStoreIndex:
        """Create a new vector store index or load an existing one.
        
        Returns:
            VectorStoreIndex instance.
        """
        if self.index is None:
            # Create new index with embedding model
            self.index = VectorStoreIndex(
                nodes=[],
                embed_model=self.embedding_model,
            )
        return self.index

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to ingest.
            
        Returns:
            Number of documents added.
        """
        if self.index is None:
            self.create_or_load_index()

        nodes = [
            TextNode(text=doc.get_content(), metadata=doc.metadata)
            for doc in documents
        ]
        self.index.insert_nodes(nodes)
        return len(nodes)

    def generate_query_embedding(self, query_str: str) -> list[float]:
        """Generate an embedding vector for a query string."""
        return self.embedding_model.get_query_embedding(query_str)

    def ingest_from_source(self, source: str = DEFAULT_PASSAGES_SOURCE, limit: int | None = None) -> int:
        """Load documents from a source and add them to the vector store.

        Args:
            source: Source path or URI to parquet data.
            limit: Optional row limit for partial ingestion.

        Returns:
            Number of documents ingested.
        """
        documents = load_documents_from_parquet(source=source, limit=limit)
        return self.add_documents(documents)

    def get_status(self) -> dict[str, Any]:
        """Get current vector store status.
        
        Returns:
            Dictionary with status information including document count.
        """
        if self.index is None:
            return {
                "initialized": False,
                "document_count": 0,
                "index_info": None,
            }

        doc_count = len(self.index.docstore.docs)
        return {
            "initialized": True,
            "document_count": doc_count,
            "index_info": "LlamaIndex VectorStoreIndex",
            "persist_dir": str(self.persist_dir),
        }

    def retrieve(
        self,
        query_str: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar documents for a query.

        Args:
            query_str: Query string.
            top_k: Number of top results to return.
            query_embedding: Optional precomputed embedding for explainable flow.

        Returns:
            List of retrieved document results.
        """
        if self.index is None:
            return []

        normalized_query = self._normalize_query(query_str)
        top_k = max(1, top_k)

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        query_bundle = QueryBundle(query_str=normalized_query, embedding=query_embedding)
        results = retriever.retrieve(query_bundle)

        return [
            {
                "text": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
            }
            for node in results
        ]

    def query(self, query_str: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query the vector store for similar documents."""
        if self.index is None:
            return []

        normalized_query = self._normalize_query(query_str)
        query_embedding = self.generate_query_embedding(normalized_query)
        return self.retrieve(query_str=normalized_query, top_k=top_k, query_embedding=query_embedding)

    def build_augmented_prompt(self, query_str: str, retrieved_docs: list[dict[str, Any]]) -> str:
        """Build a context-augmented prompt from retrieved results."""
        normalized_query = self._normalize_query(query_str)
        if not retrieved_docs:
            context_block = "No supporting context was retrieved."
        else:
            context_parts: list[str] = []
            for idx, doc in enumerate(retrieved_docs, start=1):
                title = doc.get("metadata", {}).get("title") or "Untitled"
                text = str(doc.get("text", ""))[: self.context_char_limit]
                context_parts.append(f"[{idx}] Title: {title}\n{text}")
            context_block = "\n\n".join(context_parts)

        return (
            "You are a helpful assistant. Answer the user question using only the provided context. "
            "If the context does not contain enough information, say you do not know. "
            "Respond in 1-2 concise sentences.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {normalized_query}\n"
            "Answer:"
        )

    def generate_answer_text(self, augmented_prompt: str) -> str:
        """Generate an answer from the augmented prompt using the chat model."""
        response = self.chat_model.complete(augmented_prompt)
        if hasattr(response, "text"):
            return str(response.text)
        return str(response)

    def answer_query(self, query_str: str, top_k: int = 5) -> dict[str, Any]:
        """Run retrieval-augmented generation for a query and return full flow details."""
        normalized_query = self._normalize_query(query_str)
        cache_key = (normalized_query, max(1, top_k))

        with self._cache_lock:
            cached = self._answer_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        query_embedding = self.generate_query_embedding(normalized_query)
        results = self.retrieve(query_str=normalized_query, top_k=top_k, query_embedding=query_embedding)
        augmented_prompt = self.build_augmented_prompt(query_str=normalized_query, retrieved_docs=results)
        answer = self.generate_answer_text(augmented_prompt)

        result = {
            "query": normalized_query,
            "top_k": top_k,
            "embedding": query_embedding,
            "results": results,
            "prompt": augmented_prompt,
            "answer": answer,
        }

        with self._cache_lock:
            if len(self._answer_cache) >= self.answer_cache_size:
                # Simple bounded cache eviction: remove oldest inserted key.
                oldest_key = next(iter(self._answer_cache))
                self._answer_cache.pop(oldest_key, None)
            self._answer_cache[cache_key] = result.copy()

        return result
