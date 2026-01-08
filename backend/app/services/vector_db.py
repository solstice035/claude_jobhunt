"""
Vector Database Service - ChromaDB Interface

Provides vector storage and similarity search for job embeddings.
Uses ChromaDB in embedded mode for development, with option for
pgvector in production.

Key Features:
- Persistent storage with configurable path
- HNSW index for fast similarity search
- Metadata filtering (status, score thresholds)
- Batch operations for bulk updates

Usage:
    db = get_vector_db()

    # Add jobs
    db.upsert(
        ids=["job-1", "job-2"],
        embeddings=[[0.1, ...], [0.2, ...]],
        metadatas=[{"title": "Developer"}, {"title": "Engineer"}],
        documents=["Job description 1", "Job description 2"]
    )

    # Query similar jobs
    results = db.query(
        query_embeddings=[cv_embedding],
        n_results=50,
        where={"status": "new"}
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorDB:
    """
    ChromaDB-backed vector database for job embeddings.

    Provides CRUD operations and similarity search with
    metadata filtering support.

    Attributes:
        client: ChromaDB persistent client
        collection: Jobs collection with HNSW index
    """

    def __init__(self, persist_directory: str):
        """
        Initialize vector database.

        Args:
            persist_directory: Path to store database files
        """
        self.persist_directory = persist_directory

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Get or create jobs collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name="jobs",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"VectorDB initialized at {persist_directory}")

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> None:
        """
        Insert or update job embeddings.

        Args:
            ids: List of job UUIDs
            embeddings: List of embedding vectors (1536-dim each)
            metadatas: Optional metadata dicts (title, company, etc.)
            documents: Optional job descriptions (for retrieval)
        """
        if not ids:
            return

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.debug(f"Upserted {len(ids)} embeddings")
        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 50,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find similar jobs by embedding similarity.

        Args:
            query_embeddings: Query vectors (e.g., CV embedding)
            n_results: Maximum results to return
            where: Metadata filter (e.g., {"status": "new"})
            include: Fields to include ("metadatas", "documents", "distances")

        Returns:
            Dict with ids, distances, metadatas, documents
        """
        if include is None:
            include = ["metadatas", "distances"]

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            raise

    def get(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get embeddings by ID.

        Args:
            ids: List of job UUIDs to retrieve

        Returns:
            Dict with ids, embeddings, metadatas, documents
        """
        try:
            return self.collection.get(
                ids=ids,
                include=["metadatas", "embeddings", "documents"]
            )
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def delete(self, ids: List[str]) -> None:
        """
        Delete embeddings by ID.

        Args:
            ids: List of job UUIDs to delete
        """
        if not ids:
            return

        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} embeddings")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise

    def count(self) -> int:
        """
        Get total number of embeddings stored.

        Returns:
            Count of stored embeddings
        """
        return self.collection.count()

    def find_similar(
        self,
        embedding: List[float],
        n_results: int = 50,
        min_score: Optional[float] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find jobs similar to a given embedding.

        Convenience method with common filtering options.

        Args:
            embedding: Query embedding (e.g., CV)
            n_results: Maximum results
            min_score: Minimum match score filter
            status: Job status filter (e.g., "new")

        Returns:
            List of job dicts with id, distance, metadata
        """
        # Build where clause
        where = None
        if min_score is not None and status is not None:
            where = {
                "$and": [
                    {"match_score": {"$gte": min_score}},
                    {"status": status}
                ]
            }
        elif min_score is not None:
            where = {"match_score": {"$gte": min_score}}
        elif status is not None:
            where = {"status": status}

        results = self.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances", "documents"]
        )

        # Format results
        jobs = []
        if results["ids"] and results["ids"][0]:
            for i, job_id in enumerate(results["ids"][0]):
                job = {
                    "id": job_id,
                    "similarity": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                }
                if results.get("metadatas"):
                    job["metadata"] = results["metadatas"][0][i]
                if results.get("documents"):
                    job["document"] = results["documents"][0][i]
                jobs.append(job)

        return jobs

    def update_metadata(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Update metadata for existing embeddings.

        Args:
            ids: Job UUIDs to update
            metadatas: New metadata dicts
        """
        try:
            self.collection.update(
                ids=ids,
                metadatas=metadatas
            )
            logger.debug(f"Updated metadata for {len(ids)} embeddings")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise

    def reset(self) -> None:
        """
        Delete all data and reset the collection.

        WARNING: This is destructive and cannot be undone.
        """
        try:
            self.client.delete_collection("jobs")
            self.collection = self.client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("Vector database reset")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise


# ==================== Factory Function ====================

_vector_db_instance: Optional[VectorDB] = None


def get_vector_db(persist_directory: Optional[str] = None) -> VectorDB:
    """
    Get or create VectorDB singleton.

    Args:
        persist_directory: Optional path (uses settings if not provided)

    Returns:
        VectorDB instance
    """
    global _vector_db_instance

    if _vector_db_instance is None:
        path = persist_directory or get_settings().chroma_persist_directory
        _vector_db_instance = VectorDB(persist_directory=path)

    return _vector_db_instance


def reset_vector_db() -> None:
    """Reset the singleton instance (for testing)."""
    global _vector_db_instance
    _vector_db_instance = None
