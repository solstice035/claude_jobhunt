"""
Tests for embedding provider abstraction.

Tests cover:
- Abstract embedding interface
- OpenAI embeddings provider
- Local embeddings provider (sentence-transformers)
- Provider factory and configuration

Run with: cd backend && pytest tests/test_embedding_providers.py -v
"""
import pytest
from typing import List
from unittest.mock import Mock, AsyncMock, patch


class TestEmbeddingProviderInterface:
    """Tests for the abstract embedding provider interface."""

    def test_interface_defines_required_methods(self):
        """Interface should define embed, embed_batch, and dimensions."""
        from app.services.embedding_providers import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "embed_batch")
        assert hasattr(EmbeddingProvider, "dimensions")

    def test_all_providers_implement_interface(self):
        """All providers should implement EmbeddingProvider."""
        from app.services.embedding_providers import (
            EmbeddingProvider,
            OpenAIEmbeddings,
            LocalEmbeddings,
        )

        # Check that providers have required attributes/methods
        openai_provider = OpenAIEmbeddings(api_key="test")
        local_provider = LocalEmbeddings(lazy_load=True)

        # Both should have dimensions property and embed methods
        assert hasattr(openai_provider, "dimensions")
        assert hasattr(openai_provider, "embed")
        assert hasattr(openai_provider, "embed_batch")

        assert hasattr(local_provider, "dimensions")
        assert hasattr(local_provider, "embed")
        assert hasattr(local_provider, "embed_batch")


class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings provider."""

    def test_openai_provider_initialization(self):
        """Should initialize with default model."""
        from app.services.embedding_providers import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key="test-key")
        assert provider.model == "text-embedding-3-small"
        assert provider.dimensions == 1536

    def test_openai_provider_custom_model(self):
        """Should accept custom model name."""
        from app.services.embedding_providers import OpenAIEmbeddings

        provider = OpenAIEmbeddings(
            api_key="test-key",
            model="text-embedding-3-large"
        )
        assert provider.model == "text-embedding-3-large"
        assert provider.dimensions == 3072

    @pytest.mark.asyncio
    async def test_openai_embed_single_text(self):
        """Should embed a single text."""
        from app.services.embedding_providers import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key="test-key")

        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]

        with patch.object(provider, "_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            embedding = await provider.embed("Test text")

            assert len(embedding) == 1536
            mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_embed_batch(self):
        """Should embed multiple texts efficiently."""
        from app.services.embedding_providers import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key="test-key")

        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]

        with patch.object(provider, "_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            embeddings = await provider.embed_batch(["Text 1", "Text 2"])

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536

    @pytest.mark.asyncio
    async def test_openai_handles_empty_text(self):
        """Should return zero vector for empty text."""
        from app.services.embedding_providers import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key="test-key")

        embedding = await provider.embed("")

        assert len(embedding) == 1536
        assert all(v == 0.0 for v in embedding)


class TestLocalEmbeddings:
    """Tests for local sentence-transformers embeddings."""

    def test_local_provider_initialization(self):
        """Should initialize with default model."""
        from app.services.embedding_providers import LocalEmbeddings

        provider = LocalEmbeddings(lazy_load=True)
        assert "nomic" in provider.model_name.lower() or "bge" in provider.model_name.lower()

    def test_local_provider_custom_model(self):
        """Should accept custom model name."""
        from app.services.embedding_providers import LocalEmbeddings

        provider = LocalEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            lazy_load=True
        )
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_local_provider_dimensions_property(self):
        """Should report correct embedding dimensions."""
        from app.services.embedding_providers import LocalEmbeddings

        # Test with known model dimensions
        provider = LocalEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            lazy_load=True
        )
        assert provider.dimensions == 768

    @pytest.mark.asyncio
    async def test_local_embed_single_text(self):
        """Should embed a single text."""
        import numpy as np
        from app.services.embedding_providers import LocalEmbeddings

        provider = LocalEmbeddings(lazy_load=True)

        # Mock the model - return numpy array like real model does
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 768)

        provider._model = mock_model

        embedding = await provider.embed("Test text")

        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_local_embed_batch(self):
        """Should embed multiple texts."""
        import numpy as np
        from app.services.embedding_providers import LocalEmbeddings

        provider = LocalEmbeddings(lazy_load=True)

        # Mock the model - return numpy array like real model does
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])

        provider._model = mock_model

        embeddings = await provider.embed_batch(["Text 1", "Text 2"])

        assert len(embeddings) == 2

    @pytest.mark.asyncio
    async def test_local_handles_empty_text(self):
        """Should return zero vector for empty text."""
        from app.services.embedding_providers import LocalEmbeddings

        provider = LocalEmbeddings(lazy_load=True)
        provider._model = Mock()  # Prevent model loading

        embedding = await provider.embed("")

        assert len(embedding) == provider.dimensions
        assert all(v == 0.0 for v in embedding)


class TestEmbeddingProviderFactory:
    """Tests for embedding provider factory."""

    def test_factory_creates_openai_provider(self):
        """Should create OpenAI provider."""
        from app.services.embedding_providers import get_embedding_provider

        provider = get_embedding_provider(
            provider_name="openai",
            api_key="test-key"
        )
        assert provider is not None

    def test_factory_creates_local_provider(self):
        """Should create local provider."""
        from app.services.embedding_providers import get_embedding_provider

        provider = get_embedding_provider(
            provider_name="local",
            lazy_load=True
        )
        assert provider is not None

    def test_factory_invalid_provider(self):
        """Should raise error for unknown provider."""
        from app.services.embedding_providers import get_embedding_provider

        with pytest.raises(ValueError):
            get_embedding_provider(provider_name="unknown")

    def test_factory_default_provider(self):
        """Should use OpenAI as default."""
        from app.services.embedding_providers import get_embedding_provider

        provider = get_embedding_provider(api_key="test-key")
        from app.services.embedding_providers import OpenAIEmbeddings
        assert isinstance(provider, OpenAIEmbeddings)


class TestEmbeddingProviderConfig:
    """Tests for embedding provider configuration."""

    def test_provider_has_model_info(self):
        """Providers should expose model information."""
        from app.services.embedding_providers import OpenAIEmbeddings, LocalEmbeddings

        openai = OpenAIEmbeddings(api_key="test")
        assert hasattr(openai, "model")
        assert hasattr(openai, "dimensions")

        local = LocalEmbeddings(lazy_load=True)
        assert hasattr(local, "model_name")
        assert hasattr(local, "dimensions")

    def test_available_models_constant(self):
        """Should have list of available models."""
        from app.services.embedding_providers import AVAILABLE_MODELS

        assert "openai" in AVAILABLE_MODELS
        assert "local" in AVAILABLE_MODELS

        # Each provider should list supported models
        assert "text-embedding-3-small" in AVAILABLE_MODELS["openai"]
        assert len(AVAILABLE_MODELS["local"]) > 0


class TestEmbeddingComparison:
    """Tests for comparing embeddings from different providers."""

    @pytest.mark.asyncio
    async def test_embeddings_have_correct_dimensions(self):
        """Embeddings should have documented dimensions."""
        import numpy as np
        from app.services.embedding_providers import OpenAIEmbeddings, LocalEmbeddings

        openai = OpenAIEmbeddings(api_key="test")
        local = LocalEmbeddings(lazy_load=True)

        # Mock OpenAI
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        openai._client = Mock()
        openai._client.embeddings.create = AsyncMock(return_value=mock_response)

        # Mock local - return numpy array like real model does
        local._model = Mock()
        local._model.encode.return_value = np.array([0.1] * 768)

        openai_emb = await openai.embed("test")
        local_emb = await local.embed("test")

        assert len(openai_emb) == openai.dimensions
        assert len(local_emb) == local.dimensions


class TestExperimentFramework:
    """Tests for A/B testing embedding providers."""

    def test_experiment_config_structure(self):
        """Should support A/B experiment configuration."""
        from app.services.embedding_providers import EmbeddingExperiment

        experiment = EmbeddingExperiment(
            name="test_nomic",
            provider_name="local",
            sample_percentage=0.2
        )

        assert experiment.name == "test_nomic"
        assert experiment.sample_percentage == 0.2

    def test_experiment_routing_by_id(self):
        """Should consistently route same ID to same experiment."""
        from app.services.embedding_providers import route_to_experiment

        experiments = [
            {"name": "control", "percentage": 0.8},
            {"name": "treatment", "percentage": 0.2},
        ]

        # Same ID should always route to same experiment
        result1 = route_to_experiment("job-123", experiments)
        result2 = route_to_experiment("job-123", experiments)

        assert result1 == result2

    def test_experiment_routing_distribution(self):
        """Should approximately match configured percentages."""
        from app.services.embedding_providers import route_to_experiment

        experiments = [
            {"name": "control", "percentage": 0.8},
            {"name": "treatment", "percentage": 0.2},
        ]

        # Route many IDs and check distribution
        results = {"control": 0, "treatment": 0}
        for i in range(1000):
            result = route_to_experiment(f"job-{i}", experiments)
            results[result] += 1

        # Should be roughly 80/20 (with some variance)
        assert 700 < results["control"] < 900
        assert 100 < results["treatment"] < 300


class TestMockEmbeddingProvider:
    """Tests for mock embedding provider (for testing)."""

    @pytest.mark.asyncio
    async def test_mock_provider_returns_deterministic_embeddings(self):
        """Mock should return consistent embeddings for same input."""
        from app.services.embedding_providers import MockEmbeddingProvider

        provider = MockEmbeddingProvider()

        emb1 = await provider.embed("test")
        emb2 = await provider.embed("test")

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_mock_provider_different_inputs_different_outputs(self):
        """Mock should return different embeddings for different inputs."""
        from app.services.embedding_providers import MockEmbeddingProvider

        provider = MockEmbeddingProvider()

        emb1 = await provider.embed("text one")
        emb2 = await provider.embed("text two")

        assert emb1 != emb2

    def test_mock_provider_configurable_dimensions(self):
        """Mock should support configurable dimensions."""
        from app.services.embedding_providers import MockEmbeddingProvider

        provider = MockEmbeddingProvider(dimensions=768)
        assert provider.dimensions == 768
