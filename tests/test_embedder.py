"""
Unit tests for the FaceEmbedder module (InceptionResnetV1 & L2-Normalization).
"""

import numpy as np
import pytest
import torch
from src.config import EMBEDDING_DIM
from src.embedder import FaceEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Shared FaceEmbedder fixture for module tests."""
    return FaceEmbedder()


def test_embedding_output_shape(embedder):
    """Ensure a single (3, 160, 160) face tensor yields a 512-D vector."""
    dummy_face = torch.randn(3, 160, 160)
    embedding = embedder.compute_embedding(dummy_face)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32


def test_embedding_l2_normalization(embedder):
    """Ensure output embeddings have unit L2 length (||v||_2 == 1.0)."""
    dummy_face = torch.randn(3, 160, 160)
    embedding = embedder.compute_embedding(dummy_face)

    norm = float(np.linalg.norm(embedding))
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_batch_embedding_computation(embedder):
    """Ensure batch of 4 face tensors returns (4, 512) normalized vectors."""
    batch_faces = torch.randn(4, 3, 160, 160)
    embeddings = embedder.compute_embedding(batch_faces)

    assert embeddings.shape == (4, EMBEDDING_DIM)
    for i in range(4):
        norm = float(np.linalg.norm(embeddings[i]))
        assert pytest.approx(norm, abs=1e-4) == 1.0


def test_deterministic_embedding(embedder):
    """Given identical inputs, the embedder must produce identical vectors."""
    face = torch.ones(3, 160, 160)
    emb1 = embedder.compute_embedding(face)
    emb2 = embedder.compute_embedding(face)

    assert np.allclose(emb1, emb2, atol=1e-6)
