"""
Unit tests for the FaceMatcher module (Cosine Similarity & UNKNOWN Rejection).
"""

import numpy as np
import pytest
from src.matcher import FaceMatcher, MatchResult, MatchCandidate
from src.config import DEFAULT_SIMILARITY_THRESHOLD, UNKNOWN_LABEL


def test_cosine_similarity_identical():
    """Identical non-zero vectors should have cosine similarity = 1.0."""
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sim = FaceMatcher.cosine_similarity(v, v)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have cosine similarity = 0.0."""
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim = FaceMatcher.cosine_similarity(u, v)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_cosine_similarity_opposite():
    """Directly opposite vectors should have cosine similarity = -1.0."""
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = -u
    sim = FaceMatcher.cosine_similarity(u, v)
    assert pytest.approx(sim, abs=1e-5) == -1.0


def test_batch_cosine_similarity():
    """Verify vectorized matrix dot product matches elementwise similarity."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db = np.array([
        [1.0, 0.0, 0.0],   # Identical -> 1.0
        [0.0, 1.0, 0.0],   # Orthogonal -> 0.0
        [0.7071, 0.7071, 0.0], # 45 degrees -> ~0.7071
    ], dtype=np.float32)

    sims = FaceMatcher.batch_cosine_similarity(query, db)
    assert len(sims) == 3
    assert pytest.approx(sims[0], abs=1e-4) == 1.0
    assert pytest.approx(sims[1], abs=1e-4) == 0.0
    assert pytest.approx(sims[2], abs=1e-4) == 0.7071


def test_match_known_identity():
    """Query with similarity above threshold should be recognized with candidate name."""
    matcher = FaceMatcher(threshold=0.65)
    enrolled_names = ["Alice", "Bob", "Charlie"]
    enrolled_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],   # Alice
        [0.0, 1.0, 0.0, 0.0],   # Bob
        [0.0, 0.0, 1.0, 0.0],   # Charlie
    ], dtype=np.float32)

    # Query very close to Alice
    query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    result = matcher.match(query, enrolled_names, enrolled_embeddings, threshold=0.65)

    assert result.is_known is True
    assert result.identity == "Alice"
    assert result.similarity > 0.65
    assert result.top_candidate == "Alice"
    assert result.rejection_reason is None


def test_unknown_rejection_below_threshold():
    """Query whose highest similarity is below threshold must be rejected as UNKNOWN."""
    matcher = FaceMatcher(threshold=0.75)
    enrolled_names = ["Alice", "Bob"]
    enrolled_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    # Query with max similarity ~0.50 (below 0.75)
    query = np.array([0.5, 0.5, 0.707], dtype=np.float32)
    result = matcher.match(query, enrolled_names, enrolled_embeddings, threshold=0.75)

    assert result.is_known is False
    assert result.identity == UNKNOWN_LABEL
    assert result.similarity < 0.75
    assert result.rejection_reason is not None
    assert "below the configured threshold" in result.rejection_reason


def test_match_empty_database():
    """Matching against an empty database must gracefully return UNKNOWN."""
    matcher = FaceMatcher(threshold=0.65)
    query = np.random.randn(512).astype(np.float32)

    result = matcher.match(query, [], np.empty((0, 512), dtype=np.float32))
    assert result.is_known is False
    assert result.identity == UNKNOWN_LABEL
    assert "No identities currently registered" in result.rejection_reason
