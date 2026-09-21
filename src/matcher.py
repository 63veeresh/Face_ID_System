"""
Similarity Matching & Identity Classification Module.
Implements Cosine Similarity, Top-K Candidate Ranking, and Threshold-Based UNKNOWN Rejection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .config import DEFAULT_SIMILARITY_THRESHOLD, UNKNOWN_LABEL


@dataclass
class MatchCandidate:
    """Represents a candidate match from the enrolled database."""
    name: str
    similarity: float
    rank: int


@dataclass
class MatchResult:
    """
    Comprehensive result of matching a query face embedding against enrolled identities.
    Designed for full explainability during technical evaluations and interview review.
    """
    identity: str               # Predicted identity ("UNKNOWN" or person's name)
    is_known: bool              # True if similarity >= threshold, False otherwise
    similarity: float           # Highest cosine similarity score [-1.0, 1.0]
    threshold: float            # Threshold used for decision boundary
    margin: float               # Difference between top match and runner-up score
    top_candidate: str          # Name of the closest candidate (even if rejected)
    all_candidates: List[MatchCandidate] # Ranked list of all candidates
    rejection_reason: Optional[str] = None # Human-readable explanation if rejected


class FaceMatcher:
    """
    Matches query face embeddings against an enrolled database of facial embeddings
    using Cosine Similarity on L2-normalized 512-D vectors.
    """

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.threshold = float(threshold)

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes cosine similarity between two 1D vectors u and v.
        cos(theta) = (u . v) / (||u|| * ||v||)
        """
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0 or norm_v == 0:
            return 0.0

        sim = float(np.dot(u, v) / (norm_u * norm_v))
        return float(np.clip(sim, -1.0, 1.0))

    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, database_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized cosine similarity between a single query (512,) and database (N, 512).
        Returns array of similarities of shape (N,).
        """
        if database_matrix.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        # Normalize query
        norm_q = np.linalg.norm(query)
        q_normed = query / norm_q if norm_q > 0 else query

        # Normalize database rows
        norms_db = np.linalg.norm(database_matrix, axis=1, keepdims=True)
        norms_db[norms_db == 0] = 1e-8
        db_normed = database_matrix / norms_db

        # Matrix-vector dot product
        sims = np.dot(db_normed, q_normed)
        return np.clip(sims, -1.0, 1.0).astype(np.float32)

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_names: List[str],
        enrolled_embeddings: np.ndarray,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """
        Identifies the person corresponding to query_embedding.
        If no enrolled identities exist, immediately rejects as UNKNOWN.
        If max similarity < threshold, rejects as UNKNOWN.
        """
        current_threshold = float(threshold if threshold is not None else self.threshold)

        # Handle empty enrolled database
        if len(enrolled_names) == 0 or enrolled_embeddings.shape[0] == 0:
            return MatchResult(
                identity=UNKNOWN_LABEL,
                is_known=False,
                similarity=0.0,
                threshold=current_threshold,
                margin=0.0,
                top_candidate="None",
                all_candidates=[],
                rejection_reason="No identities currently registered in the database.",
            )

        # Compute cosine similarity against all enrolled identities
        similarities = self.batch_cosine_similarity(query_embedding, enrolled_embeddings)

        # Rank all enrolled candidates by descending similarity
        sorted_indices = np.argsort(-similarities)
        ranked_candidates: List[MatchCandidate] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            ranked_candidates.append(
                MatchCandidate(
                    name=enrolled_names[idx],
                    similarity=float(similarities[idx]),
                    rank=rank,
                )
            )

        top_match = ranked_candidates[0]
        top_name = top_match.name
        top_sim = top_match.similarity

        # Compute margin between #1 and #2 match
        if len(ranked_candidates) > 1:
            margin = top_sim - ranked_candidates[1].similarity
        else:
            margin = top_sim

        # Threshold decision boundary
        if top_sim >= current_threshold:
            return MatchResult(
                identity=top_name,
                is_known=True,
                similarity=top_sim,
                threshold=current_threshold,
                margin=margin,
                top_candidate=top_name,
                all_candidates=ranked_candidates,
                rejection_reason=None,
            )
        else:
            reason = (
                f"Similarity score {top_sim:.4f} is below the configured threshold "
                f"{current_threshold:.4f} (closest match was '{top_name}')."
            )
            return MatchResult(
                identity=UNKNOWN_LABEL,
                is_known=False,
                similarity=top_sim,
                threshold=current_threshold,
                margin=margin,
                top_candidate=top_name,
                all_candidates=ranked_candidates,
                rejection_reason=reason,
            )
