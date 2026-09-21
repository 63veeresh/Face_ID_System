"""
Face Embedding Module using InceptionResnetV1 pretrained on VGGFace2.
Extracts 512-dimensional L2-normalized feature representations of facial images.
"""

from typing import Union, List
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from .config import DEVICE, EMBEDDING_DIM, PRETRAINED_WEIGHTS


class FaceEmbedder:
    """
    Extracts deep facial feature embeddings using InceptionResnetV1.
    Trained with Triplet / Softmax loss on VGGFace2, mapping human faces to a
    512-dimensional hypersphere where Euclidean / Cosine distances directly reflect identity similarity.
    """

    def __init__(
        self,
        pretrained: str = PRETRAINED_WEIGHTS,
        device: str = DEVICE,
    ):
        self.device = device
        self.pretrained = pretrained
        self.embedding_dim = EMBEDDING_DIM

        # Load InceptionResnetV1 model in evaluation mode
        self.model = InceptionResnetV1(pretrained=pretrained, classify=False).eval().to(self.device)

        # Freeze all parameters to prevent accidental gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_embedding(
        self, face_tensor: Union[torch.Tensor, List[torch.Tensor]]
    ) -> np.ndarray:
        """
        Computes L2-normalized 512-D embedding for a single face tensor or a batch.

        Args:
            face_tensor: A torch.Tensor of shape (3, 160, 160) or (N, 3, 160, 160),
                         or a list of (3, 160, 160) tensors.

        Returns:
            np.ndarray: Embedding vector of shape (512,) for single input,
                        or (N, 512) for batch input.
        """
        if isinstance(face_tensor, list):
            if len(face_tensor) == 0:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            # Stack list of tensors into batch (N, 3, 160, 160)
            tensor_batch = torch.stack([t.to(self.device) for t in face_tensor])
            is_batch = True
        elif isinstance(face_tensor, torch.Tensor):
            if face_tensor.dim() == 3:
                # Single face: (3, 160, 160) -> (1, 3, 160, 160)
                tensor_batch = face_tensor.unsqueeze(0).to(self.device)
                is_batch = False
            elif face_tensor.dim() == 4:
                tensor_batch = face_tensor.to(self.device)
                is_batch = True
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got shape {face_tensor.shape}")
        else:
            raise TypeError(f"Unsupported face_tensor type: {type(face_tensor)}")

        # Ensure float32 tensor
        tensor_batch = tensor_batch.float()

        # Forward pass through InceptionResnetV1
        raw_embeddings = self.model(tensor_batch)

        # Apply L2 normalization across dim=1 so ||v||_2 = 1.0
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        # Convert to numpy float32
        embeddings_np = normalized_embeddings.cpu().numpy().astype(np.float32)

        if not is_batch:
            return embeddings_np[0]
        return embeddings_np
