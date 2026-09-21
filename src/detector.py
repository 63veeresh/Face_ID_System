"""
Face Detection and Alignment Module using MTCNN (Multi-task Cascaded Convolutional Networks).
Handles face localization, landmark detection, alignment, and normalized face cropping.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from .config import (
    DEVICE,
    FACE_IMAGE_SIZE,
    DETECTION_MARGIN,
    DETECTION_MIN_CONFIDENCE,
    MTCNN_MIN_FACE_SIZE,
)


class DetectedFace:
    """Represents a detected face with its bounding box, confidence score, landmarks, and cropped image."""

    def __init__(
        self,
        box: np.ndarray,
        score: float,
        landmarks: Optional[np.ndarray],
        face_tensor: torch.Tensor,
        face_image: Image.Image,
    ):
        self.box = [int(coord) for coord in box]  # [x1, y1, x2, y2]
        self.score = float(score)
        self.landmarks = landmarks
        self.face_tensor = face_tensor  # (3, 160, 160) normalized tensor
        self.face_image = face_image  # PIL Image of cropped face

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


class FaceDetector:
    """
    MTCNN-based face detector.
    Three-stage cascaded architecture:
      1. P-Net (Proposal Network): Fast candidate window generator.
      2. R-Net (Refine Network): Candidate filtering and false-positive rejection.
      3. O-Net (Output Network): Final bounding box refinement and 5 facial landmarks.
    """

    def __init__(
        self,
        device: str = DEVICE,
        image_size: int = FACE_IMAGE_SIZE,
        margin: int = DETECTION_MARGIN,
        min_face_size: int = MTCNN_MIN_FACE_SIZE,
        min_confidence: float = DETECTION_MIN_CONFIDENCE,
    ):
        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.min_confidence = min_confidence

        # Initialize MTCNN for both bounding box/landmark extraction and tensor cropping
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=True,  # Standardizes tensor to [-1, 1]
            device=self.device,
            keep_all=True,  # Detect all faces in an image
        )

    def _to_pil_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Converts various image input formats (path, numpy, cv2) to RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Check if input is likely BGR from OpenCV or RGB
            if len(image_input.shape) == 2:  # Grayscale
                img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 3:
                img = Image.fromarray(image_input)
            elif image_input.shape[2] == 4:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def detect_faces(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[DetectedFace]:
        """
        Detects all faces in the given image.
        Returns a list of DetectedFace objects sorted by bounding box area (largest first).
        """
        pil_img = self._to_pil_image(image_input)

        # Run MTCNN detection
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        # Extract cropped and aligned face tensors
        face_tensors = self.mtcnn.extract(pil_img, boxes, save_path=None)

        results = []
        img_w, img_h = pil_img.size

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < self.min_confidence:
                continue

            # Clip coordinates to image boundary
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(img_w, int(box[2]))
            y2 = min(img_h, int(box[3]))

            # Discard invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Get landmark points for this face if available
            lm = landmarks[i] if landmarks is not None else None

            # Get cropped PIL image for visualization
            cropped_img = pil_img.crop((x1, y1, x2, y2)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )

            # Extract corresponding tensor
            if face_tensors is not None and i < len(face_tensors):
                tensor = face_tensors[i]
            else:
                # Fallback manual crop and standardize
                tensor = self._manual_preprocess(cropped_img)

            results.append(
                DetectedFace(
                    box=np.array([x1, y1, x2, y2]),
                    score=float(prob),
                    landmarks=lm,
                    face_tensor=tensor,
                    face_image=cropped_img,
                )
            )

        # Sort faces by area descending (primary/prominent face first)
        results.sort(key=lambda f: f.width * f.height, reverse=True)
        return results

    def detect_single_face(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Optional[DetectedFace]:
        """
        Convenience method for enrollment or single-face verification.
        Returns the primary (largest/most confident) face detected, or None.
        """
        faces = self.detect_faces(image_input)
        if not faces:
            return None
        return faces[0]

    def _manual_preprocess(self, pil_face: Image.Image) -> torch.Tensor:
        """Fallback to standardize a PIL face crop to [-1, 1] tensor."""
        arr = np.float32(np.asarray(pil_face))
        # Standardize: (x - 127.5) / 128.0
        standardized = (arr - 127.5) / 128.0
        # Transpose to (C, H, W)
        tensor = torch.from_numpy(standardized).permute(2, 0, 1).float()
        return tensor
