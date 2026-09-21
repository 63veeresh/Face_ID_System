"""
Unified End-to-End Face Recognition & Identification Pipeline.
Connects MTCNN detection, InceptionResnetV1 embedding extraction, database management, and similarity matching.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    DEVICE,
    DEFAULT_SIMILARITY_THRESHOLD,
    UNKNOWN_LABEL,
    BOX_COLOR_MATCH,
    BOX_COLOR_UNKNOWN,
)
from .detector import FaceDetector, DetectedFace
from .embedder import FaceEmbedder
from .database import FaceDatabase
from .matcher import FaceMatcher, MatchResult


class FaceRecognitionPipeline:
    """
    Unified Pipeline for Face Recognition.
    Workflow:
      Image -> MTCNN Detection -> Crop & Alignment -> InceptionResnetV1 512-D Embedding
            -> Cosine Matching against Enrolled Database -> Decision (Identity or UNKNOWN)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        device: str = DEVICE,
        db_file: Optional[Path] = None,
    ):
        self.device = device
        self.threshold = threshold

        # Initialize sub-modules
        self.detector = FaceDetector(device=self.device)
        self.embedder = FaceEmbedder(device=self.device)
        self.database = FaceDatabase() if db_file is None else FaceDatabase(db_file=db_file)
        self.matcher = FaceMatcher(threshold=self.threshold)

    def enroll_person(
        self,
        name: str,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        notes: str = "",
    ) -> Tuple[bool, str, Optional[DetectedFace]]:
        """
        Enrolls a new person into the database from an image.
        Detects primary face, extracts 512-D embedding, and registers the identity.
        """
        detected = self.detector.detect_single_face(image_input)
        if detected is None:
            return False, "No face detected in the provided image. Please provide a clearer photo.", None

        if detected.face_tensor is None:
            return False, "Could not extract face region properly.", None

        # Compute 512-D normalized embedding
        embedding = self.embedder.compute_embedding(detected.face_tensor)

        # Store in persistent database
        self.database.enroll(
            name=name,
            embedding=embedding,
            reference_image=detected.face_image,
            notes=notes,
            allow_update=True,
        )

        return True, f"Successfully enrolled '{name}' (sample count: {self.database.get(name)['sample_count']}).", detected

    def identify_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Tuple[DetectedFace, MatchResult]]:
        """
        Detects all faces in the input image and matches each against the enrolled database.
        Returns a list of (DetectedFace, MatchResult) tuples.
        """
        detected_faces = self.detector.detect_faces(image_input)
        if not detected_faces:
            return []

        enrolled_names, enrolled_embeddings = self.database.get_enrolled_embeddings()
        results: List[Tuple[DetectedFace, MatchResult]] = []

        for face in detected_faces:
            if face.face_tensor is None:
                continue

            # Compute embedding for detected face
            embedding = self.embedder.compute_embedding(face.face_tensor)

            # Match against database
            match_res = self.matcher.match(
                query_embedding=embedding,
                enrolled_names=enrolled_names,
                enrolled_embeddings=enrolled_embeddings,
                threshold=threshold if threshold is not None else self.threshold,
            )
            results.append((face, match_res))

        return results

    def verify_identity(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        claimed_name: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        1:1 Verification: Verifies if the face in the image matches the claimed identity.
        """
        record = self.database.get(claimed_name)
        if not record:
            return False, 0.0, f"Claimed identity '{claimed_name}' is not enrolled in the database."

        detected = self.detector.detect_single_face(image_input)
        if detected is None:
            return False, 0.0, "No face detected in the image."

        query_embedding = self.embedder.compute_embedding(detected.face_tensor)
        target_embedding = record["embedding"]

        sim = self.matcher.cosine_similarity(query_embedding, target_embedding)
        applied_threshold = threshold if threshold is not None else self.threshold

        is_match = sim >= applied_threshold
        if is_match:
            msg = f"VERIFIED: Face matches '{claimed_name}' with similarity {sim:.4f} >= {applied_threshold:.4f}."
        else:
            msg = f"REJECTED: Similarity {sim:.4f} is below threshold {applied_threshold:.4f}."

        return is_match, sim, msg

    def annotate_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        results: List[Tuple[DetectedFace, MatchResult]],
        show_landmarks: bool = True,
    ) -> Image.Image:
        """
        Draws bounding boxes, identity tags, similarity scores, and landmarks on the image.
        Known faces are marked in green; UNKNOWN faces are marked in red.
        """
        pil_img = self.detector._to_pil_image(image_input).copy()
        draw = ImageDraw.Draw(pil_img)

        # Basic default font
        font = ImageFont.load_default()

        for face, match_res in results:
            x1, y1, x2, y2 = face.box
            color = BOX_COLOR_MATCH if match_res.is_known else BOX_COLOR_UNKNOWN

            # Draw bounding box with thickness
            for offset in range(3):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color,
                )

            # Draw 5 facial landmarks (eyes, nose, mouth corners) if requested
            if show_landmarks and face.landmarks is not None:
                for pt in face.landmarks:
                    px, py = int(pt[0]), int(pt[1])
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 215, 0))

            # Header tag text
            if match_res.is_known:
                tag_text = f"{match_res.identity} ({match_res.similarity:.2f})"
            else:
                tag_text = f"{UNKNOWN_LABEL} ({match_res.similarity:.2f})"

            # Calculate text background box
            bbox = draw.textbbox((x1, y1), tag_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            tag_y1 = max(0, y1 - text_h - 6)
            tag_y2 = y1
            draw.rectangle([x1, tag_y1, x1 + text_w + 8, tag_y2], fill=color)
            draw.text((x1 + 4, tag_y1 + 2), tag_text, fill=(255, 255, 255), font=font)

        return pil_img
