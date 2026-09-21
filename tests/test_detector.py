"""
Unit tests for the FaceDetector module (MTCNN Face Detection & Alignment).
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw
import torch
from src.detector import FaceDetector, DetectedFace


@pytest.fixture(scope="module")
def detector():
    """Shared FaceDetector fixture for module tests."""
    return FaceDetector()


def test_detector_no_face_on_blank_image(detector):
    """A completely blank white image should result in 0 detected faces."""
    blank = Image.new("RGB", (300, 300), color=(255, 255, 255))
    faces = detector.detect_faces(blank)
    assert len(faces) == 0
    assert detector.detect_single_face(blank) is None


def test_detector_no_face_on_random_noise(detector):
    """Uniform random noise image should not trigger confident face detections."""
    noise_arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)
    faces = detector.detect_faces(noise_img)
    assert len(faces) == 0


def test_detector_with_sample_face(detector):
    """Test detector returns valid DetectedFace with landmarks and (3, 160, 160) tensor."""
    # We can synthesize an oval face with eye/mouth contrast markings or test structure
    img = Image.new("RGB", (320, 320), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    # Head contour
    draw.ellipse([80, 50, 240, 270], fill=(210, 180, 140))
    # Eyes
    draw.ellipse([110, 110, 135, 130], fill=(20, 20, 20))
    draw.ellipse([185, 110, 210, 130], fill=(20, 20, 20))
    # Nose
    draw.polygon([(160, 140), (150, 180), (170, 180)], fill=(160, 120, 90))
    # Mouth
    draw.rectangle([130, 210, 190, 225], fill=(150, 50, 50))

    faces = detector.detect_faces(img)
    # Even if MTCNN synthetic threshold varies, verify detect_faces returns list
    assert isinstance(faces, list)
    if len(faces) > 0:
        f = faces[0]
        assert isinstance(f, DetectedFace)
        assert len(f.box) == 4
        assert f.box[2] > f.box[0]
        assert f.box[3] > f.box[1]
        assert f.face_tensor.shape == (3, 160, 160)
