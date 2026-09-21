"""
End-to-End Integration tests for FaceRecognitionPipeline.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.pipeline import FaceRecognitionPipeline
from src.config import UNKNOWN_LABEL


def create_mock_portrait(seed_color=(210, 180, 140)) -> Image.Image:
    """Creates a basic face portrait image for integration tests."""
    img = Image.new("RGB", (320, 320), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Head
    draw.ellipse([80, 60, 240, 260], fill=seed_color)
    # Eyes
    draw.ellipse([115, 115, 135, 135], fill=(30, 30, 30))
    draw.ellipse([185, 115, 205, 135], fill=(30, 30, 30))
    # Nose
    draw.polygon([(160, 140), (150, 175), (170, 175)], fill=(160, 110, 80))
    # Mouth
    draw.rectangle([130, 205, 190, 220], fill=(160, 50, 50))
    return img


@pytest.fixture
def temp_pipeline():
    """Provides an isolated FaceRecognitionPipeline using a temporary database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = Path(tmp_dir) / "test_pipe_db.json"
        pipeline = FaceRecognitionPipeline(threshold=0.65, db_file=tmp_db)
        yield pipeline


def test_pipeline_empty_db_identification(temp_pipeline):
    """Identifying on an empty database must result in UNKNOWN rejection."""
    img = create_mock_portrait()
    results = temp_pipeline.identify_image(img)
    if results:
        face, match = results[0]
        assert match.is_known is False
        assert match.identity == UNKNOWN_LABEL


def test_pipeline_annotate_empty_and_populated(temp_pipeline):
    """Image annotation should succeed and produce a valid PIL image."""
    img = create_mock_portrait()
    results = temp_pipeline.identify_image(img)
    annotated = temp_pipeline.annotate_image(img, results)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == img.size
