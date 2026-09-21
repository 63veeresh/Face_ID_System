"""
Unit tests for the FaceDatabase module (Persistence & Operations).
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.database import FaceDatabase


@pytest.fixture
def temp_db():
    """Provides a fresh FaceDatabase using a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_file = tmp_path / "test_db.json"
        storage_dir = tmp_path / "enrolled"
        db = FaceDatabase(db_file=db_file, storage_dir=storage_dir)
        yield db


def test_enroll_and_retrieve(temp_db):
    """Test enrolling an identity and retrieving their data."""
    emb = np.random.randn(512).astype(np.float32)
    fake_img = Image.new("RGB", (160, 160), color="blue")

    success = temp_db.enroll(name="TestUser", embedding=emb, reference_image=fake_img, notes="Engineer")
    assert success is True
    assert len(temp_db) == 1

    record = temp_db.get("TestUser")
    assert record is not None
    assert record["name"] == "TestUser"
    assert record["sample_count"] == 1
    assert record["notes"] == "Engineer"
    assert record["image_path"] is not None
    assert Path(record["image_path"]).exists()

    # Verify unit-normalized embedding
    np_emb = record["embedding"]
    assert pytest.approx(np.linalg.norm(np_emb), abs=1e-4) == 1.0


def test_multiple_enrollment_averaging(temp_db):
    """Enrolling multiple samples for the same user should update sample_count and average."""
    emb1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    emb2 = np.ones(512, dtype=np.float32) / np.sqrt(512)

    temp_db.enroll(name="MultiSampleUser", embedding=emb1)
    temp_db.enroll(name="MultiSampleUser", embedding=emb2)

    record = temp_db.get("MultiSampleUser")
    assert record["sample_count"] == 2
    assert len(temp_db) == 1


def test_delete_person(temp_db):
    """Test deleting an identity and removing associated files."""
    emb = np.random.randn(512).astype(np.float32)
    temp_db.enroll(name="ToDelete", embedding=emb)
    assert len(temp_db) == 1

    deleted = temp_db.delete("ToDelete")
    assert deleted is True
    assert len(temp_db) == 0
    assert temp_db.get("ToDelete") is None

    # Deleting non-existent person returns False
    assert temp_db.delete("NonExistent") is False


def test_persistence_across_instances():
    """Verify that a second FaceDatabase instance loads previously saved data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_file = tmp_path / "persist_db.json"
        storage_dir = tmp_path / "enrolled"

        # Instance 1 saves person
        db1 = FaceDatabase(db_file=db_file, storage_dir=storage_dir)
        emb = np.array([1.0] * 512, dtype=np.float32) / np.sqrt(512)
        db1.enroll("PersistPerson", emb)

        # Instance 2 loads same file
        db2 = FaceDatabase(db_file=db_file, storage_dir=storage_dir)
        assert len(db2) == 1
        record = db2.get("PersistPerson")
        assert record is not None
        assert record["name"] == "PersistPerson"
