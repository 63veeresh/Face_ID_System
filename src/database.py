"""
Enrolled Face Database Management Module.
Handles persistent storage of enrolled identities, embeddings, reference photos, and metadata.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from .config import DATABASE_FILE, ENROLLED_DIR


class FaceDatabase:
    """
    Manages persistent local storage of enrolled identities.
    Maintains a JSON catalog of identities, metadata, and references to saved embedding arrays.
    """

    def __init__(self, db_file: Path = DATABASE_FILE, storage_dir: Path = ENROLLED_DIR):
        self.db_file = Path(db_file)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Loads records from the JSON database file and restores numpy embeddings."""
        if not self.db_file.exists():
            self.records = {}
            return

        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.records = {}
            for name, entry in data.items():
                emb_path = self.storage_dir / entry.get("embedding_file", f"{name}.npy")
                if emb_path.exists():
                    embedding = np.load(emb_path)
                else:
                    embedding = np.array(entry.get("embedding", []), dtype=np.float32)

                self.records[name] = {
                    "name": name,
                    "embedding": embedding,
                    "image_path": entry.get("image_path"),
                    "enrolled_at": entry.get("enrolled_at"),
                    "sample_count": entry.get("sample_count", 1),
                    "notes": entry.get("notes", ""),
                }
        except Exception as e:
            print(f"[Warning] Failed to load database from {self.db_file}: {e}")
            self.records = {}

    def _save(self) -> None:
        """Saves current database state to JSON and persists numpy embedding files."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        catalog = {}

        for name, entry in self.records.items():
            emb_filename = f"{name}_embedding.npy"
            emb_path = self.storage_dir / emb_filename
            np.save(emb_path, entry["embedding"].astype(np.float32))

            catalog[name] = {
                "name": name,
                "embedding_file": emb_filename,
                "image_path": str(entry.get("image_path")) if entry.get("image_path") else None,
                "enrolled_at": entry.get("enrolled_at"),
                "sample_count": entry.get("sample_count", 1),
                "notes": entry.get("notes", ""),
            }

        temp_file = self.db_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        shutil.move(temp_file, self.db_file)

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        reference_image: Optional[Image.Image] = None,
        notes: str = "",
        allow_update: bool = True,
    ) -> bool:
        """
        Enrolls a new person or updates an existing person's embedding.
        If person already exists and allow_update=True: updates running average centroid of embeddings.
        """
        name = name.strip()
        if not name:
            raise ValueError("Identity name cannot be empty.")

        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm  # Ensure unit norm

        image_path = None
        if reference_image is not None:
            safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).lower()
            img_filename = f"{safe_name}_ref.jpg"
            save_dest = self.storage_dir / img_filename
            reference_image.convert("RGB").save(save_dest, quality=95)
            image_path = str(save_dest)

        if name in self.records and allow_update:
            # Update embedding by computing running mean vector and re-normalizing
            old_entry = self.records[name]
            old_emb = old_entry["embedding"]
            old_count = old_entry.get("sample_count", 1)
            new_count = old_count + 1

            # Running average
            updated_emb = (old_emb * old_count + embedding) / new_count
            updated_emb = updated_emb / np.linalg.norm(updated_emb)

            self.records[name] = {
                "name": name,
                "embedding": updated_emb,
                "image_path": image_path or old_entry.get("image_path"),
                "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": new_count,
                "notes": notes or old_entry.get("notes", ""),
            }
        else:
            self.records[name] = {
                "name": name,
                "embedding": embedding,
                "image_path": image_path,
                "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": 1,
                "notes": notes,
            }

        self._save()
        return True

    def delete(self, name: str) -> bool:
        """Removes a person from the database."""
        if name not in self.records:
            return False

        entry = self.records.pop(name)
        # Clean up files if they exist
        emb_path = self.storage_dir / f"{name}_embedding.npy"
        if emb_path.exists():
            try:
                emb_path.unlink()
            except OSError:
                pass

        if entry.get("image_path"):
            img_path = Path(entry["image_path"])
            if img_path.exists() and img_path.parent == self.storage_dir:
                try:
                    img_path.unlink()
                except OSError:
                    pass

        self._save()
        return True

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves record for a specific identity."""
        return self.records.get(name)

    def list_all(self) -> List[Dict[str, Any]]:
        """Returns list of all enrolled records."""
        return list(self.records.values())

    def get_enrolled_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Returns a tuple (names_list, embeddings_matrix) where:
          - names_list is a list of enrolled person names of length N.
          - embeddings_matrix is a numpy array of shape (N, 512).
        """
        if not self.records:
            return [], np.empty((0, 512), dtype=np.float32)

        names = list(self.records.keys())
        embeddings = np.array([self.records[name]["embedding"] for name in names], dtype=np.float32)
        return names, embeddings

    def clear(self) -> None:
        """Clears all enrolled identities."""
        self.records = {}
        if self.db_file.exists():
            self.db_file.unlink()
        for f in self.storage_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.records)
