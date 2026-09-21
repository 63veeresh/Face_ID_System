"""
Data Setup & Evaluation Dataset Provisioning Module.
Downloads or generates clean evaluation sets featuring genuine pairs and impostor/unknown cases.
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from .config import DATA_DIR, EVAL_DIR, TEST_SAMPLES_DIR


# Curated public domain sample portraits from Wikimedia Commons for reproducible local evaluation
SAMPLE_DATASET_CONFIG = {
    "enrolled": [
        {
            "name": "Alan_Turing",
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Alan_Turing_Aged_16.jpg",
            "filename": "alan_turing_1.jpg",
        },
        {
            "name": "Ada_Lovelace",
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg",
            "filename": "ada_lovelace_1.jpg",
        },
        {
            "name": "Grace_Hopper",
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Commodore_Grace_M._Hopper%2C_USN_%28covered%29.jpg",
            "filename": "grace_hopper_1.jpg",
        },
    ],
    "genuine_test": [
        {
            "name": "Alan_Turing",
            "url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Alan_Turing_az_1930-as_%C3%A9vekben.jpg",
            "filename": "alan_turing_2.jpg",
        },
        {
            "name": "Ada_Lovelace",
            "url": "https://upload.wikimedia.org/wikipedia/commons/b/b7/Ada_Byron_daguerreotype_by_Antoine_Claudet_1843_or_1850_-_cropped.png",
            "filename": "ada_lovelace_2.jpg",
        },
        {
            "name": "Grace_Hopper",
            "url": "https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg",
            "filename": "grace_hopper_2.jpg",
        },
    ],
    "impostor_test": [
        {
            "name": "Claude_Shannon",
            "url": "https://upload.wikimedia.org/wikipedia/commons/9/99/ClaudeShannon_MFO3807.jpg",
            "filename": "claude_shannon.jpg",
        },
        {
            "name": "Nikola_Tesla",
            "url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Tesla_circa_1890.jpeg",
            "filename": "nikola_tesla.jpg",
        },
    ],
}


def download_file(url: str, dest_path: Path) -> bool:
    """Downloads a file with standard user-agent header."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FaceRecognitionProject/1.0 (Educational Assessment)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"[Warning] Could not download {url}: {e}")
        return False


def load_or_create_evaluation_dataset(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads an evaluation dataset manifest or populates the default Wikimedia benchmark suite.
    """
    manifest_file = Path(manifest_path) if manifest_path else DATA_DIR / "eval_manifest.json"

    # If manifest already exists on disk, load and verify files
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            # Check if all files exist
            all_exist = True
            for group in ("enrolled", "genuine_test", "impostor_test"):
                for item in manifest.get(group, []):
                    if not Path(item["image"]).exists():
                        all_exist = False
                        break
            if all_exist:
                return manifest
        except Exception:
            pass

    # Build standard benchmark set
    benchmark_dir = EVAL_DIR / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {"enrolled": [], "genuine_test": [], "impostor_test": []}

    print("[*] Downloading benchmark evaluation face images...")
    for group_name, items in SAMPLE_DATASET_CONFIG.items():
        for item in items:
            target_path = benchmark_dir / item["filename"]
            success = download_file(item["url"], target_path)
            if success:
                # Standardize to RGB JPEG if needed
                try:
                    with Image.open(target_path) as img:
                        rgb = img.convert("RGB")
                        rgb.save(target_path, "JPEG")
                    manifest_data[group_name].append({
                        "name": item["name"],
                        "image": str(target_path.resolve()),
                    })
                except Exception as e:
                    print(f"Error processing {target_path}: {e}")

    # Save manifest
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[+] Benchmark evaluation dataset initialized with {len(manifest_data['enrolled'])} enrolled, "
          f"{len(manifest_data['genuine_test'])} genuine, and {len(manifest_data['impostor_test'])} impostor images.")
    return manifest_data


if __name__ == "__main__":
    ds = load_or_create_evaluation_dataset()
    print("Dataset setup completed successfully.")
