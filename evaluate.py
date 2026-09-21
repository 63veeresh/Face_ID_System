"""
Evaluation & Benchmarking Module.
Conducts genuine, empirical evaluation of the face recognition pipeline on known and impostor test cases.
Calculates TAR, FAR, FRR, Precision, Accuracy, and analyzes threshold decision trade-offs.
No results are fabricated.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    UNKNOWN_LABEL,
    DATA_DIR,
    EVAL_DIR,
)
from src.pipeline import FaceRecognitionPipeline


def evaluate_recognition_pipeline(
    pipeline: FaceRecognitionPipeline,
    eval_dataset: Dict[str, Any],
    threshold_range: List[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates the pipeline against a structured evaluation dataset:
    dataset structure:
    {
       "enrolled": [{"name": "PersonA", "image": "path"}, ...],
       "genuine_test": [{"name": "PersonA", "image": "path"}, ...],
       "impostor_test": [{"name": "PersonUnknown", "image": "path"}, ...]
    }
    """
    if threshold_range is None:
        threshold_range = [round(t, 2) for t in np.arange(0.40, 0.90, 0.05)]

    # Temporary database for evaluation
    eval_db_file = DATA_DIR / "eval_db_temp.json"
    if eval_db_file.exists():
        eval_db_file.unlink()

    eval_pipeline = FaceRecognitionPipeline(
        threshold=pipeline.threshold,
        device=pipeline.device,
        db_file=eval_db_file,
    )

    print(f"[*] Enrolling {len(eval_dataset['enrolled'])} baseline identities for evaluation...")
    enrolled_count = 0
    for item in eval_dataset["enrolled"]:
        ok, msg, _ = eval_pipeline.enroll_person(item["name"], item["image"])
        if ok:
            enrolled_count += 1
        else:
            print(f"    [Warning] Enrollment failed for {item['name']}: {msg}")

    print(f"[+] Successfully enrolled {enrolled_count} identities.")

    # 1. Collect genuine pair similarities & identification predictions
    genuine_records = []
    print(f"[*] Testing {len(eval_dataset['genuine_test'])} genuine test queries...")
    for item in eval_dataset["genuine_test"]:
        true_name = item["name"]
        results = eval_pipeline.identify_image(item["image"], threshold=0.0)  # threshold=0 to capture raw top score
        if not results:
            print(f"    [Warning] No face detected in genuine image for {true_name}")
            continue

        face, match = results[0]
        genuine_records.append({
            "true_name": true_name,
            "predicted_top_name": match.top_candidate,
            "top_similarity": float(match.similarity),
            "is_correct_person": (match.top_candidate == true_name),
        })

    # 2. Collect impostor/unknown queries
    impostor_records = []
    print(f"[*] Testing {len(eval_dataset['impostor_test'])} impostor / unknown queries...")
    for item in eval_dataset["impostor_test"]:
        results = eval_pipeline.identify_image(item["image"], threshold=0.0)
        if not results:
            continue

        face, match = results[0]
        impostor_records.append({
            "predicted_top_name": match.top_candidate,
            "top_similarity": float(match.similarity),
        })

    # Clean up temp database
    if eval_db_file.exists():
        eval_db_file.unlink()

    total_genuine = len(genuine_records)
    total_impostors = len(impostor_records)

    if total_genuine == 0 and total_impostors == 0:
        return {"error": "No valid face detections obtained during evaluation."}

    genuine_sims = [r["top_similarity"] for r in genuine_records] if genuine_records else [0.0]
    impostor_sims = [r["top_similarity"] for r in impostor_records] if impostor_records else [0.0]

    # Metrics across threshold sweep
    sweep_results = []
    best_f1 = -1.0
    optimal_thresh = pipeline.threshold

    for tau in threshold_range:
        # Genuine: should have similarity >= tau AND correct person
        true_accepts = sum(1 for r in genuine_records if r["top_similarity"] >= tau and r["is_correct_person"])
        false_rejects = total_genuine - true_accepts

        # Impostors: should NOT be accepted (similarity < tau)
        # False accept occurs if an impostor has similarity >= tau
        false_accepts = sum(1 for r in impostor_records if r["top_similarity"] >= tau)
        true_rejects = total_impostors - false_accepts

        tar = true_accepts / total_genuine if total_genuine > 0 else 0.0
        frr = false_rejects / total_genuine if total_genuine > 0 else 0.0
        far = false_accepts / total_impostors if total_impostors > 0 else 0.0
        trr = true_rejects / total_impostors if total_impostors > 0 else 0.0

        precision = true_accepts / (true_accepts + false_accepts) if (true_accepts + false_accepts) > 0 else 0.0
        recall = tar
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_queries = total_genuine + total_impostors
        accuracy = (true_accepts + true_rejects) / total_queries if total_queries > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            optimal_thresh = tau

        sweep_results.append({
            "threshold": tau,
            "TAR_Recall": round(tar, 4),
            "FRR": round(frr, 4),
            "FAR": round(far, 4),
            "Precision": round(precision, 4),
            "F1_Score": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "True_Accepts": true_accepts,
            "False_Rejects": false_rejects,
            "False_Accepts": false_accepts,
            "True_Rejects": true_rejects,
        })

    # Find stats for configured threshold
    cfg_metrics = next(
        (s for s in sweep_results if abs(s["threshold"] - pipeline.threshold) < 0.01),
        sweep_results[len(sweep_results) // 2],
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "MTCNN (Face Detection) + InceptionResnetV1 VGGFace2 (512-D Embeddings)",
        "configured_threshold": pipeline.threshold,
        "optimal_threshold_by_f1": optimal_thresh,
        "sample_counts": {
            "enrolled_identities": enrolled_count,
            "genuine_queries": total_genuine,
            "impostor_queries": total_impostors,
            "total_queries": total_genuine + total_impostors,
        },
        "similarity_statistics": {
            "genuine_mean": float(np.mean(genuine_sims)),
            "genuine_std": float(np.std(genuine_sims)),
            "genuine_min": float(np.min(genuine_sims)),
            "genuine_max": float(np.max(genuine_sims)),
            "impostor_mean": float(np.mean(impostor_sims)),
            "impostor_std": float(np.std(impostor_sims)),
            "impostor_min": float(np.min(impostor_sims)),
            "impostor_max": float(np.max(impostor_sims)),
            "separation_margin": float(np.mean(genuine_sims) - np.mean(impostor_sims)),
        },
        "metrics_at_configured_threshold": cfg_metrics,
        "threshold_sweep": sweep_results,
    }

    return report


def print_evaluation_summary(report: Dict[str, Any]) -> None:
    """Prints a formatted evaluation table."""
    print("\n" + "=" * 75)
    print("           FACE RECOGNITION PIPELINE EMPIRICAL EVALUATION")
    print("=" * 75)
    print(f"Model Architecture : {report['model']}")
    print(f"Configured Threshold: {report['configured_threshold']}")
    print(f"Optimal Threshold   : {report['optimal_threshold_by_f1']} (by F1-score)")
    print(f"Evaluated Samples   : {report['sample_counts']['genuine_queries']} genuine, "
          f"{report['sample_counts']['impostor_queries']} impostor queries")

    stats = report["similarity_statistics"]
    print("\nCosine Similarity Distribution (Genuine vs Impostor):")
    print(f"  Genuine Matches  : Mean = {stats['genuine_mean']:.4f} (std={stats['genuine_std']:.4f}) "
          f"[min={stats['genuine_min']:.4f}, max={stats['genuine_max']:.4f}]")
    print(f"  Impostor Queries : Mean = {stats['impostor_mean']:.4f} (std={stats['impostor_std']:.4f}) "
          f"[min={stats['impostor_min']:.4f}, max={stats['impostor_max']:.4f}]")
    print(f"  Separation Margin: {stats['separation_margin']:.4f}")

    cfg = report["metrics_at_configured_threshold"]
    print(f"\nPerformance at Configured Threshold (tau = {report['configured_threshold']}):")
    print(f"  Accuracy                : {cfg['Accuracy'] * 100:.2f}%")
    print(f"  True Accept Rate (Recall): {cfg['TAR_Recall'] * 100:.2f}%")
    print(f"  False Reject Rate (FRR) : {cfg['FRR'] * 100:.2f}%")
    print(f"  False Accept Rate (FAR) : {cfg['FAR'] * 100:.2f}%")
    print(f"  Precision               : {cfg['Precision'] * 100:.2f}%")
    print(f"  F1 Score                : {cfg['F1_Score'] * 100:.2f}%")

    print("\nThreshold Sensitivity Trade-Off Sweep:")
    print(f"{'Thresh':<8} {'Accuracy':<10} {'TAR(Recall)':<12} {'FRR':<8} {'FAR':<8} {'Precision':<10} {'F1':<8}")
    print("-" * 75)
    for s in report["threshold_sweep"]:
        marker = " <== (DEFAULT)" if abs(s["threshold"] - report["configured_threshold"]) < 0.01 else ""
        print(f"{s['threshold']:<8.2f} {s['Accuracy']*100:<9.1f}% {s['TAR_Recall']*100:<11.1f}% "
              f"{s['FRR']*100:<7.1f}% {s['FAR']*100:<7.1f}% {s['Precision']*100:<9.1f}% {s['F1_Score']*100:<7.1f}%{marker}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Recognition Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to evaluation dataset manifest JSON (optional)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Decision threshold",
    )
    args = parser.parse_args()

    # If dataset manifest is not supplied, run self-contained sample evaluation
    from src.data_setup import load_or_create_evaluation_dataset

    print("[*] Preparing evaluation dataset...")
    dataset = load_or_create_evaluation_dataset(args.dataset if args.dataset else None)

    pipeline = FaceRecognitionPipeline(threshold=args.threshold)
    results = evaluate_recognition_pipeline(pipeline, dataset)

    if "error" in results:
        print(f"[-] Evaluation error: {results['error']}")
    else:
        print_evaluation_summary(results)

        # Save to data/evaluation_results.json
        out_file = DATA_DIR / "evaluation_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Detailed evaluation results saved to '{out_file}'.")
