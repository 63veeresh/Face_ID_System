"""
Command-Line Interface for the Face Recognition & Identification System.
Supports enrollment, identification, verification, listing, and database management.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_SIMILARITY_THRESHOLD, UNKNOWN_LABEL
from src.pipeline import FaceRecognitionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Local Face Recognition & Identification System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a person into the database")
    enroll_parser.add_argument("--name", type=str, required=True, help="Full name or identity tag")
    enroll_parser.add_argument("--image", type=str, required=True, help="Path to face image")
    enroll_parser.add_argument("--notes", type=str, default="", help="Optional notes or department")

    # Command: identify
    id_parser = subparsers.add_parser("identify", help="Identify faces in an input image")
    id_parser.add_argument("--image", type=str, required=True, help="Path to query image")
    id_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold for recognition [0.0 to 1.0]",
    )
    id_parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional destination path to save annotated output image",
    )

    # Command: verify
    verify_parser = subparsers.add_parser("verify", help="1:1 Verification of claimed identity")
    verify_parser.add_argument("--name", type=str, required=True, help="Claimed identity name")
    verify_parser.add_argument("--image", type=str, required=True, help="Path to verification image")
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold [0.0 to 1.0]",
    )

    # Command: list
    subparsers.add_parser("list", help="List all enrolled identities")

    # Command: delete
    del_parser = subparsers.add_parser("delete", help="Delete an enrolled person")
    del_parser.add_argument("--name", type=str, required=True, help="Name of identity to remove")

    # Command: clear
    subparsers.add_parser("clear", help="Clear all enrolled identities from database")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = FaceRecognitionPipeline()

    if args.command == "enroll":
        print(f"[*] Processing enrollment for '{args.name}'...")
        success, message, face = pipeline.enroll_person(
            name=args.name,
            image_input=args.image,
            notes=args.notes,
        )
        if success:
            print(f"[+] SUCCESS: {message}")
            if face:
                print(f"    Detected face bbox: {face.box}, detection confidence: {face.score:.3f}")
        else:
            print(f"[-] FAILED: {message}")
            sys.exit(1)

    elif args.command == "identify":
        print(f"[*] Analyzing '{args.image}' (threshold = {args.threshold})...")
        results = pipeline.identify_image(args.image, threshold=args.threshold)

        if not results:
            print("[-] No faces detected in the provided image.")
            return

        print(f"[+] Detected {len(results)} face(s):")
        for i, (face, match) in enumerate(results, start=1):
            status = "RECOGNIZED" if match.is_known else "REJECTED (UNKNOWN)"
            print(f"  Face #{i}:")
            print(f"    Bounding Box: {face.box} (conf: {face.score:.2f})")
            print(f"    Predicted Identity: {match.identity}")
            print(f"    Status: {status}")
            print(f"    Cosine Similarity: {match.similarity:.4f} (Threshold: {match.threshold:.4f})")
            print(f"    Closest Match Candidate: {match.top_candidate}")
            if match.rejection_reason:
                print(f"    Rejection Reason: {match.rejection_reason}")

        if args.output:
            annotated = pipeline.annotate_image(args.image, results)
            out_path = Path(args.output)
            annotated.save(out_path)
            print(f"[+] Saved annotated visualization to '{out_path}'.")

    elif args.command == "verify":
        print(f"[*] Verifying claim: identity is '{args.name}'...")
        is_match, sim, msg = pipeline.verify_identity(
            image_input=args.image,
            claimed_name=args.name,
            threshold=args.threshold,
        )
        if is_match:
            print(f"[+] {msg}")
        else:
            print(f"[-] {msg}")

    elif args.command == "list":
        records = pipeline.database.list_all()
        print(f"\nEnrolled Database Records ({len(records)} total):")
        print("-" * 65)
        if not records:
            print("No identities currently enrolled.")
        else:
            print(f"{'Name':<25} {'Samples':<10} {'Enrolled Date':<20} {'Notes'}")
            print("-" * 65)
            for r in records:
                print(f"{r['name']:<25} {r.get('sample_count', 1):<10} {str(r.get('enrolled_at', 'N/A')):<20} {r.get('notes', '')}")

    elif args.command == "delete":
        if pipeline.database.delete(args.name):
            print(f"[+] Successfully deleted '{args.name}' from database.")
        else:
            print(f"[-] Identity '{args.name}' was not found in the database.")

    elif args.command == "clear":
        pipeline.database.clear()
        print("[+] All enrolled records and cached files have been cleared.")


if __name__ == "__main__":
    main()
