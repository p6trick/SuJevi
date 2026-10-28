"""Run the image questions from request.json with native Jev-Omni output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi import LocalJevOmni


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "examples" / "cat.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Jev-Omni the same image questions used by the OneJev example."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"local image path (default: {DEFAULT_IMAGE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"image does not exist: {image_path}\n"
            "Copy the test image there or pass --image /path/to/image."
        )

    # Load once and reuse the resident model for all three questions.
    model = LocalJevOmni()
    state = "A picture a user uploaded to a pet adoption site."
    questions = {
        "animal": {
            "question": "The photo shows an animal.",
            "options": ["Yes", "No"],
        },
        "species": {
            "question": "Which animal is it?",
            "options": ["cat", "dog", "rabbit", "other"],
        },
        "quality": {
            "question": "How good is the photo for an adoption listing?",
            "options": ["unusable", "poor", "fine", "great"],
        },
    }

    answers = {}
    for name, item in questions.items():
        answers[name] = model.predict(
            state=state,
            question=item["question"],
            options=item["options"],
            media=image_path,
            modality="image",
        )

    print(json.dumps({"model": "Jev-Omni", "answers": answers}, indent=2))


if __name__ == "__main__":
    main()
