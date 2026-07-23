"""
One-time (and re-runnable) script: uploads ground_truth.json test cases into a
Langfuse Dataset. Run this once to create the dataset, and again any time you
add new test cases (e.g. after a false-alert incident, per Section 5 of the POC).

Requires:
    pip install langfuse

Environment variables (self-hosted Langfuse, e.g. via Docker Compose):
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000
"""

import json
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

GROUND_TRUTH_PATH = "ground_truth.json"
DATASET_NAME = "chatbot-validation-v1"


def load_ground_truth(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def main():
    langfuse = get_client()
    cases = load_ground_truth(GROUND_TRUTH_PATH)

    # create_dataset is idempotent - safe to call even if it already exists
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Chatbot response validation ground truth (valid / off-topic / jailbreak / leakage / edge cases)",
    )

    for case in cases:
        # both "reference_answer" (valid questions) and "expected_behavior"
        # (everything else) collapse into one "reference" field the judge prompt uses
        reference = case.get("reference_answer", case.get("expected_behavior"))

        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"question": case["question"], "category": case["category"]},
            expected_output=reference,
            metadata={"test_id": case["id"], "category": case["category"]},
            # id here refers to the dataset item's own id derived from test_id,
            # which lets you re-run this script to update an existing case
            # rather than duplicating it - see Langfuse docs on upserting items
        )
        print(f"Uploaded {case['id']} ({case['category']})")

    print(f"\nDone. {len(cases)} test cases in dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()