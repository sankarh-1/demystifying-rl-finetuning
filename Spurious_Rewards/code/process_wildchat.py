import argparse
import json
import os
import sys

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/WildChat")
    parser.add_argument("--deepscaler_test", default="data/DeepScaleR_mv_labeled_qwen2.5_math_7b/test.json",
                        help="Path to DeepScaleR test.json to copy as test split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading WildChat dataset...")
    ds = load_dataset("allenai/WildChat", split="train")

    data = []
    for example in ds:
        if example["language"] != "English":
            continue
        if example["toxic"] or example["redacted"]:
            continue
        conv = example["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        prompt = conv[0]["content"].strip()
        if not prompt:
            continue
        data.append({
            "prompt": prompt,
            "answer": "",
            "source": "wildchat",
            "id": example["conversation_id"],
        })

    if not data:
        print("ERROR: No examples after filtering. Check HF authentication or dataset availability.", file=sys.stderr)
        sys.exit(1)

    print(f"Total examples after filtering: {len(data)}")

    train_path = os.path.join(args.output_dir, "train.json")
    with open(train_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {train_path}")

    if args.deepscaler_test:
        import shutil
        test_path = os.path.join(args.output_dir, "test.json")
        shutil.copy(args.deepscaler_test, test_path)
        print(f"Copied DeepScaleR test split to {test_path}")


if __name__ == "__main__":
    main()
