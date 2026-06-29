import os
from datasets import load_dataset

def main():
    os.makedirs("dataset", exist_ok=True)
    print("Loading dataset...")

    dataset = load_dataset("nvidia/OpenMathInstruct-2", "default", split="train")

    print("Saving to disk...")
    dataset.save_to_disk("dataset/openmath_raw")

    print("Done.")

if __name__ == "__main__":
    main()