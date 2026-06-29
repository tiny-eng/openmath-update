from datasets import load_from_disk, concatenate_datasets
import glob

shard_paths = sorted(glob.glob("dataset/updated_openmath/model_label_shard_*"))
datasets = [load_from_disk(p) for p in shard_paths]
full_dataset = concatenate_datasets(datasets)
full_dataset.save_to_disk("dataset/openmath_merged_dataset")
print("Merged dataset saved.")