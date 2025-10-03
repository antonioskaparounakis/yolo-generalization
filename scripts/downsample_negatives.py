import argparse
import random
import shutil
from pathlib import Path


def downsample_negatives(src_dataset_dir: Path, keep_fraction: float, seed: int = 42):
    dst_dataset_dir = src_dataset_dir.parent / f"{src_dataset_dir.stem}_downsampled"

    random.seed(seed)

    for split in ["train", "val", "test"]:
        src_img_dir = src_dataset_dir / "images" / split
        src_lbl_dir = src_dataset_dir / "labels" / split
        dst_img_dir = dst_dataset_dir / "images" / split
        dst_lbl_dir = dst_dataset_dir / "labels" / split

        # Skip missing splits
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        negatives, positives = [], []

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in src_img_dir.glob(ext):
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.stat().st_size == 0:  # empty -> negative
                    negatives.append((img_path, lbl_path))
                else:
                    positives.append((img_path, lbl_path))

        # Downsample negatives
        num_keep = int(len(negatives) * keep_fraction)
        kept_negatives = random.sample(negatives, num_keep) if negatives else []

        # Combine positives with kept negatives
        final_samples = positives + kept_negatives

        # Print stats
        print(
            f"[{split}] "
            f"positives: {len(positives)}, "
            f"negatives: {len(negatives)} → kept {len(kept_negatives)}, "
            f"final total: {len(final_samples)}"
        )

        # Copy images and labels
        for img_path, lbl_path in final_samples:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    for yaml_file in src_dataset_dir.glob("*.yaml"):
        new_name = yaml_file.stem + "_downsampled" + yaml_file.suffix
        shutil.copy2(yaml_file, dst_dataset_dir / new_name)

    return dst_dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Downsample negatives in YOLO dataset"
    )
    parser.add_argument(
        "src_dataset_dir",
        type=Path,
        help="Path to the source YOLO dataset directory",
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.2,
        help="Fraction of negatives to keep (default: 0.2)",
    )

    args = parser.parse_args()

    dst_dataset_dir = downsample_negatives(
        src_dataset_dir=args.src_dataset_dir,
        keep_fraction=args.keep_fraction,
    )

    print(f"Saved downsampled YOLO dataset at: {dst_dataset_dir}")


if __name__ == "__main__":
    main()
