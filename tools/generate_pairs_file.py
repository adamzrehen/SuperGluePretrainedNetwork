# save this as generate_pairs.py

import argparse
import os


def generate_image_pairs(n: int, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i in range(1, n - 30):
            img1 = f"{i:07d}.png"
            img2 = f"{i + 30 + 1:07d}.png"
            f.write(f"{img1} {img2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate consecutive image filename pairs.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output text file.")
    parser.add_argument("--n", type=int, default=600, help="Number of image files to process (default: 600).")

    args = parser.parse_args()

    generate_image_pairs(args.n, args.output)
