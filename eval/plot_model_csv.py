import argparse
import csv
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Inference Plotter")
    parser.add_argument("csv", help="the path to the csv.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    in_sizes = []
    inference_times = []
    out_sizes = []

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            in_sizes.append(int(row["in_size"]) / 1024)
            inference_times.append(float(row["inference_time"]))
            out_sizes.append(int(row["out_size"]) / 1024)
            

    x_index = list(range(len(inference_times)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows, 1 column

    # 1) Fragment index vs inference time
    axes[0].plot(x_index, inference_times, marker="o")
    axes[0].set_title("Inference Time")
    # axes[0].set_xlabel("Fragment Index")
    axes[0].set_ylabel("Milliseconds (ms)")
    axes[0].grid(True)

    # 2) Input size vs inference time
    axes[1].plot(x_index, in_sizes, marker="o")
    axes[1].set_title("Input-Fragment Size")
    # axes[1].set_xlabel("Fragment Index")
    axes[1].set_ylabel("Size (KB)")
    axes[1].grid(True)

    # 3) Output size vs inference time
    axes[2].plot(x_index, out_sizes, marker="o")
    axes[2].set_title("Output-Fragment Size")
    # axes[2].set_xlabel("Fragment Index")
    axes[2].set_ylabel("Size (KB)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
