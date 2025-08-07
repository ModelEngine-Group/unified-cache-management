import json
import argparse
import matplotlib.pyplot as plt
import math
import os

def run(context_lengths: list[str], result_path: str, connector: str, model: str, device: str):

    x_axis = [math.log(int(context_length), 2) for context_length in context_lengths]

    y_axis_raw = []
    y_axis_cache = []

    model_name = os.path.basename(model)

    for context_length in context_lengths:
        filename = f"{result_path}/docqa_TTFT_{context_length}k_{model_name}_{connector}_connector_{device}.jsonl"
        with open(filename, 'r') as file:
            count = 0
            for line in file:
                data = json.loads(line)
                if count == 0:
                    y_axis_raw.append(data[0]["mean ttft"])
                elif count == 1:
                    y_axis_cache.append(data[0]["mean ttft"])
                else:
                    print("ERROR! invalid result file format!")
                count += 1

    y_axis_ratio = [raw / cache for raw, cache in zip(y_axis_raw, y_axis_cache)]

    _, (ax1, ax2) = plt.subplots(2,1,figsize=(8,10))

    ax1.plot(x_axis, y_axis_raw, label="raw reasoning", marker='o')
    ax1.plot(x_axis, y_axis_cache, label="use cache", marker='o')
    ax1.set_title("TTFT_curve_doc_qa")
    ax1.set_xlabel("log_2(context_length[K tokens])")
    ax1.set_ylabel("TTFT(s)")
    ax1.legend(["raw_reasoning", "use cache"])
    ax1.grid(True)

    bars = ax2.bar(x_axis, y_axis_ratio, width=0.4, label="increase factor")
    ax2.set_title("TTFT increase factor of using cache w.r.t. raw reasoning")
    ax2.set_xlabel("log_2(context_length[K tokens])")
    ax2.set_ylabel("Increase factor")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, round(height, 3), ha='center', va='bottom', fontsize=10, color='black')

    plt.savefig(f"{result_path}/docqa_TTFT_{model_name}_{connector}_connector_{device}.png")

    print(f"TTFT figure has been saved to {result_path}/docqa_TTFT_{model_name}_{connector}_connector_{device}.png")

args = argparse.ArgumentParser()

args.add_argument(
    "--context-lengths", type=str, required=True, help="All context lengths to plot TTFT. Use commas to separate them. "
)

args.add_argument(
    "--result-path", type=str, required=True, help="Result path"
)

args.add_argument(
    "--model", type=str, required=True, help="Model to evaluate on"
)

args.add_argument(
    "--connector", type=str, required=True, help="Connector to use"
)

args.add_argument(
    "--device", type=str, required=True, help="Compute device to use"
)

if __name__ == "__main__":

    args = args.parse_args()

    context_lengths = args.context_lengths.split(',')
    result_path = args.result_path
    connector = args.connector
    model = args.model
    device = args.device

    run(context_lengths, result_path, connector, model, device)