import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter


def read_server_agg_metrics(csv_path="server_agg_metrics.csv"):
    csv_path = Path(csv_path)
    rows_by_round = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("server_round"):
                continue
            r = int(row["server_round"])
            rows_by_round[r] = row

    rounds = sorted(rows_by_round.keys())
    return {
        "rounds": rounds,
        "eval_loss": [float(rows_by_round[r]["eval_loss"]) for r in rounds],
        "eval_acc": [float(rows_by_round[r]["eval_acc"]) for r in rounds],
        "eval_precision": [float(rows_by_round[r]["eval_precision"]) for r in rounds],
        "eval_recall": [float(rows_by_round[r]["eval_recall"]) for r in rounds],
        "eval_f1": [float(rows_by_round[r]["eval_f1"]) for r in rounds],
    }



def read_client_metrics_timeseries(csv_path="client_metrics.csv", metric="eval_acc"):
    csv_path = Path(csv_path)
    client_round_val = defaultdict(dict) 

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("server_round"):
                continue
            val_str = row.get(metric, "")
            if val_str in ("", None):
                continue

            r = int(row["server_round"])
            cid = int(row["client_id"])
            client_round_val[cid][r] = float(val_str)

    client_data = {}
    for cid, rv in client_round_val.items():
        rounds = sorted(rv.keys())
        client_data[cid] = {"rounds": rounds, "values": [rv[r] for r in rounds]}
    return client_data



def plot_server_utility_metric(
    metric: str = "eval_acc",
    csv_path: str | Path = "server_agg_metrics.csv",
    save_path: str | None = None,
) -> None:
    data = read_server_agg_metrics(csv_path)
    rounds = data["rounds"]

    if metric not in data:
        raise ValueError(f"Metric '{metric}' not found in server_agg_metrics.csv")

    values = data[metric]

    plt.figure(figsize=(6, 4))
    plt.plot(rounds, values, marker="o")
    plt.xlabel("Server round")
    plt.ylabel(metric)
    plt.title(f"Server aggregated {metric} vs rounds")
    plt.grid(True)

    ax = plt.gca()

    if metric == "eval_loss":
        ymin = 0.0
        ymax = max(values) * 1.1 if values else 1.0
    else:
        vmax = max(values) if values else 1.0
        if vmax <= 1.0:
            ymin, ymax = 0.0, 1.0
        else:
            ymin, ymax = 0.0, vmax * 1.1

    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    plt.tight_layout()

    if save_path is None:
        save_path = f"../res_img/server_{metric}_vs_round.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] Saved server {metric} curve to: {save_path}")


def plot_client_and_server_utility(
    metric: str = "eval_acc",
    client_csv: str | Path = "client_metrics.csv",
    server_csv: str | Path = "server_agg_metrics.csv",
    save_path: str | None = None,
) -> None:
    server_data = read_server_agg_metrics(server_csv)
    rounds = server_data["rounds"]
    if metric not in server_data:
        raise ValueError(f"Metric '{metric}' not found in server_agg_metrics.csv")
    server_vals = server_data[metric]

    client_series = read_client_metrics_timeseries(client_csv, metric)

    plt.figure(figsize=(7, 5))

    for cid, series in sorted(client_series.items(), key=lambda x: x[0]):
        plt.plot(
            series["rounds"],
            series["values"],
            marker="o",
            linewidth=1,
            label=f"Client {cid}",
        )

    plt.plot(
        rounds,
        server_vals,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Server aggregated",
    )

    plt.xlabel("Server round")
    plt.ylabel(metric)
    plt.title(f"Client vs Server {metric} per round")
    plt.grid(True)

    ax = plt.gca()

    all_vals = list(server_vals)
    for series in client_series.values():
        all_vals.extend(series["values"])

    if not all_vals:
        ymin, ymax = 0.0, 1.0
    else:
        if metric == "eval_loss":
            ymin = 0.0
            ymax = max(all_vals) * 1.1
        else:
            vmax = max(all_vals)
            if vmax <= 1.0:
                ymin, ymax = 0.0, 1.0
            else:
                ymin, ymax = 0.0, vmax * 1.1

    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = f"../res_img/client_server_{metric}_vs_round.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] Saved client+server {metric} plot to: {save_path}")


def main():
    metrics = ["eval_loss", "eval_acc", "eval_precision", "eval_recall", "eval_f1"]

    for m in metrics:
        plot_server_utility_metric(metric=m, csv_path="server_agg_metrics.csv")
        plot_client_and_server_utility(
            metric=m,
            client_csv="client_metrics.csv",
            server_csv="server_agg_metrics.csv",
        )


if __name__ == "__main__":
    main()
