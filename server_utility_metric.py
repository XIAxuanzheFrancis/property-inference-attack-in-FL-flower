import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt



def read_server_agg_metrics(csv_path: str | Path = "server_agg_metrics.csv") -> Dict[str, List[float]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path.resolve()}")

    rounds: List[int] = []
    loss: List[float] = []
    acc: List[float] = []
    prec: List[float] = []
    rec: List[float] = []
    f1: List[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("server_round", "") == "":
                continue
            r = int(row["server_round"])
            rounds.append(r)
            loss.append(float(row["eval_loss"]))
            acc.append(float(row["eval_acc"]))
            prec.append(float(row["eval_precision"]))
            rec.append(float(row["eval_recall"]))
            f1.append(float(row["eval_f1"]))

    order = sorted(range(len(rounds)), key=lambda i: rounds[i])
    rounds = [rounds[i] for i in order]

    return {
        "rounds": rounds,
        "eval_loss": [loss[i] for i in order],
        "eval_acc": [acc[i] for i in order],
        "eval_precision": [prec[i] for i in order],
        "eval_recall": [recs for recs in [rec[i] for i in order]],
        "eval_f1": [f1[i] for i in order],
    }


def read_client_metrics_for_metric(
    csv_path: str | Path = "client_metrics.csv",
    metric: str = "eval_acc",
) -> Dict[int, List[float]]:

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path.resolve()}")

    round_to_values: Dict[int, List[float]] = defaultdict(list)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("server_round", "") == "":
                continue
            r = int(row["server_round"])
            val_str = row.get(metric, "")
            if val_str == "" or val_str is None:
                continue
            round_to_values[r].append(float(val_str))

    return round_to_values


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

    client_round_to_vals = read_client_metrics_for_metric(client_csv, metric)

    plt.figure(figsize=(7, 5))

    xs = []
    ys = []
    for r in rounds:
        vals = client_round_to_vals.get(r, [])
        xs.extend([r] * len(vals))
        ys.extend(vals)
    if xs:
        plt.scatter(xs, ys, alpha=0.4, label="Clients", s=20)

    plt.plot(rounds, server_vals, marker="o", label="Server aggregated", linewidth=2)

    plt.xlabel("Server round")
    plt.ylabel(metric)
    plt.title(f"Client vs Server {metric} per round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = f"../res_img/client_server_{metric}_vs_round.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] Saved client+server {metric} plot to: {save_path}")



def main():
    for m in ["eval_acc", "eval_f1"]:
        plot_server_utility_metric(metric=m, csv_path="server_agg_metrics.csv")
        plot_client_and_server_utility(
            metric=m,
            client_csv="client_metrics.csv",
            server_csv="server_agg_metrics.csv",
        )


if __name__ == "__main__":
    main()
