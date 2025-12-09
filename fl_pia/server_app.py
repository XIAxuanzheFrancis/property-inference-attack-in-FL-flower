"""fl-pia: A Flower / PyTorch app."""

import torch
import csv
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
# from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_pia.task import Net


METRICS_CSV = Path("client_metrics.csv")

# Create ServerApp
app = ServerApp()

def evaluate_metrics_aggr_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:

    header = [
        "server_round",
        "client_id",
        "num_examples",
        "eval_loss",
        "eval_acc",
        "eval_precision",
        "eval_recall",
        "eval_f1",
    ]

    file_exists = METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        total_examples = 0
        sum_loss = sum_acc = sum_prec = sum_rec = sum_f1 = 0.0

        for record in records:
            for metric_record in record.metric_records.values():
                server_round = metric_record.get("server_round", -1)
                client_id = metric_record.get("client_id", -1)
                num_examples = metric_record.get("num-examples", 0)

                row = [
                    server_round,
                    client_id,
                    num_examples,
                    metric_record.get("eval_loss", None),
                    metric_record.get("eval_acc", None),
                    metric_record.get("eval_precision", None),
                    metric_record.get("eval_recall", None),
                    metric_record.get("eval_f1", None),
                ]
                writer.writerow(row)

                total_examples += num_examples
                sum_loss += metric_record.get("eval_loss", 0.0) * num_examples
                sum_acc += metric_record.get("eval_acc", 0.0) * num_examples
                sum_prec += metric_record.get("eval_precision", 0.0) * num_examples
                sum_rec += metric_record.get("eval_recall", 0.0) * num_examples
                sum_f1 += metric_record.get("eval_f1", 0.0) * num_examples

    if total_examples == 0:
        return MetricRecord({})

    aggregated_metrics = {
        "eval_loss": sum_loss / total_examples,
        "eval_acc": sum_acc / total_examples,
        "eval_precision": sum_prec / total_examples,
        "eval_recall": sum_rec / total_examples,
        "eval_f1": sum_f1 / total_examples,
    }
    return MetricRecord(aggregated_metrics)

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train,
                      evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
