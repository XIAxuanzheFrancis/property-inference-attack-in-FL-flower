"""fl-pia: A Flower / PyTorch app."""

import torch
import csv
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
# from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_pia.task import Net, load_data
from fl_pia.task import test as test_fn



METRICS_CSV = Path("client_metrics.csv")
SERVER_AGG_CSV = Path("server_agg_metrics.csv")

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

    agg_header = [
        "server_round",
        "num_examples",
        "eval_loss",
        "eval_acc",
        "eval_precision",
        "eval_recall",
        "eval_f1",
    ]

    agg_exists = SERVER_AGG_CSV.exists()
    with SERVER_AGG_CSV.open("a", newline="") as f_agg:
        agg_writer = csv.writer(f_agg)
        if not agg_exists:
            agg_writer.writerow(agg_header)

        agg_row = [
            server_round,
            total_examples,
            aggregated_metrics["eval_loss"],
            aggregated_metrics["eval_acc"],
            aggregated_metrics["eval_precision"],
            aggregated_metrics["eval_recall"],
            aggregated_metrics["eval_f1"],
        ]
        agg_writer.writerow(agg_row)

    return MetricRecord(aggregated_metrics)

def get_server_evaluate_fn(num_partitions: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate_fn(server_round: int, arrays: ArrayRecord):
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        model.eval()

        total_examples = 0
        sum_loss = sum_acc = sum_prec = sum_rec = sum_f1 = 0.0
        for cid in range(num_partitions):
            _, valloader = load_data(partition_id=cid, num_partitions=num_partitions)

            eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = test_fn(
                model,
                valloader,
                device,
            )

            n = len(valloader.dataset)
            total_examples += n
            sum_loss += eval_loss * n
            sum_acc += eval_acc * n
            sum_prec += eval_precision * n
            sum_rec += eval_recall * n
            sum_f1 += eval_f1 * n

        if total_examples == 0:
            return None

        metrics = {
            "eval_loss": sum_loss / total_examples,
            "eval_acc": sum_acc / total_examples,
            "eval_precision": sum_prec / total_examples,
            "eval_recall": sum_rec / total_examples,
            "eval_f1": sum_f1 / total_examples,
        }

        print(
            f"[Server centralized eval | round {server_round}] "
            f"loss={metrics['eval_loss']:.4f}, acc={metrics['eval_acc']:.4f}, "
            f"precision={metrics['eval_precision']:.4f}, "
            f"recall={metrics['eval_recall']:.4f}, "
            f"f1={metrics['eval_f1']:.4f}"
        )
        return MetricRecord(metrics)

    return evaluate_fn



@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    num_partitions: int = context.run_config.get("num-partitions", 10)

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
        evaluate_fn=get_server_evaluate_fn(num_partitions),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
