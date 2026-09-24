import csv
from datetime import datetime
from pathlib import Path

"""
MetricsLogger is a context manager that handles logging of training metrics to a CSV file.
"""
class MetricsLogger:

    def __init__(self, path: Path):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path = path

        Path(self.path).mkdir(parents=True, exist_ok=True)


    def __enter__(self):
        self.openLog(self.path)
        self.writeHeader()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closeLog()

    def openLog(self, path: Path):
        self.log = open(Path(path) / f"training_log_{self.timestamp}.csv", 'x', encoding='UTF8', newline='')       
        self.writer = csv.writer(self.log)

    def closeLog(self):
        self.log.close()

    def writeHeader(self):
        header = ["epoch", "batch", "epoch_elapsed_sec", "batch_elapsed_sec", "loss",
                  "positive_energy", "negative_energy", "active_pair_fraction", "gradient_norm",
                  "mix_weight_spread", "var_min", "var_max", "var_mean"]
        self.writer.writerow(header)

    def writeRow(self, epoch, batch, epoch_sec, batch_sec, loss, pos_energy, neg_energy, active_pair, grad_norm,
                 mix_weight, var_min, var_max, var_mean):
        self.writer.writerow([epoch, batch, epoch_sec, batch_sec, loss, pos_energy, neg_energy, active_pair, grad_norm,
                              mix_weight, var_min, var_max, var_mean])

    def flush(self):
        self.log.flush()
