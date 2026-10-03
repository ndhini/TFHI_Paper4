import os
import csv


class TrainingLogger:
    """
    Research Grade CSV Logger
    """

    def __init__(self, log_dir="logs"):

        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(
            log_dir,
            "training_log.csv"
        )

        if not os.path.exists(self.log_file):

            with open(self.log_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "Epoch",
                    "Train Loss",
                    "Validation Accuracy",
                    "Learning Rate"
                ])

    def log(
        self,
        epoch,
        train_loss,
        val_accuracy,
        lr
    ):

        with open(self.log_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_accuracy, 4),
                lr
            ])