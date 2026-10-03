import os
import torch


class CheckpointManager:

    def __init__(self,
                 save_dir="checkpoints"):

        self.save_dir = save_dir

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        self.best_path = os.path.join(
            save_dir,
            "best_model.pth"
        )

        self.latest_path = os.path.join(
            save_dir,
            "latest_model.pth"
        )

    def save_best(
        self,
        model,
        optimizer,
        epoch,
        accuracy
    ):

        torch.save({

            "epoch": epoch,

            "accuracy": accuracy,

            "model_state_dict":
            model.state_dict(),

            "optimizer_state_dict":
            optimizer.state_dict()

        }, self.best_path)

    def save_latest(
        self,
        model,
        optimizer,
        epoch,
        accuracy
    ):

        torch.save({

            "epoch": epoch,

            "accuracy": accuracy,

            "model_state_dict":
            model.state_dict(),

            "optimizer_state_dict":
            optimizer.state_dict()

        }, self.latest_path)