import os

import matplotlib.pyplot as plt


class TrainingPlotter:

    def __init__(self):

        os.makedirs(
            "figures",
            exist_ok=True
        )

    def plot_loss(self, losses):

        plt.figure(figsize=(7,5))

        plt.plot(losses)

        plt.title("Training Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.savefig(
            "figures/loss_curve.png"
        )

        plt.close()

    def plot_accuracy(self, acc):

        plt.figure(figsize=(7,5))

        plt.plot(acc)

        plt.title(
            "Validation Accuracy"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.grid(True)

        plt.savefig(
            "figures/accuracy_curve.png"
        )

        plt.close()