class EarlyStopping:

    def __init__(
        self,
        patience=10,
        min_delta=0.001
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None

        self.counter = 0

        self.stop = False

    def __call__(self, score):

        if self.best_score is None:

            self.best_score = score

            return

        if score > self.best_score + self.min_delta:

            self.best_score = score

            self.counter = 0

        else:

            self.counter += 1

            print(
                f"EarlyStopping Counter : "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.stop = True