class ExperimentConfig:
    def __init__(self, freeze, lr, scale, epochs):
        self.freeze = freeze
        self.lr = lr
        self.scale = scale
        self.epochs = epochs
