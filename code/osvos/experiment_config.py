class ExperimentConfig:
    def __init__(self, freeze, lr, scale, epochs):
        self.freeze = freeze
        self.lr = lr
        self.scale = scale
        self.epochs = epochs

    def __str__(self):
        return f'Freeze: {self.freeze} Lr: {self.lr} Scale: {self.scale}'
