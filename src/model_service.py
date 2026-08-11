from src.model import load_model


class ModelService:

    def __init__(self):
        self.model = None

    def load(self):
        self.model = load_model()

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been loaded")

        return self.model.predict(X)