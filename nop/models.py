from sklearn.linear_model import Lasso, LinearRegression

class ModelWrapper:
    def __init__(self, estimator):
        self.estimator = estimator

    def train_model(self, X, y):
        self.estimator.fit(X, y)

    def generate_predictions(self, X):
        return self.estimator.predict(X)

def build_model(model_type="lasso"):
    if model_type == "linear":
        return ModelWrapper(LinearRegression())
    return ModelWrapper(Lasso(alpha=0.1))

def evaluate_model(actual, predicted):
    mse = ((actual - predicted) ** 2).mean()
    return {"mse": float(mse)}
