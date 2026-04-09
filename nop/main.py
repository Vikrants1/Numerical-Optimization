import numpy as np
from data_utils import load_dataset, preprocess_dataset
from regression_models import build_model
from plot_utils import plot_results

def get_user_input():
    print("\nEnter car details for resale price prediction")
    year = int(input("Manufacturing Year: "))
    present_price = float(input("Original Price (in lakhs): "))
    kms_driven = float(input("Kilometers Driven: "))
    owner = int(input("Number of Previous Owners: "))
    fuel_type = int(input("Fuel Type [0=Petrol, 1=Diesel, 2=CNG]: "))
    seller_type = int(input("Seller Type [0=Dealer, 1=Individual]: "))
    transmission = int(input("Transmission [0=Manual, 1=Automatic]: "))
    age = 2026 - year
    return np.array([[year, present_price, kms_driven, owner, fuel_type, seller_type, transmission, age]])

def execute_pipeline():
    print("Starting car price prediction pipeline...")
    data = load_dataset()
    X, y = preprocess_dataset(data)

    lasso_model = build_model(model_type="lasso")
    lasso_model.train_model(X, y)
    lasso_predictions = lasso_model.generate_predictions(X)

    linear_model = build_model(model_type="linear")
    linear_model.train_model(X, y)
    linear_predictions = linear_model.generate_predictions(X)

    lasso_mse = np.mean((y - lasso_predictions) ** 2)
    linear_mse = np.mean((y - linear_predictions) ** 2)

    print(f"LASSO MSE: {lasso_mse:.4f}")
    print(f"Linear Regression MSE: {linear_mse:.4f}")
    print("Better model:", "LASSO" if lasso_mse <= linear_mse else "Linear Regression")

    user_data = get_user_input()
    estimated_price = lasso_model.generate_predictions(user_data)
    print(f"\nEstimated Resale Price: {float(estimated_price[0]):.2f} lakhs")

    plot_results(y, lasso_predictions)

if __name__ == "__main__":
    execute_pipeline()
