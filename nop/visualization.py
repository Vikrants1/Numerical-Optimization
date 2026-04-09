import matplotlib.pyplot as plt

def plot_results(actual_values, predicted_values):
    plt.figure(figsize=(8, 5))
    plt.plot(actual_values[:50], label="Actual Prices")
    plt.plot(predicted_values[:50], label="Predicted Prices")
    plt.title("Car Price Prediction Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Price (lakhs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("car_price_prediction_plot.png")
    plt.close()
