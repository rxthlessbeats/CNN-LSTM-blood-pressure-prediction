import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model_performance(y_true, y_pred, label="Model"):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    if len(y_true) == 0:
        raise ValueError("no samples to evaluate")
    # Calculate standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true - y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    # Evaluate BHS Protocol
    bhs_results = evaluate_bhs_protocol(y_true, y_pred)
    grade = assign_bhs_grade(bhs_results)

    # Print results
    print(f"{label} Performance Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Standard Deviation of Errors: {std_dev:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("BHS Protocol Results:")
    print(f"Within 5 mmHg: {bhs_results['within_5mmHg']:.2f}%")
    print(f"Within 10 mmHg: {bhs_results['within_10mmHg']:.2f}%")
    print(f"Within 15 mmHg: {bhs_results['within_15mmHg']:.2f}%")
    print(f"{label} BHS Grade: {grade}")
    print("-" * 50)

    return (
        mae,
        rmse,
        std_dev,
        r2,
        bhs_results["within_5mmHg"],
        bhs_results["within_10mmHg"],
        bhs_results["within_15mmHg"],
        grade,
    )


def evaluate_bhs_protocol(y_true, y_pred):
    """
    Evaluate predictions using the BHS Protocol.

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - results: dict, percentages of predictions within absolute difference ranges of 5, 10, and 15 mmHg
    """
    import numpy as np

    # Calculate absolute differences
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    abs_diff = np.abs(y_true - y_pred)

    # Total number of predictions
    total_predictions = len(y_true)
    if total_predictions == 0:
        raise ValueError("no samples to evaluate")

    # Number of predictions within each specific range
    num_within_5 = np.count_nonzero(abs_diff <= 5)
    num_within_10 = np.count_nonzero(abs_diff <= 10)
    num_within_15 = np.count_nonzero(abs_diff <= 15)

    # Calculate percentages
    percent_within_5 = (num_within_5 / total_predictions) * 100
    percent_within_10 = (num_within_10 / total_predictions) * 100
    percent_within_15 = (num_within_15 / total_predictions) * 100

    results = {
        "within_5mmHg": percent_within_5,
        "within_10mmHg": percent_within_10,
        "within_15mmHg": percent_within_15,
    }

    return results


def assign_bhs_grade(bhs_results):
    within_5 = bhs_results["within_5mmHg"]
    within_10 = bhs_results["within_10mmHg"]
    within_15 = bhs_results["within_15mmHg"]

    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        grade = "A"
    elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        grade = "B"
    elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        grade = "C"
    else:
        grade = "D"

    return grade
