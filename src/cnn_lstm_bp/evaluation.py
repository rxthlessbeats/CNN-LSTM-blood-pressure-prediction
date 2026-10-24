import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model_performance(y_true, y_pred, label="Model"):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    if len(y_true) == 0:
        raise ValueError("no samples to evaluate")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true - y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    bhs_results = evaluate_bhs_protocol(y_true, y_pred)
    grade = assign_bhs_grade(bhs_results)

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
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    abs_diff = np.abs(y_true - y_pred)

    total_predictions = len(y_true)
    if total_predictions == 0:
        raise ValueError("no samples to evaluate")

    num_within_5 = np.count_nonzero(abs_diff <= 5)
    num_within_10 = np.count_nonzero(abs_diff <= 10)
    num_within_15 = np.count_nonzero(abs_diff <= 15)

    percent_within_5 = (num_within_5 / total_predictions) * 100
    percent_within_10 = (num_within_10 / total_predictions) * 100
    percent_within_15 = (num_within_15 / total_predictions) * 100

    return {
        "within_5mmHg": percent_within_5,
        "within_10mmHg": percent_within_10,
        "within_15mmHg": percent_within_15,
    }


def assign_bhs_grade(bhs_results):
    within_5 = bhs_results["within_5mmHg"]
    within_10 = bhs_results["within_10mmHg"]
    within_15 = bhs_results["within_15mmHg"]

    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        return "A"
    if within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        return "B"
    if within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        return "C"
    return "D"
