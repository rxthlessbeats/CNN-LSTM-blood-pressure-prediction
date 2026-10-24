from cnn_lstm_bp.evaluation import assign_bhs_grade


def test_evaluation():
    assert (
        assign_bhs_grade({"within_5mmHg": 60, "within_10mmHg": 85, "within_15mmHg": 95})
        == "A"
    )
    assert (
        assign_bhs_grade({"within_5mmHg": 59, "within_10mmHg": 85, "within_15mmHg": 95})
        == "B"
    )
    assert (
        assign_bhs_grade({"within_5mmHg": 0, "within_10mmHg": 0, "within_15mmHg": 0})
        == "D"
    )
