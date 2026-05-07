from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "brier_score": brier_score_loss(y_test, probs),
        "log_loss": log_loss(y_test, probs),
    }