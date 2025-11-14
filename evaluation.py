from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    return {"F1 Score": f1, "ROC-AUC": roc_auc}

# Example usage:
# metrics = evaluate_model(y_val, y_pred, y_proba)
# print(metrics)
