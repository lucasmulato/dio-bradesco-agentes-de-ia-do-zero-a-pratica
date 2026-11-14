from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    print(classification_report(y_test, preds, target_names=['Normal', 'Fraude']))

    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
    plt.title(f"Matriz - Threshold {threshold}")
    plt.show()
    
    precision, recall, _ = precision_recall_curve(y_test, probs)
    print(f"AUC-PR: {auc(recall, precision):.4f}")

def find_best_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    for p, r, t in zip(precision, recall, thresholds):
        if r >= 0.85:
            return t, p, r
    return 0.5, 0, 0
