import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.title("SHAP - Features mais importantes para fraude")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()
