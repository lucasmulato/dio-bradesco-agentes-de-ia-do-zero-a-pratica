from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models(y_train):
    scale = (y_train == 0).sum() / (y_train == 1).sum() # ~577
    
    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=scale, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    return models
