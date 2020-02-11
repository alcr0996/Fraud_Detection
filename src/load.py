import pickle
# from xgboost import XGBClassifier, plot_tree, plot_importance

def load_model(model):
    loaded_model = pickle.load(open('..src/models/model_all_features_fin.pkl', 'rb'))
    return loaded_model

def preds_new_data_point(new_data, model):
    y_pred = model.predict(new_data)
    y_prob = model.predict_proba(new_data)
    return y_pred, y_prob