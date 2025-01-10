import joblib
import numpy as np
from helpers.cleaning_helper import testing_data_cleaner, single_customer_data_cleaner


def predict_callback(data):
    model = joblib.load('src/lr_model.joblib')
    class_names = np.array(['None', 'Churn'])

    clean_df = single_customer_data_cleaner(data)

    model_pred = model.predict(clean_df)
    class_name = class_names[model_pred]
    return {'predicted_class': class_name.tolist()}  # fastapi can't return native numpyarrays - converted to list

def batch_predict_callback(data):
    model = joblib.load('src/lr_model.joblib')
    class_names = np.array(['None', 'Churn'])

    X_test = testing_data_cleaner(data)

    model_pred = model.predict(X_test)
    class_name = class_names[model_pred]
    return {'predicted_class': class_name.tolist()}  # fastapi can't return native numpyarrays - converted to list
