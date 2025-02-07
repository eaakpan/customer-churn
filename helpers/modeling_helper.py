import joblib
import numpy as np
from helpers.cleaning_helper import testing_data_cleaner, single_customer_data_cleaner
from helpers.data_structures import class_names


def predict_callback(data):
    model = joblib.load('runtime_data/for_models/lr_model.joblib')
    np_class_name = np.array(class_names)

    X_test = single_customer_data_cleaner(data)

    model_pred = model.predict(X_test)
    class_name = np_class_name[model_pred]
    return {'predicted_class': class_name.tolist()}  # fastapi can't return native numpyarrays - converted to list

def batch_predict_callback(data):
    model = joblib.load('runtime_data/for_models/lr_model.joblib')
    np_class_name = np.array(class_names)

    X_test = testing_data_cleaner(data)

    model_pred = model.predict(X_test)
    class_name = np_class_name[model_pred]
    return {'predicted_class': class_name.tolist()}  # fastapi can't return native numpyarrays - converted to list
