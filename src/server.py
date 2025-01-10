from fastapi import FastAPI
import joblib
import numpy as np
from config import MyDatabase
from helpers.cleaning_helper import training_data_cleaner

model = joblib.load('src/lr_model.joblib')

class_names = np.array(['None', 'Churn'])

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}

@app.post('/predict')
def predict():
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4]}

    Returns:
        dict: A dictionary containing the predicted class.
    """

    db = MyDatabase()
    df = db.query(''' select * from churnset.customers''')

    X_test = training_data_cleaner(df)

    # features = np.array(data['features']).reshape(1,-1)
    # prediction = model.predict(features)

    model_pred = model.predict(X_test)
    class_name = class_names[model_pred]
    return {'predicted_class': class_name.tolist()} #fastapi can't return native numpyarrays - converted to list
