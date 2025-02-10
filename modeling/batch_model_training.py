import joblib
from sklearn.linear_model import LogisticRegression
from database.delimited_sql_queries import select_from_customers_for_batch_prediction
from helpers.cleaning_helper import object_to_int, batch_training_data_cleaner
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_model_training():
    df = select_from_customers_for_batch_prediction()

    X_train_array, X_test_array, y_train, y_test = batch_training_data_cleaner(df)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_array, y_train)

    accuracy_lr = lr_model.score(X_test_array, y_test)
    logging.info(f"Logistic Regression accuracy is : {accuracy_lr}")

    # Save the trained model
    joblib.dump(lr_model, 'runtime_data/for_models/lr_model.joblib')

    logging.info("batch model has been completed and saved")
    return print("batch model has been completed and saved")


if __name__ == '__main__':
    batch_model_training()

