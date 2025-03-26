import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from database.delimited_sql_queries import select_from_customers_for_batch_prediction
from helpers.cleaning_helper import object_to_int, batch_training_data_cleaner,y_label_encoder
from helpers.modeling_helper import (cm_plot_2x2, xgboost_feature_selection, classification_model_evaluation_dict,
                                     xgboost_plot_importance)
import matplotlib; matplotlib.use('TkAgg')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_model_training(model):
    df = select_from_customers_for_batch_prediction()

    X_train_array, X_test_array, y_train, y_test = batch_training_data_cleaner(df)




    if model == 'logisticRegression':
        # Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train_array, y_train)
        y_pred = lr_model.predict(X_test_array)

        lr_model_evaluation = classification_model_evaluation_dict(model=lr_model,
                                                                        X_test_array=X_test_array,
                                                                        y_test=y_test,
                                                                        y_pred=y_pred,
                                                                        average='binary',
                                                                        pos_label="Yes")


        logging.info(f"Logistic Regression model evaluation is : {lr_model_evaluation}")

        cm_plot_2x2(lr_model_evaluation['confusion_matrix'])

        # Save the trained model
        joblib.dump(lr_model, 'runtime_data/for_models/lr_model.joblib')


    elif model == 'xgBoost':
        xgb_model = XGBClassifier()

        y_train, y_test = y_label_encoder(y_train,y_test)

        xgb_model.fit(X_train_array, y_train)

        xgboost_plot_importance(xgb_model)


        # feature selection
        xgboost_feature_selection(xgb_model, X_train_array, y_train, X_test_array, y_test)

        # predictions
        y_pred = xgb_model.predict(X_test_array)




        cm_plot_2x2(cm)

        logistic_regression_model_evaluation_dict(xgb_model, X_test_array, y_test, y_train, y_pred,
                                                  average='binary', pos_label="Yes")
        xgb_model_evaluation = classification_model_evaluation_dict(model=xgb_model,
                                                                        X_test_array=X_test_array,
                                                                        y_test=y_test,
                                                                        y_pred=y_pred,
                                                                        average='binary',
                                                                        pos_label=1)
        cm_plot_2x2(xgb_model_evaluation['confusion_matrix'])
    else:
        return "No model declared, batch training did not run"
    logging.info("batch model has been completed and saved")
    return print("batch model has been completed and saved")


if __name__ == '__main__':
    batch_model_training()

