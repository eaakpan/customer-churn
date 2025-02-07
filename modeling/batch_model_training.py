import joblib
from sklearn.linear_model import LogisticRegression
from database.delimited_sql_queries import select_from_customers_for_batch_prediction
from helpers.cleaning_helper import object_to_int, batch_training_data_cleaner




def batch_model_training():
    df = select_from_customers_for_batch_prediction().apply(lambda x: object_to_int(x))

    X_train, X_test, y_train, y_test = batch_training_data_cleaner(df)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    accuracy_lr = lr_model.score(X_test, y_test)
    print("Logistic Regression accuracy is :", accuracy_lr)

    # Save the trained model
    joblib.dump(lr_model, 'runtime_data/for_models/lr_model.joblib')

    return print("batch model has been completed and saved")


if __name__ == '__main__':
    batch_model_training()

