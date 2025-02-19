from helpers.cleaning_helper import batch_training_data_cleaner
from database.delimited_sql_queries import select_from_customers_for_batch_prediction
from helpers.modeling_helper import estimators, plot_estimators, tune_param
from sklearn.pipeline import Pipeline
import matplotlib; matplotlib.use('TkAgg')
def run_pipeline(X_train_array, X_test_array, y_train, y_test, show_plots = False):

    model = estimators[0]
    pipe = Pipeline(steps=[
         model,
        ])
    pipe.fit(X_train_array,y_train)
    pipe.predict(X_train_array)
    pipe.steps[0][0]

    # Model Scoring
    pipes = {}
    for model in estimators:
        pipe = Pipeline(steps=[model])
        pipe.fit(X_train_array, y_train)
        pipes[pipe.steps[0][0]] = pipe

    plot_estimators(pipes=pipes,
                    data=X_train_array,
                    target=y_train,
                    estimators_list=estimators,
                    n_splits=5,
                    metrics=['f1', 'auc', 'accuracy', 'logloss'],
                    show_plots=show_plots)

    # Hyper parameter tuning
    pipes['xgb'].named_steps['xgb'].get_params()

    param_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900]}

    tune_param(model='xgb',
               pipes=pipes,
               param_grid=param_grid,
               data=X_train_array,
               target=y_train,
               refit='auc',
               chart='line',
               cv=5,
               show_plots=show_plots)

if __name__ == '__main__':
    df = select_from_customers_for_batch_prediction()

    X_train_array, X_test_array, y_train, y_test = batch_training_data_cleaner(df)

    run_pipeline(X_train_array, X_test_array, y_train, y_test, show_plots=False)
