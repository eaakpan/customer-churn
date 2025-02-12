from ensemble_pipline.data_transformation import data_transformation
from helpers.transofrmation_helpers import correct_dtypes, transformer, cleaning_transformer
from helpers.modeling_helper import estimators, plot_estimators, tune_param
from sklearn.pipeline import Pipeline

def run_pipeline(X_train_array, X_test_array, y_train, y_test):

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
                    metrics=['f1', 'auc', 'accuracy', 'logloss'])

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
               cv=5)

