import joblib
import numpy as np
from helpers.cleaning_helper import testing_data_cleaner, single_customer_data_cleaner
from helpers.data_structures import class_names_dict
import logging
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (precision_score, recall_score, confusion_matrix,make_scorer, f1_score,
                             roc_auc_score,log_loss)
from sklearn.model_selection import KFold,cross_validate, GridSearchCV
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_callback(data):
    model = joblib.load('runtime_data/for_models/xgbcv_model.joblib')

    X_test_array = single_customer_data_cleaner(data)

    model_pred = model.predict(X_test_array)
    class_name = class_names_dict[model_pred.tolist()[0]]
    return {'predicted_class': class_name}  # fastapi can't return native numpyarrays - converted to list

def batch_predict_callback(data):
    model = joblib.load('runtime_data/for_models/xgbcv_model.joblib')
    data.totalcharges = data.totalcharges.replace(' ', '0.0').astype(float)
    X_test_array = testing_data_cleaner(data)

    model_pred = model.predict(X_test_array)
    model_pred_list = list(map(lambda x: class_names_dict[0] if x == 0 else (class_names_dict[1] if x == 1  else  x), model_pred))

    return {'predicted_class': model_pred_list}  # fastapi can't return native numpyarrays - converted to list


def cm_plot_2x2(cnf_matrix):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.show()

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def roc_curve(model,X_test,y_test):
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

def xgboost_feature_selection(model,X_train,y_train,X_test,y_test):
    thresholds = np.sort(model.feature_importances_)
    summary = []
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = metrics.accuracy_score(y_test, predictions)
        summary.append("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

    return pd.DataFrame({"summary": summary})

def classification_model_evaluation_dict(model, X_test_array, y_test,
                                              y_pred, average='binary', pos_label="Yes"):
    accuracy_lr = model.score(X_test_array, y_test)
    logging.info(f"Logistic Regression accuracy is : {accuracy_lr}")

    precision_lr = precision_score(y_test, y_pred, average=average, pos_label=pos_label)
    logging.info(f"Logistic Regression precision_score is : {precision_lr}")

    recall_lr = recall_score(y_test, y_pred, average=average, pos_label=pos_label)
    logging.info(f"Logistic Regression recall_score is : {recall_lr}")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    logging.info(f"Logistic Regression confusion_matrix is : {cm}")


    return {'accuracy': accuracy_lr,'precision': precision_lr,
            'recall':recall_lr, 'confusion_matrix': cm }


def xgboost_plot_importance(xgb_model):
    plot_importance(xgb_model)
    plt.show()


estimators=[
    ('logistic',LogisticRegression(solver='liblinear',penalty='l2')),
    ('xgb',XGBClassifier(objective='reg:logistic')),
]



def plot_estimators(pipes, data, target, estimators_list, n_splits=5, metrics=['f1', 'auc', 'accuracy', 'logloss'], show_plots=True):
    estimators = [model[0] for model in estimators_list]

    metrics_dict = {'f1': make_scorer(f1_score), 'auc': make_scorer(roc_auc_score),
                    'accuracy': 'accuracy', 'logloss': make_scorer(log_loss)}

    metrics = {key: metrics_dict[key] for key in metrics}
    scorers = []
    labels = []
    for pipe_name in pipes.keys():
        if pipe_name in estimators:
            logging.info(f"Found in estimators beginning pipeline for : {pipe_name}")
            pipe = pipes[pipe_name]
            labels.append(pipe_name)
            kf = KFold(n_splits)
            model_score = cross_validate(pipe, data, target, scoring=metrics, cv=kf)
            scorers.append(model_score)
            logging.info(f"{pipe_name} model score  :  {model_score}")

    score_lists = {}
    for metric in metrics:
        score_lists[metric] = [score['test_' + metric] for score in scorers]

    if show_plots:
        for i, (title, _list) in enumerate(score_lists.items()):
            plt.figure(i)
            plot = sns.boxplot(data=_list).set_xticklabels(labels, rotation=45)
            plt.title(title)
        plt.show()

def tune_param(model, pipes, param_grid,data, target, refit='auc', chart=None, cv=5, metrics=['f1', 'auc', 'accuracy', 'logloss'], show_plots=True):
    param_grid = {model + '__' + key: param_grid[key] for key in param_grid.keys()}

    metrics_dict = {'auc': make_scorer(roc_auc_score)}

    logging.info("beginning grid search cv")
    xgbcv = GridSearchCV(pipes[model], param_grid, scoring=metrics_dict, refit=refit, cv=cv)
    logging.info("beginning model fit")
    xgbcv.fit(data, target)

    logging.info(f'best score: {str(xgbcv.best_score_)}')
    logging.info(f'best params: {str(xgbcv.best_params_)}')

    # Save the trained model
    logging.info(f'saving the fitted xgbcv to runtime_data/for_models/xgbcv_model.joblib')
    joblib.dump(xgbcv, 'runtime_data/for_models/xgbcv_model.joblib')
    results = pd.DataFrame(xgbcv.cv_results_)
    if show_plots:
        if 'line' in chart:
            for i, param in enumerate(param_grid.keys()):
                graph_data = results[['param_' + param, 'mean_test_' + refit]]
                graph_data = graph_data.rename(columns={'mean_test_' + refit: 'test'})
                graph_data = graph_data.melt('param_' + param, var_name='type', value_name=refit)
                plt.figure(i)
                plot = sns.lineplot(x='param_' + param, y=refit, hue='type', data=graph_data)

        if 'heatmap' in chart:
            assert len(param_grid) == 2, 'heatmap only works with 2 params, {} passed'.format(str(len(param_grid)))

            param1 = list(param_grid.keys())[0]
            param2 = list(param_grid.keys())[1]

            graph_data = results[['param_' + param1, 'param_' + param2, 'mean_test_' + refit]]
            graph_data = graph_data.pivot(index='param_' + param1, columns='param_' + param2, values='mean_test_' + refit)
            sns.heatmap(graph_data, annot=True, xticklabels=True, yticklabels=True).set(xlabel=param2, ylabel=param1)

