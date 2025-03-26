import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def object_to_int(dataframe_series):
    if dataframe_series.dtype == 'object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


class NumericalFeatureCleaner(BaseEstimator, TransformerMixin):
    '''
    This Class performs all the transformation jobs on the numerical features.
    '''
    def __init__(self, scaler=None):
        if scaler:
            self._scalar = scaler
        else:
            self._scalar = StandardScaler()

    def fit(self, X, y=None, file_path = 'runtime_data/for_models/scaler.joblib'):
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        logging.info(f"The number of numerical columns are: {len(self.num_cols)}")

        self._scalar.fit(X[self.num_cols])

        joblib.dump(self._scalar, filename=file_path)
        logging.info(f"StandardScaler has been saved to : {file_path}")

        return self


    def transform(self, X, y=None):
        # When Transform is called, it uses the calculations from fit method.
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        logging.info(f"Beginning transform for: {self.num_cols}")
        X[self.num_cols] = self._scalar.transform(X[self.num_cols])

        return X[self.num_cols]

class CategoricalFeatureCleaner(BaseEstimator, TransformerMixin):
    '''
    This Class performs all the transformation jobs on the numerical features.
    In my use case, I have to do housekeeping tasks for Outliers and data normalizatoion.
    I have used RobustScalar to preserve the outliers and
    clipped outliers with their 20th and 80th percentile values.
    '''

    def __init__(self, enc=None, le=None):
        if enc:
            self._enc = enc
        else:
            self._enc = OneHotEncoder(sparse=False)
        if le:
            self._le = le
        else:
            self._le = LabelEncoder()


    def fit(self, X, y=None, file_path = 'runtime_data/for_models/enc.joblib'):
        self.cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        logging.info(f"The number of dummy columns are: {len(self.cat_cols)}")

        self._enc.fit(X[self.cat_cols])
        if y is not None:
         self._le.fit(y)

        joblib.dump(self._enc, filename=file_path)
        logging.info(f"OneHotEncoder has been saved to : {file_path}")

        return self

    def transform(self, X, y=None):
        self.cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        # When Transform is called, it uses the calculations from fit method.
        logging.info(f"Beginning transform for: {self.cat_cols}")
        self.cat_cols_transformed = self._enc.transform(X[self.cat_cols])

        if y is not None:
            self.y_transformed = self._le.transform(y)
            return self.cat_cols_transformed, self.y_transformed
        else:
            return self.cat_cols_transformed

def load_fitted_cleaning_objects():
    scaler = joblib.load('runtime_data/for_models/scaler.joblib')
    enc = joblib.load('runtime_data/for_models/enc.joblib')

    return scaler, enc


def cleaning_transformer_test(scaler, enc):
    pipe = Pipeline([
            ('features', FeatureUnion(n_jobs=1, transformer_list=[
                ('numericals', Pipeline([
                    ('selector', NumericalFeatureCleaner(scaler=scaler)),
                ])),  # numericals close

                ('categoricals', Pipeline([
                    ('selector', CategoricalFeatureCleaner(enc=enc)),
                ]))  # categoricals close

            ])),  # features close
        ])  # pipeline close
    return pipe



def jsonb_col_cleaner(df, col_list: list):
    """
    :param df:
    :param col_list:
    :return: json formatted col ready for postgres table upload

    """

    for col in col_list:
        df[col] = df[df[col].notnull()][col].map(json.dumps).astype('string')
        df.loc[df[col].isna()] = df.loc[df[col].isna()].fillna(json.dumps('{}'))

    return df



# def training_data_cleaner(df):
#
#     df = df.apply(lambda x: object_to_int(x))
#     X = df.drop(columns=['churn'])
#     y = df['churn'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)
#
#     # data transformation
#     num_cols = ["tenure", 'monthlycharges', 'totalcharges']
#
#
#
#     scaler = StandardScaler()
#
#     X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
#     X_test[num_cols] = scaler.transform(X_test[num_cols])
#     joblib.dump(scaler, 'runtime_data/for_models/scaler.joblib')# dump standard scaler to use for test data
#
#     # update columns
#     old_columns = pd.read_csv('runtime_data/example_csv/WA_Fn-UseC_-Telco-Customer-Churn.csv').drop(columns=['Churn', 'customerID']).columns
#     X_test.columns = old_columns
#     return X_test

def batch_training_data_cleaner(df):

    # df = df.apply(lambda x: object_to_int(x))
    X = df.drop(columns=['churn'])
    y = df['churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    cleaning_transformer_train = Pipeline([
        ('features', FeatureUnion(n_jobs=1, transformer_list=[
            ('numericals', Pipeline([
                ('selector', NumericalFeatureCleaner()),
            ])),  # numericals close

            ('categoricals', Pipeline([
                ('selector', CategoricalFeatureCleaner()),
            ]))  # categoricals close

        ])),  # features close
    ])  # pipeline close

    cleaning_transformer_train.fit(X_train, y_train)

    X_train_array = cleaning_transformer_train.transform(X_train)

    enc = joblib.load('runtime_data/for_models/enc.joblib')
    scaler = joblib.load('runtime_data/for_models/scaler.joblib')

    cleaning_transformer_test = Pipeline([
        ('features', FeatureUnion(n_jobs=1, transformer_list=[
            ('numericals', Pipeline([
                ('selector', NumericalFeatureCleaner(scaler=scaler)),
            ])),  # numericals close

            ('categoricals', Pipeline([
                ('selector', CategoricalFeatureCleaner(enc=enc)),
            ]))  # categoricals close

        ])),  # features close
    ])  # pipeline close

    X_test_array = cleaning_transformer_test.transform(X_test)

    y_train, y_test = y_label_encoder(y_train, y_test)
    return X_train_array, X_test_array, y_train, y_test

def single_customer_data_cleaner(X_test):
    scaler, enc = load_fitted_cleaning_objects()
    pipe = cleaning_transformer_test(scaler, enc)

    X_test_array = pipe.transform(X_test)
    return X_test_array


def testing_data_cleaner(X_test):
    scaler, enc = load_fitted_cleaning_objects()
    pipe = cleaning_transformer_test(scaler, enc)

    X_test_array = pipe.transform(X_test)
    return X_test_array


def y_label_encoder(y_train,y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test
