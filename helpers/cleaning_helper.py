import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from helpers.data_structures import num_cols


def object_to_int(dataframe_series):
    if dataframe_series.dtype == 'object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

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



def training_data_cleaner(df):

    df = df.apply(lambda x: object_to_int(x))
    X = df.drop(columns=['churn'])
    y = df['churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    # data transformation
    num_cols = ["tenure", 'monthlycharges', 'totalcharges']



    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])


    # update columns
    old_columns = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv').drop(columns=['Churn', 'customerID']).columns
    X_test.columns = old_columns
    return X_test

def batch_training_data_cleaner(df):

    df = df.apply(lambda x: object_to_int(x))
    X = df.drop(columns=['churn'])
    y = df['churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    # data transformation
    cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService']  # those that need one-hot encoding
    cat_cols_le = list(set(X_train.columns) - set(num_cols) - set(cat_cols_ohe))  # those that need label encoding

    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])


    return X_train, X_test, y_train, y_test

def single_customer_data_cleaner(df):
    # df['customerID'] = '45e42fc8-ab63-4fe5-9c78-3db32fedf6a7'
    df = df.apply(lambda x: object_to_int(x))

    return df


def testing_data_cleaner(df):

    X_test = df.apply(lambda x: object_to_int(x))

    # data transformation
    scaler = StandardScaler()

    X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

    return X_test