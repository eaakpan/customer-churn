import pandas as pd



def list_value_counts(list):
    return pd.Series((v for v in list)).value_counts().reset_index()