import numpy as np
import pandas as pd
from config import MyDatabase, churn_rates
from database.delimited_sql_queries import select_null_churn_from_customers
import uuid

def update_churn_values(input_df):
    """
    Generates synthetic data based on the distributions of the existing dataset.

    Parameters:
        input_df (pd.DataFrame): Original DataFrame to sample distributions from.
        n_samples (int): Number of synthetic rows to generate.

    Returns:
        pd.DataFrame: Synthetic DataFrame with the same columns as the input DataFrame.
    """
    output_df = input_df.copy()

    output_df['churn'] = np.random.choice(
        list(churn_rates.keys()),
        size=len(input_df),
        p=(list(churn_rates.values()))
    )

    return output_df

if __name__ == '__main__':
    # Load the uploaded CSV
    data = select_null_churn_from_customers()

    # Generate synthetic data based on the existing dataset
    updated_data = update_churn_values(data)

    # Bulk insert data to db
    db = MyDatabase()
    db.update(updated_data,'customers')


# # Write to csv
# synthetic_data.columns = synthetic_data.columns.str.lower()
# synthetic_data.drop(columns = ['churn', 'customerid']).to_csv('data/batch_upload_example.csv', index=False)

