import numpy as np
import pandas as pd
from config import MyDatabase
import uuid

def generate_synthetic_data(input_df, n_samples):
    """
    Generates synthetic data based on the distributions of the existing dataset.

    Parametrs:
        input_df (pd.DataFrame): Original DataFrame to sample distributions from.
        n_samples (int): Number of synthetic rows to generate.

    Returns:
        pd.DataFrame: Synthetic DataFrame with the same columns as the input DataFrame.
    """
    synthetic_data = pd.DataFrame()

    for column in input_df.columns:
        if column == 'customerID':
            # Generate unique customer IDs
            synthetic_data[column] = [str(uuid.uuid4()) for _ in range(n_samples)]
        elif input_df[column].dtype == 'object':
            # For categorical columns, sample based on value counts
            synthetic_data[column] = np.random.choice(
                input_df[column].dropna().unique(),
                size=n_samples,
                p=(input_df[column].value_counts(normalize=True).values)
            )
        elif np.issubdtype(input_df[column].dtype, np.number):
            if column == 'SeniorCitizen':
                # Ensure SeniorCitizen is binary (0 or 1)
                synthetic_data[column] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
            elif column == 'tenure':
                # Ensure tenure is an integer value
                synthetic_data[column] = np.random.randint(
                    low=int(input_df[column].min()),
                    high=int(input_df[column].max()) + 1,
                    size=n_samples
                )
            else:
                # For other numerical columns, sample based on a normal distribution
                synthetic_data[column] = np.random.normal(
                    loc=input_df[column].mean(),
                    scale=input_df[column].std(),
                    size=n_samples
                )
        else:
            # For any other data types, fill with NaN or placeholder
            synthetic_data[column] = [None] * n_samples

    return synthetic_data


# Load the uploaded CSV
file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(file_path)

# Generate synthetic data based on the existing dataset
synthetic_data = generate_synthetic_data(data, n_samples=1000)

# Bulk insert data to db
db = MyDatabase()
db.insert(synthetic_data,'customers')


# Write to csv
# synthetic_data.drop(columns = ['Churn', 'customerID']).to_csv('data/batch_upload_example.csv', index=False)

