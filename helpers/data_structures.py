from config import MyDatabase
from database.delimited_sql_queries import select_all_from_customers

contract_options = ['Month-to-month', 'One year', 'Two year']

customers_options = {
    'binary': [{'label': 'No', 'value': 'No'}, {'label': 'Yes', 'value': 'Yes'}],
    'multipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'paymenMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],

}

class_names = ['Churn', 'No Churn']

class_names_dict = {'Yes': 'Churn', 'No': 'No Churn'}

num_cols = ["tenure", 'monthlycharges', 'totalcharges']


batch_prediction_cols = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',\
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',\
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',\
                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',\
                         'totalcharges', 'churn']
