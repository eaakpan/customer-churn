from config import MyDatabase
from database.delimited_sql_queries import select_all_from_customers

contract_options = ['Month-to-month', 'One year', 'Two year']

customers_options = {
    'binary': [{'label': 'No', 'value': 'No'}, {'label': 'Yes', 'value': 'Yes'}],
    'seniorCitizen': [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
    'multipleLines': ['Yes', 'No', 'No phone service'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'paymenMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],

}
