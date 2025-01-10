from config import MyDatabase


def get_customer_columns(without=None):
    db = MyDatabase()
    df = db.query(''' select * from churnset.customers''')
    if without:
        df.drop(columns=without, inplace=True)
    return df.columns


contract_options = ['Month-to-month', 'One year', 'Two year']

customers_options = {
    'binary': [{'label': 'No', 'value': 'No'}, {'label': 'Yes', 'value': 'Yes'}],
    'seniorCitizen': [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
    'multipleLines': ['Yes', 'No', 'No phone service'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'paymenMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],

}
