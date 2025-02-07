from config import MyDatabase

db = MyDatabase()


def select_all_from_customers():
    df = db.query(''' 
    SELECT
        customerID,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
        Churn,
        created_at,
        last_modified
    from churnset.customers
    ''')

    return df


def select_input_from_customers():
    df = db.query(''' 
    SELECT
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    from churnset.customers
    ''')

    return df

def select_from_customers_for_batch_prediction():
    df = db.query(''' 
    SELECT
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        totalcharges,
        churn
    from churnset.customers
    where created_at >= '2025-02-01'
    ''')

    return df


def select_null_churn_from_customers():
    df = db.query(''' 
    SELECT
        customerID,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
        churn,
        created_at,
        last_modified
    from churnset.customers
    WHERE churn is null
    ''')

    return df
