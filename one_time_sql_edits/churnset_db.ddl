CREATE TABLE IF NOT EXISTS customers (
    customerID VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    SeniorCitizen VARCHAR(3) customers_seniorcitizen_check CHECK (SeniorCitizen IN ('Yes', 'No')),
    Partner VARCHAR(3) customers_partner_check CHECK (Partner IN ('Yes', 'No')),
    Dependents VARCHAR(3) customers_dependents_check CHECK (Dependents IN ('Yes', 'No')),
    tenure INTEGER customers_tenure_check CHECK (tenure >= 0),
    PhoneService VARCHAR(3) customers_phoneservice_check CHECK (PhoneService IN ('Yes', 'No')),
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),
    OnlineSecurity VARCHAR(20),
    OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20),
    TechSupport VARCHAR(20),
    StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20),
    Contract VARCHAR(20),
    PaperlessBilling VARCHAR(3),
    PaymentMethod VARCHAR(30),
    MonthlyCharges NUMERIC(10, 2),
    TotalCharges NUMERIC(15, 2),
    Churn VARCHAR(3) customers_churn_check CHECK (Churn IN ('Yes', 'No')),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    last_modified TIMESTAMP NOT NULL DEFAULT now()
);
