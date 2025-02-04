from config import MyDatabase

schema = 'churnset'

# init db connector
db = MyDatabase()

# drop old schema
db.drop_schema(schema)

# create schema
db.create_schema(schema)

# Create tables

db.create_table(f'''CREATE TABLE IF NOT EXISTS {schema}.customers (
    customerID VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    SeniorCitizen SMALLINT CHECK (SeniorCitizen IN (0, 1)),
    Partner VARCHAR(3) CHECK (Partner IN ('Yes', 'No')),
    Dependents VARCHAR(3) CHECK (Dependents IN ('Yes', 'No')),
    tenure INTEGER CHECK (tenure >= 0),
    PhoneService VARCHAR(3) CHECK (PhoneService IN ('Yes', 'No')),
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
    Churn VARCHAR(3) CHECK (Churn IN ('Yes', 'No')),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    last_modified TIMESTAMP NOT NULL DEFAULT now()
);''')
