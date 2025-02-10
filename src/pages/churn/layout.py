import dash
from dash import dcc, html, Input, Output, State
from helpers.data_structures import customers_options


# Initialize the Dash src
app = dash.Dash(__name__)
app.title = "Churn Prediction"


# Layout of the src
layout = html.Div([
    html.H1("Customer Churn Prediction", style={'textAlign': 'center'}),

    # Section for single customer input
    html.Div([
        html.H3("Input Single Customer Details"),
        html.Div([
            html.Label("Gender:"),
            dcc.Dropdown(
                id='input-gender',
                options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                placeholder='Select Gender',
                value='Male'
            ),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Senior Citizen :"),
            dcc.Dropdown(id='input-senior',  options=customers_options['seniorCitizen'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Partner:"),
            dcc.Dropdown(id='input-partner', options=customers_options['binary'], placeholder='Enter Partner', value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Dependents:"),
            dcc.Dropdown(id='input-dependents', options=customers_options['binary'], placeholder='Yes or No', value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Tenure (Months):"),
            dcc.Input(id='input-tenure', type='number', placeholder='Enter Tenure', min=0, step=1, value = 8),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Phone Service:"),
            dcc.Dropdown(id='input-phone-service', options=customers_options['binary'], placeholder='Yes or No', value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Multiple Lines:"),
            dcc.Dropdown(id='input-multiple-lines', options=customers_options['multipleLines'],
                         placeholder='Select Multiple Lines', value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Internet Service:"),
            dcc.Dropdown(id='input-internet-service', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Online Security:"),
            dcc.Dropdown(id='input-online-security', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Online Backup:"),
            dcc.Dropdown(id='input-online-backup', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Device Protection:"),
            dcc.Dropdown(id='input-device-protection', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Tech Support:"),
            dcc.Dropdown(id='input-tech-support', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Streaming TV:"),
            dcc.Dropdown(id='input-streaming-tv', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Streaming Movies:"),
            dcc.Dropdown(id='input-streaming-movies', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Contract:"),
            dcc.Dropdown(id='input-contract', options=customers_options['contract'], placeholder='Select Contract',
                         value='Month-to-month'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Paperless Billing:"),
            dcc.Dropdown(id='input-paperless-billing', options=customers_options['binary'], placeholder='Yes or No',
                         value='Yes'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Payment Method:"),
            dcc.Dropdown(id='input-payment-method', options=customers_options['paymenMethod'],
                         placeholder='Select Payment Method', value='Bank transfer (automatic)'),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Monthly Charges:"),
            dcc.Input(id='input-monthly-charges', type='number', placeholder='Enter Monthly Charges', min=0,
                      value=27.82),
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label("Total Charges:"),
            dcc.Input(id='input-total-charges', type='number', placeholder='Enter Total Charges', min=0,
                      value=98.23),
        ], style={'margin-bottom': '10px'}),

        html.Button("Predict Churn", id='predict-button', n_clicks=0),
        html.Div(id='single-prediction-output', style={'margin-top': '20px'})
    ], style={'margin-bottom': '50px', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '5px'}),

    # Section for batch upload
    html.Div([
        html.H3("Upload Batch of Customers for Prediction"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin-bottom': '10px'
            },
            multiple=False
        ),
        html.Div(id='batch-prediction-output')
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '5px'})
])
