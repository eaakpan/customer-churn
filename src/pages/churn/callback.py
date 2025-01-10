import dash
import base64
from dash import dcc, html, Input, Output, State
import plotly.express as px
from helpers.data_structures import get_customer_columns
from helpers.modeling_helper import predict_callback, batch_predict_callback
from helpers.data_viz_helper import list_value_counts
import pandas as pd
import json
import io

def register_callbacks(app):
    # Callback for single customer prediction
    @app.callback(
        Output('single-prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        State('input-gender', 'value'),
        State('input-senior', 'value'),
        State('input-partner', 'value'),
        State('input-dependents', 'value'),
        State('input-tenure', 'value'),
        State('input-phone-service', 'value'),
        State('input-multiple-lines', 'value'),
        State('input-internet-service', 'value'),
        State('input-online-security', 'value'),
        State('input-online-backup', 'value'),
        State('input-device-protection', 'value'),
        State('input-tech-support', 'value'),
        State('input-streaming-tv', 'value'),
        State('input-streaming-movies', 'value'),
        State('input-contract', 'value'),
        State('input-paperless-billing', 'value'),
        State('input-payment-method', 'value'),
        State('input-monthly-charges', 'value'),
        State('input-total-charges', 'value')
    )
    def input_customer(n_clicks, gender, senior,partner,dependents,tenure,phoneService,multipleLines,internetService,
                       onlineSecurity,onlineBackup,deviceProtection,techSupport,streamingTv,streamingMovies,contract,
                       paperlessBilling,paymentMethod,monthlyCharges, totalCharges):
        if n_clicks > 0:
            input_list = [gender, senior,partner,dependents,tenure,phoneService,multipleLines,internetService,
                           onlineSecurity,onlineBackup,deviceProtection,techSupport,streamingTv,streamingMovies,contract,
                           paperlessBilling,paymentMethod,monthlyCharges, totalCharges]

            customer_columns = get_customer_columns(['churn', 'customerid'])

            customer_df = pd.DataFrame(columns=customer_columns)
            customer_df.loc[0] = [gender, senior,partner,dependents,tenure,phoneService,multipleLines,internetService,
                           onlineSecurity,onlineBackup,deviceProtection,techSupport,streamingTv,streamingMovies,contract,
                           paperlessBilling,paymentMethod,monthlyCharges, totalCharges]




            # Predict
            result = predict_callback(customer_df)
            return json.dumps(result, indent=4)
        return ""


    # @app.callback(
    #     Output('single-prediction-output', 'children'),
    #     Input('customer-df', 'data'),
    # )
    # def predict_single_customer(customer_df):
    #         # Placeholder for prediction logic
    #     return f"Prediction: Churn probability for the given customer is X% (Replace with model output)."

    # Callback for batch upload prediction
    @app.callback(
        Output('batch-prediction-output', 'children'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def predict_batch(contents, filename):
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            try:
                # Read the content into a Pandas DataFrame
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(io.BytesIO(decoded))
                else:
                    return html.Div(["Unsupported file type. Please upload a CSV or Excel file."])


                # batch prediction
                result = batch_predict_callback(df)
                churn_count = list_value_counts(result['predicted_class'])
                churn_count.columns = ['predicted_class', 'count']

                fig = px.pie(churn_count, values='count', names='predicted_class', title='Churn Prediction Breakdown')


            except Exception as e:
                return html.Div([f"Error processing file: {str(e)}"])


            return html.Div([
                html.H5(f"File uploaded: {filename}"),
                html.P("Predictions for the batch are generated (Replace with actual model output)."),
                dcc.Graph(figure=fig)
            ])

        return html.Div(["No file uploaded."])