from dash import Dash
from src.pages.churn.layout import layout
from src.pages.churn.callback import register_callbacks

# Initialize the src
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Churn Prediction"
app.layout = layout
server = app.server
# Import callbacks (this ensures they are registered)
register_callbacks(app)

if __name__ == "__main__":
    app.run( host='127.0.0.1',port='5433',debug=True)






# from dash import Dash, html, dcc
# import plotly.express as px
# import pandas as pd
#
# app = Dash()
#
# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
#
# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
#
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for your data.
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])
#
# if __name__ == '__main__':
#     app.run(debug=True,port='5433')