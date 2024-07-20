from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import datetime

import pandas as pd
import numpy as np

app = Dash(__name__)

df = pd.read_csv('./data/benchmark_curve.csv', index_col=0)
df.index = pd.to_datetime(df.index)

def get_wtd_butterfly(my_df, left, belly, right, left_weight=0.5, right_weight=0.5):
    '''
    pass in butterfly tenors, ex: 2y
    without speciying weights, assume 50:50 weighted butterfly

    return time series of the butterfly
    '''

    # returns a 1:2:1 fly, in bp by default
    return 200 * (my_df[belly] - (my_df[left] * left_weight + my_df[right] * right_weight))



app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                df.columns,
                '2y',
                id='left_wing'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df.columns,
                '3y',
                id='belly'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df.columns,
                '5y',
                id='right_wing'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
    ]),

    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df.index[0],
            max_date_allowed=df.index[-1],
            start_date=df.index[-63],
            end_date=df.index[-1]
        ),
        html.Div(id='output-container-date-picker-range')
    ]),

    dcc.Graph(id='graph'),

])


@callback(
    Output('graph', 'figure'),
    Input('left_wing', 'value'),
    Input('belly', 'value'),
    Input('right_wing', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
)
def update_graph(left_wing, belly, right_wing, start_date, end_date):

    df_select = df.loc[start_date:end_date]

    y = get_wtd_butterfly(df_select, left_wing, belly, right_wing)
    x = df_select[belly]
    colors = np.arange(0, len(y))

    fly_name = '%s_%s_%s fly' % (left_wing, belly, right_wing)
    fig = px.scatter(x=x, y=y, 
        # hover_data={'color': False},
        color_continuous_scale=px.colors.sequential.Teal, 
        trendline="ols"
    )
    fig.update_layout(xaxis_title=belly, yaxis_title=fly_name, showlegend=False)
    fig.add_trace(go.Scatter(x=x.iloc[-1:], y=y.iloc[-1:],
        mode='markers',  
        marker_color='orange',
        hoverinfo='skip')
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)
