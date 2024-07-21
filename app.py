from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import datetime
import statsmodels.api as sm
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

app = Dash(__name__)

df = pd.read_csv('./data/benchmark_curve.csv', index_col=0)
df.index = pd.to_datetime(df.index)

def get_wtd_butterfly(my_df, left, belly, right, left_weight=-1, right_weight=-1, scale=True):
    '''
    pass in butterfly tenors, ex: 2y
    without speciying weights, assume 1:2:1 weighted butterfly

    return time series of the butterfly
    '''
    scale = 100 if scale else 1  # whether to scale by 100 to convert to bp
    # returns a 1:2:1 fly, in bps, by default
    return scale * (2 * my_df[belly] + (my_df[left] * left_weight + my_df[right] * right_weight))

def _get_xaxis_for_plot(my_df, left, belly, right, method='duration', pc=0):
    '''
    helper function that returns the Y and x vairable for visualization
    '''
    if method == 'duration':
        if pc == 0:
            x = my_df[belly]
        else:
            x = my_df[right] - my_df[left]
        return x 
    elif method == 'pca':
        my_pca = PCA(n_components=3)
        my_pca.fit_transform(my_df[[left, belly, right]].diff().dropna()) # innately assumes change in yields have mean 0 and follow a normal distribution
        
        my_pcs = my_df[[left, belly, right]] @ my_pca.components_.T 
        my_pc = my_pcs[pc] # retrieve either the 1st or 2nd pc, indicated by the pc param
        
        # adjust the sign if the pc happens to be inverted
        sign_adj = my_pc.corr(_get_xaxis_for_plot(my_df, left, belly, right, method='duration', pc=0)) 
        return sign_adj * my_pc

    else:
        # should not reach here
        return

def get_duration_neutral_butterfly(my_df, left, belly, right, method='duration'):
    '''
    input: "regression" or "pca", and tenors of the fly
    output: (w1, w2), weights for w1_2_w2 fly that neutralizes duration and curve

    for regression:
        -1 * left + 2 * belly - 1 * right = B1 * belly + B2 * (right - left) + Err
        -1 * left + 2 * belly - 1 * right = B1 * belly + B2 * right - B2 * left + Err
        (-1 + B2) * left + (2 - B1) * belly + (-1 - B2) * right = Err
        (-1 + B2) * 2 / (2 - B1) * left + 2 * belly + (-1 - B2) * 2 / (2 - B1) * right = Err
        so return [(-1 + B2) * 2 / (2 - B1),  (-1 - B2) * 2 / (2 - B1)]

    for PCA:
        we generate the 1st and 2nd principal component, then solve the system of equations to hedge out both exposures
        PC1 = [pc1_1, pc1_2, pc1_3]
        PC2 = [pc2_1, pc2_2, pc2_3]
        
        [w1, 2, w2] @ [pc1_1, pc1_2, pc1_3] = 0
        [w1, 2, w2] @ [pc2_1, pc2_2, pc2_3] = 0

        w1 * pc1_1 + 2 * pc1_2 + w2 * pc1_3 = 0
        w1 * pc2_1 + 2 * pc2_2 + w2 * pc2_3 = 0

        w1 * pc1_1 + w2 * pc1_3 = -2 * pc1_2 
        w1 * pc2_1 + w2 * pc2_3 = -2 * pc2_2 

        can be rewritten as Ax = b and using a solver to find unique solutions
    '''

    fly = get_wtd_butterfly(my_df, left, belly, right, scale=False)
    left_ts = my_df[left]
    belly_ts = my_df[belly]
    right_ts = my_df[right]

    if method == 'duration':
        X = pd.concat([belly_ts, right_ts - left_ts], axis=1, keys=['duration', 'curve'])
        y = fly
        ## fit a OLS model with intercept on TV and Radio 
        X = sm.add_constant(X) 
        ols = sm.OLS(y, X).fit() 

        B1 = ols.params['duration']
        B2 = ols.params['curve']
        w1, w2 = ((-1 + B2) * 2 / (2 - B1),  (-1 - B2) * 2 / (2 - B1))
        w1 = round(w1, 2)
        w2 = round(w2, 2)
        return w1, w2
    elif method == 'pca':
        my_pca = PCA(n_components=3)
        my_pca.fit_transform(my_df[[left, belly, right]].diff().dropna()) # innately assumes change in yields have mean 0 and follow a normal distribution
        pc0 = my_pca.components_[0, :]
        pc1 = my_pca.components_[1, :]

        # sets up the system of equations for deriving a unique solution
        A = np.array([[pc0[0], pc0[2]], [pc1[0], pc1[2]]])
        b = -2 * np.array([pc0[1], pc1[1]]).T
        w1, w2 = (np.linalg.solve(A, b))

        return w1, w2
    else:
        # should not reach here
        return

#TODO: loop through all available fly combinations and find the 3 richest and 3 cheapest, sorted by resid / vol

app.layout = html.Div([
    # TODO: add html label for left, belly, and right
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
        dcc.RadioItems(['duration', 'pca'], 'duration', id='metric_selector'),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df.index[0],
            max_date_allowed=df.index[-1],
            start_date=df.index[-126],
            end_date=df.index[-1]
        ),
        html.Div(id='output-container-date-picker-range')
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='regression-graph1'),
        ], style={'height': '50%', 'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='regression-graph2'),
        ], style={'height': '50%', 'width': '50%', 'display': 'inline-block'}),
    ]),

    html.Div([
        dcc.Graph(id='residual-graph'),
    ]),

])


@callback(
    Output('regression-graph1', 'figure'),
    Output('regression-graph2', 'figure'),
    Input('left_wing', 'value'),
    Input('belly', 'value'),
    Input('right_wing', 'value'),
    Input('metric_selector', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
)
# TODO: allow x axis to be 1st PC or 
def update_regression_graphs(left_wing, belly, right_wing, metric, start_date, end_date):

    df_select = df.loc[start_date:end_date]
    
    ### FIG 1 ###
    y = get_wtd_butterfly(df_select, left_wing, belly, right_wing)
    x = _get_xaxis_for_plot(df_select, left_wing, belly, right_wing, method=metric, pc=0)
    colors = np.arange(0, len(y))

    fly_name = '1_2_1 %s_%s_%s fly' % (left_wing, belly, right_wing)
    # TODO: update the marker so that it displays the date instead of an integer
    fig1 = px.scatter(x=x, y=y, color=colors,
        # hover_data={'color': False},
        color_continuous_scale=px.colors.sequential.Teal, 
        trendline="ols"
    )
    xaxis_title = belly if metric == 'duration' else 'PC1'
    fig1.update_layout(xaxis_title=xaxis_title, yaxis_title=fly_name, showlegend=False)
    fig1.update(layout_coloraxis_showscale=False)
    fig1.add_trace(go.Scatter(x=x.iloc[-1:], y=y.iloc[-1:],
        mode='markers',  
        marker_color='orange',
        hoverinfo='skip')
    )

    ### FIG 2 ###
    x2 = _get_xaxis_for_plot(df_select, left_wing, belly, right_wing, method=metric, pc=1)
    fig2 = px.scatter(x=x2, y=y, color=colors,
        # hover_data={'color': False},
        color_continuous_scale=px.colors.sequential.Teal, 
        trendline="ols"
    )
    xaxis_title = '%s / %s curve' % (left_wing, right_wing) if metric == 'duration' else 'PC2'
    fig2.update_layout(xaxis_title=xaxis_title, yaxis_title=fly_name, showlegend=False)
    fig2.update(layout_coloraxis_showscale=False)
    fig2.add_trace(go.Scatter(x=x2.iloc[-1:], y=y.iloc[-1:],
        mode='markers',  
        marker_color='orange',
        hoverinfo='skip')
    )

    return fig1, fig2

@callback(
    Output('residual-graph', 'figure'),
    Input('left_wing', 'value'),
    Input('belly', 'value'),
    Input('right_wing', 'value'),
    Input('metric_selector', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
)
def update_residual_graph(left_wing, belly, right_wing, metric, start_date, end_date):
    df_select = df.loc[start_date:end_date]
    w1, w2 = get_duration_neutral_butterfly(df_select, left_wing, belly, right_wing, method=metric)
    wtd_fly = get_wtd_butterfly(df_select, left_wing, belly, right_wing, left_weight=w1, right_weight=w2)
    fly_avg = wtd_fly.mean()
    resid = wtd_fly[-1] - fly_avg
    rlzd_vol = wtd_fly.diff().dropna().std()
    vol_adj_resid = resid / rlzd_vol
    my_range = wtd_fly.max() - wtd_fly.min()
    my_title = 'residual: %.1fbp; vol adjusted residual: %.1fbp; range: %.1fbp' % (resid, vol_adj_resid, my_range)

    fig = px.line(x=wtd_fly.index, y=wtd_fly, title=my_title)
    fly_name = '%.2f_2_%.2f %s_%s_%s fly' % (w1, w2, left_wing, belly, right_wing)
    fig.update_layout(yaxis_title=fly_name, showlegend=False)
    fig.add_hline(y=wtd_fly.mean())

    return fig


if __name__ == '__main__':
    app.run(debug=True)
