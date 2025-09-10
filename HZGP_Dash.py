#%%
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from HZGP import transportrate
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
import io
import base64
from minimumVc import options, default_option
# from minimumVc_private import options, default_option

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

tab_geometry_content= html.Div([ 
    html.Table([
        html.Tr([
            # html.Label('Openhole:', style={'marginRight': '10px', 'marginTop': '10px','textAlign': 'right'}
            html.Td('Openhole(in):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='OH', type='number', value=8.5, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Screen OD(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='ScreenOD', type='number', value=6.125, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Screen ID(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='ScreenID', type='number', value=4.545, style={'marginTop': '10px'}))
        ]), 

        html.Tr([
            html.Td('Screen Centralizer(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='ScreenCD', type='number', value=6.75,  style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Washpipe OD(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='WashPipeOD', type='number', value=4.0, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Washpipe ID(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='WashPipeID', type='number', value=3.8, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('OH roughness(in):',style={'paddingLeft': '40px', 'marginTop': '10px','textAlign': 'right'}),
            html.Td(dcc.Input(id='E_OH', type='number', value=0.05, style={'marginTop': '10px'}))
        ]),

        html.Tr([
            html.Td('Washpipe-screen roughness(in):',style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='E_WP', type='number', value=0.005, style={'marginTop': '10px'}))
        ])
    ], style={'borderCollapse': 'separate', 'borderSpacing': '10px 5px'})
]) 
tab_slurry_content= html.Div([
    html.Table([
        html.Tr([
            html.Td('Fluid density(ppg):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='Rho_f', type='number', value=9.0, step=0.05, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Fluid Viscosity(cP):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='Mu_f', type='number', value=1.0, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('DR channel 0-1:', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='drag_r_o', type='number', value=0, min=0, max=1, step=0.01, 
                              style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('DR wash-pipe annulus 0-1:', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='drag_r_w', type='number', value=0, min=0, max=1, step=0.01, 
                              style={'marginTop': '10px'}))
        ]),        
        html.Tr([
            html.Td('Gravel size(um):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='D_p', type='number', value=650, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Gravel density(sg):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='SG', type='number', value=2.65, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Slurry conc.(ppa):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='ppa', type='number', value=1.0, step=0.1, style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td('Deposition model:', style={'textAlign': 'right'}),
            html.Td(dcc.Dropdown(id='Model',
                                 options= options,
                                 value= default_option,
                                 style={'marginTop': '10px', 'width': '200px'}))
        ]),
        html.Tr([
            html.Td('Constant Vc (ft/s):', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='Vconstant', type='number', value=5, step=0.1, style={'marginTop': '10px'}))
        ]),        
        html.Tr([
            html.Td('Name:', style={'textAlign': 'right'}),
            html.Td(dcc.Input(id='Name', type='text', style={'marginTop': '10px'}))
        ]),
        html.Tr([
            html.Td(html.Button('Run', id='run-button', n_clicks=0, style={'marginTop': '10px', 'marginBottom': '10px', 'width': '200px', 'textAlign': 'center'}),
                    style={'textAlign': 'left'}),
            html.Td(html.Button('Clear', id='clear-button', n_clicks=0, style={'marginTop': '10px', 'marginBottom': '10px', 'width': '200px', 'textAlign': 'center'}),
                    style={'textAlign': 'left'}),
        ])
    ], style={'borderCollapse': 'separate', 'borderSpacing': '10px 5px'})
])


tabs = dbc.Tabs([
    dbc.Tab(tab_geometry_content, label='Geometry'),
    dbc.Tab(tab_slurry_content, label='Slurry')
], style={'marginTop': '20px', 'marginLeft': '20px', 'width': '500px'}) 

savebutton = html.Div([
    html.Table([
        html.Tr([html.Td('Save Model', colSpan=2, style={'textAlign': 'center'})]),
        html.Tr([
            html.Td(dcc.Dropdown(id='curve-save', multi=False, style={'width': '300px', 'marginTop': '10px'})),
            html.Td(html.Button('Save', id='btn_excel', n_clicks=0, 
                    style={'marginTop': '10px', 'marginBottom': '10px', 'width': '200px', 'textAlign': 'center'}))
        ])
    ], style={'borderCollapse': 'separate', 'borderSpacing': '10px 5px'}),
    dcc.Download(id="curve-dataframe-xlsx"),
], style={'marginTop': '20px', 'marginLeft': '20px'})


tab_resutls_content = html.Div([
    html.H3("Results"),
    dbc.Tabs([
        dbc.Tab([html.Div(id='outputbox', style={'marginTop': '20px'}),
                html.Table([
                html.Tr([
                    html.Td('Graph:', style={'marginRight': '10px', 'marginTop': '10px', 'textAlign': 'right', "vertical-align":"bottom"}),
                    html.Td(dcc.RadioItems(id='graph-item',
                                    options={ 'height':'rate-height',
                                            'dpdx':'rate-dp/dx alpha',
                                            'dpdx_w':'rate-dpdx beta'}, 
                                            value='height', inline=True, 
                                    labelStyle={'marginRight': '20px', 'marginLeft': '20px'},
                                    inputStyle={'marginRight': '10px'}),
                                    style={"vertical-align":"bottom"})
                    ]),
                html.Tr([
                    html.Td('Units:', style={'marginRight': '10px', 'marginTop': '10px', 'textAlign': 'right', "vertical-align":"bottom"}),
                    html.Td(dcc.RadioItems(id='unit-item',
                                    options={'gpm':'gpm',
                                             'bpm':'bpm'}, 
                                            value='gpm', inline=True, 
                                    labelStyle={'marginRight': '20px', 'marginLeft': '20px'},
                                    inputStyle={'marginRight': '10px'}),
                                    style={"vertical-align":"bottom"})
                    ])
                ])
            ], tab_id='graph-tab', label='Graph'),
        dbc.Tab([html.Div(id='tabular-results'),
                 savebutton],
                tab_id='table-tab', label='Table')
    ], id='tabs-results', active_tab='graph-tab', style={'marginTop': '20px'}),
])

app.layout = html.Div(
    [dbc.Row(
        [dbc.Col(tabs,style={'borderRight': '1px solid #ddd', 'paddingRight': '20px', 'width': '500px'}),
         dbc.Col(tab_resutls_content, style={'paddingLeft': '20px', 'width': 'calc(100% - 500px)'})],
        style={'textAlign': 'left', 'marginTop': '20px'}
    ),
    dcc.Store(id='curve-data', data=[]),]
)

def createcurve(oh,screen_od,screen_id,screen_cd,washp_od,washp_id,e_oh,e_wp,
                     rho_f,mu_f,d_p,sg,ppa,vc, dro, drw, model, name):
    # Create a curve for transport rate vs height
    # This function is called to generate the curve data    
    hs = np.arange(start=oh*0.60, stop=oh*0.95,step=oh*0.35/60)
    qs = [transportrate(model=model, 
                        oh=oh,
                        ds_o=screen_od,
                        ds_c=screen_cd,
                        ds_i=screen_id,
                        dw_o=washp_od,
                        dw_i=washp_id,
                        h=h,
                        ppa=ppa,
                        dp=d_p/25400, 
                        rhof=rho_f,
                        SG=sg,
                        muf=mu_f,
                        vc = vc, 
                        e_o=e_oh, 
                        e_w=e_wp,dr_o=dro, dr_w=drw) 
                        for h in hs]
    qvsh = pd.DataFrame(data = {'h':hs, 'q': [q[0] for q in qs], 'dpdx': [q[1] for q in qs], 'dpdx_w': [q[2] for q in qs]})
    qmax = qs[0][0]
    qvsh = qvsh[[q <= qmax and pd.notna(q) for q in qvsh.q.values]]
    hmax = qvsh[qvsh.q.min()==qvsh.q].h.values[0]
    showvalues1 = (qvsh.h <= hmax).values
    showvalues2 = (qvsh.h >= hmax).values
    curve = {
        'h': qvsh[showvalues1].h.tolist(),
        'q': qvsh[showvalues1].q.tolist(),
        'dpdx': qvsh[showvalues1].dpdx.tolist(),
        'dpdx_w': qvsh[showvalues1].dpdx_w.tolist(),
        'h_' :qvsh[showvalues2].h.tolist(),
        'q_' :qvsh[showvalues2].q.tolist(),
        'dpdx_': qvsh[showvalues2].dpdx.tolist(),
        'qmin': qvsh.q.min(),
        'hmax': hmax,
        'qmax': qmax,
        'oh' :oh,
        'screen_od':screen_od,
        'screen_id':screen_id,
        'screen_cd':screen_cd,
        'washp_od':washp_od,
        'e_oh':e_oh,
        'e_wp': e_wp,
        'rho_f': rho_f,
        'mu_f': mu_f,
        'd_p': d_p,
        'sg': sg,
        'dr_o': dro,
        'dr_w': drw,
        'ppa': ppa,
        'model': model,
        'name': f'{model}/{ppa:.2f}' if name is None or name == '' else name
    }
    return curve
   

@app.callback(
    Output('curve-data', 'data'),
    Output('tabs-results', 'active_tab'),
    State('OH', 'value'),
    State('ScreenOD', 'value'),
    State('ScreenID', 'value'),
    State('ScreenCD', 'value'), 
    State('WashPipeOD', 'value'),
    State('WashPipeID', 'value'),
    State('E_OH', 'value'),
    State('E_WP', 'value'),
    State('Rho_f', 'value'),
    State('Mu_f', 'value'),
    State('drag_r_o', 'value'),
    State('drag_r_w', 'value'),
    State('D_p', 'value'),
    State('SG', 'value'),
    State('ppa', 'value'),
    State('Vconstant', 'value'),
    State('Model', 'value'),
    State('Name', 'value'),
    State('curve-data', 'data'),
    Input('run-button', 'n_clicks'),
    Input('clear-button', 'n_clicks')
)
def callbackinputbox(oh,screen_od,screen_id,screen_cd,washp_od,washp_id,e_oh,e_wp,
                     rho_f, mu_f, dro, drw, d_p, sg, ppa,vc, model, name, curvedata, n_clicks_run, n_clicks_clear):
    ctx = dash.callback_context
    if ctx.triggered:
        if 'clear-button' == ctx.triggered[0]['prop_id'].split('.')[0]:
            curvedata = None
    curve = createcurve(oh, screen_od, screen_id, screen_cd, washp_od, washp_id, e_oh, e_wp,
                   rho_f, mu_f, d_p, sg, ppa, vc, dro, drw, model, name)
    if curvedata == None:        
        return [curve], 'graph-tab'
    else:
        curvedata.append(curve)
        return curvedata, 'graph-tab'
    

@app.callback(
    Output("curve-dataframe-xlsx", "data"),
    Input("btn_excel", "n_clicks"),
    State("curve-save", "value"),
    State('curve-data', 'data'),
    prevent_initial_call=True)
def generate_excel(n_clicks, curvename, curves):
    if curvename is None or curvename == '':
        return dash.no_update
    curve = next((c for c in curves if c['name'] == curvename), None)
    if curve is None:
        return dash.no_update
    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Write the selected curve to an Excel sheet
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_curve = pd.DataFrame({
            'Q (gal/min)': curve['q'],
            'h (inches)': curve['h'],
            'dp/dx-alpha (psi/ft)': curve['dpdx'],
            'dp/dx-beta (psi/ft)': curve['dpdx_w']   
        })
        df_curve.to_excel(writer, sheet_name='Curve Data', index=False)
        
        # Write parameters to another sheet
        df_params = pd.DataFrame({
            'Parameter': ['Openhole (in)', 'Screen OD (in)', 'Screen ID (in)', 'Screen Centralizer (in)', 
                          'Washpipe OD (in)', 'OH roughness (in)', 'Washpipe-screen roughness (in)', 
                          'Fluid density (ppg)', 'Fluid Viscosity (cP)',
                          'Drag reduction channel:', 
                          'Drag reduction wash-pipe annulus:',
                          'Gravel size (um)', 
                          'Gravel density (sg)', 'Slurry conc. (ppa)', 'Minimum transport slurry rate (gpm)',
                          'Maximum bed height (in)', 'Deposition model'],
            'Value': [curve['oh'], curve['screen_od'], curve['screen_id'], curve['screen_cd'], 
                      curve['washp_od'], curve['e_oh'], curve['e_wp'], 
                      curve['rho_f'], curve['mu_f'], 
                      curve['dr_o'], curve['dr_w'],
                      curve['d_p'], 
                      curve['sg'], curve['ppa'], curve['qmin'], curve['hmax'],
                      curve['model']]
        })
        df_params.to_excel(writer, sheet_name='Parameters', index=False)

    buffer.seek(0)

    # Encode to base64 for Dash download
    return dcc.send_bytes(buffer.getvalue(), filename=f"{curvename}_data.xlsx")
@app.callback(Output('outputbox', 'children'),
              Output('tabular-results', 'children'),
              Output('curve-save', 'options'),
              Input('curve-data', 'data'),
              Input('graph-item', 'value'),
              Input('unit-item', 'value'))  
def update_outputbox(curves, graphitem, unititem):
    fig = go.Figure()
    if not curves or len(curves) == 0:
        return html.Div("No curves to display. Please input parameters and click 'Run'."), html.Div(), []
    colors= px.colors.qualitative.Plotly
    cellstyle = {'border': '1px solid black', 'padding': '5px', 'textAlign': 'center'}
    tablerows = [html.Tr([
        html.Th('Curve Name', style=cellstyle),
        html.Th('Qmin (gal/min)', style=cellstyle),
        html.Th('hmax (inches)', style=cellstyle),
        html.Th('Conc. (ppa)', style=cellstyle),
        html.Th('Model' , style=cellstyle)
    ])]
    color_i = 0
    if unititem == 'bpm':
        for curve in curves:
            curve['q'] = [q/42 for q in curve['q']]
            curve['q_'] = [q/42 for q in curve['q_']]
            curve['qmin'] = curve['qmin']/42
            curve['qmax'] = curve['qmax']/42
        x_axis_title = 'Slurry Rate (bbl/min)'
    else:
        x_axis_title = 'Slurry Rate (gal/min)'

    for curve in curves:
        if graphitem == 'dpdx':
            t1 = go.Scatter(x=curve['q'], y=curve['dpdx'], type='scatter', mode='lines', name=curve['name'], 
                            line=dict(color=colors[color_i]), 
                            legendgroup=curve['name'])
            t2 = go.Scatter(x=curve['q_'], y=curve['dpdx_'], type='scatter', mode='lines', name=curve['name'], 
                            line=dict(dash='dash', color=colors[color_i]),
                             showlegend=False, legendgroup=curve['name'])
            t3 = go.Scatter(x=[curve['qmin']], y=[curve['dpdx'][curve['h'].index(curve['hmax'])]], mode='markers', 
                            marker=dict(color=colors[color_i], size=10, symbol='circle'), name=f"{curve['name']} Qmin",
                            showlegend=False, legendgroup=curve['name'])
            fig.add_traces([t1,t2,t3])
            Fig_title = 'Rate vs Alpha Pressure Gradient'
        elif graphitem == 'dpdx_w':
            t1 = go.Scatter(x=curve['q'], y=curve['dpdx_w'], type='scatter', mode='lines', name=curve['name'], 
                            line=dict(color=colors[color_i]), 
                            legendgroup=curve['name'])            
            t3 = go.Scatter(x=[curve['qmin']], y=[curve['dpdx_w'][curve['h'].index(curve['hmax'])]], mode='markers', 
                            marker=dict(color=colors[color_i], size=10, symbol='circle'), name=f"{curve['name']} Qmin",
                            showlegend=False, legendgroup=curve['name'])
            fig.add_traces([t1,t3])
            Fig_title = 'Rate vs Beta Pressure Gradient'
        else:
            t1 = go.Scatter(x=curve['q'], y=curve['h'], type='scatter', mode='lines', name=curve['name'] , 
                            line=dict(color=colors[color_i]), 
                            legendgroup=curve['name'])
            t2 = go.Scatter(x=curve['q_'], y=curve['h_'], type='scatter', mode='lines', name=curve['name'], 
                            line=dict(dash='dash', color=colors[color_i]),
                             showlegend=False, legendgroup=curve['name'])
            t3 = go.Scatter(x=[curve['qmin']], y=[curve['hmax']], mode='markers', 
                            marker=dict(color=colors[color_i], size=10, symbol='circle'), name=f"{curve['name']} Qmin",
                            showlegend=False, legendgroup=curve['name'])
            fig.add_traces([t1,t2,t3])
            Fig_title = 'Rate vs Bed Height'
        color_i+=1
        
        tablerows.append(html.Tr([
            html.Td(curve['name'], style=cellstyle),
            html.Td(f"{curve['qmin']:.2f}", style=cellstyle),
            html.Td(f"{curve['hmax']:.2f}", style=cellstyle), 
            html.Td(f"{curve['ppa']:.2f}", style=cellstyle),
            html.Td(curve['model'], style=cellstyle)
        ]))
    options = [dict(value=curve['name'], label=curve['name']) for curve in curves]
    fig.update_layout(title=Fig_title,
                      xaxis_title=x_axis_title,
                      yaxis_title='Bed Height (inches)',
                      xaxis=dict(range=[0, 1.2 * max([c['qmax'] for c in curves])]),
                      width=800, height=600)
    if graphitem == 'dpdx':
        fig.update_layout(yaxis_title='dp/dx-alpha (psi/ft)')
        fig.update_yaxes(range=[0, 1.2*max([max(c['dpdx'])for c in curves])])
    elif graphitem == 'dpdx_w':
        fig.update_layout(yaxis_title='dp/dx-beta (psi/ft)')
        fig.update_yaxes(range=[0, 1.2*max([max(c['dpdx_w'])for c in curves])]) 
    else:
        fig.update_yaxes(range=[0.3*min([c['oh'] for c in curves]), max([c['oh'] for c in curves])])

    return dcc.Graph(figure=fig),html.Table([row for row in tablerows]), options 


if __name__ == "__main__":
    app.run(debug=True, port=8050)


