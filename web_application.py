# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause

import webbrowser
from typing import Dict, List, Any

import dash
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objs as go
import vaex
from dash import dcc, Output, Input, State
from dash import html

import tkinter
from tkinter import filedialog

def sort_by(df: vaex.DataFrame, key: str) -> vaex.DataFrame:
    """Sort input dataframe by entries in given column"""
    return df.sort(df[key])


def transform_value(value):
    return 10 ** value


def plot_boat_trajectory(data_df: vaex.DataFrame) -> go.Figure:
    """
    Extract (lat, long) coordinates from dataframe and group them by MMSI.
    NOTE: Assumes that the input data is sorted by in time
    """
    input = {"lat": data_df["lat"].evaluate(),
             "lon": data_df["lon"].evaluate(),
             "mmsi": data_df["mmsi"].evaluate()}
    fig = px.line_mapbox(input,
                         lat="lat", lon="lon",
                         color="mmsi")

    fig2 = px.density_mapbox(input, lat='lat', lon='lon', z='mmsi', radius=5)

    fig.add_trace(fig2.data[0])
    fig.update_coloraxes(showscale=False)

    fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=4)

    return fig


def count_number_of_messages(data: vaex.DataFrame) -> vaex.DataFrame:
    """Given a set of AIS messages, accumulate number of messages per mmsi identifier"""
    return data.groupby("mmsi", agg="count")


class WebApplication():
    """
    Base class for Dash web applications
    """

    _app: dash.Dash
    _data: Dict[str, vaex.DataFrame]
    _layout: List[Any]
    _port: str
    _server: str

    def __init__(self, df: Dict[str, vaex.DataFrame],
                 header: str,
                 server: str = "0.0.0.0", port: str = "8888"):
        self._data = df
        self._app = dash.Dash(__name__)
        self.app.title = "ML App for the Maritime Domain"
        self._port = port
        self._server = server
        self._layout = []
        self.set_header(header)

    def set_header(self, header: str):
        self._layout.append(dash.html.Div(
            children=[dash.html.H1(children=header)]))

    def run_server(self, debug: bool = True):
        self._app.layout = dash.html.Div(self._layout)
        self._app.run(debug=debug, host=self._server, port=self._port)

    def open_browser(self):
        webbrowser.open(f"http://{self._host}:{self._port}")

    @property
    def app(self) -> dash.Dash:
        return self._app


class AISApp(WebApplication):
    _messages_per_mmsi: vaex.DataFrame

    def __init__(self, ais: vaex.DataFrame, port: str = "8888"):
        super().__init__({"ais": ais}, "AIS Anomaly Detection", port=port)
        self._messages_per_mmsi = count_number_of_messages(self._data["ais"])
        self.callbacks()

    def filter_boats_by_mmsi(self, MMSI: List[int]) -> vaex.DataFrame:
        """Filter boats by MMSI identifier"""
        return self._data["ais"][self._data["ais"]["mmsi"].isin(MMSI)]

    def boats_with_min_messages(self, min_messages: float) -> npt.NDArray[np.int32]:
        """Get the boats (MMSI-identifiers) that has more than `min_messages`
        messages in database

        Args:
            min_messages (np.float64): Minimal number of messages in database

        Returns:
            _type_: MMSI identifiers
        """

        messages = int(transform_value(min_messages))
        min_message_boats = self._messages_per_mmsi[self._messages_per_mmsi["count"]
                                                    >= messages]["mmsi"]
        return min_message_boats.unique()

    def update_map(self, boat_ids: List[np.int32]) -> go.Figure:
        """
        Plot input boats in map (sorted by time-stamp)

        Args:
            boat_ids (List[np.int32]): List of boat identifiers

        Returns:
            go.Figure: Figure with heatmap and group per MMSI
        """
        if boat_ids is None:
            boat_ids = []
        data = self.filter_boats_by_mmsi(boat_ids)
        sorted_data = sort_by(data, "date_time_utc")
        return plot_boat_trajectory(sorted_data)

    def callbacks(self):
        """Define input/output of the different dash applications
        """

        self._app.callback(dash.dependencies.Output("dropdown", "options"),
                           dash.dependencies.Input("num_messages", "value"))(self.boats_with_min_messages)

        # self._app.callback(dash.dependencies.Output("plot_map", "figure"),
        #                   dash.dependencies.Input("dropdown", "value"), prevent_initial_call=True)(self.update_map)

        def parse_contents(contents, filename, date):
            content_type, content_string = contents.split(',')

            # decoded = base64.b64decode(content_string)
            # try:
            #    if 'csv' in filename:
            #        # Assume that the user uploaded a CSV file
            #        df = pd.read_csv(
            #            io.StringIO(decoded.decode('utf-8')))
            #    elif 'xls' in filename:
            #        # Assume that the user uploaded an excel file
            #        df = pd.read_excel(io.BytesIO(decoded))
            # except Exception as e:
            #    print(e)
            #    return html.Div([
            #        'There was an error processing this file.'
            #    ])

            return html.Div([
                html.H5(filename),
                # html.H6(datetime.datetime.fromtimestamp(date)),

                # dash_table.DataTable(
                #    df.to_dict('records'),
                #    [{'name': i, 'id': i} for i in df.columns]
                # ),

                # html.Hr(),  # horizontal line

                ## For debugging, display the raw contents provided by the web browser
                # html.Div('Raw Content'),
                # html.Pre(contents[0:200] + '...', style={
                #    'whiteSpace': 'pre-wrap',
                #    'wordBreak': 'break-all'
                # })
            ])

        @self._app.callback(Output('output-data-upload', 'children'),
                            Input('upload-data', 'contents'),
                            State('upload-data', 'filename'),
                            State('upload-data', 'last_modified'))
        def update_output(list_of_contents, list_of_names, list_of_dates):
            if list_of_contents is not None:
                children = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
                return children

        @self._app.callback(
            Output('container-button-basic', 'children'),
            Input('submit-val', 'n_clicks'),
            State('input-on-submit', 'value')
        )
        def update_output(n_clicks, value):
            if n_clicks > 0:
                root = tkinter.Tk()
                root.withdraw()
                directory = filedialog.askdirectory()
                root.destroy()

                return f'The input value was {directory} - clicks {n_clicks}'

    def tab_experiments(self) -> dcc.Tab:
        upload = html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Experiments - ',
                    html.A('Select Directory')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
        ])

        button = html.Div([
            html.Div(dcc.Input(id='input-on-submit', type='text')),
            html.Button('Submit', id='submit-val', n_clicks=0),
            html.Div(id='container-button-basic',
                     children='Enter a value and press submit')
        ])


        return dcc.Tab(label='Experiments', children=[
            upload,
            button,
            dcc.Graph(
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2],
                         'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5],
                         'type': 'bar', 'name': u'Montréal'},
                    ]
                }
            )
        ])

    def tab_prediction(self) -> dcc.Tab:
        # Generate slider for filtering min number of messages per MMSi
        min_messages = np.log(
            self._messages_per_mmsi["count"].min()) / np.log(10)
        max_messages = np.log(
            self._messages_per_mmsi["count"].max()) / np.log(10)
        markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
        children = [
            dash.html.H2(children="Minimum number of MMSI messages"),
            dash.dcc.Slider(id='num_messages',
                            min=min_messages, max=max_messages,
                            value=max_messages // 2, marks={i: f"{int(10 ** i)}" for i in
                                                            markers}),
            dash.html.Div(children=["Boat identifiers (MMSI)",
                                    dash.dcc.Dropdown(id="dropdown", multi=True)]),
            dash.html.Div(id='num_messages-output-container', style={'margin-top': 20}),
            dash.dcc.Graph(id="plot_map", figure=self.update_map([]))
        ]
        return dcc.Tab(label="Prediction",
                       children=children,
                       className="tab")

    def generate_layout(self):
        """Generate dashboard layout
        """
        self._layout += [
            html.Div([
                dcc.Tabs([
                    self.tab_experiments(),
                    self.tab_prediction()
                ])
            ])
        ]
