# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause
import base64
import datetime
import json
import multiprocessing
import os
import socket
import tempfile
import tkinter
import webbrowser
from collections import OrderedDict
from pathlib import Path
from time import sleep
from timeit import default_timer as timer
from tkinter import filedialog
from typing import Dict, List, Any

import dash
import diskcache
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import vaex
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.data_handling.accessors import SequenceIterator
from damast.ml.experiments import Experiment
from damast.ml.scheduler import JobScheduler, Job
from dash import dash_table, State, DiskcacheManager
from dash import dcc, Output, Input
from dash import html

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

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

        external_scripts = [
            "https://code.jquery.com/jquery-2.2.4.js"
        ]
        self._app = dash.Dash(__name__,
                              external_scripts=external_scripts,
                              # Allow to define callbacks on dynamically defined components
                              suppress_callback_exceptions=False,
                              )
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

    #: Set available set of model that can be used for prediction
    _models: Dict[str, Any]
    _pipeline: DataProcessingPipeline

    _experiment_folder: Path

    _current_model_name: str
    _current_model: Any
    _current_data: AnnotatedDataFrame
    _current_mmsi: int

    _job_scheduler: JobScheduler

    _log_messages: List[str]

    select_experiment: html.Div

    def __init__(self, ais: vaex.DataFrame, port: str = "8888"):
        super().__init__({"ais": ais}, "AIS Anomaly Detection", port=port)
        self._messages_per_mmsi = count_number_of_messages(self._data["ais"])

        self._experiment_folder = None
        self._models = {}
        self._log_messages = []

        self._job_scheduler = JobScheduler()

        self.callbacks()

        experiment_button = html.Button(children='Pick experiment',
                                        id='button-select-experiment-directory',
                                        n_clicks=0,
                                        style={
                                            "position": "relative",
                                            "display": "inline-block",
                                            "backgroundColor": "white",
                                            "color": "darkgray",
                                            "textAlign": "center",
                                            "fontSize": "1.2em",
                                            "width": "100%",
                                            "borderStyle": "dashed",
                                            "borderRadius": "5px",
                                            "borderWidth": "1px",
                                            "margin": "10px",
                                        })

        data_button = html.Button(children='Pick dataset',
                                  id='button-select-data-directory',
                                  n_clicks=0,
                                  style={
                                      "position": "relative",
                                      "display": "inline-block",
                                      "backgroundColor": "white",
                                      "color": "darkgray",
                                      "textAlign": "center",
                                      "fontSize": "1.2em",
                                      "width": "100%",
                                      "borderStyle": "dashed",
                                      "borderRadius": "5px",
                                      "borderWidth": "1px",
                                      "margin": "10px",
                                  })

        self.select_experiment = html.Div([
            experiment_button,
            # html.Div(dcc.Input(id='input-experiment-directory', type='hidden', value="<experiment folder>")),
            html.Div(id='select-models'),
            data_button,
            html.Div(id='data-preview'),
            html.Div(id='prediction-results')
        ])

    def log(self, message: str):
        timestamp = datetime.datetime.utcnow()
        self._log_messages.append((timestamp, message))

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
        return sorted(min_message_boats.unique())

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

        self._app.callback(dash.dependencies.Output("plot_map", "figure"),
                           dash.dependencies.Input("dropdown", "value"), prevent_initial_call=True)(self.update_map)

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

            # @self._app.callback(Output('output-data-upload', 'children'),
            #                     Input('upload-data', 'contents'),
            #                     State('upload-data', 'filename'),
            #                     State('upload-data', 'last_modified'))
            # def update_output(list_of_contents, list_of_names, list_of_dates):
            if list_of_contents is not None:
                children = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
                return children

        @self._app.callback(
            Output('experiment-model-predict', 'children'),
            [Input({'component_id': 'model-dropdown'}, 'value')]
        )
        def update_model(value):
            """
            Set the currently select model internally for further processing

            :param value:
            :return:
            """
            if value in self._models:
                self._current_model_name = value
                self._current_model = self._models[value]

                predict_button = html.Button(children=f'Predict with {value}',
                                             id={'component_id': 'button-predict-with-model'},
                                             n_clicks=0,
                                             style={
                                                 "position": "relative",
                                                 "display": "inline-block",
                                                 "backgroundColor": "green",
                                                 "color": "lightgray",
                                                 "textAlign": "center",
                                                 "fontSize": "1em",
                                                 "width": "20%",
                                                 "borderStyle": "solid",
                                                 "borderRadius": "5px",
                                                 "borderWidth": "1px",
                                                 "margin": "10px",
                                             })
                return html.Div(children=[
                    predict_button
                ])
            else:
                return None

        @self._app.callback(
            [
                Output('button-select-experiment-directory', 'children'),
                Output('select-models', 'children'),
            ],
            [
                Input('button-select-experiment-directory', 'n_clicks'),
                State('button-select-experiment-directory', 'children'),
                State('select-models', 'children'),
            ]
        )
        def update_model_selection(n_clicks, state_button_children, state_models_children):
            """
            All to select one of the available model from a dropdown list
            :param n_clicks:
            :param state_button_children:
            :param state_models_children:
            :return:
            """
            default_value = state_button_children, ""

            if n_clicks > 0:
                root = tkinter.Tk()
                root.withdraw()
                directory = filedialog.askdirectory()
                root.destroy()

                if not isinstance(directory, str):
                    return state_button_children, state_models_children

                self._experiment_folder = Path(directory)

                self.log(message=f"Loading Experiment from {directory}")
                self._models = Experiment.from_directory(directory)

                row_models = []
                for model_name, keras_model in self._models.items():
                    row_models.append(model_name)

                data = OrderedDict([
                    ("Models", row_models)
                ])

                df = pd.DataFrame(data)
                table = dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'id': c, 'name': c} for c in df.columns],
                    # https://dash.plotly.com/datatable/style
                    style_cell={'textAlign': 'center', 'padding': "5px"},
                    style_header={'backgroundColor': 'lightgray',
                                  'color': 'white',
                                  'fontWeight': 'bold'}
                )
                model_dropdown = dash.html.Div(children=[html.H3("Models"),
                                                         dash.dcc.Dropdown(id={'component_id': "model-dropdown"},
                                                                           placeholder="Select a model",
                                                                           multi=False,
                                                                           options=list(self._models.keys()))],
                                               style={
                                                   "display": "inline-block",
                                                   "width": "100%"
                                               })

                # self.experiment_button.children = html.H2(Path(directory).stem)

                return Path(directory).stem, [model_dropdown, html.Div(id={'component_id': 'mmsi-selection'})]

            return default_value

        @self._app.callback(
            Output({'component_id': 'mmsi-stats'}, 'children'),
            Input({'component_id': 'select-mmsi-dropdown'}, 'value'),
            State({'component_id': 'mmsi-selection'}, 'children')
        )
        def select_mmsi(value, dropdown_mmsi_selection):
            if value is not None:
                self._current_mmsi = int(value)

                df_stats = self._current_data[self._current_data.mmsi == self._current_mmsi]
                mean_lat = df_stats.mean("lat")
                mean_lon = df_stats.mean("lon")
                var_lat = df_stats.var("lat")
                var_lon = df_stats.var("lon")
                length = df_stats.count()

                data = {"Length": length,
                        "Lat": f"{mean_lat:.2f} +/- {var_lat:.3f}",
                        "Lon": f"{mean_lon:.2f} +/- {var_lon:.3f}"
                        }

                mmsi_stats_table = dash_table.DataTable(
                    data=[data],
                    columns=[{'id': c, 'name': c} for c in data.keys()],
                    # https://dash.plotly.com/datatable/style
                    style_cell={'textAlign': 'center', 'padding': "5px"},
                    style_header={'backgroundColor': 'lightgray',
                                  'color': 'white',
                                  'fontWeight': 'bold'}
                )

                return mmsi_stats_table
            return None

        @self._app.callback(
            [
                Output('button-select-data-directory', 'children'),
                Output('data-preview', 'children'),
            ],
            [
                Input('button-select-data-directory', 'n_clicks'),
                State('button-select-data-directory', 'children'),
                State('data-preview', 'children'),
            ]
        )
        def update_data(n_clicks, state_data_button, state_data_preview):
            """
            Set the current data that shall be used for prediction

            :param n_clicks:
            :param state_data_button:
            :param state_data_preview:
            :return:
            """
            if n_clicks > 0:
                if n_clicks > 0:
                    root = tkinter.Tk()
                    root.withdraw()
                    filename = filedialog.askopenfilename(title="Select data files",
                                                          filetypes=[('HDF5', '*.hdf5'), ('H5', '*.h5')])
                    root.destroy()

                    if not isinstance(filename, str):
                        return state_data_button, state_data_preview

                    self._current_data = AnnotatedDataFrame.from_file(filename=filename)

                    df = self._current_data[:5].to_pandas_df()
                    data_preview_table = dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in df.columns],
                        # https://dash.plotly.com/datatable/style
                        style_cell={'textAlign': 'center', 'padding': "5px"},
                        style_header={'backgroundColor': 'lightgray',
                                      'color': 'white',
                                      'fontWeight': 'bold'}
                    )
                    select_mmsi_dropdown = dcc.Dropdown(
                        placeholder="Select MSSI for prediction",
                        id={'component_id': "select-mmsi-dropdown"},
                        multi=False,
                    )
                    if "mmsi" in df.columns:
                        select_mmsi_dropdown.options = sorted(self._current_data.mmsi.unique())
                    else:
                        self.log("No column 'mmsi' in the dataframe - did you select the right data?")

                    data_preview = [
                        html.H3("Data Preview"),
                        data_preview_table,
                        html.H3("Select Vessel by MMSI"),
                        select_mmsi_dropdown,
                        html.Div(id={"component_id": "mmsi-stats"})
                    ]
                    return Path(filename).name, data_preview
            else:
                return state_data_button, state_data_preview

        @self._app.callback(
            [
                Output('prediction-job', 'data'),
                Output({'component_id': 'button-predict-with-model'}, 'children'),
                Output({'component_id': 'button-predict-with-model'}, 'style')
            ],
            [
                Input({'component_id': 'button-predict-with-model'}, 'n_clicks'),
                State({'component_id': 'button-predict-with-model'}, 'children'),
                State({'component_id': 'button-predict-with-model'}, 'style'),
                State('prediction-job', 'data')
            ]
        )
        def predict_with_model(n_clicks, button_label, button_style, predict_job):
            if button_label.startswith("Cancel") and n_clicks > 0:
                job_dict = json.loads(predict_job)
                self._job_scheduler.stop(job_dict["id"])
                style = button_style
                style["backgroundColor"] = "green"
                return json.dumps(None), f"Predict with {self._current_model_name}",  style
            elif n_clicks > 0 and self._current_model is not None:

                # 1. Get all mmsi based data from the dataframe
                # 2. Allow to pick from an mmsi
                # 3. Create a job to request the prediction
                if self._current_mmsi is None:
                    return json.dumps(None), button_label, button_style

                start_time = timer()
                adf = self._current_data

                # self._pipeline = DataProcessingPipeline.load(dir=self._experiment_folder)
                # prepared_df = self._pipeline.transform(df=adf)
                self.log("predict: load and apply state to dataframe")
                DataProcessingPipeline.load_state(df=adf, dir=self._experiment_folder)
                prepared_df = adf._dataframe

                self.log("predict: loading dataframe and converting to pandas")
                df = prepared_df[prepared_df.mmsi == self._current_mmsi]
                if df.count() < 51:
                    self.log(f"MMSI is too short/has insufficient length")
                    # RUN PREDICTION AND PRESENT THE RESULT -- ASYNC
                    return json.dumps(None), button_label, button_style

                dash.callback_context.record_timing('predict:prepare', timer() - start_time, 'pipeline: transform data')

                self.log("predict: preparing prediction job")
                tmpdir = tempfile.mkdtemp(prefix='.damast-ais-showcase.')
                tmpfile = Path(tmpdir) / f"mmsi-{self._current_mmsi}.hdf5"
                df.export(tmpfile)
                self.log(f"df: exported to {tmpfile}")
                if not tmpfile.exists():
                    raise FileNotFoundError("Failed to create temporary data file")

                features = ["latitude_x", "longitude_x",
                            "latitude_y", "longitude_y"]

                job = Job(
                    id=0,
                    experiment_dir=str(self._experiment_folder),
                    model_name=self._current_model_name,
                    features=features,
                    target=features,
                    sequence_length=50,
                    data_filename=str(tmpfile)
                )

                self._job_scheduler.start(job)

                style = button_style
                style["backgroundColor"] = "red"
                return json.dumps(job.__dict__), f"Cancel (job id: {job.id})", button_style

            return json.dumps(None), button_label, button_style

        @self._app.callback(
            [
                Output('prediction-results', 'children'),
                Output('logging-console', 'children')
            ],
            [
                Input('logging-console-interval', 'n_intervals'),
                State('prediction-job', 'data')
            ]
        )
        def log_content(log_interval, prediction_job_data):
            prediction_result = None
            if prediction_job_data is not None:
                json_data = json.loads(prediction_job_data)
                if json_data is not None:
                    job_id = json_data["id"]
                    responses, status = self._job_scheduler.get_status(job_id)

                    timepoints = []
                    losses = []
                    for response in responses:
                        timepoints.append(response.timepoint)
                        losses.append(response.loss)

                    data = zip(timepoints, losses)
                    df = pd.DataFrame(data=data, columns=["timepoint", "loss"])
                    fig = px.scatter(df, x="timepoint", y="loss", title="Forecast loss",
                                     range_y=[0, 1])

                    prediction_result = dcc.Graph(figure=fig)

            # take in the messages through some function
            log_entries = []
            for timestamp, msg in self._log_messages:
                log_entries.append(html.Tr(children=[
                    html.Td(timestamp.strftime("%Y%m%d-%H%M%S")),
                    html.Td(msg)])
                )

            logging_table = html.Table(title="Logging Console",
                                       children=[
                                           html.Thead(
                                               children=[
                                                   html.Tr(
                                                       children=[
                                                           html.Th("timestamp"),
                                                           html.Th("message"),
                                                       ]
                                                   )
                                               ]
                                           ),
                                           html.Tbody(
                                               children=log_entries
                                           )
                                       ]
                                       )
            return prediction_result, [html.H3("Logging Console"), logging_table]

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

        # button = html.Div([
        #    html.Div(
        #        dcc.Input(id='input-on-submit', type='file', style={"display": "none"})
        #    ),
        #    html.Button('Experiment folder', id='set-experiment-folder', n_clicks=0),
        #    html.Div(id='container-button-basic',
        #             children='Enter a value and press submit')
        # ])

        return dcc.Tab(label='Experiments', children=[
            # upload,
            self.select_experiment,
            html.Div(id="experiment-model-predict"),
            html.Div(id='logging-console',
                     children=[html.H3("Logging Console")],
                     style={
                         'background': 'lightyellow',
                         'borderStyle': 'solid',
                         'borderRadius': '5px',
                         'borderWidth': '1px',
                         'height': '300px',
                         'overflow': 'auto'
                     }),
            # Create an interval for which the logging console is updated
            dcc.Interval("logging-console-interval", interval=1000),
            # A store to trigger the prediction background job
            dcc.Store(id='prediction-job', storage_type="session"),
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
