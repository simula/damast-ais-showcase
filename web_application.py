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
from damast.ml.scheduler import JobScheduler, Job, ResponseCollector
from dash import dash_table, State, DiskcacheManager
from dash import dcc, Output, Input
from dash import html, ctx

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

    _job_scheduler: JobScheduler

    _log_messages: List[str]

    select_experiment: html.Div

    def __init__(self, ais: vaex.DataFrame, port: str = "8888"):
        super().__init__({"ais": ais}, "AIS Anomaly Detection", port=port)

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

    def callbacks(self):
        """Define input/output of the different dash applications
        """
        @self._app.callback(
            [
                Output('model-name', 'data'),
                Output('experiment-model-predict', 'children')
            ],
            [Input({'component_id': 'model-dropdown'}, 'value')],
            prevent_initial_callbacks=True
        )
        def update_model(value):
            """
            Set the currently select model internally for further processing

            :param value:
            :return:
            """
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
            return value, html.Div(children=[
                predict_button
            ])

        @self._app.callback(
            [
                Output('button-select-experiment-directory', 'children'),
                Output('experiment-directory', 'data'),
                Output('select-models', 'children'),
            ],
            [
                Input('button-select-experiment-directory', 'n_clicks'),
                State('button-select-experiment-directory', 'children'),
                State('experiment-directory', 'data')
            ]
        )
        def load_experiment(n_clicks, state_button_children, state_experiment_directory):
            """
            Allow to select the experiment and populate the available models for the dropdown list.

            :param n_clicks:
            :param state_button_children:
            :param state_models_children:
            :return:
            """
            default_value = state_button_children, state_experiment_directory, ""

            if n_clicks > 0:
                root = tkinter.Tk()
                root.withdraw()
                directory = filedialog.askdirectory()
                root.destroy()

                if not isinstance(directory, str):
                    return default_value

                self.log(message=f"Loading Experiment from {directory}")
                models = Experiment.from_directory(directory)

                row_models = []
                for model_name, keras_model in models.items():
                    row_models.append(model_name)

                model_dropdown = dash.html.Div(children=[html.H3("Models"),
                                                         dash.dcc.Dropdown(id={'component_id': "model-dropdown"},
                                                                           placeholder="Select a model",
                                                                           multi=False,
                                                                           options=list(models.keys()))],
                                               style={
                                                   "display": "inline-block",
                                                   "width": "100%"
                                               })
                return Path(directory).stem,\
                    directory, \
                    [html.Div(id={'component_id': 'mmsi-selection'}), model_dropdown]

            return default_value

        @self._app.callback(
            Output({'component_id': 'select-mmsi-dropdown'}, 'options'),
            Input({'component_id': 'filter-mmsi-min-length'}, 'value'),
            State('data-filename', 'data'),
            prevent_initial_callbacks=True
        )
        def filter_mmsi(min_length, data_filename):
            adf = AnnotatedDataFrame.from_file(json.loads(data_filename))
            messages_per_mmsi = adf.groupby("mmsi", agg="count")
            filtered_mmsi = messages_per_mmsi[messages_per_mmsi["count"] > min_length]
            selectable_mmsis = sorted(filtered_mmsi.mmsi.unique())
            return selectable_mmsis

        @self._app.callback(
            [
                Output({'component_id': 'mmsi-stats'}, 'children'),
                Output('mmsi', 'data')
            ],
            Input({'component_id': 'select-mmsi-dropdown'}, 'value'),
            State('data-filename', 'data'),
        )
        def select_mmsi(value, data_filename):
            if value is not None:
                current_mmsi = int(value)
                adf = AnnotatedDataFrame.from_file(json.loads(data_filename))
                mmsi_df = adf[adf.mmsi == current_mmsi]

                mean_lat = mmsi_df.mean("lat")
                mean_lon = mmsi_df.mean("lon")
                var_lat = mmsi_df.var("lat")
                var_lon = mmsi_df.var("lon")
                length = mmsi_df.count()

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

                trajectory_plot = dash.dcc.Graph(id="mmsi-plot-map",
                                                 figure=plot_boat_trajectory(mmsi_df))

                return [mmsi_stats_table, trajectory_plot], json.dumps(value)
            return None, json.dumps(None)

        @self._app.callback(
            [
                Output('button-select-data-directory', 'children'),
                Output('data-preview', 'children'),
                Output('data-filename', 'data'),
            ],
            [
                Input('button-select-data-directory', 'n_clicks'),
                State('button-select-data-directory', 'children'),
                State('data-preview', 'children'),
                State('data-filename', 'data'),
            ],
            prevent_initial_callbacks=True
        )
        def update_data(n_clicks, state_data_button, state_data_preview, state_data_filename):
            """
            Set the current data that shall be used for prediction

            :param n_clicks:
            :param state_data_button:
            :param state_data_preview:
            :return:
            """
            if n_clicks <= 0:
                return state_data_button, state_data_preview, state_data_filename

            root = tkinter.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(title="Select data files",
                                                  filetypes=[('HDF5', '*.hdf5'), ('H5', '*.h5')])
            root.destroy()

            if not isinstance(filename, str):
                return state_data_button, state_data_preview, state_data_filename

            adf = AnnotatedDataFrame.from_file(filename=filename)
            df = adf[:5].to_pandas_df()
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

            min_messages = 0
            max_messages = 10000
            #markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
            filter_mmsi_slider = dash.dcc.Slider(id={'component_id': 'filter-mmsi-min-length'},
                                                 min=min_messages, max=max_messages,
                                                 value=50,
                                                 # marks={i: f"{int(10 ** i)}" for i in
                                                 #       markers},
                                                 )
            if "mmsi" in df.columns:
                select_mmsi_dropdown.options = sorted(adf.mmsi.unique())
            else:
                self.log("No column 'mmsi' in the dataframe - did you select the right data?")

            data_preview = [
                html.H3("Data Preview"),
                data_preview_table,
                html.H2("Select Vessel"),
                html.Div(id="mmsi-filter", children=[
                    html.H3("Minimum sequence length"),
                    filter_mmsi_slider
                ]),
                html.H3("MMSI"),
                select_mmsi_dropdown,
                html.Div(id={"component_id": "mmsi-stats"})
            ]
            return Path(filename).name, data_preview, json.dumps(filename)

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
                State('mmsi', 'data'),  # current_mmsi
                State('prediction-job', 'data'),
                State('prediction-thread-status', 'data'),
                State('experiment-directory', 'data'),
                State('model-name', 'data'), # model_name
                State('data-filename', 'data')
            ],
            prevent_initial_callbacks=True
        )
        def predict_with_model(n_clicks, button_label, button_style,
                               mmsi,
                               predict_job, prediction_thread_status,
                               experiment_directory,
                               model_name,
                               data_filename):
            """
            Handle the Button event on the predict button.

            1. Trigger the execution of a prediction job for a particular mmsi

            :param n_clicks:
            :param prediction_thread_status:
            :param button_label:
            :param button_style:
            :param predict_job:
            :return:
            """
            if button_label.startswith("Cancel"):
                # When the button holds the label cancel, a prediction has been triggered
                # but if the actual corresponding thread is not running anymore, the
                # button should switch back to the default anyway
                thread_status = json.loads(prediction_thread_status)
                if thread_status == ResponseCollector.Status.RUNNING.value:
                    job_dict = json.loads(predict_job)
                    self._job_scheduler.stop(job_dict["id"])

                style = button_style
                style["backgroundColor"] = "green"
                return json.dumps(None), f"Predict with {model_name}", style

            # If mmsi is not set - there is no need to trigger the prediction
            if mmsi is None:
                style = button_style
                style["backgroundColor"] = "green"
                return predict_job, f"Predict with {model_name}", style

            current_mmsi = json.loads(mmsi)

            if model_name is not None:
                # 1. Get all mmsi based data from the dataframe
                # 2. Allow to pick from an mmsi
                # 3. Create a job to request the prediction
                start_time = timer()

                adf = AnnotatedDataFrame.from_file(json.loads(data_filename))

                self.log(f"predict (mmsi: {current_mmsi}): load and apply state to dataframe")
                DataProcessingPipeline.load_state(df=adf, dir=experiment_directory)
                prepared_df = adf._dataframe

                self.log("predict: loading dataframe and converting to pandas")
                df = prepared_df[prepared_df.mmsi == current_mmsi]
                if df.count() < 51:
                    self.log(f"Data fro MMSI ({current_mmsi}) is too short/has insufficient length")
                    # RUN PREDICTION AND PRESENT THE RESULT -- ASYNC
                    return json.dumps(None), button_label, button_style

                dash.callback_context.record_timing('predict:prepare', timer() - start_time, 'pipeline: transform data')

                self.log("predict: preparing prediction job")
                # Temporarily store this sequence to disk - so that the worker can pick it up
                tmpdir = tempfile.mkdtemp(prefix='.damast-ais-showcase.')
                tmpfile = Path(tmpdir) / f"mmsi-{current_mmsi}.hdf5"
                df.export(tmpfile)
                self.log(f"df: exported to {tmpfile}")
                if not tmpfile.exists():
                    raise FileNotFoundError("Failed to create temporary data file")

                # region RETRIEVE MODEL INFO
                # FIXME:
                # The features and target which are used for prediction here - are part of the trained model.
                # So we would have to get the information from the trained model to make this
                # work properly
                features = ["latitude_x", "longitude_x",
                            "latitude_y", "longitude_y"]

                sequence_length = 50
                # If there is any forecast at all
                sequence_forecast = 1
                # endregion

                # Create the prediction job
                job = Job(
                    id=0,
                    experiment_dir=experiment_directory,
                    model_name=model_name,
                    features=features,
                    target=features,
                    sequence_length=50,
                    data_filename=str(tmpfile)
                )
                # Start the prediction job
                self._job_scheduler.start(job)

                # Ensure that the use is informed about the running job
                style = button_style
                style["backgroundColor"] = "red"
                return json.dumps(job.__dict__), f"Cancel (job id: {job.id})", button_style

            return json.dumps(None), button_label, button_style

        @self._app.callback(
            [
                Output('logging-console-display', 'data')
            ],
            [
                Input({'component_id': 'button-logging-console'}, 'n_clicks'),
                State('logging-console-display', 'data')
            ],
            prevent_initial_callbacks=True
        )
        def logging_console_toggle(n_clicks, logging_console_display):
            """
            Callback to setting the state for the show/hide of the logging console

            :param n_clicks:
            :param logging_console_display:
            :return: Store value for the logging-console-display
            """
            if ctx.triggered_id == 'logging-console-display' or n_clicks is None:
                return [logging_console_display]

            if logging_console_display == "show":
                return ["hide"]
            else:
                return ["show"]

        @self._app.callback(
            [
                Output('prediction-results', 'children'),
                Output('prediction-thread-status', 'data'),
                Output('logging-console', 'children'),
            ],
            [
                Input('update-interval', 'n_intervals'),
                State('prediction-job', 'data'),
                State('logging-console-display', 'data')
            ],
        )
        def interval_update(n_intervals, prediction_job_data, logging_console_display):
            """

            :param n_intervals: current trigger of intervals
            :param prediction_job_data:
            :return:
            """
            prediction_result = None
            prediction_thread_status = json.dumps(ResponseCollector.Status.NOT_STARTED.value)
            if prediction_job_data is not None:
                json_data = json.loads(prediction_job_data)
                if json_data is not None:
                    job_id = json_data["id"]
                    responses, status = self._job_scheduler.get_status(job_id)
                    prediction_thread_status = json.dumps(status.value)

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

            logging_table = None
            if logging_console_display == "show":
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
                                           ],
                                           style={
                                               'background': 'lightyellow',
                                               'borderStyle': 'solid',
                                               'borderRadius': '5px',
                                               'borderWidth': '1px',
                                               'width': '100%',
                                               'maxHeight': '20%',
                                               'overflow': 'auto'
                                           }
                                           )
            button_label = "show"
            if logging_console_display == "show":
                button_label = "hide"

            button_logging_display = html.Button(button_label,
                                                 id={'component_id': "button-logging-console"},
                                                 style={
                                                     'borderStyle': 'none',
                                                     'borderRadius': '5px'
                                                 })

            return prediction_result, prediction_thread_status, [
                html.Hr(),
                html.H3("Logging Console ",
                        style={
                            "display": "inline"
                        }),
                button_logging_display,
                logging_table
            ]

    def tab_predict(self) -> dcc.Tab:
        return dcc.Tab(label='Predict', children=[
            # upload,
            self.select_experiment,
            html.Div(id="experiment-model-predict"),
            html.Div(id='logging-console',
                     children=[html.H3("Logging Console"),
                               html.Button("Hide", id={'component_id': 'button-logging-console'}, n_clicks=0)],
                     style={
                         "position": "fixed",
                         "bottom": 0,
                         "width": "100%"
                     }),
            # Create an interval from which regular updates are trigger, e.g.,
            # the logging console is updated - interval is set in milliseconds
            dcc.Interval("update-interval", interval=1000),
            dcc.Store(id='experiment-directory', storage_type='session'),
            dcc.Store(id='data-filename', storage_type="session"),
            dcc.Store(id='model-name', storage_type="session"),
            dcc.Store(id='mmsi', storage_type='session'),
            # A store to trigger the prediction background job - with storage type
            # memory, the data will be cleared with a refresh
            dcc.Store(id='prediction-job', storage_type="memory"),
            dcc.Store(id='prediction-thread-status', storage_type="memory"),

            # Control showing of control
            dcc.Store(id='logging-console-display',
                      data="hide",
                      storage_type="session"),
        ])

    def tab_explore(self) -> dcc.Tab:
        return dcc.Tab(label="Explore",
                       children=[],
                       className="tab")

    def generate_layout(self):
        """Generate dashboard layout
        """
        self._layout += [
            html.Div([
                dcc.Tabs([
                    self.tab_explore(),
                    self.tab_predict()
                ])
            ])
        ]
