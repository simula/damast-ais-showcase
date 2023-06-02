# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause
import datetime
import logging
import tempfile
import webbrowser
from logging import DEBUG, INFO, WARNING, Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash

# Using dash_logger in combination with dash_mantine_components
# https://community.plotly.com/t/logging-in-dash-logtransform/61173/17
# https://www.dash-extensions.com/transforms/log_transform
import dash_mantine_components as dmc
import diskcache
import vaex
from damast.ml.scheduler import JobScheduler
from dash import DiskcacheManager, dcc, html
from dash_extensions.enrich import DashLogger, DashProxy, LogTransform

logging.basicConfig()
_log: Logger = getLogger(__name__)
_log.setLevel(DEBUG)

cache = diskcache.Cache( Path(tempfile.gettempdir()) / "damast-ais-showcase-cache")
background_callback_manager = DiskcacheManager(cache)


def sort_by(df: vaex.DataFrame, key: str) -> vaex.DataFrame:
    """Sort input dataframe by entries in given column"""
    return df.sort(df[key])

def transform_value(value):
    return 10 ** value

def count_number_of_messages(data: vaex.DataFrame) -> vaex.DataFrame:
    """Given a set of AIS messages, accumulate number of messages per passage_plan_id identifier"""
    return data.groupby("passage_plan_id", agg="count")

class WebApplication:
    """
    Base class for Dash web applications
    """

    _app: dash.Dash

    _layout: List[Any]
    _port: str
    _server: str

    def __init__(self,
                 header: str,
                 server: str = "0.0.0.0", port: str = "8888"):

        external_scripts = [
            "https://code.jquery.com/jquery-2.2.4.js"
        ]
        self._app = DashProxy(__name__,
                              transforms=[LogTransform()],
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
        webbrowser.open(f"http://{self._server}:{self._port}")

    @property
    def app(self) -> dash.Dash:
        return self._app

    @property
    def callback(self):
        return self._app.callback

    def log(self, message: str,
            level=INFO,
            dash_logger: DashLogger = None
            ):

        if dash_logger:
            if level == INFO:
                dash_logger.info(message, autoClose=10000)
            else:
                dash_logger.log(level=level, message=message)

        _log.log(level=level, msg=message)



class AISApp(WebApplication):
    #: Allow to run predictions as background job
    _job_scheduler: JobScheduler
    _log_messages: List[str]

    select_experiment: html.Div

    def __init__(self, port: str = "8888"):
        super().__init__("AIS Anomaly Detection", port=port)

        self._log_messages = []
        self._job_scheduler = JobScheduler()

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
        ])

    def generate_layout(self):
        """Generate dashboard layout
        """
        from .tabs.explore import ExploreTab
        from .tabs.predict import PredictionTab

        self._layout += [
            html.Div([
                # upload,
                self.select_experiment,
                # Create an interval from which regular updates are trigger, e.g.,
                # the logging console is updated - interval is set in milliseconds
                dcc.Interval("update-interval", interval=1000),
                # Control showing of control
                html.Div(id="tab-spacer",
                         style={
                             "height": "2em"
                         }),
                dcc.Tabs([
                    ExploreTab.create(app=self),
                    #PredictionTab.create(app=self)
                ],
                    # Start with the second tab
                    #value="tab-explore"
                )
            ])
        ]