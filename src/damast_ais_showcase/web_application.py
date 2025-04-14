# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause
import datetime
import logging
import tempfile
import webbrowser
from logging import DEBUG, INFO, WARNING, Logger, getLogger
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

import dash
import dash_uploader as du

# Using dash_logger in combination with dash_mantine_components
# https://community.plotly.com/t/logging-in-dash-logtransform/61173/17
# https://www.dash-extensions.com/transforms/log_transform
import dash_mantine_components as dmc
import diskcache
from damast.core.dataframe import AnnotatedDataFrame
from damast.ml.scheduler import JobScheduler
from dash import DiskcacheManager, dcc, html
from dash_extensions.enrich import DashLogger, DashProxy, LogTransform

logging.basicConfig()
_log: Logger = getLogger(__name__)
_log.setLevel(DEBUG)

cache = diskcache.Cache( Path(tempfile.gettempdir()) / "damast-ais-showcase-cache")
background_callback_manager = DiskcacheManager(cache)


DATA_UPLOAD_PATH = Path(tempfile.gettempdir()) / "damast-ais-showcase"


class WebApplication:
    """
    Base class for Dash web applications
    """

    _app: dash.Dash

    _layout: List[Any]
    _port: str
    _server: str

    data_upload_path: Union[str, Path]

    def __init__(self,
                 header: str,
                 server: str = "0.0.0.0",
                 port: str = "8888",
                 upload_path = DATA_UPLOAD_PATH):

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

        self.data_upload_path = upload_path
        du.configure_upload(self._app, self.data_upload_path,
                            use_upload_id=True)

    def set_header(self, header: str):
        self._layout.append(dash.html.Div(
            children=[dash.html.H1(children=header)]))

    def run_server(self, debug: bool = True):
        self._app.layout = dmc.MantineProvider(
            dash.html.Div(self._layout)
        )
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

    def generate_layout(self):
        """Generate dashboard layout
        """
        from .tabs.explore import ExploreTab
        #from .tabs.predict import PredictionTab

        self._layout += [
            html.Div([
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
