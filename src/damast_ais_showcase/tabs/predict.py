import dash
from dash import html, dcc, Output, Input, State
from dash_extensions.enrich import DashLogger
import json
import tempfile

from timeit import default_timer as timer

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.ml.experiments import Experiment
from damast.ml.scheduler import Job, ResponseCollector

from pathlib import Path

import tkinter
from tkinter import filedialog
from logging import WARNING

import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.express as px


from ..figures import (
    create_figure_data_preview_table
)
from ..web_application import AISApp


class PredictionTab:
    @classmethod
    def register_callbacks(cls, app: AISApp):
        @app.callback(
            [
                Output('model-name', 'data'),
                Output('prediction-trigger', 'children')
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
            
        @app.callback(
            [
                Output({'component_id': 'button-predict-with-model'}, 'children'),
                Output({'component_id': 'button-predict-with-model'}, 'style'),
                Output({'component_id': 'button-predict-with-model'}, 'disabled')
            ],
            [
                Input('prediction-thread-status', 'data'),
                State('model-name', 'data'),
                State({'component_id': 'button-predict-with-model'}, 'style')
            ],
            prevent_initial_callbacks=True
        )
        def update_predict_button(prediction_thread_status, model_name, button_style):
            style = button_style.copy()
            if prediction_thread_status == Job.Status.RUNNING.value:
                style["backgroundColor"] = "red"
                return f"Cancel - Predict with {model_name}", style, False
            else:
                if model_name is not None:
                    style["backgroundColor"] = "green"
                    return f"Predict with {model_name}", style, False
                else:
                    style["backgroundColor"] = "gray"
                    return "Predict", style, True

        @app.callback(
            [
                Output('button-select-experiment-directory', 'children'),
                Output('experiment-directory', 'data'),
                Output('select-models', 'children'),
                Output('model-name', 'clear_data'),
            ],
            [
                Input('button-select-experiment-directory', 'n_clicks'),
                State('button-select-experiment-directory', 'children'),
                State('experiment-directory', 'data')
            ],
            prevent_initial_callbacks=True,
            log=True
        )
        def load_experiment(n_clicks, state_button_children, state_experiment_directory, dash_logger: DashLogger):
            """
            Allow to select the experiment and populate the available models for the dropdown list.

            :param n_clicks:
            :param state_button_children:
            :param state_models_children:
            :return:
            """
            default_value = state_button_children, state_experiment_directory, "", False

            directory = ''
            if n_clicks > 0:
                root = tkinter.Tk()
                root.withdraw()

                initial_directory = None
                if state_experiment_directory is not None:
                    initial_directory = str(Path(state_experiment_directory).parent)

                result_directory = filedialog.askdirectory(initialdir=initial_directory)
                if isinstance(result_directory, str):
                    directory = result_directory
                root.destroy()
            else:
                directory = state_experiment_directory

            if directory == '' or not isinstance(directory, str):
                return default_value

            app.log(f"Loading Experiment from {directory}",
                     dash_logger=dash_logger)

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
                                               "width": "100%",
                                           })
            return Path(directory).stem, \
                directory, \
                [html.Div(id={'component_id': 'prediction-passage_plan_id-selection'}), model_dropdown], \
                True  # Clear model_name

        @app.callback(
            [
                Output('button-select-data-directory', 'children'),
                Output('prediction-data-preview', 'children'),
                Output('prediction-passage_plan_id-selection', 'children'),
                Output('prediction-data-filename', 'data'),
                Output('prediction-passage_plan_id', 'clear_data'),
            ],
            [
                Input('button-select-data-directory', 'n_clicks'),
                State('button-select-data-directory', 'children'),
                State('prediction-passage_plan_id-selection', 'children'),
                State('prediction-data-preview', 'children'),
                State('prediction-data-filename', 'data'),
            ],
            prevent_initial_callbacks=True,
            log=True
        )
        def update_data(n_clicks, state_data_button, state_prediction_passage_plan_id_selection, state_data_preview,
                        state_data_filename, dash_logger: DashLogger):
            """
            Set the current data that shall be used for prediction

            :param n_clicks:
            :param state_data_button:
            :param state_data_preview:
            :return:
            """
            filename = ''
            if n_clicks > 0:
                initial_directory = str(Path().resolve())
                if state_data_filename is not None:
                    initial_directory = str(Path(state_data_filename).parent)

                root = tkinter.Tk()
                root.withdraw()
                result_filename = filedialog.askopenfilename(title="Select data files",
                                                             initialdir=initial_directory,
                                                             filetypes=[('HDF5', '*.hdf5'), ('H5', '*.h5')])
                if isinstance(result_filename, str):
                    filename = result_filename

                root.destroy()
            else:
                filename = state_data_filename

            if filename == '' or not isinstance(filename, str):
                return state_data_button, state_data_preview, state_prediction_passage_plan_id_selection, \
                    state_data_filename, False

            adf = AnnotatedDataFrame.from_file(filename=filename)
            data_preview_table = create_figure_data_preview_table(adf.dataframe)

            select_passage_plan_id_dropdown = dcc.Dropdown(
                placeholder="Select passage plan for prediction",
                id={'component_id': "select-prediction-passage_plan_id-dropdown"},
                multi=False,
            )

            df = adf[:5].to_pandas_df()
            feature_highlight_options = [{'label': c, 'value': c} for c in df.columns if is_numeric_dtype(df.dtypes[c])]
            feature_highlight_options.append({'label': 'no highlighting', 'value': ''})
            select_feature_highlight_dropdown = dcc.Dropdown(
                placeholder="Select a feature to highlight in the plot",
                id={'component_id': "select-feature-highlight-dropdown"},
                options=feature_highlight_options,
                multi=False,
            )

            min_messages = 0
            grouped = adf.groupby('passage_plan_id', agg={'sequence_length': 'count'})
            max_messages = max(grouped.sequence_length.values)
            # markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
            filter_passage_plan_id_slider = dash.dcc.RangeSlider(id={'component_id': 'filter-prediction-passage_plan_id-min-max-length'},
                                                 min=min_messages, max=max_messages,
                                                 value=[0,max_messages],
                                                 allowCross=False,
                                                 tooltip={'placement': 'bottom', 'always_visible': True}
                                                 # marks={i: f"{int(10 ** i)}" for i in
                                                 #       markers},
                                                 )
            if "passage_plan_id" in df.columns:
                select_passage_plan_id_dropdown.options = sorted(adf.passage_plan_id.unique())
            else:
                app.log("No column 'passage_plan_id' in the dataframe - did you select the right data?",
                         level=WARNING,
                         dash_logger=dash_logger)

            select_for_prediction = [
                html.H2("Select Vessel"),
                html.Div(id="prediction-passage_plan_id-filter", children=[
                    html.H3("Minimum-Maximum sequence length"),
                    filter_passage_plan_id_slider
                ]),
                select_passage_plan_id_dropdown,
                html.Br(),
                select_feature_highlight_dropdown,
                html.Div(id={"component_id": "prediction-passage_plan_id-stats"},
                         children=[
                             html.Div(
                                 children=[
                                     dash.dcc.Graph(id={'component_id': 'prediction-passage_plan_id-plot-map'})
                                     ],
                                 hidden=True
                             )
                            ]
                )
            ]
            return Path(filename).name, data_preview_table, select_for_prediction, \
                filename, True

        @app.callback(
            Output('prediction-job', 'data'),
            [
                Input({'component_id': 'button-predict-with-model'}, 'n_clicks'),
                State({'component_id': 'button-predict-with-model'}, 'children'),
                State({'component_id': 'button-predict-with-model'}, 'style'),
                State('prediction-passage_plan_id', 'data'),  # current_passage_plan_id
                State('prediction-job', 'data'),
                State('prediction-thread-status', 'data'),
                State('experiment-directory', 'data'),
                State('model-name', 'data'),  # model_name
                State('prediction-data-filename', 'data')
            ],
            prevent_initial_callbacks=True,
            log=True
        )
        def predict_with_model(n_clicks, button_label, button_style,
                               passage_plan_id,
                               predict_job, prediction_thread_status,
                               experiment_directory,
                               model_name,
                               data_filename,
                               dash_logger: DashLogger):
            """
            Handle the Button event on the predict button.

            1. Trigger the execution of a prediction job for a particular passage_plan_id

            :param n_clicks:
            :param prediction_thread_status:
            :param button_label:
            :param button_style:
            :param predict_job:
            :return:
            """
            if n_clicks <= 0:
                return predict_job

            # When the button holds the label cancel, a prediction has been triggered
            # but if the actual corresponding thread is not running anymore, the
            # button should switch back to the default anyway
            if prediction_thread_status == Job.Status.RUNNING.value:
                job_dict = json.loads(predict_job)
                app._job_scheduler.stop(job_dict["id"])
                return predict_job

            # If passage_plan_id is not set - there is no need to trigger the prediction
            if passage_plan_id is None:
                return predict_job

            current_passage_plan_id = json.loads(passage_plan_id)

            if model_name is not None:
                # 1. Get all passage_plan_id based data from the dataframe
                # 2. Allow to pick from an passage_plan_id
                # 3. Create a job to request the prediction
                start_time = timer()

                adf = AnnotatedDataFrame.from_file(data_filename)

                app.log(f"predict (passage_plan_id: {current_passage_plan_id}): load and apply state to dataframe",
                         dash_logger=dash_logger)
                DataProcessingPipeline.load_state(df=adf, dir=experiment_directory)
                prepared_df = adf._dataframe

                app.log("predict: loading dataframe and converting to pandas",
                         dash_logger=dash_logger)
                df = prepared_df[prepared_df.passage_plan_id == current_passage_plan_id]
                if df.count() < 51:
                    app.log(f"Data from passage_plan_id ({current_passage_plan_id}) is too short/has insufficient length",
                             level=WARNING,
                             dash_logger=dash_logger)
                    # RUN PREDICTION AND PRESENT THE RESULT -- ASYNC
                    return json.dumps(None)

                dash.callback_context.record_timing('predict:prepare', timer() - start_time, 'pipeline: transform data')

                app.log("predict: preparing prediction job",
                         dash_logger=dash_logger)

                # Temporarily store this sequence to disk - so that the worker can pick it up
                tmpdir = tempfile.mkdtemp(prefix='.damast-ais-showcase.')
                tmpfile = Path(tmpdir) / f"passage_plan_id-{current_passage_plan_id}.hdf5"
                df.export(tmpfile)
                app.log(f"df: exported to {tmpfile}",
                         dash_logger=dash_logger)

                if not tmpfile.exists():
                    raise FileNotFoundError("Failed to create temporary data file")

                # region RETRIEVE MODEL INFO
                # FIXME:
                # The features and target which are used for prediction here - are part of the trained model.
                # So we would have to get the information from the trained model to make this
                # work properly
                features = ["Latitude_x", "Longitude_x",
                            "Latitude_y", "Longitude_y"]

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
                app._job_scheduler.start(job)
                return json.dumps(job.__dict__)

            return json.dumps(None)

        @app.callback(
            [
                Output('prediction-results', 'children'),
                Output('prediction-thread-status', 'data'),
            ],
            [
                Input('update-interval', 'n_intervals'),
                State('prediction-job', 'data'),
            ],
        )
        def interval_update(n_intervals, prediction_job_data):
            """

            :param n_intervals: current trigger of intervals
            :param prediction_job_data:
            :return:
            """
            prediction_result = None
            prediction_thread_status = Job.Status.NOT_STARTED.value
            if prediction_job_data is not None:
                json_data = json.loads(prediction_job_data)
                if json_data is not None:
                    job_id = json_data["id"]
                    responses, status = app._job_scheduler.get_status(job_id)
                    if status.value != prediction_thread_status:
                        prediction_thread_status = status.value

                    timepoints = []
                    losses = []
                    for response in responses:
                        timepoints.append(response.timepoint)
                        losses.append(response.loss)

                    data = zip(timepoints, losses)
                    df = pd.DataFrame(data=data, columns=["timepoint", "loss"])
                    fig = px.scatter(df, x="timepoint", y="loss", title="Forecast loss")

                    prediction_result = dcc.Graph(figure=fig)

            return prediction_result, prediction_thread_status


    @classmethod
    def create(cls, app: AISApp) -> dcc.Tab:
        cls.register_callbacks(app=app)

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

        select_experiment = html.Div([
            experiment_button,
            # html.Div(dcc.Input(id='input-experiment-directory', type='hidden', value="<experiment folder>")),
            html.Div(id='select-models'),
            data_button,
            html.Div(id='data-preview'),
        ])

        
        return dcc.Tab(label='Predict',
                       value="tab-predict",
                       children=[
                           select_experiment,
                           html.Div(id='prediction-passage_plan_id-selection'),
                           html.Div(id='prediction-data-preview'),
                           html.Div(id='prediction-trigger'),
                           html.Div(id='prediction-results'),
                           

                           dcc.Store(id='experiment-directory', storage_type='session'),
                           dcc.Store(id='model-name', storage_type="session"),
                           dcc.Store(id='prediction-data-filename', storage_type="session"),
                           dcc.Store(id='prediction-passage_plan_id', storage_type='session'),
                           # A store to trigger the prediction background job - with storage type
                           # memory, the data will be cleared with a refresh
                           dcc.Store(id='prediction-job', storage_type="session"),
                           dcc.Store(id='prediction-thread-status', storage_type="session"),
                       ])


        