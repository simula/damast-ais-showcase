import dash
import dash_daq as daq
from dash import dash_table, State
from dash import dcc, Output, Input
from dash import html

from dash_extensions.enrich import DashLogger
from enum import Enum
import tkinter
from tkinter import filedialog
import json
from pathlib import Path

from pandas.api.types import is_numeric_dtype
from logging import WARNING

from damast.core.dataframe import AnnotatedDataFrame
from ..figures import (
    create_div_histogram,
    create_div_metadata,
    create_div_statistic,
    create_figure_feature_correlation_heatmap,
    create_figure_trajectory,
    create_figure_data_preview_table
)

from ..web_application import AISApp

class VisualizationType(str, Enum):
    Histogram = "Histogram"
    Statistics = "Statistics"
    Metadata = "Metadata"

class ExploreTab:
    @classmethod
    def register_callbacks(cls, app: AISApp):

        @app.callback(
            Output('explore-dataset', 'children'),
            Input({'component_id': 'explore-data-visualization-dropdown'}, 'value'),
            Input({'component_id': 'explore-data-columns-dropdown'}, 'value'),
            Input('explore-passage_plan_id', 'data'),
            State('explore-data-filename', 'data'),
            log=True
        )
        def update_explore_dataset(dropdown_data_visualization,
                                   state_data_columns, state_passage_plan_id,
                                   state_data_filename,
                                   dash_logger: DashLogger):

            if state_data_filename is None or dropdown_data_visualization is None:
                return []

            adf = AnnotatedDataFrame.from_file(filename=state_data_filename)

            # If a passage plan id has been selected, then limit the visualisation to this passage plan id
            if state_passage_plan_id and state_passage_plan_id != 'null':
                current_passage_plan_id = int(state_passage_plan_id)
                adf._dataframe = adf.dataframe[adf.passage_plan_id == current_passage_plan_id]

            explore_dataset_children = []
            for column_name in state_data_columns:
                children = []
                if VisualizationType.Histogram.value in dropdown_data_visualization:
                    children.extend(create_div_histogram(adf=adf, column_name=column_name))

                if VisualizationType.Statistics.value in dropdown_data_visualization:
                    children.extend(create_div_statistic(adf=adf, column_name=column_name))

                if VisualizationType.Metadata.value in dropdown_data_visualization:
                    children.extend(create_div_metadata(adf=adf, column_name=column_name))

                explore_dataset_children.append(html.H3(f"Explore '{column_name}'"))
                explore_dataset_children.append(
                    html.Div(
                        id={'component_id': f'explore-column-{column_name}'},
                        children=children,
                        style={
                            'backgroundColor': 'lightblue',
                            'borderRadius': '10px',
                            'borderStyle': 'solid',
                            'borderWidth': '3px',
                            # 'margin': '10px'
                        }
                    )
                )

            return explore_dataset_children

        @app.callback(
            Output({'component_id': 'explore-data-columns-dropdown'}, 'options'),
            Output({'component_id': 'feature-correlation-map'}, 'children'),
            Input('explore-data-filename', 'data'),
            Input({'component_id': 'explore-data-columns-dropdown'}, 'value'),
        )
        def update_exploration_data(state_data_filename, state_data_columns):
            if state_data_filename is None:
                return [], []


            adf = AnnotatedDataFrame.from_file(filename=state_data_filename)
            analyse_df = adf
            if state_data_columns and state_data_columns != 'null':
                analyse_df = adf[state_data_columns]

            feature_correlation_heatmap = dash.dcc.Graph(id='feature-correlation-heatmap',
                           figure=create_figure_feature_correlation_heatmap(data_df=analyse_df),
                           style={
                               "width": "80%",
                               "height": "75%",
                           }
            )

            return adf.column_names, [html.Div(id="div-feature-correlation-map",
                                               children=[feature_correlation_heatmap],
                                               style={
                                                 "textAlign": 'center'
                                               }
            )]

        @app.callback(
            Output({'component_id': 'select-explore-passage_plan_id-dropdown'}, 'options'),
            Input({'component_id': 'filter-explore-passage_plan_id-min-max-length'}, 'value'),
            State('explore-data-filename', 'data'),
            prevent_initial_callbacks=True
        )
        def filter_passage_plan_id(min_max_length, data_filename):
            min_length, max_length = min_max_length
            adf = AnnotatedDataFrame.from_file(data_filename)
            messages_per_passage_plan_id = adf.groupby("passage_plan_id", agg="count")
            filtered_passage_plan_id = messages_per_passage_plan_id[messages_per_passage_plan_id["count"] > min_length]
            filtered_passage_plan_id = filtered_passage_plan_id[filtered_passage_plan_id["count"] < max_length]
            selectable_passage_plan_ids = sorted(filtered_passage_plan_id.passage_plan_id.unique())
            return selectable_passage_plan_ids

        @app.callback(
            [
                Output({'component_id': 'explore-passage_plan_id-stats'}, 'children'),
                Output('explore-passage_plan_id', 'data'),
                Output('explore-dataset-preview', 'children')
            ],
            Input({'component_id': 'select-explore-passage_plan_id-dropdown'}, 'value'),
            Input({'component_id': 'select-feature-highlight-dropdown'}, 'value'),
            Input({'component_id': 'explore-radius-factor'}, 'value'),
            State('explore-data-filename', 'data'),
            State('explore-passage_plan_id', 'data'),
            State({'component_id': 'explore-passage_plan_id-plot-map'}, 'figure'),
            State('explore-dataset-preview', 'children'),
            prevent_initial_callback=True
        )
        def select_passage_plan_id(passage_plan_id,
                                   feature,
                                   radius_factor,
                                   data_filename,
                                   prev_passage_plan_id,
                                   plot_map_cfg,
                                   current_data_preview,
                                   ):
            if passage_plan_id is not None:
                current_passage_plan_id = int(passage_plan_id)
                adf = AnnotatedDataFrame.from_file(data_filename)
                passage_plan_id_df = adf[adf.passage_plan_id == current_passage_plan_id]

                mean_lat = passage_plan_id_df.mean("Latitude")
                mean_lon = passage_plan_id_df.mean("Longitude")
                var_lat = passage_plan_id_df.var("Latitude")
                var_lon = passage_plan_id_df.var("Longitude")
                length = passage_plan_id_df.count()

                data = {"Length": length,
                        "Lat": f"{mean_lat:.2f} +/- {var_lat:.3f}",
                        "Lon": f"{mean_lon:.2f} +/- {var_lon:.3f}"
                        }

                passage_plan_id_stats_table = dash_table.DataTable(
                    data=[data],
                    columns=[{'id': c, 'name': c} for c in data.keys()],
                    # https://dash.plotly.com/datatable/style
                    style_cell={'textAlign': 'center', 'padding': "5px"},
                    style_header={'backgroundColor': 'lightgray',
                                  'color': 'white',
                                  'fontWeight': 'bold'}
                )

                zoom_factor = 4
                center = None
                if prev_passage_plan_id and prev_passage_plan_id != 'null':
                    if current_passage_plan_id == int(prev_passage_plan_id) and plot_map_cfg:
                        zoom_factor = plot_map_cfg["layout"]["mapbox"]["zoom"]
                        center = plot_map_cfg["layout"]["mapbox"]["center"]

                trajectory_plot = dash.dcc.Graph(id={'component_id': 'explore-passage_plan_id-plot-map'},
                                                 figure=create_figure_trajectory(passage_plan_id_df, density_by=feature,
                                                                                 zoom_factor=zoom_factor,
                                                                                 radius_factor=float(radius_factor),
                                                                                 center=center),
                                                 style={
                                                     "width": "100%",
                                                     "height": "70%"
                                                 })

                return [passage_plan_id_stats_table, trajectory_plot], json.dumps(passage_plan_id), create_figure_data_preview_table(data_df=adf.dataframe, passage_plan_id=passage_plan_id)
            return [html.Div(children=[dash.dcc.Graph(id={'component_id': 'explore-passage_plan_id-plot-map'})],
                             hidden=True)], json.dumps(None), current_data_preview

        @app.callback(
            [
                Output('button-select-data-directory', 'children'),
                Output('explore-dataset-preview', 'children'),
                Output('explore-passage_plan_id-selection', 'children'),
                Output('explore-data-filename', 'data'),
                Output('explore-passage_plan_id', 'clear_data'),
            ],
            [
                Input('button-select-data-directory', 'n_clicks'),
                State('button-select-data-directory', 'children'),
                State('explore-passage_plan_id-selection', 'children'),
                State('explore-dataset-preview', 'children'),
                State('explore-data-filename', 'data'),
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
                id={'component_id': "select-explore-passage_plan_id-dropdown"},
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
            select_feature_highlight_radius = html.Div(id="div-radius-scaling-factor",
                                                       children=[
                                                           html.H4("Highlight radius scaling factor:"),
                                                           daq.NumericInput(id={'component_id': 'explore-radius-factor'},
                                                               min=1.0,
                                                               max=25.0,
                                                               value=10.0),
                                                        html.Br()
                                                       ])

            min_messages = 0
            grouped = adf.groupby('passage_plan_id', agg={'sequence_length': 'count'})
            max_messages = max(grouped.sequence_length.values)
            # markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
            filter_passage_plan_id_slider = dash.dcc.RangeSlider(id={'component_id': 'filter-explore-passage_plan_id-min-max-length'},
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
                html.Div(id="explore-passage_plan_id-filter", children=[
                    html.H3("Minimum-Maximum sequence length"),
                    filter_passage_plan_id_slider
                ]),
                select_passage_plan_id_dropdown,
                html.Br(),
                select_feature_highlight_dropdown,
                select_feature_highlight_radius,
                html.Div(id={"component_id": "explore-passage_plan_id-stats"},
                         children=[
                             html.Div(
                                 children=[
                                     dash.dcc.Graph(id={'component_id': 'explore-passage_plan_id-plot-map'})
                                     ],
                                 hidden=True
                             )
                            ]
                )
            ]
            return Path(filename).name, data_preview_table, select_for_prediction, \
                filename, True




    @classmethod
    def create(cls, app: AISApp) -> dcc.Tab:
        cls.register_callbacks(app)

        data_column_dropdown = html.Div(
            children=[html.H3("Column"),
            dcc.Dropdown(id={'component_id': "explore-data-columns-dropdown"},
                                placeholder="Select a data column",
                                multi=True)],
            style={
                "display": "inline-block",
                "width": "100%",
            },
        )

        visualization_type_dropdown = html.Div(
            children=[
                html.H3("Visualization Type"),
                dcc.Dropdown(
                    id={'component_id': "explore-data-visualization-dropdown"},
                    placeholder="Select a visualization type column",
                    multi=True,
                    options=list(VisualizationType)
                    )
                    ],
            style={
                "display": "inline-block",
                "width": "100%",
            },
        )

        # Allow to visualize the feature correlation table
        dataset_stats = html.Div(children=[
            html.H3("Feature Correlations (numeric columns only)"),
            html.Div(id={'component_id': 'feature-correlation-map'}),
            html.Br(),
        ])

        return dcc.Tab(label="Explore",
                       value="tab-explore",
                       children=[
                           html.Br(),
                           html.Div(id='explore-passage_plan_id-selection'),
                           html.Div(id='explore-dataset-preview'),
                           dataset_stats,
                           data_column_dropdown,
                           visualization_type_dropdown,
                           html.Div(id='explore-dataset'),

                           dcc.Store(id='explore-data-filename', storage_type="session"),
                           dcc.Store(id='explore-passage_plan_id', storage_type='session'),
                       ],
                       className="tab")