import uuid
import dash
import dash_daq as daq
from dash import dash_table, State
from dash import dcc, Output, Input
from dash import html

import dash_uploader as du
from dash_extensions.enrich import DashLogger
from enum import Enum

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

        # FIXME: packing version 23.0 has no LegacyVersion
        # which dash_uploader is still using
        # It will set prevent_initial_call
        import packaging
        setattr(packaging.version, "LegacyVersion", float)

        @du.callback(
            Output({'component_id': 'explore-datasets-dropdown'}, 'options'),
            id='explore-dataset-upload'
        )
        def upload(status: du.UploadStatus):
            return { str(x): x.name for  x in (Path(app.data_upload_path) / "datasets").glob("*") if x.is_file()}

        @app.callback(
            Output('explore-dataset', 'children'),
            Input({'component_id': 'explore-data-visualization-dropdown'}, 'value'),
            Input({'component_id': 'explore-data-columns-dropdown'}, 'value'),
            Input('explore-sequence_id', 'data'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'),
            log=True
        )
        def update_explore_dataset(dropdown_data_visualization,
                                   state_data_columns, state_sequence_id,
                                   state_data_filename, state_sequence_id_column,
                                   dash_logger: DashLogger):

            if not state_data_filename or not dropdown_data_visualization:
                return []

            adf = AnnotatedDataFrame.from_file(filename=state_data_filename)

            # If a passage plan id has been selected, then limit the visualisation to this passage plan id
            if state_sequence_id and state_sequence_id != 'null':
                try:
                    current_sequence_id = int(state_sequence_id)
                except Exception:
                    # Looks like this is a string based id
                    current_sequence_id = state_sequence_id

                adf._dataframe = adf.dataframe[adf.dataframe[state_sequence_id_column] == current_sequence_id]

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
            Input({'component_id': 'explore-datasets-dropdown'}, 'value'),
            Input({'component_id': 'explore-data-columns-dropdown'}, 'value'),
        )
        def update_exploration_data(state_data_filename, state_data_columns):

            if not state_data_filename:
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
            Output({'component_id': 'select-explore-sequence_id-dropdown'}, 'options'),
            Input({'component_id': 'filter-explore-sequence_id-min-max-length'}, 'value'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'),
            prevent_initial_callbacks=True
        )
        def filter_by_sequence_id(min_max_length, data_filename, sequence_id_column):
            if not data_filename or not sequence_id_column:
                return []

            min_length, max_length = min_max_length
            adf = AnnotatedDataFrame.from_file(data_filename)
            messages_per_sequence_id = adf.groupby(sequence_id_column, agg="count")
            filtered_sequence_id = messages_per_sequence_id[messages_per_sequence_id["count"] > min_length]
            filtered_sequence_id = filtered_sequence_id[filtered_sequence_id["count"] < max_length]
            selectable_sequence_ids = sorted(filtered_sequence_id[sequence_id_column].unique())
            return selectable_sequence_ids

        @app.callback(
            [
                Output({'component_id': 'explore-sequence_id-stats'}, 'children'),
                Output('explore-sequence_id', 'data'),
                Output('explore-dataset-preview', 'children')
            ],
            Input({'component_id': 'select-explore-sequence_id-dropdown'}, 'value'),
            Input({'component_id': 'select-feature-highlight-dropdown'}, 'value'),
            Input({'component_id': 'explore-radius-factor'}, 'value'),
            Input({'component_id': 'explore-use-absolute-value'}, 'value'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State('explore-sequence_id', 'data'),
            State({'component_id': 'explore-sequence_id-plot-map'}, 'figure'),
            State('explore-dataset-preview', 'children'),
            State({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'),
            prevent_initial_callback=True
        )
        def select_sequence_id(sequence_id,
                               features,
                               radius_factor,
                               use_absolute_value,
                               data_filename,
                               prev_sequence_id,
                               plot_map_cfg,
                               current_data_preview,
                               sequence_id_column,
                               ):
            if sequence_id is not None:
                try:
                    current_sequence_id = int(sequence_id)
                except ValueError:
                    # Looks like the sequence id is a str
                    current_sequence_id = sequence_id

                adf = AnnotatedDataFrame.from_file(data_filename)
                sequence_id_df = adf[adf.dataframe[sequence_id_column] == current_sequence_id]

                mean_lat = sequence_id_df.mean("Latitude")
                mean_lon = sequence_id_df.mean("Longitude")
                var_lat = sequence_id_df.var("Latitude")
                var_lon = sequence_id_df.var("Longitude")
                length = sequence_id_df.count()

                data = {"Length": length,
                        "Lat": f"{mean_lat:.2f} +/- {var_lat:.3f}",
                        "Lon": f"{mean_lon:.2f} +/- {var_lon:.3f}"
                        }

                sequence_id_stats_table = dash_table.DataTable(
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
                if prev_sequence_id and prev_sequence_id != 'null':
                    try:
                        prev_sequence_id = int(prev_sequence_id)
                    except ValueError:
                        pass

                    if current_sequence_id == prev_sequence_id and plot_map_cfg:
                        zoom_factor = plot_map_cfg["layout"]["mapbox"]["zoom"]
                        center = plot_map_cfg["layout"]["mapbox"]["center"]

                trajectory_plot = dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'},
                                                 figure=create_figure_trajectory(sequence_id_df, densities_by=features,
                                                                                 zoom_factor=zoom_factor,
                                                                                 radius_factor=float(radius_factor),
                                                                                 center=center,
                                                                                 use_absolute_value=use_absolute_value),
                                                 style={
                                                     "width": "100%",
                                                     "height": "70%"
                                                 })

                return [sequence_id_stats_table, trajectory_plot], json.dumps(sequence_id), create_figure_data_preview_table(data_df=adf.dataframe,
                                                                                                                             sequence_id_column=sequence_id_column,
                                                                                                                             sequence_id=sequence_id)
            return [html.Div(children=[dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'})],
                             hidden=True)], json.dumps(None), current_data_preview

        @app.callback(
            Output('explore-sequence_id-column', 'children'),
            Input({'component_id': 'explore-datasets-dropdown'}, 'value'),
            prevent_initial_callbacks=True,
            log=True
        )
        def update_data(explore_data_filename,
                        dash_logger: DashLogger):
            """
            Set the current data that shall be used for prediction
            :return:
            """
            filename = explore_data_filename
            options = {}
            if filename and filename != '' and isinstance(filename, str):
                adf = AnnotatedDataFrame.from_file(filename=filename)
                for c in adf.column_names:
                    label = c
                    try:
                        metadata = adf.metadata[c]
                        if metadata.description != '':
                            label += f" - {metadata.description}"
                        options[c] = label
                    except KeyError:
                        pass

            select_sequence_column_dropdown = dcc.Dropdown(
                placeholder="Select column of sequence id",
                id={'component_id': "explore-sequence_id-column-dropdown"},
                multi=False,
                options=options
            )
            return select_sequence_column_dropdown


        @app.callback(
            [
                Output('explore-dataset-preview', 'children'),
                Output('explore-sequence_id-selection', 'children'),
                Output('explore-sequence_id', 'clear_data'),
            ],
            [
                State({'component_id': 'explore-datasets-dropdown'}, 'value'),
                Input({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'),
                State('explore-sequence_id-selection', 'children'),
                State('explore-dataset-preview', 'children'),
            ],
            prevent_initial_callbacks=True,
            log=True
        )
        def update_data(explore_data_filename, explore_sequence_id_column,
                        state_prediction_sequence_id_selection, state_data_preview,
                        dash_logger: DashLogger):
            """
            Set the current data that shall be used for prediction

            :param n_clicks:
            :param state_data_button:
            :param state_data_preview:
            :return:
            """
            filename = explore_data_filename
            if not filename or filename == '' or not isinstance(filename, str):
                return state_data_preview, state_prediction_sequence_id_selection, False

            adf = AnnotatedDataFrame.from_file(filename=filename)
            data_preview_table = create_figure_data_preview_table(adf.dataframe)

            select_sequence_id_dropdown = dcc.Dropdown(
                placeholder="Select sequence for prediction",
                id={'component_id': "select-explore-sequence_id-dropdown"},
                multi=False,
            )

            df = adf[:5].to_pandas_df()
            feature_highlight_options = [{'label': c, 'value': c} for c in df.columns if is_numeric_dtype(df.dtypes[c])]
            feature_highlight_options.append({'label': 'no highlighting', 'value': ''})
            # Allow to select multiple features to highlight
            select_feature_highlight_dropdown = dcc.Dropdown(
                placeholder="Select a feature to highlight in the plot",
                id={'component_id': "select-feature-highlight-dropdown"},
                options=feature_highlight_options,
                multi=True,
            )
            select_feature_highlight_options = html.Div(id="div-feature-highlight-options",
                                                       children=[
                                                           html.H4("Highlight radius scaling factor:"),
                                                           daq.NumericInput(id={'component_id': 'explore-radius-factor'},
                                                               min=1.0,
                                                               max=25.0,
                                                               value=10.0),
                                                           html.H4("Use absolute values:"),
                                                           daq.BooleanSwitch(id={'component_id': 'explore-use-absolute-value'}, on=True),
                                                           html.Br()
                                                       ])

            min_messages = 0

            grouped = adf.groupby(explore_sequence_id_column, agg={'sequence_length': 'count'})
            max_messages = max(grouped.sequence_length.values)
            # markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
            filter_sequence_id_slider = dash.dcc.RangeSlider(id={'component_id': 'filter-explore-sequence_id-min-max-length'},
                                                 min=min_messages, max=max_messages,
                                                 value=[0,max_messages],
                                                 allowCross=False,
                                                 tooltip={'placement': 'bottom', 'always_visible': True}
                                                 # marks={i: f"{int(10 ** i)}" for i in
                                                 #       markers},
                                                 )

            select_for_prediction = [
                html.H2("Select sequence:"),
                html.Div(id="explore-sequence_id-filter", children=[
                    html.H3("Minimum-Maximum sequence length"),
                    filter_sequence_id_slider
                ]),
                select_sequence_id_dropdown,
                html.Br(),
                select_feature_highlight_dropdown,
                select_feature_highlight_options,
                html.Div(id={"component_id": "explore-sequence_id-stats"},
                         children=[
                             html.Div(
                                 children=[
                                     dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'})
                                     ],
                                 hidden=True
                             )
                            ]
                )
            ]
            return data_preview_table, select_for_prediction, True


    @classmethod
    def create(cls, app: AISApp) -> dcc.Tab:
        cls.register_callbacks(app)

        datasets_dropdown = html.Div(
            children=[html.H3("Available datasets"),
            dcc.Dropdown(id={'component_id': "explore-datasets-dropdown"},
                         options={ str(x): x.name for  x in (Path(app.data_upload_path) / "datasets").glob("*") if x.is_file()},
                         placeholder="Select a dataset",
                         multi=False)],
            style={
                "display": "inline-block",
                "width": "100%",
            },
        )

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
                           html.Div(id='explore-upload',
                                    children=[du.Upload(
                                        text="Drag and drop to upload dataset",
                                        text_completed='',
                                        id='explore-dataset-upload',
                                        max_file_size=10000, # 10 000 Mb,
                                        cancel_button=True,
                                        pause_button=True,
                                        filetypes=['h5', 'hdf5'],
                                        upload_id='datasets',
                                        max_files=1,
                                    ),
                                    ],
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
                                    }),
                           datasets_dropdown,
                           html.Div(id='explore-sequence_id-column',
                                    children=[
                                        dcc.Dropdown(id={"component_id": "explore-sequence_id-column-dropdown"})
                                    ]
                           ),
                           html.Div(id='explore-sequence_id-selection'),
                           html.Div(id='explore-dataset-preview'),
                           dataset_stats,
                           data_column_dropdown,
                           visualization_type_dropdown,
                           html.Div(id='explore-dataset'),

                           dcc.Store(id='explore-data-filename', storage_type="session"),
                           dcc.Store(id='explore-sequence_id', storage_type='session'),
                       ],
                       className="tab")