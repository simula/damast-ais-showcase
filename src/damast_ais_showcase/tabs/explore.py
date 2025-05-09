import uuid
import logging
import dash
import dash_daq as daq
from dash import dash_table, State, dcc, Output, Input, html, ALL, MATCH, Patch
import hashlib

import dash_uploader as du
from dash_extensions.enrich import DashLogger
from enum import Enum

import os
import json
from pathlib import Path
import numpy as np
import polars as pl
from typing import List, Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_ns_dtype
from logging import WARNING

from threading import Lock

from damast.core.dataframe import AnnotatedDataFrame, DAMAST_SUPPORTED_FILE_FORMATS
from ..figures import (
    create_div_histogram,
    create_div_metadata,
    create_div_statistic,
    create_figure_feature_correlation_heatmap,
    create_figure_trajectory,
    create_figure_data_preview_table
)

from ..web_application import AISApp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_ANNOTATED_DATAFRAMES_CACHE_MUTEX = Lock()

def get_annotated_dataframe(filename: Path | str | list[str]):
    if type(filename) != list:
        filename = [filename]

    if not filename:
        raise ValueError("Filename list cannot be empty or None")

    key = hashlib.md5(''.join(filename).encode('UTF-8')).hexdigest()
    cache_dir = Path("/tmp/damast-ais-showcase/datasets-cache/")
    cache_file = cache_dir / f"{key}.parquet"

    with _ANNOTATED_DATAFRAMES_CACHE_MUTEX:
        if not cache_file.exists():
            logger.info(f"Trying to load: {filename}")
            adf = AnnotatedDataFrame.from_files(files=filename, metadata_required=False)
            logger.info(f"Successfully loaded: {filename}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching: {filename} as {cache_file}")
            adf.save(filename=cache_file)
        else:
            logger.info(f"Trying to load from cache: {cache_file}")
            adf = AnnotatedDataFrame.from_files(files=[cache_file], metadata_required=False)
        return adf

def has_files(filename: str | list[str]):
 return filename and filename != [] and isinstance(filename, list)

class VisualizationType(str, Enum):
    Histogram = "Histogram"
    Statistics = "Statistics"
    Metadata = "Metadata"

ALL_SEQUENCES_OPTION = "all"
SUPPORTED_FILE_FORMATS: list[str] = [x for suffixes in DAMAST_SUPPORTED_FILE_FORMATS.values() for x in suffixes]

def pandas_slice(adf: AnnotatedDataFrame) -> pd.DataFrame:
   return adf.slice(offset=0, length=5).collect().to_pandas()


def get_lat_lon_col(adf: AnnotatedDataFrame) -> dict[str, str]:
    latitude_candidates = []
    longitude_candidates = []
    for col in adf.column_names:
        if col.lower().startswith("lon"):
            longitude_candidates.append(col)

        if col.lower().startswith("lat"):
            latitude_candidates.append(col)

    mapping = {}
    mapping['Latitude']  = latitude_candidates[0]
    mapping['Longitude']  = longitude_candidates[0]

    if len(latitude_candidates) != 1:
        logger.warn(f"Multiple candidates for latitude column: {latitude_candidates=}, "
            f"using {latitude_candidates[0]}")

    if len(longitude_candidates) != 1:
        logger.warn(f"Multiple candidates for longitude column: {longitude_candidates=}, "
            f"using {longitude_candidates[0]}")

    return mapping

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
            id='explore-dataset-upload',
        )
        def upload(status: du.UploadStatus):
            datasets = { str(x): x.name for x in (Path(app.data_upload_path) / "datasets").glob("*") if x.is_file() and x.suffix in SUPPORTED_FILE_FORMATS}
            for dataset, name in datasets.items():
                # Loading
                get_annotated_dataframe(dataset)
            return datasets


        @app.callback(
            Output('explore-dataset', 'children'),
            Output('explore-column-filter-state', 'clear_data'),
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
            """
            Update the dataset that will be explored

            Args:
                dropdown_data_visualization (_type_): _description_
                state_data_columns (_type_): _description_
                state_sequence_id (_type_): _description_
                state_data_filename (_type_): _description_
                state_sequence_id_column (_type_): _description_
                dash_logger (DashLogger): _description_

            Returns:
                _type_: _description_
            """

            if not state_data_filename:
                return [], True

            adf = get_annotated_dataframe(filename=state_data_filename)

            # If a group id has been selected, then limit the visualisation to this group id
            if state_sequence_id_column and state_sequence_id and state_sequence_id != 'null':
                logger.info(f"Sequence {state_sequence_id=} of {state_sequence_id_column=} selected")
                try:
                    current_sequence_id = int(state_sequence_id)
                except Exception:
                    # Looks like this is a string based id
                    current_sequence_id = state_sequence_id

                adf._dataframe = adf.dataframe.filter(pl.col(state_sequence_id_column) == current_sequence_id)

            if dropdown_data_visualization:
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

                return explore_dataset_children, False
            else:
                return [], False

        @app.callback(
            Output({'component_id': 'explore-data-columns-dropdown'}, 'options'),
            Output({'component_id': 'feature-correlation-map'}, 'children'),
            Input({'component_id': 'explore-datasets-dropdown'}, 'value'),
            Input({'component_id': 'explore-data-columns-dropdown'}, 'value'),
        )
        def update_exploration_data(state_data_filename, state_data_columns):
            """
            Load the dataset from a given filename and update the the correlation heatmap

            Args:
                state_data_filename (_type_): _description_
                state_data_columns (_type_): _description_

            Returns:
                _type_: _description_
            """

            if not state_data_filename:
                return [], []

            adf = get_annotated_dataframe(filename=state_data_filename)
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

        def apply_column_filter(adf: AnnotatedDataFrame, column_filter_state = {}):
            """
            Filter the annotated dataframe, based on the column filter state

            Args:
                adf (AnnotatedDataFrame): _description_
                column_filter_state (dict, optional): _description_. Defaults to {}.
            """
            # Apply extra column filtering
            if column_filter_state:
                for _, filter_status in column_filter_state.items():
                    column_name = filter_status['column_name']
                    min_value = filter_status['value'][0]
                    max_value = filter_status['value'][1]

                    # FIXME: handle issue on Mac OS
                    if min_value == "None" or max_value == "None":
                        continue

                    dtype = pandas_slice(adf).dtypes[column_name]
                    if is_datetime64_ns_dtype(dtype):
                        min_value = np.datetime64(min_value[:10])
                        max_value = np.datetime64(max_value[:10])

                    adf._dataframe = (adf.dataframe
                                        .filter(pl.col(column_name) >= min_value)
                                        .filter(pl.col(column_name) <= max_value)
                                    )

        def get_selectables(adf: AnnotatedDataFrame,
                           group_column_name: str,
                           column_filter_state={},
                           min_length=0,
                           max_length=None,
        ) -> List[Any]:
            """
            Limit the selection to group that fulfill a particular critera


            Args:
                adf (AnnotatedDataFrame): _description_
                sequence_id_column (str): _description_
                column_filter_state (dict, optional): _description_. Defaults to {}.
                min_length (int, optional): _description_. Defaults to 0.
                max_length (_type_, optional): _description_. Defaults to None.

            Returns:
                _type_: _description_
            """
            apply_column_filter(adf=adf, column_filter_state=column_filter_state)

            if group_column_name is None:
                raise ValueError("Group identifier cannot be 'None'")

            logger.info(f"Group column filter: {group_column_name=}")

            entries_per_group = adf.group_by(group_column_name).agg(pl.len().alias("count"))
            filtered_groups = entries_per_group.filter(pl.col("count") > min_length)
            if max_length:
                filtered_groups = filtered_groups.filter(pl.col("count") < max_length)
            selectable_groups = filtered_groups.select(pl.col(group_column_name)).unique().sort(by=group_column_name).collect()[:,0].to_list()

            if selectable_groups:
                selectable_groups = [ALL_SEQUENCES_OPTION] + selectable_groups
            return selectable_groups

        @app.callback(
            Output({'component_id': 'select-explore-sequence_id-dropdown'}, 'options'),
            Input({'component_id': 'filter-explore-sequence_id-min-max-length'}, 'value'),
            Input('explore-column-filter-state', 'data'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'),
            prevent_initial_callbacks=True
        )
        def filter_by_sequence_id(min_max_length, column_filter_state, data_filename, sequence_id_column):
            if not data_filename or not sequence_id_column:
                return []

            min_length, max_length = min_max_length
            adf = get_annotated_dataframe(data_filename)

            logger.info(f"Filter for {sequence_id_column=} {min_length=} {max_length=}")
            return get_selectables(adf=adf,
                                   group_column_name=sequence_id_column,
                                   column_filter_state=column_filter_state,
                                   min_length=min_length,
                                   max_length=max_length
                                   )

        @app.callback(
            [
                Output({'component_id': 'explore-sequence_id-stats'}, 'children'),
                Output('explore-sequence_id', 'data'),
                Output('explore-dataset-preview', 'children'),
                Output({'component_id': 'explore-sequence-batch-number'}, 'max')
            ],
            Input({'component_id': 'select-explore-sequence_id-dropdown'}, 'value'),
            Input({'component_id': 'select-feature-highlight-dropdown'}, 'value'),
            Input({'component_id': 'explore-radius-factor'}, 'value'),
            Input({'component_id': 'explore-use-absolute-value'}, 'value'),
            Input({'component_id': 'filter-explore-sequence_id-min-max-length'}, 'value'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State('explore-sequence_id', 'data'),
            State({'component_id': 'explore-sequence_id-plot-map'}, 'figure'),
            State('explore-dataset-preview', 'children'),
            State({'component_id': 'explore-sequence_id-column-dropdown'}, 'value'), # select one or many sequences by id
            State({'component_id': 'select-explore-sequence_id-dropdown'}, 'options'), # available (and narrowed selection)
            State('explore-column-filter-state', 'data'), # Status of of the column filters are stored here
            Input({'component_id': 'explore-sequence-max-count'},'value'),
            Input({'component_id': 'explore-sequence-batch-number'},'value'), # Select the batch number when a lot of sequence should be displayed
            Input({'component_id': 'explore-plot-width'}, 'value'),
            Input({'component_id': 'explore-plot-height'}, 'value'),
        )
        def select_sequence_id(sequence_ids,
                               features,
                               radius_factor,
                               use_absolute_value,
                               filter_explore_sequence_id_min_max_length,
                               data_filename,
                               prev_sequence_ids,
                               plot_map_cfg,
                               current_data_preview,
                               sequence_id_column,
                               sequence_ids_options,
                               column_filter_state,
                               max_sequence_count,
                               batch_number,
                               plot_width,
                               plot_height,
                               ):

            if not sequence_ids:
                return [html.Div(children=[dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'})],
                                hidden=True)], json.dumps(None), current_data_preview, 0

            # per default use
            adf = get_annotated_dataframe(filename=data_filename)
            groups_df = None

            current_sequence_ids = []
            preview_sequence_id = None
            try:
                if ALL_SEQUENCES_OPTION in sequence_ids:
                    sequence_ids_options.remove(ALL_SEQUENCES_OPTION)
                    sequence_ids = [int(x) for x in sequence_ids_options]

                current_sequence_ids = [int(x) for x in sequence_ids]
                preview_sequence_id = current_sequence_ids[0]
            except ValueError:
                # Looks like the sequence id is a str
                current_sequence_ids = sequence_ids

            groups_df = adf.filter(pl.col(sequence_id_column).is_in(current_sequence_ids))

            col_mapping = get_lat_lon_col(adf)
            lat_col = pl.col(col_mapping["Latitude"])
            lon_col = pl.col(col_mapping["Longitude"])

            mean_lat = groups_df.select(lat_col).mean().collect()[0,0]
            mean_lon = groups_df.select(lon_col).mean().collect()[0,0]
            var_lat = groups_df.select(lat_col).var().collect()[0,0]
            var_lon = groups_df.select(lon_col).var().collect()[0,0]
            length = groups_df.count().collect()[0,0]

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

            # Ensure that the map remains centered where it was before (when only an addition a column)
            # is being highlighted
            zoom_factor = None
            center = None
            if prev_sequence_ids and prev_sequence_ids != 'null':
                prev_sequence_ids = json.loads(prev_sequence_ids)
                try:
                    prev_sequence_ids = [int(x) for x in prev_sequence_ids]
                except ValueError:
                    # Ids seem to be strings after all
                    pass

                if current_sequence_ids == prev_sequence_ids and plot_map_cfg:
                    zoom_factor = plot_map_cfg["layout"]["map"]["zoom"]
                    center = plot_map_cfg["layout"]["map"]["center"]

            #  Allow stepping through all using a paging-like mechanism
            all_ids = groups_df.select(sequence_id_column).unique().sort(by=sequence_id_column).collect()

            partition_count = int(len(all_ids)/max_sequence_count)
            if partition_count >= 2:
                bounded_ids = np.array_split(all_ids, partition_count)[batch_number][:,0]
                groups_df = groups_df.filter(pl.col(sequence_id_column).is_in(bounded_ids))

            trajectory_plot = dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'},
                                                figure=create_figure_trajectory(groups_df,
                                                                                densities_by=features,
                                                                                zoom_factor=zoom_factor,
                                                                                radius_factor=float(radius_factor/10.0),
                                                                                center=center,
                                                                                use_absolute_value=use_absolute_value,
                                                                                sequence_id_column=sequence_id_column,
                                                                                lat_lon_cols=col_mapping.values(),
                                                                                width=plot_width,
                                                                                height=plot_height,
                                                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "100%"
                                                })

            return [sequence_id_stats_table, trajectory_plot], json.dumps(sequence_ids), create_figure_data_preview_table(data_df=adf.dataframe,
                                                                                                                            sequence_id_column=sequence_id_column,
                                                                                                                            sequence_id=preview_sequence_id), max(0, partition_count-1)

        @app.callback(
                Input({'component_id': 'explore-sequence-batch-number'}, 'max'),
                Output({'component_id': 'explore-sequence-max-batch-number'}, 'children')
        )
        def display_max_batch_number(max_batch_number):
            return html.H4(f" / {max_batch_number}")

        @app.callback(
            Output('explore-sequence_id-column', 'children'),
            Input({'component_id': 'explore-datasets-dropdown'}, 'value'),
            prevent_initial_callbacks=True,
            log=True
        )
        def update_data(explore_data_filename,
                        dash_logger: DashLogger):
            """
            Set the current data *.hdf5 file that shall be used for analysis.

            The file has to be generated / prepared with 'damast' so that the
            data specification is contained in the hdf5 file.

            :return:
            """
            filename = explore_data_filename
            options = {}
            if has_files(filename):
                adf = get_annotated_dataframe(filename=filename)
                for c in adf.column_names:
                    label = c
                    try:
                        metadata = adf.metadata[c]
                        if metadata.description != '':
                            label += f" - {metadata.description}"
                        options[c] = label
                    except KeyError as e:
                        logger.info(f"update_data: no key {c} found in metadata")
                        pass

            select_sequence_column_dropdown = dcc.Dropdown(
                placeholder="Select column of sequence id",
                id={'component_id': "explore-sequence_id-column-dropdown"},
                multi=False,
                options=options
            )
            return select_sequence_column_dropdown

        @app.callback(
                Output({'component_id': 'explore-extra-filters'}, 'children'),
                Input({'component_id': 'explore-dynamic-add-filter'}, 'n_clicks'),
                State({'component_id': 'explore-datasets-dropdown'}, 'value'),
                State({'component_id': 'explore-extra-filters'}, 'children'),
                prevent_initial_callbacks=True
            )
        def add_filters(n_clicks, data_filename, extra_filters):
            if n_clicks <= 0:
                return []

            filename = data_filename
            if not has_files(filename):
                return []

            adf = get_annotated_dataframe(filename=filename)
            df = pandas_slice(adf)

            filter_columns_options = [{'label': c, 'value': c} for c in df.columns if is_numeric_dtype(df.dtypes[c]) or is_datetime64_ns_dtype(df.dtypes[c])]

            if extra_filters is not None:
                updated_extra_filters = extra_filters
            else:
                updated_extra_filters = []
            new_filter = html.Div(
                id={'component_id': 'filter-column-div', 'filter_id': n_clicks},
                children=[
                    html.Div(
                        html.Button("-",
                                    id={'component_id': 'explore-dynamic-remove-filter', 'filter_id': n_clicks},
                                    n_clicks=0,
                                    style={
                                        'borderRadius': '5px',
                                        'backgroundColor': 'white',
                                        'fontSize': '0.8em'
                                    }
                                ),
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id={'component_id': 'explore-column-dropdown', 'filter_id': n_clicks},
                            options=filter_columns_options,
                            placeholder="Select column",
                            multi=False,
                        ),
                        style={'display': 'inline-block', 'margin': '1em', 'verticalAlign': 'middle', 'width': '15%'}
                    ),
                    html.Div(id={'component_id' : 'explore-column-value-filter', 'filter_id': n_clicks},
                             style={'display': 'inline-block', 'verticalAlign': 'middle', 'width': '80%'}),
                    html.Br()
                ],
            )
            updated_extra_filters.append(new_filter)
            return updated_extra_filters

        @app.callback(
            Output({'component_id': 'explore-column-value-filter', 'filter_id': MATCH}, 'children'),
            Input({'component_id': 'explore-column-dropdown', 'filter_id': MATCH }, 'value'),
            State({'component_id': 'explore-column-dropdown', 'filter_id': MATCH }, 'id'),
            State({'component_id': 'explore-datasets-dropdown'}, 'value'),
            State({'component_id': 'explore-column-value-filter', 'filter_id': MATCH}, 'children'),
            State('explore-column-filter-state', 'data'),
            prevent_initial_callbacks=True
        )
        def create_filter(value, component_id, explore_data_filename, existing_value_filter, value_filter):
            # Callback might fire although filter should not change
            # and it should not be recreated
            if value_filter:
                # The filter dropdown value represents the column name - if that changed
                # the range selection component needs to be updated
                cid = str(component_id['filter_id'])
                if cid in value_filter and value_filter[cid]['column_name'] == value:
                    return existing_value_filter

            filename = explore_data_filename
            if not value or not has_files(filename):
                return []

            adf = get_annotated_dataframe(filename=filename)

            filter_id = component_id['filter_id']
            dtype = pandas_slice(adf).dtypes[value]
            if is_numeric_dtype(dtype):
                # Ensure that a value range exists
                min_value = min(adf[value].min().collect()[0,0], 0)
                max_value = max(adf[value].max().collect()[0,0], 1)
                filter_slider = dash.dcc.RangeSlider(
                    id={'component_id': 'explore-column-filter-range', 'filter_id': filter_id },
                    min=int(min_value), max=int(max_value)+1,
                    value=[0,int(max_value)+1],
                    allowCross=False,
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
                return [filter_slider]
            elif is_datetime64_ns_dtype(dtype):
                min_value = adf[value].min().collect()[0,0]
                max_value = adf[value].max().collect()[0,0]

                date_picker = dcc.DatePickerRange(
                    id={'component_id': 'explore-column-filter-range', 'filter_id': filter_id },
                    min_date_allowed=min_value,
                    max_date_allowed=max_value,
                    initial_visible_month=min_value,
                    end_date=max_value,
                )
                return [date_picker]
            return html.Div(f"Column '{value}' is not numeric")

        @app.callback(
            Output('explore-column-filter-state', 'data'),
            Input({'component_id': 'explore-column-filter-range', 'filter_id': ALL}, 'value'),
            Input({'component_id': 'explore-column-dropdown', 'filter_id': ALL}, 'value'),
            State({'component_id': 'explore-column-filter-range', 'filter_id': ALL}, 'id'),
            State('explore-column-filter-state', 'data'),
            prevent_initial_callbacks=True,
        )
        def update_filter_slider_state(range_values, column_values, filter_ids, filter_state):
            """Method to update regular slider components"""
            state = filter_state if filter_state else {}
            if filter_ids:
                for i, cid in enumerate(filter_ids):
                    value = range_values[i]
                    if value:
                        state[str(cid['filter_id'])] = {
                            'value': value,
                            'column_name': column_values[i]
                        }
            return state

        @app.callback(
            Output('explore-column-filter-state', 'data'),
            Input({'component_id': 'explore-column-filter-range', 'filter_id': ALL}, 'start_date'),
            Input({'component_id': 'explore-column-filter-range', 'filter_id': ALL}, 'end_date'),
            Input({'component_id': 'explore-column-dropdown', 'filter_id': ALL}, 'value'),
            State({'component_id': 'explore-column-filter-range', 'filter_id': ALL}, 'id'),
            State('explore-column-filter-state', 'data'),
            prevent_initial_callbacks=True,
        )
        def update_filter_date_state(start_dates, end_dates, column_values, filter_ids, filter_state):
            """Method to update state from time picker"""
            state = filter_state if filter_state else {}
            if filter_ids:
                for i, cid in enumerate(filter_ids):
                    state[cid['filter_id']] = {
                        'value': [str(start_dates[i]), str(end_dates[i])],
                        'column_name': column_values[i]
                    }
            return state


        @app.callback(
                    Output({'component_id': 'explore-extra-filters'}, 'children'),
                    Output('explore-column-filter-state', 'data'),
                    Input({'component_id': 'explore-dynamic-remove-filter', 'filter_id': ALL}, 'n_clicks'),
                    Input({'component_id': 'explore-dynamic-remove-filter', 'filter_id': ALL}, 'id'),
                    State({'component_id': 'explore-extra-filters'}, 'children'),
                    State('explore-column-filter-state', 'data'),
                    prevent_initial_callbacks=True
                )
        def remove_filter(n_clicks, ids, extra_filters, filter_state):
            if not any(n_clicks):
                return extra_filters, filter_state

            remove_ids = []
            for idx, x in enumerate(ids):
                if n_clicks[idx]:
                    remove_ids.append(x['filter_id'])

            updated_filter_state = { x: y for x,y in filter_state.items() if int(x) not in remove_ids}
            filters = []
            for f in extra_filters:
                filter_column_div = f['props']
                cid = filter_column_div['id']
                # check if this should be removed
                if 'filter_id' in cid and int(cid['filter_id']) not in remove_ids:
                    filters.append(f)
            return filters, updated_filter_state

        @app.callback(
            [
                Output('explore-dataset-preview', 'children'),
                Output('explore-sequence_id-selection', 'children'),
                Output('explore-sequence_id', 'clear_data'),
                Output('explore-column-filter-state', 'clear_data')
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
            logger.info(f"{explore_sequence_id_column=}")
            if not explore_sequence_id_column:
                return [], [], None, None

            filename = explore_data_filename
            if not has_files(filename) or explore_sequence_id_column is None:
                return state_data_preview, state_prediction_sequence_id_selection, False, True

            adf = get_annotated_dataframe(filename=filename)
            data_preview_table = create_figure_data_preview_table(adf.dataframe)

            select_sequence_id_dropdown = dcc.Dropdown(
                placeholder="Select sequence for exploration",
                id={'component_id': "select-explore-sequence_id-dropdown"},
                multi=True,
            )

            df = pandas_slice(adf)
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
                                                           html.H4("Highlight radius scaling factor:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.NumericInput(id={'component_id': 'explore-radius-factor'},
                                                                    min=1,
                                                                    max=100.0,
                                                                    value=25.0,
                                                                ),
                                                                style={'display': 'inline-block',
                                                                       'verticalAlign': 'middle',
                                                                       'padding': '2em',
                                                                }
                                                           ),
                                                           html.H4("Use absolute values:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.BooleanSwitch(
                                                                    id={'component_id': 'explore-use-absolute-value'},
                                                                    on=True,
                                                                ),
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'vertialAlign': 'middle',
                                                                    'padding': '4em'
                                                                }
                                                           ),
                                                           html.H4("Display trajectories:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.NumericInput(id={'component_id': 'explore-sequence-max-count'},
                                                                    min=1,
                                                                    max=100,
                                                                    value=10),
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'verticalAlign': 'middle',
                                                                    'padding': '2em',
                                                                }
                                                           ),
                                                           html.H4("Page:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.NumericInput(id={'component_id': 'explore-sequence-batch-number'},
                                                                    min=0,
                                                                    max=1000,
                                                                    value=0),
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'verticalAlign': 'middle',
                                                                    'padding': '2em',
                                                                }
                                                           ),
                                                           html.Div(id={'component_id': 'explore-sequence-max-batch-number'},
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'verticalAlign': 'left',
                                                                }
                                                           ),
                                                       ],
                                                )
            plot_options = html.Div(id="div-plot-options",
                                                       children=[
                                                           html.H4("Plot width in pixel:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.NumericInput(id={'component_id': 'explore-plot-width'},
                                                                    min=500,
                                                                    max=5000,
                                                                    value=2000),
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'verticalAlign': 'middle',
                                                                    'padding': '2em',
                                                                }
                                                           ),
                                                           html.H4("Plot height in pixel:", style={'display': 'inline-block'}),
                                                           html.Div(
                                                                daq.NumericInput(id={'component_id': 'explore-plot-height'},
                                                                    min=500,
                                                                    max=5000,
                                                                    value=2000),
                                                                style={
                                                                    'display': 'inline-block',
                                                                    'verticalAlign': 'middle',
                                                                    'padding': '2em',
                                                                }
                                                           ),
                                                       ],
                                                )

            min_messages = 0
            grouped = adf.group_by(explore_sequence_id_column).agg(pl.len().alias("sequence_length")).collect()
            # Ensure that a minimal range exists
            # Ensure min 1
            max_messages = 0
            if not grouped.is_empty():
                max_messages = grouped.select(pl.col("sequence_length")).max().item(0,0)

            max_messages = max(max_messages,1)
            # markers = np.linspace(min_messages, max_messages, 10, endpoint=True)
            filter_sequence_id_slider = dash.dcc.RangeSlider(id={'component_id': 'filter-explore-sequence_id-min-max-length'},
                                                 min=min_messages, max=max_messages,
                                                 value=[0,max_messages],
                                                 allowCross=False,
                                                 tooltip={'placement': 'bottom', 'always_visible': True}
                                                 # marks={i: f"{int(10 ** i)}" for i in
                                                 #       markers},
                                                 )

            select_for_exploration = [
                html.Br(),
                html.Br(),
                html.Div(id="explore-filters",
                    children=[
                        html.H3("Filter sequence(s):"),
                        html.Div(id="explore-sequence_id-filter",
                                 children=[
                                     html.H4("Minimum-Maximum sequence length"),
                                     filter_sequence_id_slider
                                ]),
                        html.Div(id="explore-add-filters",
                                 children=[
                                     html.Br(),
                                     html.Button("+",
                                                 id={'component_id': 'explore-dynamic-add-filter'},
                                                 n_clicks=0,
                                                 style={
                                                     'borderRadius': '5px',
                                                     'backgroundColor': 'white',
                                                     'fontSize': '1em'
                                                 })
                                 ]
                        ),
                        html.Div(id={'component_id': 'explore-extra-filters'}),
                        html.Br(),
                    ],
                    #style={
                    #    'borderStyle': 'dashed',
                    #    'borderWidth': '1px',
                    #    'borderRadius': '5px',
                    #}
                    ),
                html.Br(),
                select_sequence_id_dropdown,
                select_feature_highlight_dropdown,
                select_feature_highlight_options,
                plot_options,
                html.Div(id={"component_id": "explore-sequence_id-stats"},
                         children=[
                             html.Div(
                                 children=[
                                     dash.dcc.Graph(id={'component_id': 'explore-sequence_id-plot-map'})
                                     ],
                                 hidden=True
                             )
                            ]
                ),
            ]
            return data_preview_table, select_for_exploration, True, True


    @classmethod
    def create(cls, app: AISApp) -> dcc.Tab:
        cls.register_callbacks(app)

        datasets_dropdown = html.Div(
            children=[html.H3("Available datasets"),
            dcc.Dropdown(id={'component_id': "explore-datasets-dropdown"},
                         options={ str(x): x.name for  x in (Path(app.data_upload_path) / "datasets").glob("*") if x.is_file() and x.suffix in SUPPORTED_FILE_FORMATS},
                         placeholder="Select a dataset",
                         multi=True)],
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
                                        text="Drag and drop to upload dataset(s)",
                                        text_completed='',
                                        id='explore-dataset-upload',
                                        max_file_size=10000, # 10 000 Mb,
                                        cancel_button=True,
                                        pause_button=True,
                                        filetypes=[x[1:] for x in SUPPORTED_FILE_FORMATS],
                                        upload_id='datasets',
                                        max_files=20,
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
                                        dcc.Dropdown(id={"component_id": "explore-sequence_id-column-dropdown"}, options=["test"], multi=True)
                                    ]
                           ),
                           html.Div(id='explore-sequence_id-selection'),
                           html.Div(id='explore-dataset-preview'),
                           dataset_stats,
                           data_column_dropdown,
                           visualization_type_dropdown,
                           html.Div(id='explore-dataset'),

                           dcc.Store(id='explore-sequence_id', storage_type='session'),
                           dcc.Store(id='explore-column-filter-state', storage_type='session'),
                       ],
                       className="tab")
