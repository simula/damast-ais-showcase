import plotly.express as px
import plotly.graph_objs as go
from dash import dash_table, html, dcc
import numpy as np
import polars as pl
from typing import Optional, List, Any, Dict
import pandas as pd

from damast.core.dataframe import AnnotatedDataFrame

# https://github.com/vaexio/dash-120million-taxi-app/blob/master/app.py
# This has to do with layout/styling
fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)


def create_figure_histogram(x, counts, title=None, xlabel=None, ylabel=None):
    # settings
    color = 'royalblue'

    # list of traces
    traces = []

    # Create the figure
    line = go.scatter.Line(color=color, width=2)
    hist = go.Scatter(x=x, y=counts, mode='lines', line_shape='hv', line=line, name=title, fill='tozerox')
    traces.append(hist)

    # Layout
    title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
    margin = go.layout.Margin(l=0, r=0, b=0, t=30)
    legend = go.layout.Legend(orientation='h',
                              bgcolor='rgba(0,0,0,0)',
                              x=0.5,
                              y=1,
                              itemclick=False,
                              itemdoubleclick=False)
    layout = go.Layout(height=230,
                       margin=margin,
                       legend=legend,
                       title=title,
                       xaxis=go.layout.XAxis(title=xlabel),
                       yaxis=go.layout.YAxis(title=ylabel),
                       **fig_layout_defaults)

    # Now calculate the most likely value (peak of the histogram)
    # peak = np.round(x[np.argmax(counts)], 2)

    return go.Figure(data=traces, layout=layout)


def create_div_histogram(adf: AnnotatedDataFrame, column_name: str) -> go.Figure:

    histograms = []
    col = adf[column_name]
    if col.dtype == np.float16:
        col = col.astype('float32')

    min_value = col.min()
    max_value = col.max()
    counts = data_df.count(binby=col,
                           # limits=[0, 900000000],
                           shape=1000)
    unit_name = None
    try:
        unit_name = str(adf._metadata[column_name].unit)
    except KeyError:
        pass
    if unit_name is None:
        unit_name = "<no known unit>"

    fig_column = create_figure_histogram(np.linspace(min_value, max_value, 1000), counts,
                                         # title=f"\n{column_name}",
                                         xlabel=unit_name,
                                         ylabel="count")
    histograms.append(html.H4(f"Histogram: '{column_name}'"))
    histograms.append(dcc.Graph(id=f'data-histogram-{column_name}', figure=fig_column))
    return histograms


def create_div_statistic(adf: AnnotatedDataFrame, column_name: str) -> go.Figure:
    data_df = adf._dataframe
    statistics = []

    min_value = data_df.select(pl.col(column_name)).min().collect()[0,0]
    max_value = data_df.select(pl.col(column_name)).max().collect()[0,0]
    median_approx = data_df.select(pl.col(column_name)).median().collect()[0,0]
    mean = data_df.select(pl.col(column_name)).mean().collect()[0,0]
    variance = data_df.select(pl.col(column_name)).var().collect()[0,0]

    data = {
        "min": min_value,
        "max": max_value,
        "median (approx)": median_approx,
        "mean": f"{mean:.2f} +/- {variance:.3f}",
    }

    stats_table = dash_table.DataTable(
        id={'component_id': f'data-statistics-{column_name}'},
        data=[data],
        columns=[{'id': c, 'name': c} for c in data.keys()],
        # https://dash.plotly.com/datatable/style
        style_cell={'textAlign': 'center', 'padding': "5px"},
        style_header={'backgroundColor': 'lightgray',
                      'color': 'white',
                      'fontWeight': 'bold'}
    )

    statistics.append(html.H4(f"Statistics: '{column_name}'"))
    statistics.append(stats_table)
    return statistics


def create_div_metadata(adf: AnnotatedDataFrame, column_name: str) -> go.Figure:
    metadata = []
    try:
        # Ensure that values are converted to string for html presentation
        data = {x: str(y) for x,y in dict(adf._metadata[column_name]).items()}
        metadata_table = dash_table.DataTable(
            id={'component_id': f'data-metadata-{column_name}'},
            data=[data],
            columns=[{'id': c, 'name': c} for c in data.keys()],
            # https://dash.plotly.com/datatable/style
            style_cell={'textAlign': 'center', 'padding': "5px"},
            style_header={'backgroundColor': 'lightgray',
                          'color': 'white',
                          'fontWeight': 'bold'}
        )
    except KeyError:
        metadata_table = html.Div(id={'component_id': f'metadata-{column_name}-warning'},
                                  children=html.H5("No metadata available for this column"),
                                  style={
                                      'backgroundColor': 'orange'
                                  })

    metadata.append(html.H4(f"Metadata: '{column_name}'"))
    metadata.append(metadata_table)
    return metadata


def create_figure_trajectory(data_df: AnnotatedDataFrame,
                             zoom_factor: Optional[float] = None,
                             center: Optional[Dict[str, float]] = None,
                             radius_factor: float = 10.0,
                             densities_by: Optional[List[str]] = None,
                             use_absolute_value: bool = True,
                             sequence_id_column = "passage_plan_id",
                             width=2000,
                             height=2000,
                             lat_lon_cols = ["Latitude", "Longitude"],
                             sequence_reference_cols={'Latitude': 'pp_proj_lat', 'Longitude': 'pp_proj_lon'},
                             ) -> go.Figure:
    """
    Extract (lat, long) coordinates from dataframe and group them by passage_plan_id.
    NOTE: Assumes that the input data is sorted by in time

    Args:
        sequence_reference_cols (Dict[str,str]): Dictionary containing the
            reference for the actual sequence latitude / longitude columns which can
            be displayed as a reference (ground-truth) trajectory
    """
    sorted_ids = data_df.select(pl.col(sequence_id_column)).unique().sort(by=sequence_id_column).collect()[:,0].to_list()

    lat, lon = lat_lon_cols
    ref_lat = sequence_reference_cols["Latitude"]
    ref_lon = sequence_reference_cols["Longitude"]

    # Wrap the data so that transitions over the antimeridian do not lead to
    # map artifacts
    data_df_pandas = data_df.collect().to_pandas()
    data_df = data_df.collect().to_pandas()
    lat_crossings = pd.DataFrame(data_df_pandas.groupby(sequence_id_column)[lat].apply(lambda x: x.min() < -89 or x.max() > 89).reset_index())
    lon_crossings = pd.DataFrame(data_df_pandas.groupby(sequence_id_column)[lon].apply(lambda x: x.min() < -179 or x.max() > 179).reset_index())

    lat_crossings = lat_crossings[lat_crossings[lat]][sequence_id_column].to_numpy()
    data_df[lat] = data_df.apply(lambda x: x[lat] + 180 if x[sequence_id_column] in lat_crossings and x[lat] < 0 else x[lat], axis=1)
    #if ref_lat in data_df.columns:
    #    data_df[ref_lat] = data_df.apply(lambda x,y: y + 180 if x in lat_crossings and y < 0 else y, [data_df[sequence_id_column], data_df[ref_lat]])

    lon_crossings = lon_crossings[lon_crossings[lon]][sequence_id_column].to_numpy()
    data_df[lon] = data_df.apply(lambda x: x[lon] + 360 if x[sequence_id_column] in lon_crossings and x[lon] < 0 else x[lon], axis=1)

    #if ref_lon in data_df.columns:
    #    data_df[ref_lon] = data_df.apply(lambda x,y: y + 360 if x in lon_crossings and y < 0 else y, [data_df[sequence_id_column], data_df[ref_lon]])

    input_data = {
        "sequence_id": data_df[sequence_id_column]
    }
    # Add all information to the tooltip of the main line plot
    for column in data_df.columns:
        if column not in [sequence_id_column] and not column.startswith("__"):
            input_data[column] = data_df[column]

    fig = px.line_map(input_data,
                         lat=lat, lon=lon,
                         color="sequence_id",
                         category_orders={'sequence_id': sorted_ids},
                         )

    # Allow to visualize a projection / reference line for the original trajectory
    if len(sorted_ids) == 1 and ref_lat in input_data and ref_lon in input_data:
        fig_pp = px.line_map(input_data,
                            lat=ref_lat, lon=ref_lon,
                            color="sequence_id",
                            category_orders={'sequence_id': sorted_ids},
                            color_discrete_sequence=["orange"]
                            )
        fig.add_trace(fig_pp.data[0])

    if densities_by:
        for density_by in densities_by:
            if density_by not in data_df.columns:
                continue

            density_input_df = data_df.dropna(subset=[density_by])
            density_input_df[density_by] = density_input_df[density_by].astype('float32')

            density_input_data = {
                "sequence_id": density_input_df[sequence_id_column]
            }
            # Add all information to the tooltip
            for column in data_df.columns:
                if column not in [sequence_id_column] and not column.startswith("__"):
                    density_input_data[column] = density_input_df[column]

            if use_absolute_value:
                density_input_df[density_by] = density_input_df[density_by].abs()

            # 2maz
            # this will create a column 'standard_scaled_<feature-name>'
            normalized_column = f"standard_scaled_{density_by}"
            mean = density_input_df[density_by].mean()
            std_dev = density_input_df[density_by].std()
            density_input_df[normalized_column] =  (density_input_df[density_by] - mean) / std_dev
            density_input_data[density_by] = density_input_df[density_by]

            radius = []
            for x in density_input_df[normalized_column]:
                if np.isnan(x):
                    radius.append(1)
                    continue

                value = x*radius_factor
                if value < 1:
                    radius.append(1)
                else:
                    radius.append(value)

            hover_data = {k: True for k, _ in density_input_data.items()}
            fig_feature = px.density_map(density_input_data,
                                    lat=lat,
                                    lon=lon,
                                    hover_data=hover_data,
                                    color_continuous_scale="YlOrRd",
                                    #range_color=[0,10],
                                    z=density_by,
                                    radius=radius)

            fig.add_trace(fig_feature.data[0])

    fig.update_coloraxes(showscale=False)

    lat_mean = np.mean(input_data[lat])
    lon_mean = np.mean(input_data[lon])

    lon_min = np.min(input_data[lon])
    lon_max = np.max(input_data[lon])

    lat_min = np.min(input_data[lat])
    lat_max = np.max(input_data[lat])

    if not zoom_factor:
        # Upon rendering a new trajectory (set)
        # try to automatically set the zoom so that all
        # trajectories are part of this visualized map
        delta_lon = lon_max - lon_min
        delta_lat = lat_max - lat_min
        max_delta = max(delta_lat, delta_lon)
        zoom_factor = max(11.5 - np.log(max_delta*40), 0)

    fig.update_layout(height=height,
                      width=width,
                      map_style="open-street-map",
                      map_zoom=zoom_factor)

    if center:
        fig.update_layout(map_center=center)
    else:
        # If not center is given, then use the known mean values of the current data
        fig.update_layout(map_center={ 'lat': lat_mean, 'lon': lon_mean})

    return fig

def create_figure_feature_correlation_heatmap(data_df: AnnotatedDataFrame) -> go.Figure:
    """
    Create a heatmap image with the correlations of all dataframe columns

    Args:
        data_df (vaex.DataFrame): _description_

    Returns:
        go.Figure: _description_
    """
    df_correlations = data_df.collect().to_pandas().corr(numeric_only=True)
    return px.imshow(df_correlations,
                     text_auto='.2f',
                     height=1000,
                     width=1000)

def create_figure_data_preview_table(data_df: AnnotatedDataFrame,
                                     sequence_id_column: Optional[str] = None,
                                     sequence_id: Optional[int] = None,
                                     upper_bound: int = 500,
                                     page_size: int = 10)  -> List[Any]:
    """
    Create a table that show the data columns of the given dataframe

    Args:
        data_df (vaex.DataFrame): _description_
        sequence_id_column (Optional[str], optional): _description_. Defaults to None.
        sequence_id (Optional[int], optional): _description_. Defaults to None.
        upper_bound (int): max number of rows.
        page_size (int): number of rows to be shown on one page

    Returns:
        List[Any]: List of html object that shall be rendered
    """
    if sequence_id_column and sequence_id:
        data_df = data_df.filter(pl.col(sequence_id_column) == sequence_id)

    df = data_df.slice(offset=0, length=upper_bound).collect().to_pandas()
    return [html.H3("Data Preview"),
            dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            # https://dash.plotly.com/datatable/style
            style_cell={'textAlign': 'center', 'padding': "5px"},
            style_header={'backgroundColor': 'lightgray',
                            'color': 'white',
                            'fontWeight': 'bold'},
            page_size=page_size
        )]
