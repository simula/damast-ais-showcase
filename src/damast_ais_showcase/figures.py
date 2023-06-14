import plotly.express as px
import plotly.graph_objs as go
from dash import dash_table, html, dcc
import numpy as np
import vaex
from typing import Optional, List, Any, Dict
import pandas as pd
from pandas.api.types import is_numeric_dtype

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
    data_df = adf._dataframe.copy()

    histograms = []
    col = data_df[column_name]
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
    data_df = adf._dataframe.copy()
    statistics = []
    if data_df[column_name].dtype == np.float16:
        data_df[column_name] = data_df[column_name].astype('float32')
    min_value = data_df.min(column_name)
    max_value = data_df.max(column_name)
    median_approx = data_df.median_approx(column_name)
    mean = data_df.mean(column_name)
    variance = data_df.var(column_name)

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


def create_figure_trajectory(data_df: vaex.DataFrame,
                             zoom_factor: float = 4,
                             center: Optional[Dict[str, float]] = None,
                             radius_factor: float = 10.0,
                             densities_by: Optional[List[str]] = None,
                             use_absolute_value: bool = True,
                             sequence_id_column = "passage_plan_id"
                             ) -> go.Figure:
    """
    Extract (lat, long) coordinates from dataframe and group them by passage_plan_id.
    NOTE: Assumes that the input data is sorted by in time
    """
    sorted_ids = sorted(data_df[sequence_id_column].unique())

    # Wrap the data so that transitions over the antimeridian do not lead to
    # map artifacts
    data_df_pandas = data_df.to_pandas_df()
    lat_crossings = pd.DataFrame(data_df_pandas.groupby(sequence_id_column)["Latitude"].apply(lambda x: x.min() < -89 or x.max() > 89).reset_index())
    lon_crossings = pd.DataFrame(data_df_pandas.groupby(sequence_id_column)["Longitude"].apply(lambda x: x.min() < -179 or x.max() > 179).reset_index())

    lat_crossings = lat_crossings[lat_crossings["Latitude"]][sequence_id_column].to_numpy()
    data_df["Latitude"] = data_df.apply(lambda x,y: y + 180 if x in lat_crossings and y < 0 else y, [data_df[sequence_id_column], data_df["Latitude"]])

    lon_crossings = lon_crossings[lon_crossings["Longitude"]][sequence_id_column].to_numpy()
    data_df["Longitude"] = data_df.apply(lambda x,y: y + 360 if x in lon_crossings and y < 0 else y, [data_df[sequence_id_column], data_df["Longitude"]])

    input_data = {
        "lat": data_df["Latitude"].evaluate(),
        "lon": data_df["Longitude"].evaluate(),
        "sequence_id": data_df[sequence_id_column].evaluate(),
    }
    fig = px.line_mapbox(input_data,
                         lat="lat", lon="lon",
                         color="sequence_id",
                         category_orders={'sequence_id': sorted_ids},
                         )

    if densities_by:
        for density_by in densities_by:
            if density_by not in data_df.column_names:
                continue

            density_input_df = data_df.dropnan(column_names=[density_by])
            density_input_df[density_by] = density_input_df[density_by].astype('float32')

            density_input_data = {
                "lat": density_input_df["Latitude"].evaluate(),
                "lon": density_input_df["Longitude"].evaluate(),
                "sequence_id": density_input_df[sequence_id_column].evaluate(),
            }

            # Add all information to the tooltip
            for column in density_input_df.column_names:
                if column not in ["Latitude", "Longitude", sequence_id_column] and not column.startswith("__"):
                    density_input_data[column] = density_input_df[column].evaluate()

            if use_absolute_value:
                density_input_df[density_by] = density_input_df[density_by].abs()

            scaler = vaex.ml.StandardScaler(features=[density_by])
            # this will create a column 'standard_scaled_<feature-name>'
            density_input_df = scaler.fit_transform(density_input_df)
            normalized_column = f"standard_scaled_{density_by}"

            density_input_data[density_by] = density_input_df[density_by].evaluate()

            radius = []
            for x in density_input_df[normalized_column].evaluate():
                if np.isnan(x):
                    radius.append(1)
                    continue

                value = x*radius_factor
                if value < 1:
                    radius.append(1)
                else:
                    radius.append(value)

            hover_data = {k: True for k, _ in density_input_data.items()}
            fig_feature = px.density_mapbox(density_input_data,
                                    lat='lat',
                                    lon='lon',
                                    hover_data=hover_data,
                                    color_continuous_scale="YlOrRd",
                                    #range_color=[0,10],
                                    z=density_by,
                                    radius=radius)

            fig.add_trace(fig_feature.data[0])
    fig.update_coloraxes(showscale=False)

    fig.update_layout(height=1000,
                      mapbox_style="open-street-map",
                      mapbox_zoom=zoom_factor)
    if center:
        fig.update_layout(mapbox_center=center)

    return fig

def create_figure_feature_correlation_heatmap(data_df: vaex.DataFrame) -> go.Figure:
    df_correlations = data_df.to_pandas_df().corr(numeric_only=True)
    return px.imshow(df_correlations,
                     text_auto='.2f',
                     height=1000,
                     width=1000)

def create_figure_data_preview_table(data_df: vaex.DataFrame, sequence_id_column: Optional[str] = None, sequence_id: Optional[int] = None)  -> List[Any]:
    if sequence_id_column and sequence_id:
        data_df = data_df[data_df[sequence_id_column] == sequence_id]

    df = data_df[:500].to_pandas_df()
    return [html.H3("Data Preview"),
            dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            # https://dash.plotly.com/datatable/style
            style_cell={'textAlign': 'center', 'padding': "5px"},
            style_header={'backgroundColor': 'lightgray',
                            'color': 'white',
                            'fontWeight': 'bold'},
            page_size=10
        )]
