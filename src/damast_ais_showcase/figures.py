import plotly.express as px
import plotly.graph_objs as go
from dash import dash_table, html, dcc
import numpy as np
import vaex
from typing import Optional, List, Any, Dict

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
                             density_by: Optional[str] = None,
                             ) -> go.Figure:
    """
    Extract (lat, long) coordinates from dataframe and group them by passage_plan_id.
    NOTE: Assumes that the input data is sorted by in time
    """
    input_data = {
        "lat": data_df["Latitude"].evaluate(),
        "lon": data_df["Longitude"].evaluate(),
        "passage_plan_id": data_df["passage_plan_id"].evaluate(),
    }
    fig = px.line_mapbox(input_data,
                         lat="lat", lon="lon",
                         color="passage_plan_id")

    if density_by and density_by in data_df.column_names:
        # Ensure operation with float32 since float16 is not supported by vaex
        data_df[density_by] = data_df[density_by].astype('float32')
        scaler = vaex.ml.StandardScaler(features=[density_by])
        # this will create a column 'standard_scaled_<feature-name>'
        data_df = scaler.fit_transform(data_df)
        normalized_column = f"standard_scaled_{density_by}"

        input_data[density_by] = data_df[density_by].evaluate()

        radius = []
        for x in data_df[normalized_column].evaluate():
            value = x*radius_factor
            if value < 1:
                radius.append(1)
            else:
                radius.append(value)

        fig2 = px.density_mapbox(input_data,
                                lat='lat',
                                lon='lon',
                                color_continuous_scale="YlOrRd",
                                #range_color=[0,10],
                                z=density_by,
                                radius=radius)

        fig.add_trace(fig2.data[0])
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
