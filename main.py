# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause

import argparse
from pathlib import Path
from typing import Union


import vaex
from web_application import AISApp

import damast
from damast.core.datarange import CyclicMinMax
from damast.core.units import units
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement

class HDF5Export(PipelineElement):
    filename: Path = None

    def __init__(self, filename: Union[str, Path]):
        super().__init__()
        self.filename = Path(filename)

    @damast.core.input({})
    @damast.core.artifacts({
        "hdf5_export": "*.hdf5"
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        df.save(filename=self.parent_pipeline.base_dir / self.filename)
        return df


class LatLonFilter(PipelineElement):
    @damast.core.describe("Filter lat/lon to valid range")
    @damast.core.input({
        "lat": {"unit": units.deg},
        "lon": {"unit": units.deg}
    })
    @damast.core.output({
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        df["latitude"] = df["lat"]
        df._dataframe = df.filter(df.latitude >= -90.0).filter(df.latitude <= 90.0)

        df["longitude"] = df["lon"]
        df._dataframe = df.filter(df.longitude >= -180.0).filter(df.longitude <= 180.0)
        return df


class LatLonTransformer(PipelineElement):
    @damast.core.describe("Lat/Lon cyclic transformation")
    @damast.core.input({
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}
    })
    @damast.core.output({
        "latitude_x": {"unit": units.deg},
        "latitude_y": {"unit": units.deg},
        "longitude_x": {"unit": units.deg},
        "longitude_y": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = vaex.ml.CycleTransformer(features=["latitude"], n=180.0)
        lon_cyclic_transformer = vaex.ml.CycleTransformer(features=["longitude"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", dest="csv_file", required=True, type=str, help="Input csv file")
    parser.add_argument("--port", dest="port", required=True, type=int, help="Port for dash server")
    parser.add_argument("--mmsi-file", dest="mmsi_file", required=True, type=str, help="File identifying vessels")
    parser.add_argument("--delimiter", dest="delimiter", required=False, type=str, default=";",
                        help="Delimiter used in csv file")
    args = parser.parse_args()

    # We start by opening the DataFrame used in the application
    path = Path(args.csv_file)
    mmsi_path = Path(args.mmsi_file)
    if path.suffix == ".csv":
        vaex_df = vaex.read_csv(path, delimiter=args.delimiter)
    elif path.suffix == ".hdf5":
        vaex_df = vaex.open(path)
    else:
        raise ValueError("Unknown file format for input AIS data")

    if mmsi_path.suffix == ".csv":
        mmsi_df = vaex.read_csv(mmsi_path, delimiter=args.delimiter)
    elif mmsi_path.suffix == ".hdf5":
        mmsi_df = vaex.open(mmsi_path)
    else:
        raise ValueError("Unknown file format for input MMSI vessel types")

    # Only consider fishing boats
    unique_boats = mmsi_df[mmsi_df["vessel_type"] == "fishing"]["MMSI"].unique()
    boat_df = vaex_df[vaex_df["mmsi"].isin(unique_boats)]

    # Create and launch web-application
    web_app = AISApp(boat_df, port=args.port)
    web_app.generate_layout()
    web_app.run_server()
