# Copyright (C) 2023 Simula Research Laboratory
# SPDX-License-Identifier:     BSD 3-Clause

import argparse
import os
from pathlib import Path
from typing import Union

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import PipelineElement
from damast.core.datarange import CyclicMinMax
from damast.core.units import units
from damast.core.transformations import CycleTransformer

from damast_ais_showcase.web_application import AISApp


class HDF5Export(PipelineElement):
    filename: Path

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
        df._dataframe = df.with_columns(pl.col("lat").alias("latitude"), pl.col("lon").alias("longitude"))
        df._dataframe = df._dataframe.filter(df.latitude >= -90.0).filter(df.latitude <= 90.0).filter(df.longitude >= -180.0).filter(df.longitude <= 180.0)
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
        lat_cyclic_transformer = CycleTransformer(features=["latitude"], n=180.0)
        lon_cyclic_transformer = CycleTransformer(features=["longitude"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port",
                        dest="port",
                        type=int,
                        default=8888,
                        help="Port for dash server")
    parser.add_argument("--delimiter", dest="delimiter", required=False, type=str, default=";",
                        help="Delimiter used in csv file")
    args = parser.parse_args()

    # Create and launch web-application
    web_app = AISApp(port=args.port)
    web_app.generate_layout()
    web_app.run_server()


def run_worker():
    from damast.ml.worker import Worker
    w = Worker()
    w.listen_and_accept()


if __name__ == "__main__":
    return_val = os.fork()
    if return_val == 0:
        print(f"Starting worker with pid '{os.getpid()}'")
        run_worker()
    else:
        run()
