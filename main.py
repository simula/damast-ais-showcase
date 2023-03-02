# Copyright (C) 2023 Jørgen S. Dokken
#
# SPDX-License-Identifier:     BSD 3-Clause

import argparse
import pathlib

import vaex
from web_application import AISApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", dest="csv_file", required=True, type=str, help="Input csv file")
    parser.add_argument("--port", dest="port", required=True, type=int, help="Port for dash server")
    parser.add_argument("--mmsi-file", dest="mmsi_file", required=True, type=str, help="File identifying vessels")
    parser.add_argument("--delimiter", dest="delimiter", required=False, type=str, default=";",
                        help="Delimiter used in csv file")
    args = parser.parse_args()

    # We start by opening the DataFrame used in the application
    path = pathlib.Path(args.csv_file)
    mmsi_path = pathlib.Path(args.mmsi_file)
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
