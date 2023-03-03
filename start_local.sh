#!/bin/bash

SCRIPT_DIR=$(dirname $0)
VENV_DIR=$SCRIPT_DIR/../../venv

source $VENV_DIR/bin/activate

cd $SCRIPT_DIR
python3 main.py --file=data/ais_20200101.hdf5 --mmsi-file=data/mmsi2vesseltype.hdf5 --port=8888
