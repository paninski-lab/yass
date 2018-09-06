#!/usr/bin/env bash

# Set hard fail on error
set -e

# cd to the script directory to avoid pathing issues
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../examples

# run readme examples
echo "Running config_sample.yaml example"
yass sort config_sample.yaml --clean

echo "Running config_sample_complete.yaml example"
yass sort config_sample_complete.yaml --clean


# run examples in examples/pipeline
python pipeline/preprocess.py
python pipeline/detect.py
python pipeline/cluster.py
python pipeline/templates.py
python pipeline/deconvolute.py