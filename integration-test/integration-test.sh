#!/usr/bin/env bash

# Set hard fail on error
set -e

# cd to the script directory to avoid pathing issues
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../examples

# run readme examples
echo "Running config_sample.yaml example"
yass sort config_sample.yaml

echo "Running config_sample_complete.yaml example"
yass sort config_sample_complete.yaml


# run examples in examples/
echo "examples in examples/"
python preprocess.py
python detect.py
python cluster.py
python templates.py
python deconvolute.py