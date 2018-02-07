#!/usr/bin/env bash

# Set hard fail on error
set -e

# cd to the script directory to avoid pathing issues
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

# Run Examples
echo "Running config_sample.yaml example"
yass sort $DIR/../examples/config_sample.yaml

echo "Running config_sample_complete.yaml example"
yass sort $DIR/../examples/config_sample_complete.yaml
