#!/usr/bin/env bash

# Set hard fail on error
set -e

# cd to the script directory to avoid pathing issues
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../examples

# Run Examples
echo "Running config_sample.yaml example"
yass sort config_sample.yaml

echo "Running config_sample_complete.yaml example"
yass sort config_sample_complete.yaml
