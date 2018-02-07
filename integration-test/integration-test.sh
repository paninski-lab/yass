#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Examples
echo "Running config_sample.yaml example"
yass sort ../examples/config_sample.yaml

echo "Running config_sample_complete.yaml example"
yass sort ../examples/config_sample_complete.yaml
