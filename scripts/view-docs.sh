#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT/built-docs
python -m http.server
