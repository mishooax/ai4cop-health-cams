#!/bin/bash --login
set -e

conda activate base
exec "$@"
