#!/bin/bash

echo "current user is: $(whoami)"

source /opt/intel/oneapi/intelpython/latest/env/vars.sh
cd /home/szaday2/workspace/HeterogeneousComputing

python3 ./test.py
python3 ./test2.py

timestamp="$(date +%s)"

git pull origin data
git commit -am "update at $timestamp"
git push
