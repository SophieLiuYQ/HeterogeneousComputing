#!/bin/bash -x

# script to fetch data and run predictions

export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

required_lang="C.UTF-8"
if [ "$LANG" != "$required_lang" ]; then
    export LANG="$required_lang"
fi

PYPATH="/home/yl4876/envs/poclEnv/bin"
PY="$PYPATH/python3"

poclEnv() {
  pushd "$PYPATH"
  source activate
  popd
}

pushd $HOME/workspace/HeterogeneousComputing/data
git pull origin data
popd

log_file="$HOME/logs/$(date +%s).log"
current_date="$(date +"%m-%d-%Y")"

echo "$(date)" >> "$log_file"

poclEnv
pushd $HOME/workspace/HeterogeneousComputing
"$PY" ./use_for_test.py -p -d "$current_date" -t $1 | tee -a "$log_file"
"$PY" ./use_for_test.py -u -d "$current_date" -t $1
# python3 ./use_for_test.py -p -d "$current_date" -t $1 | tee -a "$log_file"
# python3 ./use_for_test.py -u -d "$current_date" -t $1 | tee -a "$log_file"
popd
deactivate

pushd $HOME/logs
git pull origin logs
git add "$log_file"
git commit -m "update $current_date $1"
git push
popd
