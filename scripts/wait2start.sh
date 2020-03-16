#!/bin/bash

if [ $# -ne 1 ]; then
echo "usage: $0 config_path"
exit -1
fi

# setup a counter
host_idle_count=0
while true; do
    host_idle_flag=1
    # check availablity (retreat if remote host is in use)
    for x in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader); do
        x="${x//[$'\t\r\n ']}"  # remove trailing whitespace
        if [ $x -gt 10 ]; then
            echo "gpu utilization $x%";
            host_idle_flag=0
            break
        fi;
    done

    if [[ $host_idle_flag == 1 ]]; then
        let host_idle_count++
    fi

    if [[ $host_idle_count -ge 3 ]]; then
        # cleanup potentially dead python process (march since we checked it)
        echo "killing $host"
        pkill python
        break
    fi
    sleep 200;
done

python detection_train.py --config $1
