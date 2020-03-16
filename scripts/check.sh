#!/bin/bash

if [ $# -ne 1 ]; then
echo "usage: $0 comma_separated_worker_hostnames"
continue
fi

hosts=$1

# constants
root_dir="/mnt/tscpfs/yuntao.chen/simpledet/simpledet_open"
sync_dir="/tmp/simpledet_sync"
singularity_image=/mnt/tscpfs/yuntao.chen/simpledet.img

# extract worker and check reachablity
IFS=, read -r -a host_array <<< $hosts
for host in ${host_array[@]}; do
    # check reachability
    echo "check reachability of $host"
    ssh -q $host exit
    if [ $? -ne 0 ]; then
        echo "$host is not reachable"
	    continue
    fi

    # check availablity (retreat if remote host is in use)
    echo "check availability of $host"
    for x in $(ssh $host nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader); do
	x="${x//[$'\t\r\n ']}"  # remove trailing whitespace
	    if [ $x -gt 10 ]; then
	        echo "$host has gpu utilization of $x%";
	        continue
        fi;
    done

    # check filesystem
    if ssh -q $host "[ -d ${root_dir} ]"; then
        echo "$host has mounted ${root_dir}"
    else
        echo "$host has not mounted ${root_dir}"
        continue
    fi

    # check singularity
    ssh -q $host singularity > /dev/null
    if [ $? -ne 0 ]; then
        echo "singularity is not installed on $host"
	    continue
    else
        echo "singularity is installed on $host"
    fi

    # check cuda
    ssh -q $host "cd /usr/local/cuda/samples/1_Utilities/deviceQuery/; make && ./deviceQuery" 2>&1 > /dev/null
    if [ $? -ne 0 ]; then
        echo "device query fails on $host"
	    continue
    else
        echo "device query succeeds on $host"
    fi
done
