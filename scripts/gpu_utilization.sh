#!/bin/bash

if [ $# -gt 1 ]; then
    echo "usage: $0 [comma_separated_worker_hostnames]"
    exit -1
fi

if [ $# -eq 1 ]; then
    hosts=$1
else
    # hosts=a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27
    hosts=a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14,a15,a16,a17,a19,a20,a21,a22,a23,a24,a25,a26,a27
fi

# extract worker and check reachablity
IFS=, read -r -a host_array <<< $hosts
for host in ${host_array[@]}; do
    # check reachability
    ssh -q $host -o ConnectTimeout=1 -o StrictHostKeyChecking=no exit
    if [ $? -ne 0 ]; then
        echo "$host is not reachable"
	continue
    fi

    # check availablity (retreat if remote host is in use)
    empty=1
    for x in $(ssh $host nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader); do 
	x="${x//[$'\t\r\n ']}"  # remove trailing whitespace
	if [ $x -gt 5 ]; then 
            empty=0
            echo "$host is busy"
	    break
        fi;  
    done

    if [ $empty -eq 1 ]; then
	echo "$host is available"
    fi
done
