#!/usr/bin/env bash

set -x
set -e

if [[ $# -lt 2 || $# -gt 4 ]]; then
	echo "usage: $0 config_file gpus [postfix] [epoch] [timeout]"
	exit -1
fi

config=$1
gpus=$2
postfix=$3
epoch=${4:-1001}
cksum=$(echo ${config}${postfix} | md5sum | awk '{print $1}')
cksum=${cksum:0:8}
timeout=${5:-1000}

config_basename=$(basename $config)
exp_path=experiments/${config_basename/\.py/}
last_epoch_file=$exp_path/last_epoch.$cksum

# read in last epoch if exist
if [[ -e $last_epoch_file ]]; then
	epoch=$(cat $last_epoch_file)
	let epoch++
fi

if [[ -n $postfix ]]; then
	postfix_args="--postfix $postfix"
	ckpt_pattern="checkpoint_$postfix"
else
	postfix_args=""
	ckpt_pattern="checkpoint"
fi

# process old checkpoints before watchdog start
while [[ -e $exp_path/$ckpt_pattern-$epoch.params ]];
do
	timeout $timeout python detection_test.py --config $config --gpus $gpus --epoch $epoch $postfix_args
	if [[ $? -eq 0 ]]; then
		echo -n $epoch > $last_epoch_file
		let epoch++
	fi
done

# process incoming checkpoints
inotifywait -e create --format %f -m $exp_path | while read res
do
	if [[ $res == $ckpt_pattern-$epoch.params ]]; then
		timeout $timeout python detection_test.py --config $config --gpus $gpus --epoch $epoch $postfix_args
		if [[ $? -eq 0 ]]; then
			echo -n $epoch > $last_epoch_file
			let epoch++
		fi
	fi
done
