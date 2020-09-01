#!/usr/bin/env bash

set -x
set -e

if [[ $# -lt 3 || $# -gt 4 ]]; then
	echo "usage: $0 config_file gpus ema [epoch]"
	exit -1
fi

config=$1
gpus=$2
ema=${3:-0}
epoch=${4:-1001}

config_basename=$(basename $config)
experiment_path=experiments/${config_basename/\.py/}

while res=$(inotifywait -e create --format %f $experiment_path); do
	if [[ $ema == true && $res == checkpoint_ema-$epoch.params ]]; then
		# python detection_test --config $config --gpus $gpus --epoch $epoch --ema
		let epoch++
	fi

	if [[ $ema == false && $res == checkpoint-$epoch.params ]]; then
		# python detection_test --config $config --gpus $gpus --epoch $epoch
		let epoch++
	fi
done
