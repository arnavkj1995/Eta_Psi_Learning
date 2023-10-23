#!/bin/bash

env=$1
python -m run_hive -c configs/${env}.yml --loggers.1.group ${env} --loggers.1.name seed_${SLURM_ARRAY_TASK_ID} --logdir logs/${env}/seed_${SLURM_ARRAY_TASK_ID}/
