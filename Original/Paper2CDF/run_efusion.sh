#!/bin/bash
set -e
# Your Env name
source activate pytorch1.10

root_dir=/path/to/data

python ./experiments/run_elasticfusion.py --base_dir=$root_dir ---depth_file=depth --rgb_file=rgb --drop_num=1
/path/to/pngtoklg -w "$root_dir" -o  "$root_dir/scene.klg" -s 4000 -t

echo "Running ElasticFusion"


/path/to/elasticfuion -l $root_dir/scene.klg -f -cal $root_dir/camera
