#!/bin/bash

. ~/.profile

# local fs
python wide_n_deep_tutorial.py --model_type=wide --train_data=data/adult.data --test_data=data/adult.test

python wide_n_deep_tutorial.py --model_type=deep --train_data=data/adult.data --test_data=data/adult.test

python wide_n_deep_tutorial.py --model_type=wide_n_deep --train_data=data/adult.data --test_data=data/adult.test

# hdfs
HADOOP_HDFS_HOME=/usr/local/hadoop python wide_n_deep_tutorial.py --model_type=wide --train_data=hdfs://localhost:9000/user/yaowq/data/adult.data --test_data=hdfs://localhost:9000/user/yaowq/data/adult.test
