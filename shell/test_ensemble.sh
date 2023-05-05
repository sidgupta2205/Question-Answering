#!/bin/bash

python ../src/ensemble_test.py --split dev -n ensemble-test/ensemble-8 --batch_size 32 --load_path temp
python ../src/ensemble_test.py --split test -n ensemble-test/ensemble-8 --batch_size 32 --load_path temp
