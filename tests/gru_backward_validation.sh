#!/bin/bash

# [seq length, batch size, input size, hidden size]
test_sizes="
    5,4,6,6  
    10,32,128,128
"
for size in $test_sizes
do
    size_str=${size//,/ }
    # gru.py param: size, num layer, bidirectional
    python gru.py $size_str 1 False
    ../build/gru_backward_validation
    python gru.py $size_str 1 True
    ../build/gru_backward_validation
    python gru.py $size_str 3 False
    ../build/gru_backward_validation
    python gru.py $size_str 3 True
    ../build/gru_backward_validation
done
