#!/bin/bash

EXPS=("dload_bag_mir" "dload_bag_rtfm" "dload_bag_anm" "dload_bag_cmala") 

for exp in "${EXPS[@]}"; do
    python main.py -cn=xdv exp=${exp}
done