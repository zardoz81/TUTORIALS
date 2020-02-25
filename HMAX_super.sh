#!/bin/bash

##sbatch bt_serial.sh 6.5

for sample_entry in {0..700..10}
do
    echo $sample_entry
    sleep 2
    sbatch HMAX_serial.sh $sample_entry
done
