#!/bin/bash

grep "AccuracyStats:"    out_stats.txt | awk -F" " '{printf("%d, %f\n", $2, $3);}' > data/accuracy_stats.csv 

grep "LoadTimeStats:"    out_stats.txt | awk -F" " '{printf("%f\n", $2, $3);}' > data/lt_stats.csv

grep "ComputeTimeStats:" out_stats.txt | awk -F" " '{printf("%s, %d, %f\n", $2, $3, $4);}' > data/ccc_stats.csv

grep "GSTimingStats:"    out_stats.txt | awk -F" " '{printf("%f\n", $2);}' > data/gs_stats.csv
