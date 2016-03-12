#!/bin/bash
./train -t 20 -e 0.00001 data/MSLR-WEB30K/Fold1/train.txt model/mslrweb30k/mslrweb30k_20.model
./predict data/MSLR-WEB30K/Fold1/test.txt model/mslrweb30k/mslrweb30k_20.model output/mslrweb30k/mslrweb30k_20.txt
