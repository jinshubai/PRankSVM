#!/bin/bash
./train -t 14 -e 0.00001 data/MSLR-WEB30K/Fold1/train.txt model/mslrweb30k/mslrweb30k_14.model
./predict data/MSLR-WEB30K/Fold1/test.txt model/mslrweb30k/mslrweb30k_14.model output/mslrweb30k/mslrweb30k_14.txt
