#!/bin/bash
./train -t 8 -e 0.00001 data/MSLR-WEB30K/Fold1/train.txt model/mslrweb30k/mslrweb30k_8.model
./predict data/MSLR-WEB30K/Fold1/test.txt model/mslrweb30k/mslrweb30k_8.model output/mslrweb30k/mslrweb30k_8.txt