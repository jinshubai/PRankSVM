#!/bin/bash
./train -t 10 -e 0.00001 data/MSLR-WEB10K/Fold1/train.txt model/mslrweb10k/mslrweb10k_10.model
./predict data/MSLR-WEB10K/Fold1/test.txt model/mslrweb10k/mslrweb10k_10.model output/mslrweb10k/mslrweb10k_10.txt
