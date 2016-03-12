#!/bin/bash
./train -t 16 -e 0.00001 data/MQ2008-list/Fold1/train.txt model/mq2008list/mq2008_16.model
./predict data/MQ2008-list/Fold1/test.txt model/mq2008list/mq2008_16.model output/mq2008list/mq2008_16.txt
