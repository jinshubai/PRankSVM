#!/bin/bash
./train -t 10 -e 0.00001 data/MQ2008-list/Fold1/train.txt model/mq2008list/mq2008_10.model
./predict data/MQ2008-list/Fold1/test.txt model/mq2008list/mq2008_10.model output/mq2008list/mq2008_10.txt
