#!/bin/bash
./train -t 12 -e 0.00001 data/MQ2007-list/Fold1/train.txt model/mq2007list/mq2007_12.model
./predict data/MQ2007-list/Fold1/test.txt model/mq2007list/mq2007_12.model output/mq2007list/mq2007_12.txt
