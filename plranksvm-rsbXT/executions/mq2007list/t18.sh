#!/bin/bash
./train -t 18 -e 0.00001 data/MQ2007-list/Fold1/train.txt model/mq2007list/mq2007_18.model
./predict data/MQ2007-list/Fold1/test.txt model/mq2007list/mq2007_18.model output/mq2007list/mq2007_18.txt
