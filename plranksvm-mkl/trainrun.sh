#!/bin/bash
./train -t 1 train.txt 
./predict test.txt train.txt.model MQ2007.txt
