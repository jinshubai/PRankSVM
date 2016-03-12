#!/bin/bash
./train -t 3 train.txt 
./predict test.txt train.txt.model MQ2007.txt
