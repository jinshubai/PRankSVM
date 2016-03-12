#!/bin/bash
./train -t 8 train.txt 
./predict test.txt train.txt.model MQ2007.txt
