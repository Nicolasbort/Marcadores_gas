#!/bin/bash

FILE="detection"

g++ $FILE.cpp ROI/ROI.cpp landingMark.cpp `pkg-config --cflags --libs opencv` -o $FILE
