#!/bin/bash

FILE="detection"

g++ $FILE.cpp ROI/ROI.cpp LandingMark/landingMark.cpp `pkg-config --cflags --libs opencv tesseract` -o $FILE
