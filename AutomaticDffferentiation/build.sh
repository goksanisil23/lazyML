#!/usr/bin/env bash

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.8/" # for boost python

g++ -g -o test_needle tests/test_needle.cpp \
-lgtest -lgtest_main -pthread -I . -isystem /usr/include/eigen3 \
-I /home/s0001734/Downloads/Fastor \
-std=c++17
