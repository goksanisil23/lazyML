#!/usr/bin/env bash

g++ -g -o test_needle autograd.cpp tests/test_needle.cpp -lgtest -lgtest_main -pthread -I . -isystem /usr/include/eigen3 -std=c++17
