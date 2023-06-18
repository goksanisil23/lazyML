#!/usr/bin/env bash

g++ -o test_needle autograd.cpp tests/test_needle.cpp -lgtest -lgtest_main -pthread -I . -I /usr/include/eigen3 -std=c++17
