#!/bin/bash

# Script to compile and run the comparison between Sormyakov methods and conventional optimization methods

echo "Compiling the comparison program..."
g++ -std=c++11 -o conventional_comparison conventional_comparison_main.cpp -lm

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "Running comparison between Sormyakov methods and conventional optimization methods..."
    echo ""
    ./conventional_comparison
else
    echo "Compilation failed!"
    exit 1
fi