#!/bin/bash

echo "Running Sormyakov Optimization Method Comparisons"
echo "================================================"

echo -e "\n1. Basic comparison across all methods and test functions:"
echo "--------------------------------------------------------"
./comparison_main

echo -e "\n2. Detailed analysis with convergence tracking:"
echo "---------------------------------------------"
./detailed_analysis

echo -e "\n3. Convergence data sample (first 10 rows):"
echo "-------------------------------------------"
head -10 convergence_data.csv

echo -e "\nComparison completed! Check SORMYAKOV_METHOD_COMPARISON.md for detailed analysis."