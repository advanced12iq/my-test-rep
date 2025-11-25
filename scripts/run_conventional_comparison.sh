#!/bin/bash

# Скрипт для компиляции и запуска сравнения между методами Сормякова и традиционными методами оптимизации

echo "Компиляция программы сравнения..."
g++ -std=c++11 -o conventional_comparison conventional_comparison_main.cpp -lm

if [ $? -eq 0 ]; then
    echo "Компиляция прошла успешно!"
    echo ""
    echo "Запуск сравнения между методами Сормякова и традиционными методами оптимизации..."
    echo ""
    ./conventional_comparison
else
    echo "Компиляция не удалась!"
    exit 1
fi