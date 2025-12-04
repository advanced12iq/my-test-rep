#include "sornyak_optimizer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "Сорняковый метод оптимизации" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    std::cout << "\nПример 1: Функция сферы (2 переменные, минимум в [0,0])" << std::endl;
    auto sphere_function = [](const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val;
        }
        return sum;
    };
    
    SornyakOptimizer optimizer1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto [result1, actual_iter1] = optimizer1.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result1.size(); i++) {
        std::cout << result1[i];
        if (i < result1.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer1.getBestFitness(result1) << std::endl;
    std::cout << "Количество итераций: " << actual_iter1 << std::endl;
    
    std::cout << "\nПример 2: Функция сферы (5 переменных, минимум в [0,0,0,0,0])" << std::endl;
    SornyakOptimizer optimizer2(sphere_function, 5, 500, 30, -5.0, 5.0);
    auto [result2, actual_iter2] = optimizer2.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result2.size(); i++) {
        std::cout << result2[i];
        if (i < result2.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer2.getBestFitness(result2) << std::endl;
    std::cout << "Количество итераций: " << actual_iter2 << std::endl;
    
    std::cout << "\nПример 3: Функция Розенброка (2 переменные, минимум в [1,1])" << std::endl;
    auto rosenbrock_function = [](const std::vector<double>& x) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size() - 1; i++) {
            double term1 = 100.0 * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]);
            double term2 = (1.0 - x[i]) * (1.0 - x[i]);
            sum += term1 + term2;
        }
        return sum;
    };
    
    SornyakOptimizer optimizer3(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto [result3, actual_iter3] = optimizer3.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result3.size(); i++) {
        std::cout << result3[i];
        if (i < result3.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer3.getBestFitness(result3) << std::endl;
    std::cout << "Количество итераций: " << actual_iter3 << std::endl;
    
    std::cout << "\nПример 4: Функция Розенброка (4 переменных, минимум в [1,1,1,1])" << std::endl;
    SornyakOptimizer optimizer4(rosenbrock_function, 4, 1000, 50, -2.0, 2.0);
    auto [result4, actual_iter4] = optimizer4.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result4.size(); i++) {
        std::cout << result4[i];
        if (i < result4.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer4.getBestFitness(result4) << std::endl;
    std::cout << "Количество итераций: " << actual_iter4 << std::endl;
    
    std::cout << "\nПример 5: Функция Растригина (2 переменных, минимум в [0,0])" << std::endl;
    auto rastrigin_function = [](const std::vector<double>& x) {
        double sum = 10.0 * x.size();
        for (double val : x) {
            sum += val * val - 10.0 * cos(2.0 * M_PI * val);
        }
        return sum;
    };
    
    SornyakOptimizer optimizer5(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto [result5, actual_iter5] = optimizer5.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result5.size(); i++) {
        std::cout << result5[i];
        if (i < result5.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer5.getBestFitness(result5) << std::endl;
    std::cout << "Количество итераций: " << actual_iter5 << std::endl;
    
    std::cout << "\nПример 6: Функция Растригина (10 переменных, минимум в [0,0,0,0,0,0,0,0,0,0])" << std::endl;
    SornyakOptimizer optimizer6(rastrigin_function, 10, 1000, 50, -5.0, 5.0);
    auto [result6, actual_iter6] = optimizer6.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result6.size(); i++) {
        std::cout << result6[i];
        if (i < result6.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer6.getBestFitness(result6) << std::endl;
    std::cout << "Количество итераций: " << actual_iter6 << std::endl;
    
    std::cout << "\nПример 7: Функция Швефеля (5 переменных, минимум в [420.9687,...,420.9687])" << std::endl;
    auto schwefel_function = [](const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * sin(sqrt(abs(val)));
        }
        return 418.9829 * x.size() - sum;
    };
    
    SornyakOptimizer optimizer7(schwefel_function, 5, 2000, 50, -500.0, 500.0);
    auto [result7, actual_iter7] = optimizer7.optimize();
    
    std::cout << "Оптимальное решение: [";
    for (size_t i = 0; i < result7.size(); i++) {
        std::cout << result7[i];
        if (i < result7.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Значение функции: " << optimizer7.getBestFitness(result7) << std::endl;
    std::cout << "Количество итераций: " << actual_iter7 << std::endl;
    std::cout << "\nТаблица результатов:" << std::endl;
    std::cout << std::setw(20) << std::left << "Функция" 
              << std::setw(10) << std::left << "Размерность" 
              << std::setw(15) << std::left << "Значение функции" 
              << std::setw(12) << std::left << "Итерации" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    std::cout << std::setw(20) << std::left << "Сфера (2D)"
              << std::setw(10) << std::left << "2"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer1.getBestFitness(result1)
              << std::setw(12) << std::left << actual_iter1 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Сфера (5D)"
              << std::setw(10) << std::left << "5"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer2.getBestFitness(result2)
              << std::setw(12) << std::left << actual_iter2 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Розенброк (2D)"
              << std::setw(10) << std::left << "2"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer3.getBestFitness(result3)
              << std::setw(12) << std::left << actual_iter3 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Розенброк (4D)"
              << std::setw(10) << std::left << "4"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer4.getBestFitness(result4)
              << std::setw(12) << std::left << actual_iter4 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Растригин (2D)"
              << std::setw(10) << std::left << "2"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer5.getBestFitness(result5)
              << std::setw(12) << std::left << actual_iter5 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Растригин (10D)"
              << std::setw(10) << std::left << "10"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer6.getBestFitness(result6)
              << std::setw(12) << std::left << actual_iter6 << std::endl;
    
    std::cout << std::setw(20) << std::left << "Швефель (5D)"
              << std::setw(10) << std::left << "5"
              << std::setw(15) << std::left << std::fixed << std::setprecision(6) << optimizer7.getBestFitness(result7)
              << std::setw(12) << std::left << actual_iter7 << std::endl;
    
    return 0;
}