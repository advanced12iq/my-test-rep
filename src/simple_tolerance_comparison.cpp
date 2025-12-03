#include "sornyak_modifications.h"
#include "conventional_optimization.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <iomanip>

// Тестовые функции
auto sphere_function = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
};

auto rosenbrock_function = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; i++) {
        double term1 = 100.0 * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]);
        double term2 = (1.0 - x[i]) * (1.0 - x[i]);
        sum += term1 + term2;
    }
    return sum;
};

auto rastrigin_function = [](const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += val * val - 10.0 * cos(2.0 * M_PI * val);
    }
    return sum;
};

// Структура для хранения результатов сравнения
struct ToleranceComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    bool reached_tolerance;
};

// Функция для запуска оптимизации с измерением времени
template<typename OptimizerType>
ToleranceComparisonResult runOptimization(std::function<double(const std::vector<double>&)> func, 
                                         OptimizerType& optimizer, 
                                         const std::string& method_name, 
                                         double target_fitness = 0.0,
                                         double tolerance = 1e-6) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result = optimizer.optimize();
    double best_fitness = optimizer.getBestFitness(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ToleranceComparisonResult res;
    res.method_name = method_name;
    res.best_fitness = best_fitness;
    res.best_solution = result;
    res.execution_time_ms = duration.count();
    res.reached_tolerance = std::abs(best_fitness - target_fitness) < tolerance;
    
    return res;
}

int main() {
    std::cout << "Сравнение времени достижения оптимального решения с tol=1e-6" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Тест на функции Сферы (2D)
    std::cout << "\nТестирование на функции Сферы (2D, минимум в [0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(74, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakWithAdaptiveSpread sornyak_adapt(sphere_function, 2, 1000, 30, -5.0, 5.0);
    auto sornyak_result = runOptimization(sphere_function, sornyak_adapt, "Сорняки адапт", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << sornyak_result.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_result.best_fitness
              << std::setw(15) << sornyak_result.execution_time_ms
              << std::setw(12) << (sornyak_result.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd(sphere_function, 2, 1000, 0.01, 1e-6, -5.0, 5.0);
    auto gd_result = runOptimization(sphere_function, gd, "Градиентный спуск", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << gd_result.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_result.best_fitness
              << std::setw(15) << gd_result.execution_time_ms
              << std::setw(12) << (gd_result.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadOptimizer nm(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    auto nm_result = runOptimization(sphere_function, nm, "Нелдер-Мид", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << nm_result.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_result.best_fitness
              << std::setw(15) << nm_result.execution_time_ms
              << std::setw(12) << (nm_result.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellOptimizer pow(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    auto pow_result = runOptimization(sphere_function, pow, "Метод Пауэлла", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << pow_result.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_result.best_fitness
              << std::setw(15) << pow_result.execution_time_ms
              << std::setw(12) << (pow_result.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Розенброка (2D)
    std::cout << "\nТестирование на функции Розенброка (2D, минимум в [1,1], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(74, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakWithAdaptiveSpread sornyak_adapt2(rosenbrock_function, 2, 2000, 50, -2.0, 2.0);
    auto sornyak_result2 = runOptimization(rosenbrock_function, sornyak_adapt2, "Сорняки адапт", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << sornyak_result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_result2.best_fitness
              << std::setw(15) << sornyak_result2.execution_time_ms
              << std::setw(12) << (sornyak_result2.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd2(rosenbrock_function, 2, 2000, 0.001, 1e-6, -2.0, 2.0);
    auto gd_result2 = runOptimization(rosenbrock_function, gd2, "Градиентный спуск", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << gd_result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_result2.best_fitness
              << std::setw(15) << gd_result2.execution_time_ms
              << std::setw(12) << (gd_result2.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadOptimizer nm2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    auto nm_result2 = runOptimization(rosenbrock_function, nm2, "Нелдер-Мид", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << nm_result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_result2.best_fitness
              << std::setw(15) << nm_result2.execution_time_ms
              << std::setw(12) << (nm_result2.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellOptimizer pow2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    auto pow_result2 = runOptimization(rosenbrock_function, pow2, "Метод Пауэлла", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << pow_result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_result2.best_fitness
              << std::setw(15) << pow_result2.execution_time_ms
              << std::setw(12) << (pow_result2.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Растригина (2D)
    std::cout << "\nТестирование на функции Растригина (2D, минимум в [0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(74, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakWithAdaptiveSpread sornyak_adapt3(rastrigin_function, 2, 2000, 50, -5.0, 5.0);
    auto sornyak_result3 = runOptimization(rastrigin_function, sornyak_adapt3, "Сорняки адапт", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << sornyak_result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_result3.best_fitness
              << std::setw(15) << sornyak_result3.execution_time_ms
              << std::setw(12) << (sornyak_result3.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd3(rastrigin_function, 2, 2000, 0.01, 1e-6, -5.0, 5.0);
    auto gd_result3 = runOptimization(rastrigin_function, gd3, "Градиентный спуск", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << gd_result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_result3.best_fitness
              << std::setw(15) << gd_result3.execution_time_ms
              << std::setw(12) << (gd_result3.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadOptimizer nm3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    auto nm_result3 = runOptimization(rastrigin_function, nm3, "Нелдер-Мид", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << nm_result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_result3.best_fitness
              << std::setw(15) << nm_result3.execution_time_ms
              << std::setw(12) << (nm_result3.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellOptimizer pow3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    auto pow_result3 = runOptimization(rastrigin_function, pow3, "Метод Пауэлла", 0.0, 1e-6);
    std::cout << std::setw(30) << std::left << pow_result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_result3.best_fitness
              << std::setw(15) << pow_result3.execution_time_ms
              << std::setw(12) << (pow_result3.reached_tolerance ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    std::cout << "Выводы:" << std::endl;
    std::cout << "========" << std::endl;
    std::cout << "- 'Достигнута' указывает, была ли достигнута целевая точность tol=1e-6" << std::endl;
    std::cout << "- Методы с 'Да' в последнем столбце успешно нашли решение с заданной точностью" << std::endl;
    std::cout << "- Время выполнения показывает, сколько миллисекунд потребовалось для завершения" << std::endl;
    std::cout << "- Ошибка показывает значение целевой функции в найденном оптимальном решении" << std::endl;
    
    return 0;
}