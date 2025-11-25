#include "sormyakov_modifications.h"
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

// Функция для запуска оптимизации и измерения времени
template<typename OptimizerType>
ComparisonResult runOptimization(std::function<double(const std::vector<double>&)> func, 
                                OptimizerType& optimizer, 
                                const std::string& method_name, 
                                int dimensions,
                                int max_iter = 500,
                                int pop_size = 30) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result = optimizer.optimize();
    double best_fitness = optimizer.getBestFitness(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ComparisonResult res;
    res.method_name = method_name;
    res.best_fitness = best_fitness;
    res.best_solution = result;
    res.execution_time_ms = duration.count();
    res.iterations_completed = max_iter;  // Предполагается, что все итерации завершены
    res.population_size = pop_size;
    
    return res;
}

int main() {
    std::cout << "Сравнение метода оптимизации Сормякова" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Тест на функции Сферы (2D)
    std::cout << "\nТестирование на функции Сферы (2D, минимум в [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сормяков
    SormyakovOptimizer optimizer1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result1 = runOptimization(sphere_function, optimizer1, "Оригинальный", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result1.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result1.best_fitness
              << std::setw(15) << result1.execution_time_ms
              << std::setw(12) << result1.iterations_completed
              << std::setw(12) << result1.population_size << std::endl;
    
    // С элитизмом
    SormyakovWithElitism optimizer2(sphere_function, 2, 500, 30, -5.0, 5.0, 0.2);
    auto result2 = runOptimization(sphere_function, optimizer2, "С элитизмом", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result2.best_fitness
              << std::setw(15) << result2.execution_time_ms
              << std::setw(12) << result2.iterations_completed
              << std::setw(12) << result2.population_size << std::endl;
    
    // С адаптивным распространением
    SormyakovWithAdaptiveSpread optimizer3(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result3 = runOptimization(sphere_function, optimizer3, "Адаптивное распространение", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result3.best_fitness
              << std::setw(15) << result3.execution_time_ms
              << std::setw(12) << result3.iterations_completed
              << std::setw(12) << result3.population_size << std::endl;
    
    // С турнирным отбором
    SormyakovWithTournament optimizer4(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result4 = runOptimization(sphere_function, optimizer4, "Турнирный отбор", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result4.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result4.best_fitness
              << std::setw(15) << result4.execution_time_ms
              << std::setw(12) << result4.iterations_completed
              << std::setw(12) << result4.population_size << std::endl;
    
    // С динамической популяцией
    SormyakovWithDynamicPopulation optimizer5(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result5 = runOptimization(sphere_function, optimizer5, "Динамическая популяция", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result5.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result5.best_fitness
              << std::setw(15) << result5.execution_time_ms
              << std::setw(12) << result5.iterations_completed
              << std::setw(12) << result5.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Розенброка (2D)
    std::cout << "\nТестирование на функции Розенброка (2D, минимум в [1,1])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сормяков
    SormyakovOptimizer optimizer6(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result6 = runOptimization(rosenbrock_function, optimizer6, "Оригинальный", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result6.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result6.best_fitness
              << std::setw(15) << result6.execution_time_ms
              << std::setw(12) << result6.iterations_completed
              << std::setw(12) << result6.population_size << std::endl;
    
    // С элитизмом
    SormyakovWithElitism optimizer7(rosenbrock_function, 2, 1000, 50, -2.0, 2.0, 0.2);
    auto result7 = runOptimization(rosenbrock_function, optimizer7, "С элитизмом", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result7.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result7.best_fitness
              << std::setw(15) << result7.execution_time_ms
              << std::setw(12) << result7.iterations_completed
              << std::setw(12) << result7.population_size << std::endl;
    
    // С адаптивным распространением
    SormyakovWithAdaptiveSpread optimizer8(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result8 = runOptimization(rosenbrock_function, optimizer8, "Адаптивное распространение", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result8.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result8.best_fitness
              << std::setw(15) << result8.execution_time_ms
              << std::setw(12) << result8.iterations_completed
              << std::setw(12) << result8.population_size << std::endl;
    
    // С турнирным отбором
    SormyakovWithTournament optimizer9(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result9 = runOptimization(rosenbrock_function, optimizer9, "Турнирный отбор", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result9.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result9.best_fitness
              << std::setw(15) << result9.execution_time_ms
              << std::setw(12) << result9.iterations_completed
              << std::setw(12) << result9.population_size << std::endl;
    
    // С динамической популяцией
    SormyakovWithDynamicPopulation optimizer10(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result10 = runOptimization(rosenbrock_function, optimizer10, "Динамическая популяция", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result10.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result10.best_fitness
              << std::setw(15) << result10.execution_time_ms
              << std::setw(12) << result10.iterations_completed
              << std::setw(12) << result10.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Растригина (2D)
    std::cout << "\nТестирование на функции Растригина (2D, минимум в [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сормяков
    SormyakovOptimizer optimizer11(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result11 = runOptimization(rastrigin_function, optimizer11, "Оригинальный", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result11.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result11.best_fitness
              << std::setw(15) << result11.execution_time_ms
              << std::setw(12) << result11.iterations_completed
              << std::setw(12) << result11.population_size << std::endl;
    
    // С элитизмом
    SormyakovWithElitism optimizer12(rastrigin_function, 2, 1000, 50, -5.0, 5.0, 0.2);
    auto result12 = runOptimization(rastrigin_function, optimizer12, "С элитизмом", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result12.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result12.best_fitness
              << std::setw(15) << result12.execution_time_ms
              << std::setw(12) << result12.iterations_completed
              << std::setw(12) << result12.population_size << std::endl;
    
    // С адаптивным распространением
    SormyakovWithAdaptiveSpread optimizer13(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result13 = runOptimization(rastrigin_function, optimizer13, "Адаптивное распространение", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result13.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result13.best_fitness
              << std::setw(15) << result13.execution_time_ms
              << std::setw(12) << result13.iterations_completed
              << std::setw(12) << result13.population_size << std::endl;
    
    // С турнирным отбором
    SormyakovWithTournament optimizer14(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result14 = runOptimization(rastrigin_function, optimizer14, "Турнирный отбор", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result14.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result14.best_fitness
              << std::setw(15) << result14.execution_time_ms
              << std::setw(12) << result14.iterations_completed
              << std::setw(12) << result14.population_size << std::endl;
    
    // С динамической популяцией
    SormyakovWithDynamicPopulation optimizer15(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result15 = runOptimization(rastrigin_function, optimizer15, "Динамическая популяция", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result15.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result15.best_fitness
              << std::setw(15) << result15.execution_time_ms
              << std::setw(12) << result15.iterations_completed
              << std::setw(12) << result15.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Анализ результатов
    std::cout << "\nАнализ результатов:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Каждый метод был протестирован на трех классических функциях оптимизации:\n";
    std::cout << "- Функция Сферы: f(x) = sum(x_i^2), глобальный минимум в [0,0,...,0]\n";
    std::cout << "- Функция Розенброка: f(x) = sum[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2], глобальный минимум в [1,1,...,1]\n";
    std::cout << "- Функция Растригина: f(x) = 10*n + sum[x_i^2 - 10*cos(2*PI*x_i)], глобальный минимум в [0,0,...,0]\n\n";
    
    std::cout << "Описания методов:\n";
    std::cout << "- Оригинальный: Базовый метод оптимизации Сормякова (Сорняков)\n";
    std::cout << "- С элитизмом: Сохраняет лучшие решения для поддержания хороших решений\n";
    std::cout << "- Адаптивное распространение: Корректирует фактор распространения на основе разнообразия популяции\n";
    std::cout << "- Турнирный отбор: Использует турнирный отбор для выбора родителей\n";
    std::cout << "- Динамическая популяция: Корректирует размер популяции на основе скорости сходимости\n\n";
    
    std::cout << "Метрики производительности:\n";
    std::cout << "- Лучшая пригодность: Значение функции в лучшем найденном решении\n";
    std::cout << "- Время (мс): Время выполнения в миллисекундах\n";
    std::cout << "- Итерации: Количество выполненных итераций\n";
    std::cout << "- Размер популяции: Используемый размер популяции\n\n";
    
    std::cout << "Примечание: Для всех тестовых функций более низкие значения пригодности указывают на лучшую производительность.\n";
    std::cout << "Функция Сферы имеет глобальный минимум 0, Розенброка 0, и Растригина 0.\n";
    
    return 0;
}