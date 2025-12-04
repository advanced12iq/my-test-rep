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
    std::cout << "Комплексное сравнение методов оптимизации" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Тест на функции Сферы (2D)
    std::cout << "\nТестирование на функции Сферы (2D, минимум в [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации
    SornyakOptimizer sorm1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result1 = runOptimization(sphere_function, sorm1, "Сорняки ориг", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result1.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result1.best_fitness
              << std::setw(15) << result1.execution_time_ms
              << std::setw(12) << result1.iterations_completed
              << std::setw(12) << result1.population_size << std::endl;
    
    // Сорняковый метод оптимизации с элитизмом
    SornyakWithElitism sorm2(sphere_function, 2, 500, 30, -5.0, 5.0, 0.2);
    auto result2 = runOptimization(sphere_function, sorm2, "Сорняки элит", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result2.best_fitness
              << std::setw(15) << result2.execution_time_ms
              << std::setw(12) << result2.iterations_completed
              << std::setw(12) << result2.population_size << std::endl;
    
    // Сорняковый метод оптимизации с адаптивным распространением
    SornyakWithAdaptiveSpread sorm3(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result3 = runOptimization(sphere_function, sorm3, "Сорняки адапт", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result3.best_fitness
              << std::setw(15) << result3.execution_time_ms
              << std::setw(12) << result3.iterations_completed
              << std::setw(12) << result3.population_size << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd1(sphere_function, 2, 500, 0.01, 1e-6, -5.0, 5.0);
    auto result4 = runOptimization(sphere_function, gd1, "Градиентный спуск", 2, 500, 1);
    std::cout << std::setw(30) << std::left << result4.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result4.best_fitness
              << std::setw(15) << result4.execution_time_ms
              << std::setw(12) << result4.iterations_completed
              << std::setw(12) << result4.population_size << std::endl;
    
    // Симплекс Нелдера-Мида
    NelderMeadOptimizer nm1(sphere_function, 2, 500, 1e-6, -5.0, 5.0);
    auto result5 = runOptimization(sphere_function, nm1, "Нелдер-Мид", 2, 500, 3);
    std::cout << std::setw(30) << std::left << result5.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result5.best_fitness
              << std::setw(15) << result5.execution_time_ms
              << std::setw(12) << result5.iterations_completed
              << std::setw(12) << result5.population_size << std::endl;
    

    
    // Случайный поиск
    RandomSearchOptimizer rs1(sphere_function, 2, 500, -5.0, 5.0);
    auto result7 = runOptimization(sphere_function, rs1, "Случайный поиск", 2, 500, 1);
    std::cout << std::setw(30) << std::left << result7.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result7.best_fitness
              << std::setw(15) << result7.execution_time_ms
              << std::setw(12) << result7.iterations_completed
              << std::setw(12) << result7.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Розенброка (2D)
    std::cout << "\nТестирование на функции Розенброка (2D, минимум в [1,1])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации
    SornyakOptimizer sorm4(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result8 = runOptimization(rosenbrock_function, sorm4, "Сорняки ориг", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result8.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result8.best_fitness
              << std::setw(15) << result8.execution_time_ms
              << std::setw(12) << result8.iterations_completed
              << std::setw(12) << result8.population_size << std::endl;
    
    // Сорняковый метод оптимизации с элитизмом
    SornyakWithElitism sorm5(rosenbrock_function, 2, 1000, 50, -2.0, 2.0, 0.2);
    auto result9 = runOptimization(rosenbrock_function, sorm5, "Сорняки элит", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result9.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result9.best_fitness
              << std::setw(15) << result9.execution_time_ms
              << std::setw(12) << result9.iterations_completed
              << std::setw(12) << result9.population_size << std::endl;
    
    // Сорняковый метод оптимизации с адаптивным распространением
    SornyakWithAdaptiveSpread sorm6(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result10 = runOptimization(rosenbrock_function, sorm6, "Сорняки адапт", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result10.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result10.best_fitness
              << std::setw(15) << result10.execution_time_ms
              << std::setw(12) << result10.iterations_completed
              << std::setw(12) << result10.population_size << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd2(rosenbrock_function, 2, 1000, 0.001, 1e-6, -2.0, 2.0);
    auto result11 = runOptimization(rosenbrock_function, gd2, "Градиентный спуск", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result11.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result11.best_fitness
              << std::setw(15) << result11.execution_time_ms
              << std::setw(12) << result11.iterations_completed
              << std::setw(12) << result11.population_size << std::endl;
    
    // Симплекс Нелдера-Мида
    NelderMeadOptimizer nm2(rosenbrock_function, 2, 1000, 1e-6, -2.0, 2.0);
    auto result12 = runOptimization(rosenbrock_function, nm2, "Нелдер-Мид", 2, 1000, 3);
    std::cout << std::setw(30) << std::left << result12.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result12.best_fitness
              << std::setw(15) << result12.execution_time_ms
              << std::setw(12) << result12.iterations_completed
              << std::setw(12) << result12.population_size << std::endl;
    

    
    // Случайный поиск
    RandomSearchOptimizer rs2(rosenbrock_function, 2, 1000, -2.0, 2.0);
    auto result14 = runOptimization(rosenbrock_function, rs2, "Случайный поиск", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result14.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result14.best_fitness
              << std::setw(15) << result14.execution_time_ms
              << std::setw(12) << result14.iterations_completed
              << std::setw(12) << result14.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Растригина (2D)
    std::cout << "\nТестирование на функции Растригина (2D, минимум в [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Размер популяции" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации
    SornyakOptimizer sorm7(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result15 = runOptimization(rastrigin_function, sorm7, "Сорняки ориг", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result15.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result15.best_fitness
              << std::setw(15) << result15.execution_time_ms
              << std::setw(12) << result15.iterations_completed
              << std::setw(12) << result15.population_size << std::endl;
    
    // Сорняковый метод оптимизации с элитизмом
    SornyakWithElitism sorm8(rastrigin_function, 2, 1000, 50, -5.0, 5.0, 0.2);
    auto result16 = runOptimization(rastrigin_function, sorm8, "Сорняки элит", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result16.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result16.best_fitness
              << std::setw(15) << result16.execution_time_ms
              << std::setw(12) << result16.iterations_completed
              << std::setw(12) << result16.population_size << std::endl;
    
    // Сорняковый метод оптимизации с адаптивным распространением
    SornyakWithAdaptiveSpread sorm9(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result17 = runOptimization(rastrigin_function, sorm9, "Сорняки адапт", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result17.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result17.best_fitness
              << std::setw(15) << result17.execution_time_ms
              << std::setw(12) << result17.iterations_completed
              << std::setw(12) << result17.population_size << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd3(rastrigin_function, 2, 1000, 0.01, 1e-6, -5.0, 5.0);
    auto result18 = runOptimization(rastrigin_function, gd3, "Градиентный спуск", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result18.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result18.best_fitness
              << std::setw(15) << result18.execution_time_ms
              << std::setw(12) << result18.iterations_completed
              << std::setw(12) << result18.population_size << std::endl;
    
    // Симплекс Нелдера-Мида
    NelderMeadOptimizer nm3(rastrigin_function, 2, 1000, 1e-6, -5.0, 5.0);
    auto result19 = runOptimization(rastrigin_function, nm3, "Нелдер-Мид", 2, 1000, 3);
    std::cout << std::setw(30) << std::left << result19.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result19.best_fitness
              << std::setw(15) << result19.execution_time_ms
              << std::setw(12) << result19.iterations_completed
              << std::setw(12) << result19.population_size << std::endl;
    

    
    // Случайный поиск
    RandomSearchOptimizer rs3(rastrigin_function, 2, 1000, -5.0, 5.0);
    auto result21 = runOptimization(rastrigin_function, rs3, "Случайный поиск", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result21.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result21.best_fitness
              << std::setw(15) << result21.execution_time_ms
              << std::setw(12) << result21.iterations_completed
              << std::setw(12) << result21.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Анализ результатов
    std::cout << "\nАнализ результатов:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Сравнение методов оптимизации Сорняковый метод оптимизацииа (Сорняков) с традиционными методами\n\n";
    
    std::cout << "Методы Сорняковый метод оптимизацииа:\n";
    std::cout << "- Сорняковый метод оптимизации ориг: Базовый метод оптимизации Сорняковый метод оптимизацииа (Сорняков)\n";
    std::cout << "- Сорняковый метод оптимизации элит: Сорняковый метод оптимизации с элитизмом (сохраняет лучшие решения)\n";
    std::cout << "- Сорняковый метод оптимизации адапт: Сорняковый метод оптимизации с адаптивным фактором распространения\n\n";
    
    std::cout << "Традиционные методы:\n";
    std::cout << "- Градиентный спуск: Итерационная оптимизация первого порядка с использованием численных градиентов\n";
    std::cout << "- Нелдер-Мид: Метод прямого поиска с использованием операций симплекса (отражение, расширение, сжатие)\n";

    std::cout << "- Случайный поиск: Простой метод, который случайным образом отбирает точки из пространства поиска\n\n";
    
    std::cout << "Метрики производительности:\n";
    std::cout << "- Ошибка: Значение функции в лучшем найденном решении\n";
    std::cout << "- Время (мс): Время выполнения в миллисекундах\n";
    std::cout << "- Итерации: Количество выполненных итераций\n";
    std::cout << "- Размер популяции: Используемый размер популяции (для методов на основе популяции)\n\n";
    
    std::cout << "Примечание: Для всех тестовых функций более низкие значения пригодности указывают на лучшую производительность.\n";
    std::cout << "Функция Сферы имеет глобальный минимум 0, Розенброка 0, и Растригина 0.\n";
    std::cout << "\nКлючевые наблюдения:\n";
    std::cout << "- Градиентный спуск быстр, но может застрять в локальных минимумах\n";
    std::cout << "- Нелдер-Мид устойчив для задач с низкой размерностью\n";

    std::cout << "- Случайный поиск обеспечивает базовый уровень, но в целом неэффективен\n";
    std::cout << "- Методы Сорняковый метод оптимизацииа вдохновлены природой и хороши для глобальной оптимизации\n";
    
    return 0;
}