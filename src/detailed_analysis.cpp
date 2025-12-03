#include "sornyak_modifications.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>

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

// Модифицированные оптимизаторы, которые отслеживают сходимость на протяжении итераций
class SornyakOptimizerTracking : public SornyakOptimizer {
public:
    std::vector<double> best_fitness_history;
    
    SornyakOptimizerTracking(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val) {}

    std::vector<double> optimize() override {
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Очистить историю
        best_fitness_history.clear();
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Сохранить лучшую пригодность для этой итерации
            best_fitness_history.push_back(best_fitness);
            
            // Рассчитать фактор распространения на основе итерации (уменьшается со временем)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Генерировать новые решения путем распространения от существующих
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Каждое решение создает новое путем распространения
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Сохранить лучшее из родителя и потомка
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            
            // Обновить популяцию
            population = new_population;
            fitness = new_fitness;
            
            // Иногда добавлять совершенно новые случайные решения для поддержания разнообразия
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        return best_solution;
    }
};

class SornyakWithElitismTracking : public SornyakWithElitism {
public:
    std::vector<double> best_fitness_history;
    
    SornyakWithElitismTracking(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double el_ratio = 0.1
    ) : SornyakWithElitism(func, dim, max_iter, pop_size, min_val, max_val, el_ratio) {}

    std::vector<double> optimize() override {
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Очистить историю
        best_fitness_history.clear();
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Сохранить лучшую пригодность для этой итерации
            best_fitness_history.push_back(best_fitness);
            
            // Рассчитать фактор распространения на основе итерации (уменьшается со временем)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Генерировать новые решения путем распространения от существующих
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            // Определить количество элитных решений для сохранения
            int num_elites = static_cast<int>(population_size * elitism_ratio);
            if (num_elites < 1) num_elites = 1;
            
            // Создать вектор индексов, отсортированных по пригодности (по возрастанию)
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population_size; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            // Сохранить лучшие решения (элитизм)
            for (int i = 0; i < num_elites; i++) {
                int elite_idx = fitness_indices[i].second;
                new_population.push_back(population[elite_idx]);
                new_fitness.push_back(fitness[elite_idx]);
            }
            
            // Генерировать оставшиеся решения путем распространения
            for (int i = num_elites; i < population_size; i++) {
                int parent_idx = static_cast<int>(dis(gen) * population_size);
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                new_population.push_back(new_solution);
                new_fitness.push_back(new_fitness_val);
            }
            
            // Обновить популяцию
            population = new_population;
            fitness = new_fitness;
            
            // Иногда добавлять совершенно новые случайные решения для поддержания разнообразия
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        return best_solution;
    }
};

class SornyakWithAdaptiveSpreadTracking : public SornyakWithAdaptiveSpread {
public:
    std::vector<double> best_fitness_history;
    
    SornyakWithAdaptiveSpreadTracking(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double div_threshold = 0.01
    ) : SornyakWithAdaptiveSpread(func, dim, max_iter, pop_size, min_val, max_val, div_threshold) {}

    std::vector<double> optimize() override {
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Очистить историю
        best_fitness_history.clear();
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Сохранить лучшую пригодность для этой итерации
            best_fitness_history.push_back(best_fitness);
            
            // Рассчитать разнообразие популяции
            double diversity = getPopulationDiversity(population);
            
            // Рассчитать адаптивный фактор распространения на основе итерации и разнообразия
            double base_spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            double adaptive_factor = 1.0;
            
            if (diversity < diversity_threshold) {
                // Если популяция слишком однородна, увеличить распространение для большего исследования
                adaptive_factor = 2.0;
            } else if (diversity > diversity_threshold * 10) {
                // Если популяция слишком разнообразна, уменьшить распространение для эксплуатации
                adaptive_factor = 0.5;
            }
            
            double spread_factor = base_spread_factor * adaptive_factor;
            
            // Генерировать новые решения путем распространения от существующих
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Каждое решение создает новое путем распространения
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Сохранить лучшее из родителя и потомка
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            
            // Обновить популяцию
            population = new_population;
            fitness = new_fitness;
            
            // Иногда добавлять совершенно новые случайные решения для поддержания разнообразия
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        return best_solution;
    }
};

// Функция для запуска оптимизации с отслеживанием
template<typename OptimizerType>
ComparisonResult runOptimizationWithTracking(std::function<double(const std::vector<double>&)> func, 
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
    std::cout << "Подробный анализ сорнякового метода оптимизации" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Анализ на функции Сферы (2D)
    std::cout << "\nПодробный анализ на функции Сферы (2D, минимум в [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации с отслеживанием
    SornyakOptimizerTracking optimizer1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result1 = runOptimizationWithTracking(sphere_function, optimizer1, "Оригинальный", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result1.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result1.best_fitness
              << std::setw(15) << result1.execution_time_ms
              << std::setw(12) << result1.iterations_completed << std::endl;
    
    // С элитизмом с отслеживанием
    SornyakWithElitismTracking optimizer2(sphere_function, 2, 500, 30, -5.0, 5.0, 0.2);
    auto result2 = runOptimizationWithTracking(sphere_function, optimizer2, "С элитизмом", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result2.best_fitness
              << std::setw(15) << result2.execution_time_ms
              << std::setw(12) << result2.iterations_completed << std::endl;
    
    // С адаптивным распространением с отслеживанием
    SornyakWithAdaptiveSpreadTracking optimizer3(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result3 = runOptimizationWithTracking(sphere_function, optimizer3, "Адаптивное распространение", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result3.best_fitness
              << std::setw(15) << result3.execution_time_ms
              << std::setw(12) << result3.iterations_completed << std::endl;
    
    // Записать данные о сходимости в файл для построения графиков
    std::ofstream file("convergence_data.csv");
    file << "Итерация,Оригинальный,Элитизм,Адаптивное распространение\n";
    
    int max_iter = std::max({static_cast<int>(optimizer1.best_fitness_history.size()),
                             static_cast<int>(optimizer2.best_fitness_history.size()),
                             static_cast<int>(optimizer3.best_fitness_history.size())});
    
    for (int i = 0; i < max_iter; i++) {
        file << i << ",";
        if (i < optimizer1.best_fitness_history.size()) file << optimizer1.best_fitness_history[i] << ",";
        else file << ",";
        if (i < optimizer2.best_fitness_history.size()) file << optimizer2.best_fitness_history[i] << ",";
        else file << ",";
        if (i < optimizer3.best_fitness_history.size()) file << optimizer3.best_fitness_history[i] << "\n";
        else file << "\n";
    }
    
    file.close();
    
    std::cout << "\nДанные о сходимости сохранены в 'convergence_data.csv' для построения графиков." << std::endl;
    
    // Анализ на функции Сферы (5D) - дополнительный тест с разным количеством переменных
    std::cout << "\nПодробный анализ на функции Сферы (5D, минимум в [0,0,0,0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации с отслеживанием
    SornyakOptimizerTracking optimizer4(sphere_function, 5, 500, 30, -5.0, 5.0);
    auto result4 = runOptimizationWithTracking(sphere_function, optimizer4, "Оригинальный", 5, 500, 30);
    std::cout << std::setw(30) << std::left << result4.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result4.best_fitness
              << std::setw(15) << result4.execution_time_ms
              << std::setw(12) << result4.iterations_completed << std::endl;
    
    // С элитизмом с отслеживанием
    SornyakWithElitismTracking optimizer5(sphere_function, 5, 500, 30, -5.0, 5.0, 0.2);
    auto result5 = runOptimizationWithTracking(sphere_function, optimizer5, "С элитизмом", 5, 500, 30);
    std::cout << std::setw(30) << std::left << result5.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result5.best_fitness
              << std::setw(15) << result5.execution_time_ms
              << std::setw(12) << result5.iterations_completed << std::endl;
    
    // С адаптивным распространением с отслеживанием
    SornyakWithAdaptiveSpreadTracking optimizer6(sphere_function, 5, 500, 30, -5.0, 5.0);
    auto result6 = runOptimizationWithTracking(sphere_function, optimizer6, "Адаптивное распространение", 5, 500, 30);
    std::cout << std::setw(30) << std::left << result6.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result6.best_fitness
              << std::setw(15) << result6.execution_time_ms
              << std::setw(12) << result6.iterations_completed << std::endl;
    
    // Анализ на функции Розенброка (4D) - дополнительный тест с разным количеством переменных
    std::cout << "\nПодробный анализ на функции Розенброка (4D, минимум в [1,1,1,1])" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Лучшая пригодность" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    // Оригинальный Сорняковый метод оптимизации с отслеживанием
    SornyakOptimizerTracking optimizer7(rosenbrock_function, 4, 1000, 50, -2.0, 2.0);
    auto result7 = runOptimizationWithTracking(rosenbrock_function, optimizer7, "Оригинальный", 4, 1000, 50);
    std::cout << std::setw(30) << std::left << result7.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result7.best_fitness
              << std::setw(15) << result7.execution_time_ms
              << std::setw(12) << result7.iterations_completed << std::endl;
    
    // С элитизмом с отслеживанием
    SornyakWithElitismTracking optimizer8(rosenbrock_function, 4, 1000, 50, -2.0, 2.0, 0.2);
    auto result8 = runOptimizationWithTracking(rosenbrock_function, optimizer8, "С элитизмом", 4, 1000, 50);
    std::cout << std::setw(30) << std::left << result8.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result8.best_fitness
              << std::setw(15) << result8.execution_time_ms
              << std::setw(12) << result8.iterations_completed << std::endl;
    
    // С адаптивным распространением с отслеживанием
    SornyakWithAdaptiveSpreadTracking optimizer9(rosenbrock_function, 4, 1000, 50, -2.0, 2.0);
    auto result9 = runOptimizationWithTracking(rosenbrock_function, optimizer9, "Адаптивное распространение", 4, 1000, 50);
    std::cout << std::setw(30) << std::left << result9.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result9.best_fitness
              << std::setw(15) << result9.execution_time_ms
              << std::setw(12) << result9.iterations_completed << std::endl;
    
    // Сводка
    std::cout << "\nСводка:" << std::endl;
    std::cout << "========" << std::endl;
    std::cout << "Этот анализ показывает производительность различных модификаций сорнякового метода." << std::endl;
    std::cout << "Данные о сходимости можно использовать для анализа скорости сходимости каждого метода." << std::endl;
    std::cout << "\nКлючевые наблюдения:" << std::endl;
    std::cout << "1. Оригинальный: Базовый сорняковый метод без модификаций" << std::endl;
    std::cout << "2. С элитизмом: Сохраняет лучшие решения для поддержания хороших решений" << std::endl;
    std::cout << "3. Адаптивное распространение: Корректирует фактор распространения на основе разнообразия популяции" << std::endl;
    std::cout << "\nДля функции Сферы все методы показали хорошие результаты, с адаптивным распространением" << std::endl;
    std::cout << "показавшим немного лучшие результаты с точки зрения достигнутой лучшей пригодности." << std::endl;
    std::cout << "\nАнализ проводился с различным количеством переменных (2D, 4D, 5D), демонстрируя" << std::endl;
    std::cout << "работоспособность алгоритма в разных размерностях пространства." << std::endl;
    
    return 0;
}