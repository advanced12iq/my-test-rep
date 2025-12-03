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

// Модифицированный класс SornyakOptimizer с поддержкой early stopping по tolerance
class SornyakOptimizerWithTolerance : public SornyakOptimizer {
private:
    double tolerance;
    double target_fitness;  // Целевое значение функции (обычно 0 для тестовых функций)

public:
    SornyakOptimizerWithTolerance(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double tol = 1e-6,
        double target_fit = 0.0
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        tolerance(tol), target_fitness(target_fit) {}

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
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Проверка достижения целевой точности
            if (std::abs(best_fitness - target_fitness) < tolerance) {
                std::cout << "Достигнута целевая точность на итерации " << iter << std::endl;
                break;
            }
            
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
    
    // Метод для получения количества итераций, выполненных до достижения цели
    int getIterationsToTolerance(std::function<double(const std::vector<double>&)> func, 
                                double target_fit = 0.0, double tol = 1e-6) {
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Проверка достижения целевой точности
            if (std::abs(best_fitness - target_fit) < tol) {
                return iter + 1;  // Возвращаем количество выполненных итераций
            }
            
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
        
        return max_iterations;  // Если не достигнута точность, вернуть максимальное количество итераций
    }
};

// Структура для хранения результатов сравнения
struct ToleranceComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    int iterations_to_tolerance;
    int max_iterations;
};

// Функция для запуска оптимизации с измерением времени до достижения tolerance
template<typename OptimizerType>
ToleranceComparisonResult runOptimizationWithTolerance(std::function<double(const std::vector<double>&)> func, 
                                                      OptimizerType& optimizer, 
                                                      const std::string& method_name, 
                                                      double target_fitness = 0.0,
                                                      double tolerance = 1e-6,
                                                      int max_iter = 1000) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result = optimizer.optimize();
    double best_fitness = optimizer.getBestFitness(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Для методов с early stopping нужно посчитать количество итераций до достижения цели
    int iterations = max_iter;  // по умолчанию - максимальное количество итераций
    
    // Если это Sornyak метод с поддержкой early stopping
    if (auto* sornyak_ptr = dynamic_cast<SornyakOptimizerWithTolerance*>(&optimizer)) {
        iterations = sornyak_ptr->getIterationsToTolerance(func, target_fitness, tolerance);
    }
    // Для градиентного спуска, Нелдера-Мида и Пауэлла - нужно модифицировать, чтобы отслеживать итерации
    else {
        // Мы не можем получить точное количество итераций для этих методов без модификации,
        // так как они уже встроены в optimize(). Поэтому будем использовать приближение.
        // В реальности, для точного измерения потребуется модифицировать каждый метод.
        iterations = max_iter; // Это будет приближенным значением
    }
    
    ToleranceComparisonResult res;
    res.method_name = method_name;
    res.best_fitness = best_fitness;
    res.best_solution = result;
    res.execution_time_ms = duration.count();
    res.iterations_to_tolerance = iterations;
    res.max_iterations = max_iter;
    
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
              << std::setw(15) << "Итерации" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // Для более точного сравнения, создадим модифицированные версии методов, которые отслеживают количество итераций
    
    // Сорняковый метод с адаптивным распространением - как пример Sornyak метода
    SornyakOptimizerWithTolerance sornyak_adapt(sphere_function, 2, 1000, 30, -5.0, 5.0, 1e-6, 0.0);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result1 = sornyak_adapt.getIterationsToTolerance(sphere_function, 0.0, 1e-6);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << "N/A"
              << std::setw(15) << duration1.count()
              << std::setw(15) << result1 << std::endl;
    
    // Для сравнения, запустим градиентный спуск с tol=1e-6
    GradientDescentOptimizer gd(sphere_function, 2, 1000, 0.01, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result = gd.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness = gd.getBestFitness(gd_result);
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness
              << std::setw(15) << duration2.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Нелдер-Мид с tol=1e-6
    NelderMeadOptimizer nm(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result = nm.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness = nm.getBestFitness(nm_result);
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness
              << std::setw(15) << duration3.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Метод Пауэлла с tol=1e-6
    PowellOptimizer pow(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result = pow.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness = pow.getBestFitness(pow_result);
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness
              << std::setw(15) << duration4.count()
              << std::setw(15) << "N/A" << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Розенброка (2D)
    std::cout << "\nТестирование на функции Розенброка (2D, минимум в [1,1], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(15) << "Итерации" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt2(rosenbrock_function, 2, 2000, 50, -2.0, 2.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto result2 = sornyak_adapt2.getIterationsToTolerance(rosenbrock_function, 0.0, 1e-6);
    end_time = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << "N/A"
              << std::setw(15) << duration5.count()
              << std::setw(15) << result2 << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd2(rosenbrock_function, 2, 2000, 0.001, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result2 = gd2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration6 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness2 = gd2.getBestFitness(gd_result2);
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness2
              << std::setw(15) << duration6.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Нелдер-Мид
    NelderMeadOptimizer nm2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result2 = nm2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration7 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness2 = nm2.getBestFitness(nm_result2);
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness2
              << std::setw(15) << duration7.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Метод Пауэлла
    PowellOptimizer pow2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result2 = pow2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration8 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness2 = pow2.getBestFitness(pow_result2);
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness2
              << std::setw(15) << duration8.count()
              << std::setw(15) << "N/A" << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Растригина (2D)
    std::cout << "\nТестирование на функции Растригина (2D, минимум в [0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(15) << "Итерации" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt3(rastrigin_function, 2, 2000, 50, -5.0, 5.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto result3 = sornyak_adapt3.getIterationsToTolerance(rastrigin_function, 0.0, 1e-6);
    end_time = std::chrono::high_resolution_clock::now();
    auto duration9 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << "N/A"
              << std::setw(15) << duration9.count()
              << std::setw(15) << result3 << std::endl;
    
    // Градиентный спуск
    GradientDescentOptimizer gd3(rastrigin_function, 2, 2000, 0.01, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result3 = gd3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration10 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness3 = gd3.getBestFitness(gd_result3);
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness3
              << std::setw(15) << duration10.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Нелдер-Мид
    NelderMeadOptimizer nm3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result3 = nm3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration11 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness3 = nm3.getBestFitness(nm_result3);
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness3
              << std::setw(15) << duration11.count()
              << std::setw(15) << "N/A" << std::endl;
    
    // Метод Пауэлла
    PowellOptimizer pow3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result3 = pow3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness3 = pow3.getBestFitness(pow_result3);
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness3
              << std::setw(15) << duration12.count()
              << std::setw(15) << "N/A" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Примечание: 'Сорняки адапт' показывает количество итераций до достижения tol=1e-6," << std::endl;
    std::cout << "в то время как другие методы показывают достигнутую ошибку и время выполнения." << std::endl;
    std::cout << "Конвенциональные методы (Градиентный спуск, Нелдер-Мид, Пауэлл) используют tol=1e-6 для early stopping." << std::endl;
    
    return 0;
}