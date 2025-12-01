#ifndef SORNYAK_MODIFICATIONS_H
#define SORNYAK_MODIFICATIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <chrono>
#include <algorithm>

/**
 * @brief Базовый сорняковый метод оптимизацииова (оригинальный)
 */
class SornyakOptimizer {
protected:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    int population_size;
    double min_value;
    double max_value;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    SornyakOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        population_size(pop_size), min_value(min_val), max_value(max_val), 
        gen(rd()), dis(0.0, 1.0) {}

    std::vector<double> generateRandomSolution() {
        std::vector<double> solution(dimension);
        for (int i = 0; i < dimension; i++) {
            solution[i] = min_value + (max_value - min_value) * dis(gen);
        }
        return solution;
    }

    std::vector<double> spreadSolution(const std::vector<double>& parent, double spread_factor) {
        std::vector<double> child = parent;
        for (int i = 0; i < dimension; i++) {
            // Добавить случайное изменение для имитации распространения
            double variation = (dis(gen) - 0.5) * 2.0 * spread_factor;
            child[i] += variation;
            
            // Оставить в пределах границ
            child[i] = std::max(min_value, std::min(max_value, child[i]));
        }
        return child;
    }

    virtual std::vector<double> optimize() {
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
    
    virtual ~SornyakOptimizer() = default;  // Виртуальный деструктор для правильного полиморфного уничтожения
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
    
    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

/**
 * @brief Модифицированный сорняковый метод оптимизацииов с элитизмом
 * Сохраняет лучшие решения из предыдущего поколения для сохранения хороших решений
 */
class SornyakWithElitism : public SornyakOptimizer {
protected:
    double elitism_ratio;

public:
    SornyakWithElitism(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double el_ratio = 0.1  // 10% популяции сохраняется
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        elitism_ratio(el_ratio) {}

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

/**
 * @brief Модифицированный сорняковый метод оптимизацииов с адаптивным фактором распространения
 * Использует адаптивный фактор распространения, который изменяется на основе разнообразия популяции
 */
class SornyakWithAdaptiveSpread : public SornyakOptimizer {
protected:
    double diversity_threshold;

public:
    SornyakWithAdaptiveSpread(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double div_threshold = 0.01  // порог для корректировки разнообразия
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        diversity_threshold(div_threshold) {}

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
            
            // Рассчитать разнообразие популяции
            double diversity = calculatePopulationDiversity(population);
            
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
    
private:
    double calculatePopulationDiversity(const std::vector<std::vector<double>>& population) {
        if (population.size() < 2) return 0.0;
        
        double total_distance = 0.0;
        int count = 0;
        
        for (size_t i = 0; i < population.size(); i++) {
            for (size_t j = i + 1; j < population.size(); j++) {
                double distance = 0.0;
                for (int k = 0; k < dimension; k++) {
                    double diff = population[i][k] - population[j][k];
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
                total_distance += distance;
                count++;
            }
        }
        
        return count > 0 ? total_distance / count : 0.0;
    }
    
public:
    double getPopulationDiversity(const std::vector<std::vector<double>>& population) {
        return calculatePopulationDiversity(population);
    }
};

/**
 * @brief Модифицированный сорняковый метод оптимизацииов с турнирным отбором
 * Использует турнирный отбор для выбора решений, от которых распространяться
 */
class SornyakWithTournament : public SornyakOptimizer {
private:
    int tournament_size;

public:
    SornyakWithTournament(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int tour_size = 3  // размер турнира для отбора
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        tournament_size(tour_size) {}

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
            
            // Рассчитать фактор распространения на основе итерации (уменьшается со временем)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Генерировать новые решения путем распространения от существующих с использованием турнирного отбора
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Выбрать родителя с использованием турнирного отбора
                int parent_idx = tournamentSelection(population, fitness);
                
                // Создать новое решение путем распространения от выбранного родителя
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Сохранить лучшее из родителя и потомка
                if (new_fitness_val < fitness[parent_idx]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[parent_idx]);
                    new_fitness.push_back(fitness[parent_idx]);
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
    
private:
    int tournamentSelection(const std::vector<std::vector<double>>& population, 
                           const std::vector<double>& fitness) {
        int best_idx = static_cast<int>(dis(gen) * population.size());
        
        for (int i = 1; i < tournament_size; i++) {
            int candidate_idx = static_cast<int>(dis(gen) * population.size());
            if (fitness[candidate_idx] < fitness[best_idx]) {
                best_idx = candidate_idx;
            }
        }
        
        return best_idx;
    }
};

/**
 * @brief Модифицированный сорняковый метод оптимизацииов с динамическим размером популяции
 * Адаптирует размер популяции динамически на основе скорости сходимости
 */
class SornyakWithDynamicPopulation : public SornyakOptimizer {
private:
    double convergence_threshold;
    int min_population_size;
    int max_population_size;
    std::vector<double> previous_best_fitnesses;

public:
    SornyakWithDynamicPopulation(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double conv_threshold = 0.001
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        convergence_threshold(conv_threshold), 
        min_population_size(std::max(10, pop_size / 4)),
        max_population_size(pop_size * 2) {
        previous_best_fitnesses.reserve(10);  // Зарезервировать место для последних 10 значений пригодности
    }

    std::vector<double> optimize() override {
        int current_population_size = population_size;
        
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(current_population_size);
        std::vector<double> fitness(current_population_size);
        
        for (int i = 0; i < current_population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < current_population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Сохранить лучшую пригодность для анализа сходимости
            if (previous_best_fitnesses.size() >= 10) {
                previous_best_fitnesses.erase(previous_best_fitnesses.begin());
            }
            previous_best_fitnesses.push_back(best_fitness);
            
            // Скорректировать размер популяции на основе сходимости
            if (previous_best_fitnesses.size() >= 10) {
                double convergence_rate = calculateConvergenceRate();
                
                if (convergence_rate < convergence_threshold && current_population_size < max_population_size) {
                    // Медленная сходимость - увеличить размер популяции для большего исследования
                    current_population_size = std::min(max_population_size, 
                                                      static_cast<int>(current_population_size * 1.1));
                    adjustPopulationSize(population, fitness, current_population_size);
                } else if (convergence_rate > convergence_threshold * 10 && current_population_size > min_population_size) {
                    // Быстрая сходимость - уменьшить размер популяции для экономии вычислений
                    current_population_size = std::max(min_population_size, 
                                                      static_cast<int>(current_population_size * 0.9));
                    adjustPopulationSize(population, fitness, current_population_size);
                }
            }
            
            // Рассчитать фактор распространения на основе итерации (уменьшается со временем)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Генерировать новые решения путем распространения от существующих
            std::vector<std::vector<double>> new_population(current_population_size);
            std::vector<double> new_fitness(current_population_size);
            
            for (int i = 0; i < current_population_size; i++) {
                // Каждое решение создает новое путем распространения
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Сохранить лучшее из родителя и потомка
                if (new_fitness_val < fitness[i]) {
                    new_population[i] = new_solution;
                    new_fitness[i] = new_fitness_val;
                } else {
                    new_population[i] = population[i];
                    new_fitness[i] = fitness[i];
                }
            }
            
            // Обновить популяцию
            population = new_population;
            fitness = new_fitness;
            
            // Иногда добавлять совершенно новые случайные решения для поддержания разнообразия
            if (iter % 100 == 0) {
                for (int i = 0; i < current_population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * current_population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        return best_solution;
    }
    
private:
    double calculateConvergenceRate() {
        if (previous_best_fitnesses.size() < 2) return 0.0;
        
        double diff = std::abs(previous_best_fitnesses.back() - previous_best_fitnesses.front());
        return diff / previous_best_fitnesses.size();
    }
    
    void adjustPopulationSize(std::vector<std::vector<double>>& population,
                              std::vector<double>& fitness,
                              int new_size) {
        if (new_size == static_cast<int>(population.size())) return;
        
        if (new_size > static_cast<int>(population.size())) {
            // Увеличить размер популяции, добавив случайные решения
            int additional = new_size - population.size();
            for (int i = 0; i < additional; i++) {
                population.push_back(generateRandomSolution());
                fitness.push_back(objective_function(population.back()));
            }
        } else {
            // Уменьшить размер популяции, сохранив лучшие решения
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population.size(); i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < new_size; i++) {
                int idx = fitness_indices[i].second;
                new_population.push_back(population[idx]);
                new_fitness.push_back(fitness[idx]);
            }
            
            population = new_population;
            fitness = new_fitness;
        }
    }
};

/**
 * @brief Структура для хранения результатов сравнения
 */
struct ComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    int iterations_completed;
    int population_size;
};

#endif // SORNYAK_MODIFICATIONS_H