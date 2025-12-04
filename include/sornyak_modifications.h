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
            double variation = (dis(gen) - 0.5) * 2.0 * spread_factor;
            child[i] += variation;
            child[i] = std::max(min_value, std::min(max_value, child[i]));
        }
        return child;
    }

    virtual std::vector<double> optimize() {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        for (int iter = 0; iter < max_iterations; iter++) {
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            population = new_population;
            fitness = new_fitness;
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
    
    virtual ~SornyakOptimizer() = default;
    
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
        double el_ratio = 0.1,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, conv_threshold), 
        elitism_ratio(el_ratio) {}

    std::vector<double> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            
            // Update best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Check for convergence
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {  // Require stability over several iterations
                    std::cout << "Converged at iteration " << iter + 1 << " with fitness " << best_fitness << std::endl;
                    break;
                }
            } else {
                convergence_count = 0;  // Reset counter if improvement occurs
            }
            
            previous_best_fitness = best_fitness;
            
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            int num_elites = static_cast<int>(population_size * elitism_ratio);
            if (num_elites < 1) num_elites = 1;
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population_size; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            for (int i = 0; i < num_elites; i++) {
                int elite_idx = fitness_indices[i].second;
                new_population.push_back(population[elite_idx]);
                new_fitness.push_back(fitness[elite_idx]);
            }
            for (int i = num_elites; i < population_size; i++) {
                int parent_idx = static_cast<int>(dis(gen) * population_size);
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                new_population.push_back(new_solution);
                new_fitness.push_back(new_fitness_val);
            }
            population = new_population;
            fitness = new_fitness;
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        std::cout << "Completed " << actual_iterations << " iterations before convergence or reaching max iterations" << std::endl;
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
        double div_threshold = 0.01,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, conv_threshold), 
        diversity_threshold(div_threshold) {}

    std::vector<double> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            
            // Update best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Check for convergence
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {  // Require stability over several iterations
                    std::cout << "Converged at iteration " << iter + 1 << " with fitness " << best_fitness << std::endl;
                    break;
                }
            } else {
                convergence_count = 0;  // Reset counter if improvement occurs
            }
            
            previous_best_fitness = best_fitness;
            
            double diversity = calculatePopulationDiversity(population);
            double base_spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            double adaptive_factor = 1.0;
            
            if (diversity < diversity_threshold) {
                adaptive_factor = 2.0;
            } else if (diversity > diversity_threshold * 10) {
                adaptive_factor = 0.5;
            }
            
            double spread_factor = base_spread_factor * adaptive_factor;
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            population = new_population;
            fitness = new_fitness;
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        std::cout << "Completed " << actual_iterations << " iterations before convergence or reaching max iterations" << std::endl;
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
        int tour_size = 3,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, conv_threshold), 
        tournament_size(tour_size) {}

    std::vector<double> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            
            // Update best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Check for convergence
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {  // Require stability over several iterations
                    std::cout << "Converged at iteration " << iter + 1 << " with fitness " << best_fitness << std::endl;
                    break;
                }
            } else {
                convergence_count = 0;  // Reset counter if improvement occurs
            }
            
            previous_best_fitness = best_fitness;
            
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                int parent_idx = tournamentSelection(population, fitness);
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                if (new_fitness_val < fitness[parent_idx]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[parent_idx]);
                    new_fitness.push_back(fitness[parent_idx]);
                }
            }
            population = new_population;
            fitness = new_fitness;
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        std::cout << "Completed " << actual_iterations << " iterations before convergence or reaching max iterations" << std::endl;
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
        double conv_threshold = 1e-6,
        double pop_conv_threshold = 0.001
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, conv_threshold), 
        convergence_threshold(pop_conv_threshold), 
        min_population_size(std::max(10, pop_size / 4)),
        max_population_size(pop_size * 2) {
        previous_best_fitnesses.reserve(10);
    }

    std::vector<double> optimize() override {
        int current_population_size = population_size;
        std::vector<std::vector<double>> population(current_population_size);
        std::vector<double> fitness(current_population_size);
        
        for (int i = 0; i < current_population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;

        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            
            // Update best solution
            for (int i = 0; i < current_population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Check for convergence
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {  // Require stability over several iterations
                    std::cout << "Converged at iteration " << iter + 1 << " with fitness " << best_fitness << std::endl;
                    break;
                }
            } else {
                convergence_count = 0;  // Reset counter if improvement occurs
            }
            
            previous_best_fitness = best_fitness;
            if (previous_best_fitnesses.size() >= 10) {
                previous_best_fitnesses.erase(previous_best_fitnesses.begin());
            }
            previous_best_fitnesses.push_back(best_fitness);
            if (previous_best_fitnesses.size() >= 10) {
                double convergence_rate = calculateConvergenceRate();
                
                if (convergence_rate < convergence_threshold && current_population_size < max_population_size) {
                    current_population_size = std::min(max_population_size, 
                                                      static_cast<int>(current_population_size * 1.1));
                    adjustPopulationSize(population, fitness, current_population_size);
                } else if (convergence_rate > convergence_threshold * 10 && current_population_size > min_population_size) {
                    current_population_size = std::max(min_population_size, 
                                                      static_cast<int>(current_population_size * 0.9));
                    adjustPopulationSize(population, fitness, current_population_size);
                }
            }
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            std::vector<std::vector<double>> new_population(current_population_size);
            std::vector<double> new_fitness(current_population_size);
            
            for (int i = 0; i < current_population_size; i++) {
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                if (new_fitness_val < fitness[i]) {
                    new_population[i] = new_solution;
                    new_fitness[i] = new_fitness_val;
                } else {
                    new_population[i] = population[i];
                    new_fitness[i] = fitness[i];
                }
            }
            population = new_population;
            fitness = new_fitness;
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
            int additional = new_size - population.size();
            for (int i = 0; i < additional; i++) {
                population.push_back(generateRandomSolution());
                fitness.push_back(objective_function(population.back()));
            }
        } else {
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