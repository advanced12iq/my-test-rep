#ifndef SORMYAKOV_MODIFICATIONS_H
#define SORMYAKOV_MODIFICATIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <chrono>
#include <algorithm>

/**
 * @brief Base Sormyakov Optimization Method (Original)
 */
class SormyakovOptimizer {
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
    SormyakovOptimizer(
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
            // Add random variation to simulate spreading
            double variation = (dis(gen) - 0.5) * 2.0 * spread_factor;
            child[i] += variation;
            
            // Keep within bounds
            child[i] = std::max(min_value, std::min(max_value, child[i]));
        }
        return child;
    }

    virtual std::vector<double> optimize() {
        // Initialize population with random solutions
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Iteratively improve the population
        for (int iter = 0; iter < max_iterations; iter++) {
            // Find current best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Calculate spread factor based on iteration (decreases over time)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Generate new solutions by spreading from existing ones
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Each solution produces a new one by spreading
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Keep the better of parent and child
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Occasionally add completely new random solutions to maintain diversity
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
    
    virtual ~SormyakovOptimizer() = default;  // Virtual destructor for proper polymorphic destruction
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
    
    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

/**
 * @brief Modified Sormyakov with Elitism
 * Keeps the best solutions from the previous generation to preserve good solutions
 */
class SormyakovWithElitism : public SormyakovOptimizer {
protected:
    double elitism_ratio;

public:
    SormyakovWithElitism(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double el_ratio = 0.1  // 10% of population preserved
    ) : SormyakovOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        elitism_ratio(el_ratio) {}

    std::vector<double> optimize() override {
        // Initialize population with random solutions
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Iteratively improve the population
        for (int iter = 0; iter < max_iterations; iter++) {
            // Find current best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Calculate spread factor based on iteration (decreases over time)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Generate new solutions by spreading from existing ones
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            // Determine number of elite solutions to preserve
            int num_elites = static_cast<int>(population_size * elitism_ratio);
            if (num_elites < 1) num_elites = 1;
            
            // Create a vector of indices sorted by fitness (ascending order)
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population_size; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            // Preserve the best solutions (elitism)
            for (int i = 0; i < num_elites; i++) {
                int elite_idx = fitness_indices[i].second;
                new_population.push_back(population[elite_idx]);
                new_fitness.push_back(fitness[elite_idx]);
            }
            
            // Generate remaining solutions by spreading
            for (int i = num_elites; i < population_size; i++) {
                int parent_idx = static_cast<int>(dis(gen) * population_size);
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                new_population.push_back(new_solution);
                new_fitness.push_back(new_fitness_val);
            }
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Occasionally add completely new random solutions to maintain diversity
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
 * @brief Modified Sormyakov with Adaptive Spread Factor
 * Uses adaptive spread factor that changes based on the diversity of the population
 */
class SormyakovWithAdaptiveSpread : public SormyakovOptimizer {
protected:
    double diversity_threshold;

public:
    SormyakovWithAdaptiveSpread(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double div_threshold = 0.01  // threshold for diversity adjustment
    ) : SormyakovOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        diversity_threshold(div_threshold) {}

    std::vector<double> optimize() override {
        // Initialize population with random solutions
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Iteratively improve the population
        for (int iter = 0; iter < max_iterations; iter++) {
            // Find current best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Calculate population diversity
            double diversity = calculatePopulationDiversity(population);
            
            // Calculate adaptive spread factor based on iteration and diversity
            double base_spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            double adaptive_factor = 1.0;
            
            if (diversity < diversity_threshold) {
                // If population is too homogeneous, increase spread to explore more
                adaptive_factor = 2.0;
            } else if (diversity > diversity_threshold * 10) {
                // If population is too diverse, decrease spread to exploit
                adaptive_factor = 0.5;
            }
            
            double spread_factor = base_spread_factor * adaptive_factor;
            
            // Generate new solutions by spreading from existing ones
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Each solution produces a new one by spreading
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Keep the better of parent and child
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Occasionally add completely new random solutions to maintain diversity
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
 * @brief Modified Sormyakov with Tournament Selection
 * Uses tournament selection to choose which solutions to spread from
 */
class SormyakovWithTournament : public SormyakovOptimizer {
private:
    int tournament_size;

public:
    SormyakovWithTournament(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int tour_size = 3  // size of tournament for selection
    ) : SormyakovOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        tournament_size(tour_size) {}

    std::vector<double> optimize() override {
        // Initialize population with random solutions
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Iteratively improve the population
        for (int iter = 0; iter < max_iterations; iter++) {
            // Find current best solution
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Calculate spread factor based on iteration (decreases over time)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Generate new solutions by spreading from existing ones using tournament selection
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Select a parent using tournament selection
                int parent_idx = tournamentSelection(population, fitness);
                
                // Create new solution by spreading from selected parent
                std::vector<double> new_solution = spreadSolution(population[parent_idx], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Keep the better of parent and child
                if (new_fitness_val < fitness[parent_idx]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[parent_idx]);
                    new_fitness.push_back(fitness[parent_idx]);
                }
            }
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Occasionally add completely new random solutions to maintain diversity
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
 * @brief Modified Sormyakov with Dynamic Population Size
 * Adjusts population size dynamically based on convergence rate
 */
class SormyakovWithDynamicPopulation : public SormyakovOptimizer {
private:
    double convergence_threshold;
    int min_population_size;
    int max_population_size;
    std::vector<double> previous_best_fitnesses;

public:
    SormyakovWithDynamicPopulation(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double conv_threshold = 0.001
    ) : SormyakovOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        convergence_threshold(conv_threshold), 
        min_population_size(std::max(10, pop_size / 4)),
        max_population_size(pop_size * 2) {
        previous_best_fitnesses.reserve(10);  // Reserve space for last 10 fitness values
    }

    std::vector<double> optimize() override {
        int current_population_size = population_size;
        
        // Initialize population with random solutions
        std::vector<std::vector<double>> population(current_population_size);
        std::vector<double> fitness(current_population_size);
        
        for (int i = 0; i < current_population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Iteratively improve the population
        for (int iter = 0; iter < max_iterations; iter++) {
            // Find current best solution
            for (int i = 0; i < current_population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Store best fitness for convergence analysis
            if (previous_best_fitnesses.size() >= 10) {
                previous_best_fitnesses.erase(previous_best_fitnesses.begin());
            }
            previous_best_fitnesses.push_back(best_fitness);
            
            // Adjust population size based on convergence
            if (previous_best_fitnesses.size() >= 10) {
                double convergence_rate = calculateConvergenceRate();
                
                if (convergence_rate < convergence_threshold && current_population_size < max_population_size) {
                    // Slow convergence - increase population size to explore more
                    current_population_size = std::min(max_population_size, 
                                                      static_cast<int>(current_population_size * 1.1));
                    adjustPopulationSize(population, fitness, current_population_size);
                } else if (convergence_rate > convergence_threshold * 10 && current_population_size > min_population_size) {
                    // Fast convergence - decrease population size to save computation
                    current_population_size = std::max(min_population_size, 
                                                      static_cast<int>(current_population_size * 0.9));
                    adjustPopulationSize(population, fitness, current_population_size);
                }
            }
            
            // Calculate spread factor based on iteration (decreases over time)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Generate new solutions by spreading from existing ones
            std::vector<std::vector<double>> new_population(current_population_size);
            std::vector<double> new_fitness(current_population_size);
            
            for (int i = 0; i < current_population_size; i++) {
                // Each solution produces a new one by spreading
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Keep the better of parent and child
                if (new_fitness_val < fitness[i]) {
                    new_population[i] = new_solution;
                    new_fitness[i] = new_fitness_val;
                } else {
                    new_population[i] = population[i];
                    new_fitness[i] = fitness[i];
                }
            }
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Occasionally add completely new random solutions to maintain diversity
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
            // Increase population size by adding random solutions
            int additional = new_size - population.size();
            for (int i = 0; i < additional; i++) {
                population.push_back(generateRandomSolution());
                fitness.push_back(objective_function(population.back()));
            }
        } else {
            // Decrease population size by keeping best solutions
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
 * @brief Structure to store comparison results
 */
struct ComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    int iterations_completed;
    int population_size;
};

#endif // SORMYAKOV_MODIFICATIONS_H