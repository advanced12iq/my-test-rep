#ifndef SORMYAKOV_OPTIMIZER_H
#define SORMYAKOV_OPTIMIZER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

/**
 * @brief Sormyakov Optimization Method (Weeds Optimization Method)
 * 
 * This implementation represents a nature-inspired optimization algorithm
 * that mimics the behavior of weeds (sorняки) in nature - their ability
 * to spread, adapt, and find optimal growing conditions.
 */
class SormyakovOptimizer {
private:
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
    /**
     * @brief Constructor for the Sormyakov Optimizer
     * @param func The objective function to minimize
     * @param dim Number of dimensions in the search space
     * @param max_iter Maximum number of iterations
     * @param pop_size Population size (number of "weeds")
     * @param min_val Minimum value for each dimension
     * @param max_val Maximum value for each dimension
     */
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

    /**
     * @brief Generate a random solution within the bounds
     */
    std::vector<double> generateRandomSolution() {
        std::vector<double> solution(dimension);
        for (int i = 0; i < dimension; i++) {
            solution[i] = min_value + (max_value - min_value) * dis(gen);
        }
        return solution;
    }

    /**
     * @brief Create a new solution by "spreading" from an existing solution
     * This mimics how weeds spread and adapt to new locations
     */
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

    /**
     * @brief Main optimization algorithm
     */
    std::vector<double> optimize() {
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
    
    /**
     * @brief Get the best fitness value found
     */
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
    
    /**
     * @brief Get optimization parameters
     */
    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

#endif // SORMYAKOV_OPTIMIZER_H