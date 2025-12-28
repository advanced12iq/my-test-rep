#ifndef SORNYAK_OPTIMIZER_H
#define SORNYAK_OPTIMIZER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

class SornyakOptimizer {
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
    double convergence_threshold;

public:

    SornyakOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double conv_threshold = 1e-6
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        population_size(pop_size), min_value(min_val), max_value(max_val), 
        convergence_threshold(conv_threshold), gen(rd()), dis(0.0, 1.0) {}

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

    std::pair<std::vector<double>, int> optimize() {
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
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            
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
        
        return {best_solution, actual_iterations};
    }

    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }

    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

#endif // SORNYAK_OPTIMIZER_H