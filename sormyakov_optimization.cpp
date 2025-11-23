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
};

// Example usage with test functions
int main() {
    std::cout << "Sormyakov Optimization Method (Weeds Optimization Method)" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Example 1: Minimize the sphere function
    std::cout << "\nExample 1: Sphere Function (min at [0,0,...,0])" << std::endl;
    auto sphere_function = [](const std::vector<double>& x) {
        double sum = 0.0;
        for (double val : x) {
            sum += val * val;
        }
        return sum;
    };
    
    SormyakovOptimizer optimizer1(sphere_function, 2, 500, 30, -5.0, 5.0);
    std::vector<double> result1 = optimizer1.optimize();
    
    std::cout << "Optimal solution: [";
    for (size_t i = 0; i < result1.size(); i++) {
        std::cout << result1[i];
        if (i < result1.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Function value: " << optimizer1.getBestFitness(result1) << std::endl;
    
    // Example 2: Minimize the Rosenbrock function
    std::cout << "\nExample 2: Rosenbrock Function (min at [1,1,...,1])" << std::endl;
    auto rosenbrock_function = [](const std::vector<double>& x) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size() - 1; i++) {
            double term1 = 100.0 * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]);
            double term2 = (1.0 - x[i]) * (1.0 - x[i]);
            sum += term1 + term2;
        }
        return sum;
    };
    
    SormyakovOptimizer optimizer2(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    std::vector<double> result2 = optimizer2.optimize();
    
    std::cout << "Optimal solution: [";
    for (size_t i = 0; i < result2.size(); i++) {
        std::cout << result2[i];
        if (i < result2.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Function value: " << optimizer2.getBestFitness(result2) << std::endl;
    
    // Example 3: Minimize the Rastrigin function
    std::cout << "\nExample 3: Rastrigin Function (min at [0,0,...,0])" << std::endl;
    auto rastrigin_function = [](const std::vector<double>& x) {
        double sum = 10.0 * x.size();
        for (double val : x) {
            sum += val * val - 10.0 * cos(2.0 * M_PI * val);
        }
        return sum;
    };
    
    SormyakovOptimizer optimizer3(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    std::vector<double> result3 = optimizer3.optimize();
    
    std::cout << "Optimal solution: [";
    for (size_t i = 0; i < result3.size(); i++) {
        std::cout << result3[i];
        if (i < result3.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Function value: " << optimizer3.getBestFitness(result3) << std::endl;
    
    return 0;
}