#ifndef CONVENTIONAL_OPTIMIZATION_H
#define CONVENTIONAL_OPTIMIZATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <limits>

class GradientDescentOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double learning_rate;
    double tolerance;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    GradientDescentOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double lr = 0.01,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        learning_rate(lr), tolerance(tol), gen(rd()), 
        dis(min_val, max_val) {}
    std::vector<double> calculateGradient(const std::vector<double>& x, double h = 1e-8) {
        std::vector<double> gradient(dimension);
        
        for (int i = 0; i < dimension; i++) {
            std::vector<double> x_plus = x;
            std::vector<double> x_minus = x;
            
            x_plus[i] += h;
            x_minus[i] -= h;
            
            gradient[i] = (objective_function(x_plus) - objective_function(x_minus)) / (2.0 * h);
        }
        
        return gradient;
    }

    std::pair<std::vector<double>, int> optimize() {
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            std::vector<double> gradient = calculateGradient(x);
            std::vector<double> new_x = x;
            for (int i = 0; i < dimension; i++) {
                new_x[i] -= learning_rate * gradient[i];
            }
            current_fitness = objective_function(new_x);
            if (std::abs(prev_fitness - current_fitness) < tolerance) {
                break;
            }
            
            x = new_x;
            prev_fitness = current_fitness;
        }
        
        return {x, actual_iterations};
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

class NelderMeadOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    NelderMeadOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        tolerance(tol), gen(rd()), dis(min_val, max_val) {}

    std::pair<std::vector<double>, int> optimize() {
        std::vector<std::vector<double>> simplex(dimension + 1, std::vector<double>(dimension));
        std::vector<double> fitness(dimension + 1);
        for (int i = 0; i <= dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                simplex[i][j] = dis(gen);
            }
            fitness[i] = objective_function(simplex[i]);
        }
        const double alpha = 1.0;
        const double gamma = 2.0;
        const double rho = 0.5;
        const double sigma = 0.5;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i <= dimension; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            int best_idx = fitness_indices[0].second;
            int worst_idx = fitness_indices[dimension].second;
            int second_worst_idx = fitness_indices[dimension - 1].second;
            std::vector<double> centroid(dimension, 0.0);
            for (int i = 0; i <= dimension; i++) {
                if (i != worst_idx) {
                    for (int j = 0; j < dimension; j++) {
                        centroid[j] += simplex[i][j];
                    }
                }
            }
            for (int j = 0; j < dimension; j++) {
                centroid[j] /= dimension;
            }
            std::vector<double> reflected = centroid;
            for (int j = 0; j < dimension; j++) {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
            }
            double reflected_fitness = objective_function(reflected);
            
            if (fitness[best_idx] <= reflected_fitness && reflected_fitness < fitness[second_worst_idx]) {
                simplex[worst_idx] = reflected;
                fitness[worst_idx] = reflected_fitness;
            } else if (reflected_fitness < fitness[best_idx]) {
                std::vector<double> expanded = centroid;
                for (int j = 0; j < dimension; j++) {
                    expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
                }
                double expanded_fitness = objective_function(expanded);
                
                if (expanded_fitness < reflected_fitness) {
                    simplex[worst_idx] = expanded;
                    fitness[worst_idx] = expanded_fitness;
                } else {
                    simplex[worst_idx] = reflected;
                    fitness[worst_idx] = reflected_fitness;
                }
            } else {
                std::vector<double> contracted = centroid;
                for (int j = 0; j < dimension; j++) {
                    contracted[j] = centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]);
                }
                double contracted_fitness = objective_function(contracted);
                
                if (contracted_fitness < fitness[worst_idx]) {
                    simplex[worst_idx] = contracted;
                    fitness[worst_idx] = contracted_fitness;
                } else {
                    for (int i = 0; i <= dimension; i++) {
                        if (i != best_idx) {
                            for (int j = 0; j < dimension; j++) {
                                simplex[i][j] = simplex[best_idx][j] + sigma * (simplex[i][j] - simplex[best_idx][j]);
                            }
                            fitness[i] = objective_function(simplex[i]);
                        }
                    }
                }
            }
            double best_fitness = fitness[best_idx];
            double worst_fitness = fitness[worst_idx];
            
            if (std::abs(worst_fitness - best_fitness) < tolerance) {
                break;
            }
        }
        int best_idx = 0;
        for (int i = 1; i <= dimension; i++) {
            if (fitness[i] < fitness[best_idx]) {
                best_idx = i;
            }
        }
        
        return {simplex[best_idx], actual_iterations};
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

class RandomSearchOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double min_value;
    double max_value;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    RandomSearchOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        min_value(min_val), max_value(max_val), gen(rd()), 
        dis(min_val, max_val) {}

    std::pair<std::vector<double>, int> optimize() {
        std::vector<double> best_solution(dimension);
        double best_fitness = std::numeric_limits<double>::max();
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            std::vector<double> current_solution(dimension);
            for (int i = 0; i < dimension; i++) {
                current_solution[i] = dis(gen);
            }
            
            double current_fitness = objective_function(current_solution);
            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_solution = current_solution;
            }
        }
        
        return {best_solution, actual_iterations};
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

class CoordinateDescentOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    double step_size;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    CoordinateDescentOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double step = 0.01,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter),
        step_size(step), tolerance(tol), gen(rd()),
        dis(min_val, max_val) {}

    std::pair<std::vector<double>, int> optimize() {
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);
        }

        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        int actual_iterations = 0;

        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int coord = 0; coord < dimension; coord++) {
                double original_value = x[coord];
                x[coord] -= step_size;
                double fitness_decrease = objective_function(x);
                x[coord] = original_value + step_size;
                double fitness_increase = objective_function(x);
                x[coord] = original_value;
                if (fitness_decrease < std::min(prev_fitness, fitness_increase)) {
                    x[coord] -= step_size;
                    current_fitness = fitness_decrease;
                } else if (fitness_increase < std::min(prev_fitness, fitness_decrease)) {
                    x[coord] += step_size;
                    current_fitness = fitness_increase;
                }
            }
            if (std::abs(prev_fitness - current_fitness) < tolerance) {
                break;
            }

            prev_fitness = current_fitness;
        }

        return {x, actual_iterations};
    }

    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};
#endif // CONVENTIONAL_OPTIMIZATION_H