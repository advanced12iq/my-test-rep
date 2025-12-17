#include "sornyak_modifications.h"
#include "conventional_optimization.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
auto rastrigin_function = [](const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += val * val - 10.0 * cos(2.0 * M_PI * val);
    }
    return sum;
};

auto schwefel_function = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * sin(sqrt(std::abs(val)));
    }
    return 418.9829 * x.size() - sum;
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
const double RASTRIGIN_OPTIMAL = 0.0;
const double SCHWEFEL_OPTIMAL = 0.0;
const double ROSENBROCK_OPTIMAL = 0.0;
struct OptimizationStep {
    std::string method;
    int iteration_step;
    double error;
    std::vector<double> solution_coordinates;
    double function_value;
    int seed;
    std::string function;
};
class SornyakOptimizerTracking : public SornyakOptimizer {
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    SornyakOptimizerTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double conv_threshold = 1e-6,
        int s = 0,
        const std::string& name = "Base_Sornyak",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : SornyakOptimizer(f, dim, max_iter, pop_size, min_val, max_val, conv_threshold),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name) {
        gen.seed(seed);
    }
    
    std::pair<std::vector<double>, int> optimize() override {
        steps.clear();
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
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = best_solution;
        step.function_value = best_fitness;
        step.error = std::abs(best_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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
};
class SornyakWithElitismTracking : public SornyakWithElitism {
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    SornyakWithElitismTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double el_ratio = 0.1,
        double conv_threshold = 1e-6,
        int s = 0,
        const std::string& name = "Sornyak_Elitism",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : SornyakWithElitism(f, dim, max_iter, pop_size, min_val, max_val, el_ratio, conv_threshold),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name) {
        gen.seed(seed);
    }
    
    std::pair<std::vector<double>, int> optimize() override {
        steps.clear();
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
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = best_solution;
        step.function_value = best_fitness;
        step.error = std::abs(best_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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
        
        return {best_solution, actual_iterations};
    }
};
class SornyakWithAdaptiveSpreadTracking : public SornyakWithAdaptiveSpread {
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    SornyakWithAdaptiveSpreadTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double div_threshold = 0.01,
        double conv_threshold = 1e-6,
        int s = 0,
        const std::string& name = "Sornyak_AdaptiveSpread",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : SornyakWithAdaptiveSpread(f, dim, max_iter, pop_size, min_val, max_val, div_threshold, conv_threshold),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name) {
        gen.seed(seed);
    }
    
    std::pair<std::vector<double>, int> optimize() override {
        steps.clear();
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
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = best_solution;
        step.function_value = best_fitness;
        step.error = std::abs(best_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            
            double diversity = getPopulationDiversity(population);
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
        
        return {best_solution, actual_iterations};
    }
};
class SornyakWithTournamentTracking : public SornyakWithTournament {
private:
    int tournament_size_local;
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    SornyakWithTournamentTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int tour_size = 3,
        double conv_threshold = 1e-6,
        int s = 0,
        const std::string& name = "Sornyak_Tournament",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : SornyakWithTournament(f, dim, max_iter, pop_size, min_val, max_val, tour_size, conv_threshold),
        tournament_size_local(tour_size), seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name) {
        gen.seed(seed);
    }
    
    std::pair<std::vector<double>, int> optimize() override {
        steps.clear();
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
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = best_solution;
        step.function_value = best_fitness;
        step.error = std::abs(best_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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
        
        return {best_solution, actual_iterations};
    }
    
private:
    int tournamentSelection(const std::vector<std::vector<double>>& population, 
                           const std::vector<double>& fitness) {
        int best_idx = static_cast<int>(dis(gen) * population.size());
        
        for (int i = 1; i < tournament_size_local; i++) {
            int candidate_idx = static_cast<int>(dis(gen) * population.size());
            if (fitness[candidate_idx] < fitness[best_idx]) {
                best_idx = candidate_idx;
            }
        }
        
        return best_idx;
    }
};
class GradientDescentOptimizerTracking {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double learning_rate;
    double tolerance;
    double min_value;
    double max_value;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
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
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    GradientDescentOptimizerTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        double lr = 0.01,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0,
        int s = 0,
        const std::string& name = "GradientDescent",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : objective_function(f), dimension(dim), max_iterations(max_iter),
        learning_rate(lr), tolerance(tol), min_value(min_val), max_value(max_val),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name), gen(s), dis(min_val, max_val) {}
    
    std::pair<std::vector<double>, int> optimize() {
        steps.clear();
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        int actual_iterations = 0;
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = x;
        step.function_value = prev_fitness;
        step.error = std::abs(prev_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            std::vector<double> gradient = calculateGradient(x);
            std::vector<double> new_x = x;
            for (int i = 0; i < dimension; i++) {
                new_x[i] -= learning_rate * gradient[i];
            }
            current_fitness = objective_function(new_x);
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = new_x;
            step.function_value = current_fitness;
            step.error = std::abs(current_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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

class NelderMeadOptimizerTracking {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    NelderMeadOptimizerTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0,
        int s = 0,
        const std::string& name = "NelderMead",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : objective_function(f), dimension(dim), max_iterations(max_iter),
        tolerance(tol), seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name),
        gen(s), dis(min_val, max_val) {}
    
    std::pair<std::vector<double>, int> optimize() {
        steps.clear();
        std::vector<std::vector<double>> simplex(dimension + 1, std::vector<double>(dimension));
        std::vector<double> fitness(dimension + 1);
        for (int i = 0; i <= dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                simplex[i][j] = dis(gen);
            }
            fitness[i] = objective_function(simplex[i]);
        }
        int best_idx_init = 0;
        for (int i = 1; i <= dimension; i++) {
            if (fitness[i] < fitness[best_idx_init]) {
                best_idx_init = i;
            }
        }
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = simplex[best_idx_init];
        step.function_value = fitness[best_idx_init];
        step.error = std::abs(fitness[best_idx_init] - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
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
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = simplex[best_idx];
            step.function_value = fitness[best_idx];
            step.error = std::abs(fitness[best_idx] - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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
        int best_idx_final = 0;
        for (int i = 1; i <= dimension; i++) {
            if (fitness[i] < fitness[best_idx_final]) {
                best_idx_final = i;
            }
        }
        
        return {simplex[best_idx_final], actual_iterations};
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

class RandomSearchOptimizerTracking {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double min_value;
    double max_value;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    RandomSearchOptimizerTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        double min_val = -10.0,
        double max_val = 10.0,
        int s = 0,
        const std::string& name = "RandomSearch",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : objective_function(f), dimension(dim), max_iterations(max_iter),
        min_value(min_val), max_value(max_val),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name),
        gen(s), dis(min_val, max_val) {}
    
    std::pair<std::vector<double>, int> optimize() {
        steps.clear();
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
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
        }
        
        return {best_solution, actual_iterations};
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

class CoordinateDescentOptimizerTracking {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double step_size;
    double tolerance;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    CoordinateDescentOptimizerTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        double step = 0.01,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0,
        int s = 0,
        const std::string& name = "CoordinateDescent",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : objective_function(f), dimension(dim), max_iterations(max_iter),
        step_size(step), tolerance(tol),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name),
        gen(s), dis(min_val, max_val) {}
    
    std::pair<std::vector<double>, int> optimize() {
        steps.clear();
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        int actual_iterations = 0;
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = x;
        step.function_value = prev_fitness;
        step.error = std::abs(prev_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
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
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = x;
            step.function_value = current_fitness;
            step.error = std::abs(current_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
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
class SornyakWithDynamicPopulationTracking : public SornyakWithDynamicPopulation {
private:
    double pop_conv_threshold_local;
    int min_population_size_local;
    int max_population_size_local;
    std::vector<double> previous_best_fitnesses_local;
    
    double calculateConvergenceRate() {
        if (previous_best_fitnesses_local.size() < 2) return 0.0;
        double diff = std::abs(previous_best_fitnesses_local.back() - previous_best_fitnesses_local.front());
        return diff / previous_best_fitnesses_local.size();
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
            for (size_t i = 0; i < population.size(); i++) {
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
    
public:
    std::vector<OptimizationStep> steps;
    int seed;
    std::string method_name;
    std::function<double(const std::vector<double>&)> func;
    double optimal_value;
    std::string function_name;
    
    SornyakWithDynamicPopulationTracking(
        std::function<double(const std::vector<double>&)> f,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double conv_threshold = 1e-6,
        double pop_conv_threshold = 0.001,
        int s = 0,
        const std::string& name = "Sornyak_DynamicPopulation",
        double opt_val = 0.0,
        const std::string& func_name = ""
    ) : SornyakWithDynamicPopulation(f, dim, max_iter, pop_size, min_val, max_val, conv_threshold, pop_conv_threshold),
        pop_conv_threshold_local(pop_conv_threshold),
        min_population_size_local(std::max(10, pop_size / 4)),
        max_population_size_local(pop_size * 2),
        seed(s), method_name(name), func(f), optimal_value(opt_val), function_name(func_name) {
        gen.seed(seed);
        previous_best_fitnesses_local.reserve(10);
    }
    
    std::pair<std::vector<double>, int> optimize() override {
        steps.clear();
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
        OptimizationStep step;
        step.method = method_name;
        step.iteration_step = 0;
        step.solution_coordinates = best_solution;
        step.function_value = best_fitness;
        step.error = std::abs(best_fitness - optimal_value);
        step.seed = seed;
        step.function = function_name;
        steps.push_back(step);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            for (int i = 0; i < current_population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            OptimizationStep step;
            step.method = method_name;
            step.iteration_step = actual_iterations;
            step.solution_coordinates = best_solution;
            step.function_value = best_fitness;
            step.error = std::abs(best_fitness - optimal_value);
            step.seed = seed;
            steps.push_back(step);
            
            if (std::abs(best_fitness - previous_best_fitness) < 1e-6) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            if (previous_best_fitnesses_local.size() >= 10) {
                previous_best_fitnesses_local.erase(previous_best_fitnesses_local.begin());
            }
            previous_best_fitnesses_local.push_back(best_fitness);
            if (previous_best_fitnesses_local.size() >= 10) {
                double convergence_rate = calculateConvergenceRate();
                
                if (convergence_rate < pop_conv_threshold_local && current_population_size < max_population_size_local) {
                    current_population_size = std::min(max_population_size_local, 
                                                      static_cast<int>(current_population_size * 1.1));
                    adjustPopulationSize(population, fitness, current_population_size);
                } else if (convergence_rate > pop_conv_threshold_local * 10 && current_population_size > min_population_size_local) {
                    current_population_size = std::max(min_population_size_local, 
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
        
        return {best_solution, actual_iterations};
    }
};
void saveStepsToCSV(const std::vector<OptimizationStep>& all_steps, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << "method,iteration_step,error,x1,x2,function_value,seed,function\n";
    for (const auto& step : all_steps) {
        file << step.method << ","
             << step.iteration_step << ","
             << std::fixed << std::setprecision(10) << step.error << ","
             << step.solution_coordinates[0] << ","
             << step.solution_coordinates[1] << ","
             << step.function_value << ","
             << step.seed << ","
             << step.function << "\n";
    }
    
    file.close();
    std::cout << "Saved " << all_steps.size() << " steps to " << filename << std::endl;
}

int main() {
    std::cout << "Comprehensive Optimization Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    std::vector<OptimizationStep> all_steps;
    const int DIM = 2;
    const int MAX_ITER = 2000;
    const int POP_SIZE = 50;
    const std::vector<int> seeds = {42, 123, 456};
    struct TestFunction {
        std::function<double(const std::vector<double>&)> func;
        std::string name;
        double optimal;
        double min_val;
        double max_val;
    };
    
    std::vector<TestFunction> test_functions = {
        {rastrigin_function, "Rastrigin", RASTRIGIN_OPTIMAL, -5.12, 5.12},
        {schwefel_function, "Schwefel", SCHWEFEL_OPTIMAL, -500.0, 500.0},
        {rosenbrock_function, "Rosenbrock", ROSENBROCK_OPTIMAL, -2.0, 2.0}
    };
    for (const auto& test_func : test_functions) {
        std::cout << "\nTesting " << test_func.name << " function..." << std::endl;
        
        for (int seed : seeds) {
            std::cout << "  Seed: " << seed << std::endl;
            {
                SornyakOptimizerTracking optimizer(
                    test_func.func, DIM, MAX_ITER, POP_SIZE,
                    test_func.min_val, test_func.max_val, 1e-6,
                    seed, "Base_Sornyak", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                SornyakWithElitismTracking optimizer(
                    test_func.func, DIM, MAX_ITER, POP_SIZE,
                    test_func.min_val, test_func.max_val, 0.2, 1e-6,
                    seed, "Sornyak_Elitism", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                SornyakWithAdaptiveSpreadTracking optimizer(
                    test_func.func, DIM, MAX_ITER, POP_SIZE,
                    test_func.min_val, test_func.max_val, 0.01, 1e-6,
                    seed, "Sornyak_AdaptiveSpread", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                SornyakWithTournamentTracking optimizer(
                    test_func.func, DIM, MAX_ITER, POP_SIZE,
                    test_func.min_val, test_func.max_val, 3, 1e-6,
                    seed, "Sornyak_Tournament", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                SornyakWithDynamicPopulationTracking optimizer(
                    test_func.func, DIM, MAX_ITER, POP_SIZE,
                    test_func.min_val, test_func.max_val, 1e-6, 0.001,
                    seed, "Sornyak_DynamicPopulation", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                double lr = (test_func.name == "Rosenbrock") ? 0.001 : 0.01;
                GradientDescentOptimizerTracking optimizer(
                    test_func.func, DIM, MAX_ITER, lr, 1e-6,
                    test_func.min_val, test_func.max_val,
                    seed, "GradientDescent", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                NelderMeadOptimizerTracking optimizer(
                    test_func.func, DIM, MAX_ITER, 1e-6,
                    test_func.min_val, test_func.max_val,
                    seed, "NelderMead", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                RandomSearchOptimizerTracking optimizer(
                    test_func.func, DIM, MAX_ITER,
                    test_func.min_val, test_func.max_val,
                    seed, "RandomSearch", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
            {
                double step = (test_func.name == "Rosenbrock") ? 0.001 : 0.01;
                CoordinateDescentOptimizerTracking optimizer(
                    test_func.func, DIM, MAX_ITER, step, 1e-6,
                    test_func.min_val, test_func.max_val,
                    seed, "CoordinateDescent", test_func.optimal, test_func.name
                );
                optimizer.optimize();
                all_steps.insert(all_steps.end(), optimizer.steps.begin(), optimizer.steps.end());
            }
        }
    }
    saveStepsToCSV(all_steps, "optimization_results.csv");
    
    std::cout << "\nTest completed! Total steps recorded: " << all_steps.size() << std::endl;
    
    return 0;
}

