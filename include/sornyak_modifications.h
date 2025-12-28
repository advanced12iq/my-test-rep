#ifndef SORNYAK_MODIFICATIONS_H
#define SORNYAK_MODIFICATIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <chrono>
#include <algorithm>
#include <numeric>

class SornyakOptimizer {
public:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    int population_size;
    double min_value;
    double max_value;
    int min_seeds;
    int max_seeds;
    double sigma_init;  // начальное значение σ
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    std::normal_distribution<double> normal_dis;
    double convergence_threshold;
    SornyakOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int min_s = 1,
        int max_s = 5,
        double sigma = 1.0,
        double conv_threshold = 1e-6
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        population_size(pop_size), min_value(min_val), max_value(max_val),
        min_seeds(min_s), max_seeds(max_s), sigma_init(sigma),
        convergence_threshold(conv_threshold), gen(rd()), 
        dis(0.0, 1.0), normal_dis(0.0, 1.0) {}

    std::vector<double> generateRandomSolution() {
        std::vector<double> solution(dimension);
        for (int i = 0; i < dimension; i++) {
            solution[i] = min_value + (max_value - min_value) * dis(gen);
        }
        return solution;
    }

    std::vector<double> spreadSolution(const std::vector<double>& parent, double sigma) {
        std::vector<double> child = parent;
        for (int i = 0; i < dimension; i++) {
            double variation = normal_dis(gen) * sigma;
            child[i] += variation;
            child[i] = std::max(min_value, std::min(max_value, child[i]));
        }
        return child;
    }

    int calculateNumSeeds(double fitness, double best_fitness, double worst_fitness) {
        if (std::abs(worst_fitness - best_fitness) < 1e-12) {
            return min_seeds;
        }
        
        double normalized = (worst_fitness - fitness) / (worst_fitness - best_fitness);
        int num_seeds = min_seeds + static_cast<int>(normalized * (max_seeds - min_seeds));
        return std::max(min_seeds, std::min(max_seeds, num_seeds));
    }

    virtual std::pair<std::vector<double>, int> optimize() {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double worst_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            best_fitness = *std::min_element(fitness.begin(), fitness.end());
            worst_fitness = *std::max_element(fitness.begin(), fitness.end());
            auto best_it = std::min_element(fitness.begin(), fitness.end());
            best_solution = population[std::distance(fitness.begin(), best_it)];
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            double sigma = sigma_init * (max_iterations - iter) / max_iterations;
            std::vector<std::vector<double>> all_seeds;
            std::vector<double> all_fitness;
            for (int i = 0; i < population_size; i++) {
                int num_seeds = calculateNumSeeds(fitness[i], best_fitness, worst_fitness);
                
                for (int j = 0; j < num_seeds; j++) {
                    std::vector<double> new_solution = spreadSolution(population[i], sigma);
                    double new_fitness_val = objective_function(new_solution);
                    
                    all_seeds.push_back(new_solution);
                    all_fitness.push_back(new_fitness_val);
                }
            }
            for (int i = 0; i < population_size; i++) {
                all_seeds.push_back(population[i]);
                all_fitness.push_back(fitness[i]);
            }
            std::vector<size_t> indices(all_seeds.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) { return all_fitness[a] < all_fitness[b]; });
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size && i < indices.size(); i++) {
                new_population.push_back(all_seeds[indices[i]]);
                new_fitness.push_back(all_fitness[indices[i]]);
            }
            while (new_population.size() < population_size) {
                std::vector<double> random_solution = generateRandomSolution();
                new_population.push_back(random_solution);
                new_fitness.push_back(objective_function(random_solution));
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
    
    virtual ~SornyakOptimizer() = default;
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
    
    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

class SornyakWithElitism : public SornyakOptimizer {
public:
    double elitism_ratio;

    SornyakWithElitism(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int min_s = 1,
        int max_s = 5,
        double sigma = 1.0,
        double el_ratio = 0.1,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, min_s, max_s, sigma, conv_threshold), 
        elitism_ratio(el_ratio) {}

    std::pair<std::vector<double>, int> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double worst_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            best_fitness = *std::min_element(fitness.begin(), fitness.end());
            worst_fitness = *std::max_element(fitness.begin(), fitness.end());
            auto best_it = std::min_element(fitness.begin(), fitness.end());
            best_solution = population[std::distance(fitness.begin(), best_it)];
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            int num_elites = static_cast<int>(population_size * elitism_ratio);
            num_elites = std::max(1, std::min(num_elites, population_size));
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population_size; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            double sigma = sigma_init * (max_iterations - iter) / max_iterations;
            std::vector<std::vector<double>> all_seeds;
            std::vector<double> all_fitness;
            for (int i = 0; i < population_size; i++) {
                int num_seeds = calculateNumSeeds(fitness[i], best_fitness, worst_fitness);
                
                for (int j = 0; j < num_seeds; j++) {
                    std::vector<double> new_solution = spreadSolution(population[i], sigma);
                    double new_fitness_val = objective_function(new_solution);
                    
                    all_seeds.push_back(new_solution);
                    all_fitness.push_back(new_fitness_val);
                }
            }
            for (int i = 0; i < population_size; i++) {
                all_seeds.push_back(population[i]);
                all_fitness.push_back(fitness[i]);
            }
            std::vector<size_t> indices(all_seeds.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) { return all_fitness[a] < all_fitness[b]; });
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            for (int i = 0; i < num_elites; i++) {
                int elite_idx = fitness_indices[i].second;
                new_population.push_back(population[elite_idx]);
                new_fitness.push_back(fitness[elite_idx]);
            }
            for (size_t i = 0; i < indices.size() && new_population.size() < population_size; i++) {
                bool is_elite = false;
                for (int j = 0; j < num_elites; j++) {
                    if (all_seeds[indices[i]] == population[fitness_indices[j].second]) {
                        is_elite = true;
                        break;
                    }
                }
                
                if (!is_elite) {
                    new_population.push_back(all_seeds[indices[i]]);
                    new_fitness.push_back(all_fitness[indices[i]]);
                }
            }
            while (new_population.size() < population_size) {
                std::vector<double> random_solution = generateRandomSolution();
                new_population.push_back(random_solution);
                new_fitness.push_back(objective_function(random_solution));
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

class SornyakWithAdaptiveSpread : public SornyakOptimizer {
public:
    double diversity_threshold;

    SornyakWithAdaptiveSpread(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int min_s = 1,
        int max_s = 5,
        double sigma = 1.0,
        double div_threshold = 0.01,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, min_s, max_s, sigma, conv_threshold), 
        diversity_threshold(div_threshold) {}

    std::pair<std::vector<double>, int> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double worst_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            best_fitness = *std::min_element(fitness.begin(), fitness.end());
            worst_fitness = *std::max_element(fitness.begin(), fitness.end());
            auto best_it = std::min_element(fitness.begin(), fitness.end());
            best_solution = population[std::distance(fitness.begin(), best_it)];
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            double diversity = calculatePopulationDiversity(population);
            double base_sigma = sigma_init * (max_iterations - iter) / max_iterations;
            double adaptive_factor = 1.0;
            
            if (diversity < diversity_threshold) {
                adaptive_factor = 2.0;  // Увеличиваем разброс при малом разнообразии
            } else if (diversity > diversity_threshold * 10) {
                adaptive_factor = 0.5;  // Уменьшаем разброс при большом разнообразии
            }
            
            double sigma = base_sigma * adaptive_factor;
            std::vector<std::vector<double>> all_seeds;
            std::vector<double> all_fitness;
            for (int i = 0; i < population_size; i++) {
                int num_seeds = calculateNumSeeds(fitness[i], best_fitness, worst_fitness);
                
                for (int j = 0; j < num_seeds; j++) {
                    std::vector<double> new_solution = spreadSolution(population[i], sigma);
                    double new_fitness_val = objective_function(new_solution);
                    
                    all_seeds.push_back(new_solution);
                    all_fitness.push_back(new_fitness_val);
                }
            }
            for (int i = 0; i < population_size; i++) {
                all_seeds.push_back(population[i]);
                all_fitness.push_back(fitness[i]);
            }
            std::vector<size_t> indices(all_seeds.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) { return all_fitness[a] < all_fitness[b]; });
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size && i < indices.size(); i++) {
                new_population.push_back(all_seeds[indices[i]]);
                new_fitness.push_back(all_fitness[indices[i]]);
            }
            while (new_population.size() < population_size) {
                std::vector<double> random_solution = generateRandomSolution();
                new_population.push_back(random_solution);
                new_fitness.push_back(objective_function(random_solution));
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
};

class SornyakWithTournament : public SornyakOptimizer {
public:
    int tournament_size;
    SornyakWithTournament(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int min_s = 1,
        int max_s = 5,
        double sigma = 1.0,
        int tour_size = 3,
        double conv_threshold = 1e-6
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, min_s, max_s, sigma, conv_threshold), 
        tournament_size(tour_size) {}

    std::pair<std::vector<double>, int> optimize() override {
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double worst_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            best_fitness = *std::min_element(fitness.begin(), fitness.end());
            worst_fitness = *std::max_element(fitness.begin(), fitness.end());
            auto best_it = std::min_element(fitness.begin(), fitness.end());
            best_solution = population[std::distance(fitness.begin(), best_it)];
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
            }
            
            previous_best_fitness = best_fitness;
            double sigma = sigma_init * (max_iterations - iter) / max_iterations;
            std::vector<std::vector<double>> all_seeds;
            std::vector<double> all_fitness;
            for (int i = 0; i < population_size; i++) {
                int parent_idx = tournamentSelection(population, fitness);
                int num_seeds = calculateNumSeeds(fitness[parent_idx], best_fitness, worst_fitness);
                for (int j = 0; j < num_seeds; j++) {
                    std::vector<double> new_solution = spreadSolution(population[parent_idx], sigma);
                    double new_fitness_val = objective_function(new_solution);
                    
                    all_seeds.push_back(new_solution);
                    all_fitness.push_back(new_fitness_val);
                }
            }
            for (int i = 0; i < population_size; i++) {
                all_seeds.push_back(population[i]);
                all_fitness.push_back(fitness[i]);
            }
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            for (size_t i = 0; i < all_seeds.size() && i < population_size; i++) {
                new_population.push_back(all_seeds[i]);
                new_fitness.push_back(all_fitness[i]);
            }
            while (new_population.size() < population_size) {
                std::vector<double> random_solution = generateRandomSolution();
                new_population.push_back(random_solution);
                new_fitness.push_back(objective_function(random_solution));
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
    
    int tournamentSelection(const std::vector<std::vector<double>>& population, 
                           const std::vector<double>& fitness) {
        std::vector<int> candidates(tournament_size);
        
        for (int i = 0; i < tournament_size; i++) {
            candidates[i] = static_cast<int>(dis(gen) * population.size());
        }
        
        int best_idx = candidates[0];
        for (int i = 1; i < tournament_size; i++) {
            if (fitness[candidates[i]] < fitness[best_idx]) {
                best_idx = candidates[i];
            }
        }
        
        return best_idx;
    }
};

class SornyakWithDynamicPopulation : public SornyakOptimizer {
public:
    double convergence_threshold;
    int min_population_size;
    int max_population_size;
    std::vector<double> previous_best_fitnesses;
    SornyakWithDynamicPopulation(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        int min_s = 1,
        int max_s = 5,
        double sigma = 1.0,
        double conv_threshold = 1e-6,
        double pop_conv_threshold = 0.001
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val, min_s, max_s, sigma, conv_threshold), 
        convergence_threshold(pop_conv_threshold), 
        min_population_size(std::max(10, pop_size / 4)),
        max_population_size(pop_size * 2) {
        previous_best_fitnesses.reserve(10);
    }

    std::pair<std::vector<double>, int> optimize() override {
        int current_population_size = population_size;
        std::vector<std::vector<double>> population(current_population_size);
        std::vector<double> fitness(current_population_size);
        for (int i = 0; i < current_population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        double worst_fitness = fitness[0];
        double previous_best_fitness = best_fitness;
        int convergence_count = 0;
        int actual_iterations = 0;

        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations = iter + 1;
            best_fitness = *std::min_element(fitness.begin(), fitness.end());
            worst_fitness = *std::max_element(fitness.begin(), fitness.end());
            auto best_it = std::min_element(fitness.begin(), fitness.end());
            best_solution = population[std::distance(fitness.begin(), best_it)];
            if (std::abs(best_fitness - previous_best_fitness) < convergence_threshold) {
                convergence_count++;
                if (convergence_count >= 10) {
                    break;
                }
            } else {
                convergence_count = 0;
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
            double sigma = sigma_init * (max_iterations - iter) / max_iterations;
            std::vector<std::vector<double>> all_seeds;
            std::vector<double> all_fitness;
            for (int i = 0; i < current_population_size; i++) {
                int num_seeds = calculateNumSeeds(fitness[i], best_fitness, worst_fitness);
                
                for (int j = 0; j < num_seeds; j++) {
                    std::vector<double> new_solution = spreadSolution(population[i], sigma);
                    double new_fitness_val = objective_function(new_solution);
                    
                    all_seeds.push_back(new_solution);
                    all_fitness.push_back(new_fitness_val);
                }
            }
            for (int i = 0; i < current_population_size; i++) {
                all_seeds.push_back(population[i]);
                all_fitness.push_back(fitness[i]);
            }
            std::vector<size_t> indices(all_seeds.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) { return all_fitness[a] < all_fitness[b]; });
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < current_population_size && i < indices.size(); i++) {
                new_population.push_back(all_seeds[indices[i]]);
                new_fitness.push_back(all_fitness[indices[i]]);
            }
            while (new_population.size() < current_population_size) {
                std::vector<double> random_solution = generateRandomSolution();
                new_population.push_back(random_solution);
                new_fitness.push_back(objective_function(random_solution));
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

struct ComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    int iterations_completed;
    int population_size;
};

#endif // SORNYAK_MODIFICATIONS_H