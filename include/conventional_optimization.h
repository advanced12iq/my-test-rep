#ifndef CONVENTIONAL_OPTIMIZATION_H
#define CONVENTIONAL_OPTIMIZATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>

/**
 * @brief Gradient Descent Optimization Method
 * A first-order iterative optimization algorithm for finding local minima
 */
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

    // Numerical gradient calculation using central difference
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

    std::vector<double> optimize() {
        // Initialize with a random starting point
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Random initial point
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Calculate gradient
            std::vector<double> gradient = calculateGradient(x);
            
            // Update x by moving in the opposite direction of the gradient
            std::vector<double> new_x = x;
            for (int i = 0; i < dimension; i++) {
                new_x[i] -= learning_rate * gradient[i];
            }
            
            // Calculate new fitness
            current_fitness = objective_function(new_x);
            
            // Check for convergence
            if (std::abs(prev_fitness - current_fitness) < tolerance) {
                break;
            }
            
            x = new_x;
            prev_fitness = current_fitness;
        }
        
        return x;
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

/**
 * @brief Nelder-Mead Simplex Optimization Method
 * A direct search method for multidimensional unconstrained optimization
 */
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

    std::vector<double> optimize() {
        // Create initial simplex (dimension + 1 points)
        std::vector<std::vector<double>> simplex(dimension + 1, std::vector<double>(dimension));
        std::vector<double> fitness(dimension + 1);
        
        // Initialize simplex with random points
        for (int i = 0; i <= dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                simplex[i][j] = dis(gen);
            }
            fitness[i] = objective_function(simplex[i]);
        }
        
        // Coefficients for Nelder-Mead
        const double alpha = 1.0;   // reflection
        const double gamma = 2.0;   // expansion
        const double rho = 0.5;     // contraction
        const double sigma = 0.5;   // shrink
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Sort simplex by fitness (ascending order)
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i <= dimension; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            // Get indices for best, second worst, and worst points
            int best_idx = fitness_indices[0].second;
            int worst_idx = fitness_indices[dimension].second;
            int second_worst_idx = fitness_indices[dimension - 1].second;
            
            // Calculate centroid of all points except the worst
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
            
            // Reflection
            std::vector<double> reflected = centroid;
            for (int j = 0; j < dimension; j++) {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
            }
            double reflected_fitness = objective_function(reflected);
            
            if (fitness[best_idx] <= reflected_fitness && reflected_fitness < fitness[second_worst_idx]) {
                // Accept reflected point
                simplex[worst_idx] = reflected;
                fitness[worst_idx] = reflected_fitness;
            } else if (reflected_fitness < fitness[best_idx]) {
                // Expansion
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
                // Contraction
                std::vector<double> contracted = centroid;
                for (int j = 0; j < dimension; j++) {
                    contracted[j] = centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]);
                }
                double contracted_fitness = objective_function(contracted);
                
                if (contracted_fitness < fitness[worst_idx]) {
                    simplex[worst_idx] = contracted;
                    fitness[worst_idx] = contracted_fitness;
                } else {
                    // Shrink
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
            
            // Check for convergence based on simplex size
            double best_fitness = fitness[best_idx];
            double worst_fitness = fitness[worst_idx];
            
            if (std::abs(worst_fitness - best_fitness) < tolerance) {
                break;
            }
        }
        
        // Find the best solution
        int best_idx = 0;
        for (int i = 1; i <= dimension; i++) {
            if (fitness[i] < fitness[best_idx]) {
                best_idx = i;
            }
        }
        
        return simplex[best_idx];
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

/**
 * @brief Powell's Method (Direction Set Method)
 * A conjugate direction method for optimization without derivatives
 */
class PowellOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    PowellOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        tolerance(tol), gen(rd()), dis(min_val, max_val) {}

    // Line search in a given direction using golden section search
    std::vector<double> line_search(const std::vector<double>& start, 
                                   const std::vector<double>& direction, 
                                   double tolerance = 1e-6) {
        const double golden_ratio = (3.0 - sqrt(5.0)) / 2.0;
        
        // Normalize direction vector
        std::vector<double> dir = direction;
        double norm = 0.0;
        for (double val : dir) norm += val * val;
        norm = sqrt(norm);
        for (int i = 0; i < dimension; i++) dir[i] /= norm;
        
        // Initial bracket
        double a = -1.0, b = 1.0;
        
        // Function to minimize along the line
        auto line_func = [&](double alpha) {
            std::vector<double> point(dimension);
            for (int i = 0; i < dimension; i++) {
                point[i] = start[i] + alpha * dir[i];
            }
            return objective_function(point);
        };
        
        // Bracket the minimum
        double fa = line_func(a);
        double fb = line_func(b);
        
        if (fa < fb) {
            double temp = a; a = b; b = temp;
            double ftemp = fa; fa = fb; fb = ftemp;
            
            // Expand the bracket
            double c = b + golden_ratio * (b - a);
            double fc = line_func(c);
            
            while (fc < fb) {
                a = b; b = c;
                fa = fb; fb = fc;
                c = b + golden_ratio * (b - a);
                fc = line_func(c);
            }
        }
        
        // Golden section search
        double x = a + golden_ratio * (b - a);
        double fx = line_func(x);
        double w = x, fw = fx, v = x, fv = fx;
        
        double d = 0, e = 0;
        
        for (int iter = 0; iter < 100; iter++) {
            double g = e;
            double u;
            
            double mid = (a + b) / 2.0;
            double tol1 = tolerance * std::abs(x) + 1e-8;
            double tol2 = 2.0 * tol1;
            
            if (std::abs(x - mid) <= (tol2 - (b - a) / 2.0)) {
                break;
            }
            
            if (std::abs(e) > tol1) {
                // Fit parabola
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                
                if (q > 0) p = -p;
                q = std::abs(q);
                
                double etemp = e;
                e = d;
                
                if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                    e = (x >= mid) ? a - x : b - x;
                    d = golden_ratio * e;
                } else {
                    d = p / q;
                    u = x + d;
                    
                    if (u - a < tol2 || b - u < tol2) {
                        d = (u < mid) ? tol1 : -tol1;
                    }
                }
            } else {
                e = (x >= mid) ? a - x : b - x;
                d = golden_ratio * e;
            }
            
            u = (std::abs(d) >= tol1) ? x + d : x + ((d > 0) ? tol1 : -tol1);
            double fu = line_func(u);
            
            if (fu <= fx) {
                if (u >= x) a = x; else b = x;
                v = w; fv = fw;
                w = x; fw = fx;
                x = u; fx = fu;
            } else {
                if (u < x) a = u; else b = u;
                if (fu <= fw || w == x) {
                    v = w; fv = fw;
                    w = u; fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u; fv = fu;
                }
            }
        }
        
        std::vector<double> result(dimension);
        for (int i = 0; i < dimension; i++) {
            result[i] = start[i] + x * dir[i];
        }
        
        return result;
    }

    std::vector<double> optimize() {
        // Initialize with a random starting point
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Random initial point
        }
        
        // Initialize direction set (usually coordinate axes)
        std::vector<std::vector<double>> directions(dimension, std::vector<double>(dimension, 0.0));
        for (int i = 0; i < dimension; i++) {
            directions[i][i] = 1.0;
        }
        
        double fx = objective_function(x);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            double fx_old = fx;
            int worst_dir = 0;
            double max_reduction = 0.0;
            
            // Minimize along each direction
            for (int i = 0; i < dimension; i++) {
                std::vector<double> new_x = line_search(x, directions[i]);
                double new_fx = objective_function(new_x);
                
                if (fx - new_fx > max_reduction) {
                    max_reduction = fx - new_fx;
                    worst_dir = i;
                }
                
                x = new_x;
                fx = new_fx;
            }
            
            // Construct new direction based on the difference between start and end of cycle
            std::vector<double> new_dir(dimension);
            for (int i = 0; i < dimension; i++) {
                new_dir[i] = x[i] - directions[0][i];  // Assuming first point was saved
            }
            
            // Minimize along the new direction
            std::vector<double> new_x = line_search(x, new_dir);
            double new_fx = objective_function(new_x);
            
            // Check for convergence
            if (std::abs(2.0 * (fx_old - new_fx)) <= tolerance * (std::abs(fx_old) + std::abs(new_fx))) {
                break;
            }
            
            x = new_x;
            fx = new_fx;
        }
        
        return x;
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

/**
 * @brief Random Search Optimization Method
 * A simple method that randomly samples the search space
 */
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

    std::vector<double> optimize() {
        std::vector<double> best_solution(dimension);
        double best_fitness = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Generate random solution
            std::vector<double> current_solution(dimension);
            for (int i = 0; i < dimension; i++) {
                current_solution[i] = dis(gen);
            }
            
            double current_fitness = objective_function(current_solution);
            
            // Update best solution if current is better
            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_solution = current_solution;
            }
        }
        
        return best_solution;
    }
    
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
};

#endif // CONVENTIONAL_OPTIMIZATION_H