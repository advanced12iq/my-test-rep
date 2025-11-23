#include "sormyakov_modifications.h"
#include "conventional_optimization.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <iomanip>

// Test functions
auto sphere_function = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
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

auto rastrigin_function = [](const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += val * val - 10.0 * cos(2.0 * M_PI * val);
    }
    return sum;
};

// Function to run optimization and measure time
template<typename OptimizerType>
ComparisonResult runOptimization(std::function<double(const std::vector<double>&)> func, 
                                OptimizerType& optimizer, 
                                const std::string& method_name, 
                                int dimensions,
                                int max_iter = 500,
                                int pop_size = 30) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result = optimizer.optimize();
    double best_fitness = optimizer.getBestFitness(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ComparisonResult res;
    res.method_name = method_name;
    res.best_fitness = best_fitness;
    res.best_solution = result;
    res.execution_time_ms = duration.count();
    res.iterations_completed = max_iter;  // Assuming all iterations completed
    res.population_size = pop_size;
    
    return res;
}

int main() {
    std::cout << "Comprehensive Optimization Method Comparison" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Test on Sphere function (2D)
    std::cout << "\nTesting on Sphere Function (2D, min at [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Sormyakov Original
    SormyakovOptimizer sorm1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result1 = runOptimization(sphere_function, sorm1, "Sormyakov Orig", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result1.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result1.best_fitness
              << std::setw(15) << result1.execution_time_ms
              << std::setw(12) << result1.iterations_completed
              << std::setw(12) << result1.population_size << std::endl;
    
    // Sormyakov with Elitism
    SormyakovWithElitism sorm2(sphere_function, 2, 500, 30, -5.0, 5.0, 0.2);
    auto result2 = runOptimization(sphere_function, sorm2, "Sormyakov Elit", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result2.best_fitness
              << std::setw(15) << result2.execution_time_ms
              << std::setw(12) << result2.iterations_completed
              << std::setw(12) << result2.population_size << std::endl;
    
    // Sormyakov with Adaptive Spread
    SormyakovWithAdaptiveSpread sorm3(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result3 = runOptimization(sphere_function, sorm3, "Sormyakov Adapt", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result3.best_fitness
              << std::setw(15) << result3.execution_time_ms
              << std::setw(12) << result3.iterations_completed
              << std::setw(12) << result3.population_size << std::endl;
    
    // Gradient Descent
    GradientDescentOptimizer gd1(sphere_function, 2, 500, 0.01, 1e-6, -5.0, 5.0);
    auto result4 = runOptimization(sphere_function, gd1, "Gradient Desc", 2, 500, 1);
    std::cout << std::setw(30) << std::left << result4.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result4.best_fitness
              << std::setw(15) << result4.execution_time_ms
              << std::setw(12) << result4.iterations_completed
              << std::setw(12) << result4.population_size << std::endl;
    
    // Nelder-Mead Simplex
    NelderMeadOptimizer nm1(sphere_function, 2, 500, 1e-6, -5.0, 5.0);
    auto result5 = runOptimization(sphere_function, nm1, "Nelder-Mead", 2, 500, 3);
    std::cout << std::setw(30) << std::left << result5.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result5.best_fitness
              << std::setw(15) << result5.execution_time_ms
              << std::setw(12) << result5.iterations_completed
              << std::setw(12) << result5.population_size << std::endl;
    
    // Powell's Method
    PowellOptimizer pow1(sphere_function, 2, 500, 1e-6, -5.0, 5.0);
    auto result6 = runOptimization(sphere_function, pow1, "Powell Method", 2, 500, 1);
    std::cout << std::setw(30) << std::left << result6.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result6.best_fitness
              << std::setw(15) << result6.execution_time_ms
              << std::setw(12) << result6.iterations_completed
              << std::setw(12) << result6.population_size << std::endl;
    
    // Random Search
    RandomSearchOptimizer rs1(sphere_function, 2, 500, -5.0, 5.0);
    auto result7 = runOptimization(sphere_function, rs1, "Random Search", 2, 500, 1);
    std::cout << std::setw(30) << std::left << result7.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result7.best_fitness
              << std::setw(15) << result7.execution_time_ms
              << std::setw(12) << result7.iterations_completed
              << std::setw(12) << result7.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Test on Rosenbrock function (2D)
    std::cout << "\nTesting on Rosenbrock Function (2D, min at [1,1])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Sormyakov Original
    SormyakovOptimizer sorm4(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result8 = runOptimization(rosenbrock_function, sorm4, "Sormyakov Orig", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result8.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result8.best_fitness
              << std::setw(15) << result8.execution_time_ms
              << std::setw(12) << result8.iterations_completed
              << std::setw(12) << result8.population_size << std::endl;
    
    // Sormyakov with Elitism
    SormyakovWithElitism sorm5(rosenbrock_function, 2, 1000, 50, -2.0, 2.0, 0.2);
    auto result9 = runOptimization(rosenbrock_function, sorm5, "Sormyakov Elit", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result9.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result9.best_fitness
              << std::setw(15) << result9.execution_time_ms
              << std::setw(12) << result9.iterations_completed
              << std::setw(12) << result9.population_size << std::endl;
    
    // Sormyakov with Adaptive Spread
    SormyakovWithAdaptiveSpread sorm6(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result10 = runOptimization(rosenbrock_function, sorm6, "Sormyakov Adapt", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result10.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result10.best_fitness
              << std::setw(15) << result10.execution_time_ms
              << std::setw(12) << result10.iterations_completed
              << std::setw(12) << result10.population_size << std::endl;
    
    // Gradient Descent
    GradientDescentOptimizer gd2(rosenbrock_function, 2, 1000, 0.001, 1e-6, -2.0, 2.0);
    auto result11 = runOptimization(rosenbrock_function, gd2, "Gradient Desc", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result11.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result11.best_fitness
              << std::setw(15) << result11.execution_time_ms
              << std::setw(12) << result11.iterations_completed
              << std::setw(12) << result11.population_size << std::endl;
    
    // Nelder-Mead Simplex
    NelderMeadOptimizer nm2(rosenbrock_function, 2, 1000, 1e-6, -2.0, 2.0);
    auto result12 = runOptimization(rosenbrock_function, nm2, "Nelder-Mead", 2, 1000, 3);
    std::cout << std::setw(30) << std::left << result12.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result12.best_fitness
              << std::setw(15) << result12.execution_time_ms
              << std::setw(12) << result12.iterations_completed
              << std::setw(12) << result12.population_size << std::endl;
    
    // Powell's Method
    PowellOptimizer pow2(rosenbrock_function, 2, 1000, 1e-6, -2.0, 2.0);
    auto result13 = runOptimization(rosenbrock_function, pow2, "Powell Method", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result13.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result13.best_fitness
              << std::setw(15) << result13.execution_time_ms
              << std::setw(12) << result13.iterations_completed
              << std::setw(12) << result13.population_size << std::endl;
    
    // Random Search
    RandomSearchOptimizer rs2(rosenbrock_function, 2, 1000, -2.0, 2.0);
    auto result14 = runOptimization(rosenbrock_function, rs2, "Random Search", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result14.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result14.best_fitness
              << std::setw(15) << result14.execution_time_ms
              << std::setw(12) << result14.iterations_completed
              << std::setw(12) << result14.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Test on Rastrigin function (2D)
    std::cout << "\nTesting on Rastrigin Function (2D, min at [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Sormyakov Original
    SormyakovOptimizer sorm7(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result15 = runOptimization(rastrigin_function, sorm7, "Sormyakov Orig", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result15.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result15.best_fitness
              << std::setw(15) << result15.execution_time_ms
              << std::setw(12) << result15.iterations_completed
              << std::setw(12) << result15.population_size << std::endl;
    
    // Sormyakov with Elitism
    SormyakovWithElitism sorm8(rastrigin_function, 2, 1000, 50, -5.0, 5.0, 0.2);
    auto result16 = runOptimization(rastrigin_function, sorm8, "Sormyakov Elit", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result16.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result16.best_fitness
              << std::setw(15) << result16.execution_time_ms
              << std::setw(12) << result16.iterations_completed
              << std::setw(12) << result16.population_size << std::endl;
    
    // Sormyakov with Adaptive Spread
    SormyakovWithAdaptiveSpread sorm9(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result17 = runOptimization(rastrigin_function, sorm9, "Sormyakov Adapt", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result17.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result17.best_fitness
              << std::setw(15) << result17.execution_time_ms
              << std::setw(12) << result17.iterations_completed
              << std::setw(12) << result17.population_size << std::endl;
    
    // Gradient Descent
    GradientDescentOptimizer gd3(rastrigin_function, 2, 1000, 0.01, 1e-6, -5.0, 5.0);
    auto result18 = runOptimization(rastrigin_function, gd3, "Gradient Desc", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result18.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result18.best_fitness
              << std::setw(15) << result18.execution_time_ms
              << std::setw(12) << result18.iterations_completed
              << std::setw(12) << result18.population_size << std::endl;
    
    // Nelder-Mead Simplex
    NelderMeadOptimizer nm3(rastrigin_function, 2, 1000, 1e-6, -5.0, 5.0);
    auto result19 = runOptimization(rastrigin_function, nm3, "Nelder-Mead", 2, 1000, 3);
    std::cout << std::setw(30) << std::left << result19.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result19.best_fitness
              << std::setw(15) << result19.execution_time_ms
              << std::setw(12) << result19.iterations_completed
              << std::setw(12) << result19.population_size << std::endl;
    
    // Powell's Method
    PowellOptimizer pow3(rastrigin_function, 2, 1000, 1e-6, -5.0, 5.0);
    auto result20 = runOptimization(rastrigin_function, pow3, "Powell Method", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result20.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result20.best_fitness
              << std::setw(15) << result20.execution_time_ms
              << std::setw(12) << result20.iterations_completed
              << std::setw(12) << result20.population_size << std::endl;
    
    // Random Search
    RandomSearchOptimizer rs3(rastrigin_function, 2, 1000, -5.0, 5.0);
    auto result21 = runOptimization(rastrigin_function, rs3, "Random Search", 2, 1000, 1);
    std::cout << std::setw(30) << std::left << result21.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result21.best_fitness
              << std::setw(15) << result21.execution_time_ms
              << std::setw(12) << result21.iterations_completed
              << std::setw(12) << result21.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Summary analysis
    std::cout << "\nSummary Analysis:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Comparison of Sormyakov (Weeds) Optimization Methods with Conventional Methods\n\n";
    
    std::cout << "Sormyakov Methods:\n";
    std::cout << "- Sormyakov Orig: Basic Sormyakov (Weeds) Optimization Method\n";
    std::cout << "- Sormyakov Elit: Sormyakov with Elitism (preserves top solutions)\n";
    std::cout << "- Sormyakov Adapt: Sormyakov with Adaptive Spread Factor\n\n";
    
    std::cout << "Conventional Methods:\n";
    std::cout << "- Gradient Desc: First-order iterative optimization using numerical gradients\n";
    std::cout << "- Nelder-Mead: Direct search method using simplex operations (reflection, expansion, contraction)\n";
    std::cout << "- Powell Method: Conjugate direction method without derivatives\n";
    std::cout << "- Random Search: Simple method that randomly samples the search space\n\n";
    
    std::cout << "Performance Metrics:\n";
    std::cout << "- Best Fitness: The function value at the best solution found\n";
    std::cout << "- Time (ms): Execution time in milliseconds\n";
    std::cout << "- Iters: Number of iterations completed\n";
    std::cout << "- Pop Size: Population size used (for population-based methods)\n\n";
    
    std::cout << "Note: For all test functions, lower fitness values indicate better performance.\n";
    std::cout << "The Sphere function has a global minimum of 0, Rosenbrock of 0, and Rastrigin of 0.\n";
    std::cout << "\nKey observations:\n";
    std::cout << "- Gradient descent is fast but may get stuck in local minima\n";
    std::cout << "- Nelder-Mead is robust for low-dimensional problems\n";
    std::cout << "- Powell's method works well for functions with some structure\n";
    std::cout << "- Random search provides a baseline but is generally inefficient\n";
    std::cout << "- Sormyakov methods are nature-inspired and good at global optimization\n";
    
    return 0;
}