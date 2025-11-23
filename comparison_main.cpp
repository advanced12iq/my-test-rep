#include "sormyakov_modifications.h"
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
    std::cout << "Sormyakov Optimization Method Comparison" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Test on Sphere function (2D)
    std::cout << "\nTesting on Sphere Function (2D, min at [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Original Sormyakov
    SormyakovOptimizer optimizer1(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result1 = runOptimization(sphere_function, optimizer1, "Original", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result1.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result1.best_fitness
              << std::setw(15) << result1.execution_time_ms
              << std::setw(12) << result1.iterations_completed
              << std::setw(12) << result1.population_size << std::endl;
    
    // With Elitism
    SormyakovWithElitism optimizer2(sphere_function, 2, 500, 30, -5.0, 5.0, 0.2);
    auto result2 = runOptimization(sphere_function, optimizer2, "With Elitism", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result2.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result2.best_fitness
              << std::setw(15) << result2.execution_time_ms
              << std::setw(12) << result2.iterations_completed
              << std::setw(12) << result2.population_size << std::endl;
    
    // With Adaptive Spread
    SormyakovWithAdaptiveSpread optimizer3(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result3 = runOptimization(sphere_function, optimizer3, "Adaptive Spread", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result3.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result3.best_fitness
              << std::setw(15) << result3.execution_time_ms
              << std::setw(12) << result3.iterations_completed
              << std::setw(12) << result3.population_size << std::endl;
    
    // With Tournament Selection
    SormyakovWithTournament optimizer4(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result4 = runOptimization(sphere_function, optimizer4, "Tournament Sel", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result4.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result4.best_fitness
              << std::setw(15) << result4.execution_time_ms
              << std::setw(12) << result4.iterations_completed
              << std::setw(12) << result4.population_size << std::endl;
    
    // With Dynamic Population
    SormyakovWithDynamicPopulation optimizer5(sphere_function, 2, 500, 30, -5.0, 5.0);
    auto result5 = runOptimization(sphere_function, optimizer5, "Dynamic Pop", 2, 500, 30);
    std::cout << std::setw(30) << std::left << result5.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result5.best_fitness
              << std::setw(15) << result5.execution_time_ms
              << std::setw(12) << result5.iterations_completed
              << std::setw(12) << result5.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Test on Rosenbrock function (2D)
    std::cout << "\nTesting on Rosenbrock Function (2D, min at [1,1])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Original Sormyakov
    SormyakovOptimizer optimizer6(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result6 = runOptimization(rosenbrock_function, optimizer6, "Original", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result6.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result6.best_fitness
              << std::setw(15) << result6.execution_time_ms
              << std::setw(12) << result6.iterations_completed
              << std::setw(12) << result6.population_size << std::endl;
    
    // With Elitism
    SormyakovWithElitism optimizer7(rosenbrock_function, 2, 1000, 50, -2.0, 2.0, 0.2);
    auto result7 = runOptimization(rosenbrock_function, optimizer7, "With Elitism", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result7.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result7.best_fitness
              << std::setw(15) << result7.execution_time_ms
              << std::setw(12) << result7.iterations_completed
              << std::setw(12) << result7.population_size << std::endl;
    
    // With Adaptive Spread
    SormyakovWithAdaptiveSpread optimizer8(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result8 = runOptimization(rosenbrock_function, optimizer8, "Adaptive Spread", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result8.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result8.best_fitness
              << std::setw(15) << result8.execution_time_ms
              << std::setw(12) << result8.iterations_completed
              << std::setw(12) << result8.population_size << std::endl;
    
    // With Tournament Selection
    SormyakovWithTournament optimizer9(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result9 = runOptimization(rosenbrock_function, optimizer9, "Tournament Sel", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result9.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result9.best_fitness
              << std::setw(15) << result9.execution_time_ms
              << std::setw(12) << result9.iterations_completed
              << std::setw(12) << result9.population_size << std::endl;
    
    // With Dynamic Population
    SormyakovWithDynamicPopulation optimizer10(rosenbrock_function, 2, 1000, 50, -2.0, 2.0);
    auto result10 = runOptimization(rosenbrock_function, optimizer10, "Dynamic Pop", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result10.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result10.best_fitness
              << std::setw(15) << result10.execution_time_ms
              << std::setw(12) << result10.iterations_completed
              << std::setw(12) << result10.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Test on Rastrigin function (2D)
    std::cout << "\nTesting on Rastrigin Function (2D, min at [0,0])" << std::endl;
    std::cout << std::setw(30) << std::left << "Method" 
              << std::setw(15) << "Best Fitness" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Iters" 
              << std::setw(12) << "Pop Size" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Original Sormyakov
    SormyakovOptimizer optimizer11(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result11 = runOptimization(rastrigin_function, optimizer11, "Original", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result11.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result11.best_fitness
              << std::setw(15) << result11.execution_time_ms
              << std::setw(12) << result11.iterations_completed
              << std::setw(12) << result11.population_size << std::endl;
    
    // With Elitism
    SormyakovWithElitism optimizer12(rastrigin_function, 2, 1000, 50, -5.0, 5.0, 0.2);
    auto result12 = runOptimization(rastrigin_function, optimizer12, "With Elitism", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result12.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result12.best_fitness
              << std::setw(15) << result12.execution_time_ms
              << std::setw(12) << result12.iterations_completed
              << std::setw(12) << result12.population_size << std::endl;
    
    // With Adaptive Spread
    SormyakovWithAdaptiveSpread optimizer13(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result13 = runOptimization(rastrigin_function, optimizer13, "Adaptive Spread", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result13.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result13.best_fitness
              << std::setw(15) << result13.execution_time_ms
              << std::setw(12) << result13.iterations_completed
              << std::setw(12) << result13.population_size << std::endl;
    
    // With Tournament Selection
    SormyakovWithTournament optimizer14(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result14 = runOptimization(rastrigin_function, optimizer14, "Tournament Sel", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result14.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result14.best_fitness
              << std::setw(15) << result14.execution_time_ms
              << std::setw(12) << result14.iterations_completed
              << std::setw(12) << result14.population_size << std::endl;
    
    // With Dynamic Population
    SormyakovWithDynamicPopulation optimizer15(rastrigin_function, 2, 1000, 50, -5.0, 5.0);
    auto result15 = runOptimization(rastrigin_function, optimizer15, "Dynamic Pop", 2, 1000, 50);
    std::cout << std::setw(30) << std::left << result15.method_name
              << std::setw(15) << std::fixed << std::setprecision(6) << result15.best_fitness
              << std::setw(15) << result15.execution_time_ms
              << std::setw(12) << result15.iterations_completed
              << std::setw(12) << result15.population_size << std::endl;
    
    std::cout << std::endl;
    
    // Summary analysis
    std::cout << "\nSummary Analysis:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Each method was tested on three classic optimization functions:\n";
    std::cout << "- Sphere function: f(x) = sum(x_i^2), global minimum at [0,0,...,0]\n";
    std::cout << "- Rosenbrock function: f(x) = sum[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2], global minimum at [1,1,...,1]\n";
    std::cout << "- Rastrigin function: f(x) = 10*n + sum[x_i^2 - 10*cos(2*PI*x_i)], global minimum at [0,0,...,0]\n\n";
    
    std::cout << "Method Descriptions:\n";
    std::cout << "- Original: Basic Sormyakov (Weeds) Optimization Method\n";
    std::cout << "- With Elitism: Preserves top solutions to maintain good solutions\n";
    std::cout << "- Adaptive Spread: Adjusts spread factor based on population diversity\n";
    std::cout << "- Tournament Sel: Uses tournament selection to choose parents\n";
    std::cout << "- Dynamic Pop: Adjusts population size based on convergence rate\n\n";
    
    std::cout << "Performance Metrics:\n";
    std::cout << "- Best Fitness: The function value at the best solution found\n";
    std::cout << "- Time (ms): Execution time in milliseconds\n";
    std::cout << "- Iters: Number of iterations completed\n";
    std::cout << "- Pop Size: Population size used\n\n";
    
    std::cout << "Note: For all test functions, lower fitness values indicate better performance.\n";
    std::cout << "The Sphere function has a global minimum of 0, Rosenbrock of 0, and Rastrigin of 0.\n";
    
    return 0;
}