#include "sormyakov_optimizer.h"
#include <iostream>
#include <cmath>

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