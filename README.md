# Sormyakov Optimization Method (Weeds Optimization Method)

This repository contains a C++ implementation of the Sormyakov Optimization Method, also known as the Weeds Optimization Method (Сорняковый метод оптимизации). This is a nature-inspired optimization algorithm that mimics the behavior of weeds (sorняки) in nature - their ability to spread, adapt, and find optimal growing conditions.

## Algorithm Description

The Sormyakov method is modeled after the natural behavior of weeds:
- Weeds spread from existing locations to new areas
- They adapt to local conditions to survive
- Better-adapted weeds are more likely to survive and reproduce
- The population evolves over time to find optimal growing conditions

## Implementation Details

The algorithm works as follows:
1. Initialize a population of random solutions (weeds)
2. Evaluate the fitness of each solution
3. For each iteration:
   - Each solution creates a new solution by "spreading" (adding random variation)
   - Keep the better of parent and child solutions
   - Gradually decrease the spread factor over time
   - Occasionally introduce new random solutions to maintain diversity
4. Return the best solution found

## Files

- `sormyakov_optimizer.h`: Header file containing the SormyakovOptimizer class
- `sormyakov_modifications.h`: Header file containing modified versions of Sormyakov method (with elitism, adaptive spread, tournament selection, dynamic population)
- `main.cpp`: Example usage with test functions
- `sormyakov_optimization.cpp`: Original combined implementation
- `conventional_optimization.h`: Header file containing conventional optimization methods (Gradient Descent, Nelder-Mead, Powell's Method, Random Search)
- `comparison_main.cpp`: Comparison of different Sormyakov modifications
- `conventional_comparison_main.cpp`: Comprehensive comparison between Sormyakov methods and conventional optimization methods
- `SORMYAKOV_METHOD_COMPARISON.md`: Detailed comparison report with conventional methods
- `run_comparison.sh`: Script to run Sormyakov modifications comparison
- `run_conventional_comparison.sh`: Script to run comprehensive comparison with conventional methods

## Test Functions

The implementation includes examples with three classic optimization test functions:
- Sphere function (min at [0,0,...,0])
- Rosenbrock function (min at [1,1,...,1])
- Rastrigin function (min at [0,0,...,0])

## Compilation and Execution

```bash
g++ -std=c++11 main.cpp -o sormyakov_main -lm
./sormyakov_main
```

## Results

The algorithm successfully finds near-optimal solutions for the test functions, demonstrating the effectiveness of the weed-inspired approach to optimization.