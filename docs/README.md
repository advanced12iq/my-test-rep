# Sornyak Optimization Method (Weeds Optimization Method)

This repository contains a C++ implementation of the Sornyak Optimization Method, also known as the Weeds Optimization Method (Сорняковый метод оптимизации). This is a nature-inspired optimization algorithm that mimics the behavior of weeds (sorняки) in nature - their ability to spread, adapt, and find optimal growing conditions.

## Algorithm Description

The Sornyak method is modeled after the natural behavior of weeds:
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

## Project Structure

```
/workspace/
├── src/                    # Source code files (.cpp)
├── include/                # Header files (.h)
├── test/                   # Test files (if any)
├── build/                  # Build artifacts
├── bin/                    # Executable files
├── docs/                   # Documentation
├── scripts/                # Script files
└── Makefile                # Build configuration
```

## Build Instructions

To build all executables:

```bash
make all
```

To build specific executables:
- `make sornyak` - Main Sornyak optimizer
- `make comparison` - Comparison of Sornyak modifications
- `make conventional_comparison` - Comparison with conventional methods
- `make detailed_analysis` - Detailed analysis of optimization methods
- `make sornyak_optimization` - Additional optimization program

To run executables:
- `make run` - Run main Sornyak optimizer
- `make run_comparison` - Run Sornyak modifications comparison
- `make run_conventional_comparison` - Run conventional methods comparison
- `make run_detailed_analysis` - Run detailed analysis
- `make run_sornyak_optimization` - Run additional optimization program

To clean build artifacts:
- `make clean` - Remove build artifacts and executables
- `make clean_all` - Remove all build artifacts

## Files

- `src/main.cpp`: Example usage with test functions
- `src/comparison_main.cpp`: Comparison of different Sornyak modifications
- `src/conventional_comparison_main.cpp`: Comprehensive comparison between Sornyak methods and conventional optimization methods
- `src/detailed_analysis.cpp`: Detailed analysis of optimization methods
- `src/sornyak_optimization.cpp`: Original combined implementation
- `include/sornyak_optimizer.h`: Header file containing the SornyakOptimizer class
- `include/sornyak_modifications.h`: Header file containing modified versions of Sornyak method (with elitism, adaptive spread, tournament selection, dynamic population)
- `include/conventional_optimization.h`: Header file containing conventional optimization methods (Gradient Descent, Nelder-Mead, Powell's Method, Random Search)
- `docs/README.md`: This documentation file
- `docs/SORNYAK_METHOD_COMPARISON.md`: Detailed comparison report with conventional methods
- `scripts/run_comparison.sh`: Script to run Sornyak modifications comparison
- `scripts/run_conventional_comparison.sh`: Script to run comprehensive comparison with conventional methods
- `Makefile`: Build configuration

## Test Functions

The implementation includes examples with three classic optimization test functions:
- Sphere function (min at [0,0,...,0])
- Rosenbrock function (min at [1,1,...,1])
- Rastrigin function (min at [0,0,...,0])

## Results

The algorithm successfully finds near-optimal solutions for the test functions, demonstrating the effectiveness of the weed-inspired approach to optimization.