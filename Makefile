# Makefile for Sormyakov Optimization Method Project

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -I./include

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
MAIN_SRC = $(SRC_DIR)/main.cpp
COMPARISON_SRC = $(SRC_DIR)/comparison_main.cpp
CONVENTIONAL_COMPARISON_SRC = $(SRC_DIR)/conventional_comparison_main.cpp
DETAILED_ANALYSIS_SRC = $(SRC_DIR)/detailed_analysis.cpp
SORMYAKOV_OPTIMIZATION_SRC = $(SRC_DIR)/sormyakov_optimization.cpp

# Executable names
MAIN_EXE = $(BIN_DIR)/sormyakov_main
COMPARISON_EXE = $(BIN_DIR)/comparison_main
CONVENTIONAL_COMPARISON_EXE = $(BIN_DIR)/conventional_comparison
DETAILED_ANALYSIS_EXE = $(BIN_DIR)/detailed_analysis
SORMYAKOV_OPTIMIZATION_EXE = $(BIN_DIR)/sormyakov_optimization

# Ensure directories exist
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

# Default target
all: sormyakov comparison conventional_comparison detailed_analysis sormyakov_optimization

# Build the main Sormyakov optimizer
sormyakov: $(MAIN_EXE)

$(MAIN_EXE): $(MAIN_SRC) $(INCLUDE_DIR)/sormyakov_optimizer.h $(INCLUDE_DIR)/sormyakov_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build the comparison program
comparison: $(COMPARISON_EXE)

$(COMPARISON_EXE): $(COMPARISON_SRC) $(INCLUDE_DIR)/sormyakov_optimizer.h $(INCLUDE_DIR)/sormyakov_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build the conventional comparison program
conventional_comparison: $(CONVENTIONAL_COMPARISON_EXE)

$(CONVENTIONAL_COMPARISON_EXE): $(CONVENTIONAL_COMPARISON_SRC) $(INCLUDE_DIR)/sormyakov_optimizer.h $(INCLUDE_DIR)/sormyakov_modifications.h $(INCLUDE_DIR)/conventional_optimization.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build the detailed analysis program
detailed_analysis: $(DETAILED_ANALYSIS_EXE)

$(DETAILED_ANALYSIS_EXE): $(DETAILED_ANALYSIS_SRC) $(INCLUDE_DIR)/sormyakov_optimizer.h $(INCLUDE_DIR)/sormyakov_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build the sormyakov optimization program
sormyakov_optimization: $(SORMYAKOV_OPTIMIZATION_EXE)

$(SORMYAKOV_OPTIMIZATION_EXE): $(SORMYAKOV_OPTIMIZATION_SRC) $(INCLUDE_DIR)/sormyakov_optimizer.h $(INCLUDE_DIR)/sormyakov_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# Run targets
run: sormyakov
	./$(MAIN_EXE)

run_comparison: comparison
	./$(COMPARISON_EXE)

run_conventional_comparison: conventional_comparison
	./$(CONVENTIONAL_COMPARISON_EXE)

run_detailed_analysis: detailed_analysis
	./$(DETAILED_ANALYSIS_EXE)

run_sormyakov_optimization: sormyakov_optimization
	./$(SORMYAKOV_OPTIMIZATION_EXE)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(BIN_DIR)/*
	rm -f convergence_data.csv

# Clean all including documentation
clean_all: clean
	rm -f convergence_data.csv

# Phony targets
.PHONY: all sormyakov comparison conventional_comparison detailed_analysis sormyakov_optimization run run_comparison run_conventional_comparison run_detailed_analysis run_sormyakov_optimization clean clean_all