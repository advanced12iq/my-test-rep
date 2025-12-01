CXX = g++

CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -I./include

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

MAIN_SRC = $(SRC_DIR)/main.cpp
COMPARISON_SRC = $(SRC_DIR)/comparison_main.cpp
CONVENTIONAL_COMPARISON_SRC = $(SRC_DIR)/conventional_comparison_main.cpp
DETAILED_ANALYSIS_SRC = $(SRC_DIR)/detailed_analysis.cpp
SORNYAK_OPTIMIZATION_SRC = $(SRC_DIR)/sornyak_optimization.cpp

MAIN_EXE = $(BIN_DIR)/sornyak_main
COMPARISON_EXE = $(BIN_DIR)/comparison_main
CONVENTIONAL_COMPARISON_EXE = $(BIN_DIR)/conventional_comparison
DETAILED_ANALYSIS_EXE = $(BIN_DIR)/detailed_analysis
SORNYAK_OPTIMIZATION_EXE = $(BIN_DIR)/sornyak_optimization

$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

all: sornyak comparison conventional_comparison detailed_analysis sornyak_optimization

sornyak: $(MAIN_EXE)

$(MAIN_EXE): $(MAIN_SRC) $(INCLUDE_DIR)/sornyak_optimizer.h $(INCLUDE_DIR)/sornyak_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

comparison: $(COMPARISON_EXE)

$(COMPARISON_EXE): $(COMPARISON_SRC) $(INCLUDE_DIR)/sornyak_optimizer.h $(INCLUDE_DIR)/sornyak_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

conventional_comparison: $(CONVENTIONAL_COMPARISON_EXE)

$(CONVENTIONAL_COMPARISON_EXE): $(CONVENTIONAL_COMPARISON_SRC) $(INCLUDE_DIR)/sornyak_optimizer.h $(INCLUDE_DIR)/sornyak_modifications.h $(INCLUDE_DIR)/conventional_optimization.h
	$(CXX) $(CXXFLAGS) -o $@ $<

detailed_analysis: $(DETAILED_ANALYSIS_EXE)

$(DETAILED_ANALYSIS_EXE): $(DETAILED_ANALYSIS_SRC) $(INCLUDE_DIR)/sornyak_optimizer.h $(INCLUDE_DIR)/sornyak_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

sornyak_optimization: $(SORNYAK_OPTIMIZATION_EXE)

$(SORNYAK_OPTIMIZATION_EXE): $(SORNYAK_OPTIMIZATION_SRC) $(INCLUDE_DIR)/sornyak_optimizer.h $(INCLUDE_DIR)/sornyak_modifications.h
	$(CXX) $(CXXFLAGS) -o $@ $<

run: sornyak
	./$(MAIN_EXE)

run_comparison: comparison
	./$(COMPARISON_EXE)

run_conventional_comparison: conventional_comparison
	./$(CONVENTIONAL_COMPARISON_EXE)

run_detailed_analysis: detailed_analysis
	./$(DETAILED_ANALYSIS_EXE)

run_sornyak_optimization: sornyak_optimization
	./$(SORNYAK_OPTIMIZATION_EXE)

clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(BIN_DIR)/*
	rm -f convergence_data.csv

clean_all: clean
	rm -f convergence_data.csv

.PHONY: all sornyak comparison conventional_comparison detailed_analysis sornyak_optimization run run_comparison run_conventional_comparison run_detailed_analysis run_sornyak_optimization clean clean_all