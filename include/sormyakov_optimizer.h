#ifndef SORMYAKOV_OPTIMIZER_H
#define SORMYAKOV_OPTIMIZER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

/**
 * @brief Метод оптимизации Сормякова (Метод оптимизации сорняков)
 * 
 * Эта реализация представляет собой вдохновленный природой алгоритм оптимизации,
 * который имитирует поведение сорняков в природе - их способность
 * распространяться, адаптироваться и находить оптимальные условия для роста.
 */
class SormyakovOptimizer {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    int population_size;
    double min_value;
    double max_value;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

public:
    /**
     * @brief Конструктор оптимизатора Сормякова
     * @param func Функция цели для минимизации
     * @param dim Количество измерений в пространстве поиска
     * @param max_iter Максимальное количество итераций
     * @param pop_size Размер популяции (количество "сорняков")
     * @param min_val Минимальное значение для каждого измерения
     * @param max_val Максимальное значение для каждого измерения
     */
    SormyakovOptimizer(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        population_size(pop_size), min_value(min_val), max_value(max_val), 
        gen(rd()), dis(0.0, 1.0) {}

    /**
     * @brief Генерировать случайное решение в пределах границ
     */
    std::vector<double> generateRandomSolution() {
        std::vector<double> solution(dimension);
        for (int i = 0; i < dimension; i++) {
            solution[i] = min_value + (max_value - min_value) * dis(gen);
        }
        return solution;
    }

    /**
     * @brief Создать новое решение путем "распространения" из существующего решения
     * Это имитирует, как сорняки распространяются и адаптируются к новым местам
     */
    std::vector<double> spreadSolution(const std::vector<double>& parent, double spread_factor) {
        std::vector<double> child = parent;
        for (int i = 0; i < dimension; i++) {
            // Добавить случайное изменение для имитации распространения
            double variation = (dis(gen) - 0.5) * 2.0 * spread_factor;
            child[i] += variation;
            
            // Оставить в пределах границ
            child[i] = std::max(min_value, std::min(max_value, child[i]));
        }
        return child;
    }

    /**
     * @brief Основной алгоритм оптимизации
     */
    std::vector<double> optimize() {
        // Инициализировать популяцию случайными решениями
        std::vector<std::vector<double>> population(population_size);
        std::vector<double> fitness(population_size);
        
        for (int i = 0; i < population_size; i++) {
            population[i] = generateRandomSolution();
            fitness[i] = objective_function(population[i]);
        }
        
        std::vector<double> best_solution = population[0];
        double best_fitness = fitness[0];
        
        // Итеративно улучшать популяцию
        for (int iter = 0; iter < max_iterations; iter++) {
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Рассчитать фактор распространения на основе итерации (уменьшается со временем)
            double spread_factor = (max_iterations - iter) * (max_value - min_value) / (2.0 * max_iterations);
            
            // Генерировать новые решения путем распространения от существующих
            std::vector<std::vector<double>> new_population;
            std::vector<double> new_fitness;
            
            for (int i = 0; i < population_size; i++) {
                // Каждое решение создает новое путем распространения
                std::vector<double> new_solution = spreadSolution(population[i], spread_factor);
                double new_fitness_val = objective_function(new_solution);
                
                // Сохранить лучшее из родителя и потомка
                if (new_fitness_val < fitness[i]) {
                    new_population.push_back(new_solution);
                    new_fitness.push_back(new_fitness_val);
                } else {
                    new_population.push_back(population[i]);
                    new_fitness.push_back(fitness[i]);
                }
            }
            
            // Обновить популяцию
            population = new_population;
            fitness = new_fitness;
            
            // Иногда добавлять совершенно новые случайные решения для поддержания разнообразия
            if (iter % 100 == 0) {
                for (int i = 0; i < population_size / 10; i++) {
                    int idx = static_cast<int>(dis(gen) * population_size);
                    population[idx] = generateRandomSolution();
                    fitness[idx] = objective_function(population[idx]);
                }
            }
        }
        
        return best_solution;
    }
    
    /**
     * @brief Получить лучшее найденное значение пригодности
     */
    double getBestFitness(const std::vector<double>& solution) {
        return objective_function(solution);
    }
    
    /**
     * @brief Получить параметры оптимизации
     */
    int getDimension() const { return dimension; }
    int getMaxIterations() const { return max_iterations; }
    int getPopulationSize() const { return population_size; }
};

#endif // SORMYAKOV_OPTIMIZER_H