#ifndef CONVENTIONAL_OPTIMIZATION_H
#define CONVENTIONAL_OPTIMIZATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <limits>

/**
 * @brief Метод оптимизации градиентного спуска
 * Итерационный алгоритм первого порядка для нахождения локальных минимумов
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

    // Численное вычисление градиента с использованием центральной разности
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
        // Инициализация случайной начальной точкой
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Случайная начальная точка
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Вычисление градиента
            std::vector<double> gradient = calculateGradient(x);
            
            // Обновление x путем движения в противоположном направлении градиента
            std::vector<double> new_x = x;
            for (int i = 0; i < dimension; i++) {
                new_x[i] -= learning_rate * gradient[i];
            }
            
            // Вычисление новой функции приспособленности
            current_fitness = objective_function(new_x);
            
            // Проверка сходимости
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
 * @brief Метод симплекса Нелдера-Мида
 * Прямой метод поиска для многомерной безусловной оптимизации
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
        // Создание начального симплекса (dimension + 1 точек)
        std::vector<std::vector<double>> simplex(dimension + 1, std::vector<double>(dimension));
        std::vector<double> fitness(dimension + 1);
        
        // Инициализация симплекса случайными точками
        for (int i = 0; i <= dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                simplex[i][j] = dis(gen);
            }
            fitness[i] = objective_function(simplex[i]);
        }
        
        // Коэффициенты для метода Нелдера-Мида
        const double alpha = 1.0;   // отражение
        const double gamma = 2.0;   // расширение
        const double rho = 0.5;     // сжатие
        const double sigma = 0.5;   // уменьшение
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Сортировка симплекса по функции приспособленности (по возрастанию)
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i <= dimension; i++) {
                fitness_indices.push_back({fitness[i], i});
            }
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            // Получение индексов для лучшей, второй худшей и худшей точек
            int best_idx = fitness_indices[0].second;
            int worst_idx = fitness_indices[dimension].second;
            int second_worst_idx = fitness_indices[dimension - 1].second;
            
            // Вычисление центроида всех точек, кроме худшей
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
            
            // Отражение
            std::vector<double> reflected = centroid;
            for (int j = 0; j < dimension; j++) {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
            }
            double reflected_fitness = objective_function(reflected);
            
            if (fitness[best_idx] <= reflected_fitness && reflected_fitness < fitness[second_worst_idx]) {
                // Принять отраженную точку
                simplex[worst_idx] = reflected;
                fitness[worst_idx] = reflected_fitness;
            } else if (reflected_fitness < fitness[best_idx]) {
                // Расширение
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
                // Сжатие
                std::vector<double> contracted = centroid;
                for (int j = 0; j < dimension; j++) {
                    contracted[j] = centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]);
                }
                double contracted_fitness = objective_function(contracted);
                
                if (contracted_fitness < fitness[worst_idx]) {
                    simplex[worst_idx] = contracted;
                    fitness[worst_idx] = contracted_fitness;
                } else {
                    // Уменьшение
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
            
            // Проверка сходимости на основе размера симплекса
            double best_fitness = fitness[best_idx];
            double worst_fitness = fitness[worst_idx];
            
            if (std::abs(worst_fitness - best_fitness) < tolerance) {
                break;
            }
        }
        
        // Найти лучшее решение
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
 * @brief Метод Пауэлла (метод направления)
 * Метод сопряженных направлений для оптимизации без производных
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

    // Поиск линии в заданном направлении с использованием поиска по золотому сечению
    std::vector<double> line_search(const std::vector<double>& start, 
                                   const std::vector<double>& direction, 
                                   double tolerance = 1e-6) {
        const double golden_ratio = (3.0 - sqrt(5.0)) / 2.0;
        
        // Нормализация вектора направления
        std::vector<double> dir = direction;
        double norm = 0.0;
        for (double val : dir) norm += val * val;
        norm = sqrt(norm);
        for (int i = 0; i < dimension; i++) dir[i] /= norm;
        
        // Начальный интервал
        double a = -1.0, b = 1.0;
        
        // Функция для минимизации вдоль линии
        auto line_func = [&](double alpha) {
            std::vector<double> point(dimension);
            for (int i = 0; i < dimension; i++) {
                point[i] = start[i] + alpha * dir[i];
            }
            return objective_function(point);
        };
        
        // Определение границ интервала
        double fa = line_func(a);
        double fb = line_func(b);
        
        if (fa < fb) {
            double temp = a; a = b; b = temp;
            double ftemp = fa; fa = fb; fb = ftemp;
            
            // Расширение интервала
            double c = b + golden_ratio * (b - a);
            double fc = line_func(c);
            
            while (fc < fb) {
                a = b; b = c;
                fa = fb; fb = fc;
                c = b + golden_ratio * (b - a);
                fc = line_func(c);
            }
        }
        
        // Поиск по золотому сечению
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
                // Подгонка параболы
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
        // Инициализация случайной начальной точкой
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Случайная начальная точка
        }
        
        // Инициализация набора направлений (обычно оси координат)
        std::vector<std::vector<double>> directions(dimension, std::vector<double>(dimension, 0.0));
        for (int i = 0; i < dimension; i++) {
            directions[i][i] = 1.0;
        }
        
        double fx = objective_function(x);
        
        for (int iter = 0; iter < max_iterations; iter++) {
            double fx_old = fx;
            int worst_dir = 0;
            double max_reduction = 0.0;
            
            // Минимизация вдоль каждого направления
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
            
            // Построение нового направления на основе разности между началом и концом цикла
            std::vector<double> new_dir(dimension);
            for (int i = 0; i < dimension; i++) {
                new_dir[i] = x[i] - directions[0][i];  // Предполагается, что первая точка была сохранена
            }
            
            // Минимизация вдоль нового направления
            std::vector<double> new_x = line_search(x, new_dir);
            double new_fx = objective_function(new_x);
            
            // Проверка сходимости
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
 * @brief Метод случайного поиска
 * Простой метод, который случайным образом отбирает точки из пространства поиска
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
            // Генерация случайного решения
            std::vector<double> current_solution(dimension);
            for (int i = 0; i < dimension; i++) {
                current_solution[i] = dis(gen);
            }
            
            double current_fitness = objective_function(current_solution);
            
            // Обновление лучшего решения, если текущее лучше
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