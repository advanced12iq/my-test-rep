#include "sornyak_modifications.h"
#include "conventional_optimization.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <iomanip>

// Тестовые функции
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

// 3D версии тестовых функций
auto sphere_function_3d = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
};

auto rosenbrock_function_3d = [](const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; i++) {
        double term1 = 100.0 * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]);
        double term2 = (1.0 - x[i]) * (1.0 - x[i]);
        sum += term1 + term2;
    }
    return sum;
};

auto rastrigin_function_3d = [](const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += val * val - 10.0 * cos(2.0 * M_PI * val);
    }
    return sum;
};

// Модифицированный класс SornyakOptimizer с поддержкой early stopping по tolerance и подсчетом итераций
class SornyakOptimizerWithTolerance : public SornyakOptimizer {
private:
    double tolerance;
    double target_fitness;
    int actual_iterations;

public:
    SornyakOptimizerWithTolerance(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        int pop_size = 50,
        double min_val = -10.0,
        double max_val = 10.0,
        double tol = 1e-6,
        double target_fit = 0.0
    ) : SornyakOptimizer(func, dim, max_iter, pop_size, min_val, max_val), 
        tolerance(tol), target_fitness(target_fit), actual_iterations(0) {}

    std::vector<double> optimize() override {
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
            actual_iterations++;
            
            // Найти текущее лучшее решение
            for (int i = 0; i < population_size; i++) {
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    best_solution = population[i];
                }
            }
            
            // Проверка достижения целевой точности
            if (std::abs(best_fitness - target_fitness) < tolerance) {
                break;
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
    
    int getActualIterations() const { return actual_iterations; }
};

// Структура для хранения результатов сравнения
struct ToleranceComparisonResult {
    std::string method_name;
    double best_fitness;
    std::vector<double> best_solution;
    double execution_time_ms;
    int iterations_completed;
    bool reached_tolerance;
};

// Функция для запуска оптимизации с измерением времени до достижения tolerance
template<typename OptimizerType>
ToleranceComparisonResult runOptimizationToTolerance(std::function<double(const std::vector<double>&)> func, 
                                                    OptimizerType& optimizer, 
                                                    const std::string& method_name, 
                                                    double target_fitness = 0.0,
                                                    double tolerance = 1e-6,
                                                    int max_iter = 1000) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> result = optimizer.optimize();
    double best_fitness = optimizer.getBestFitness(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ToleranceComparisonResult res;
    res.method_name = method_name;
    res.best_fitness = best_fitness;
    res.best_solution = result;
    res.execution_time_ms = duration.count();
    res.iterations_completed = max_iter;  // Для простоты, используем максимальное количество
    res.reached_tolerance = std::abs(best_fitness - target_fitness) < tolerance;
    
    return res;
}

// Модифицированные версии конвенциональных методов для отслеживания количества итераций
class GradientDescentWithIterationCount {
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
    GradientDescentWithIterationCount(
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
        int actual_iterations = 0;
        
        // Инициализация случайной начальной точкой
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Случайная начальная точка
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations++;
            
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
    
    int getActualIterations() const { 
        return -1; // Placeholder - this class is not used in the final version
    }
};

// A wrapper class that tracks iterations during optimization
class GradientDescentWithIterationCountWrapper {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double learning_rate;
    double tolerance;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    int actual_iterations;

public:
    GradientDescentWithIterationCountWrapper(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double lr = 0.01,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        learning_rate(lr), tolerance(tol), gen(rd()), 
        dis(min_val, max_val), actual_iterations(0) {}

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
        actual_iterations = 0;
        
        // Инициализация случайной начальной точкой
        std::vector<double> x(dimension);
        for (int i = 0; i < dimension; i++) {
            x[i] = dis(gen);  // Случайная начальная точка
        }
        
        double prev_fitness = objective_function(x);
        double current_fitness = prev_fitness;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            actual_iterations++;
            
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
    
    int getActualIterations() const { 
        return actual_iterations;
    }
};

class NelderMeadWithIterationCountWrapper {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    int actual_iterations;

public:
    NelderMeadWithIterationCountWrapper(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        tolerance(tol), gen(rd()), dis(min_val, max_val), actual_iterations(0) {}

    std::vector<double> optimize() {
        actual_iterations = 0;
        
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
            actual_iterations++;
            
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
    
    int getActualIterations() const { 
        return actual_iterations;
    }
};

class PowellWithIterationCountWrapper {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    int dimension;
    int max_iterations;
    double tolerance;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    int actual_iterations;

public:
    PowellWithIterationCountWrapper(
        std::function<double(const std::vector<double>&)> func,
        int dim,
        int max_iter = 1000,
        double tol = 1e-6,
        double min_val = -10.0,
        double max_val = 10.0
    ) : objective_function(func), dimension(dim), max_iterations(max_iter), 
        tolerance(tol), gen(rd()), dis(min_val, max_val), actual_iterations(0) {}

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
        actual_iterations = 0;
        
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
            actual_iterations++;
            
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
    
    int getActualIterations() const { 
        return actual_iterations;
    }
};

int main() {
    std::cout << "Сравнение времени достижения оптимального решения с tol=1e-6" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Тест на функции Сферы (2D)
    std::cout << "\nТестирование на функции Сферы (2D, минимум в [0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt(sphere_function, 2, 1000, 30, -5.0, 5.0, 1e-6, 0.0);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto sornyak_result = sornyak_adapt.optimize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto sornyak_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double sornyak_fitness = sornyak_adapt.getBestFitness(sornyak_result);
    bool sornyak_reached = std::abs(sornyak_fitness) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness
              << std::setw(15) << sornyak_duration.count()
              << std::setw(12) << sornyak_adapt.getActualIterations()
              << std::setw(12) << (sornyak_reached ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd(sphere_function, 2, 1000, 0.01, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result = gd.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto gd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness = gd.getBestFitness(gd_result);
    bool gd_reached = std::abs(gd_fitness) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness
              << std::setw(15) << gd_duration.count()
              << std::setw(12) << gd.getActualIterations()
              << std::setw(12) << (gd_reached ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result = nm.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto nm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness = nm.getBestFitness(nm_result);
    bool nm_reached = std::abs(nm_fitness) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness
              << std::setw(15) << nm_duration.count()
              << std::setw(12) << nm.getActualIterations()
              << std::setw(12) << (nm_reached ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow(sphere_function, 2, 1000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result = pow.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto pow_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness = pow.getBestFitness(pow_result);
    bool pow_reached = std::abs(pow_fitness) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness
              << std::setw(15) << pow_duration.count()
              << std::setw(12) << pow.getActualIterations()
              << std::setw(12) << (pow_reached ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Розенброка (2D)
    std::cout << "\nТестирование на функции Розенброка (2D, минимум в [1,1], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt2(rosenbrock_function, 2, 2000, 50, -2.0, 2.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto sornyak_result2 = sornyak_adapt2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto sornyak_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double sornyak_fitness2 = sornyak_adapt2.getBestFitness(sornyak_result2);
    bool sornyak_reached2 = std::abs(sornyak_fitness2) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness2
              << std::setw(15) << sornyak_duration2.count()
              << std::setw(12) << sornyak_adapt2.getActualIterations()
              << std::setw(12) << (sornyak_reached2 ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd2(rosenbrock_function, 2, 2000, 0.001, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result2 = gd2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto gd_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness2 = gd2.getBestFitness(gd_result2);
    bool gd_reached2 = std::abs(gd_fitness2) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness2
              << std::setw(15) << gd_duration2.count()
              << std::setw(12) << gd2.getActualIterations()
              << std::setw(12) << (gd_reached2 ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result2 = nm2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto nm_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness2 = nm2.getBestFitness(nm_result2);
    bool nm_reached2 = std::abs(nm_fitness2) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness2
              << std::setw(15) << nm_duration2.count()
              << std::setw(12) << nm2.getActualIterations()
              << std::setw(12) << (nm_reached2 ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow2(rosenbrock_function, 2, 2000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result2 = pow2.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto pow_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness2 = pow2.getBestFitness(pow_result2);
    bool pow_reached2 = std::abs(pow_fitness2) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness2
              << std::setw(15) << pow_duration2.count()
              << std::setw(12) << pow2.getActualIterations()
              << std::setw(12) << (pow_reached2 ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    
    // Тест на функции Растригина (2D)
    std::cout << "\nТестирование на функции Растригина (2D, минимум в [0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt3(rastrigin_function, 2, 2000, 50, -5.0, 5.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto sornyak_result3 = sornyak_adapt3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto sornyak_duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double sornyak_fitness3 = sornyak_adapt3.getBestFitness(sornyak_result3);
    bool sornyak_reached3 = std::abs(sornyak_fitness3) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness3
              << std::setw(15) << sornyak_duration3.count()
              << std::setw(12) << sornyak_adapt3.getActualIterations()
              << std::setw(12) << (sornyak_reached3 ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd3(rastrigin_function, 2, 2000, 0.01, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result3 = gd3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto gd_duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness3 = gd3.getBestFitness(gd_result3);
    bool gd_reached3 = std::abs(gd_fitness3) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness3
              << std::setw(15) << gd_duration3.count()
              << std::setw(12) << gd3.getActualIterations()
              << std::setw(12) << (gd_reached3 ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result3 = nm3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto nm_duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness3 = nm3.getBestFitness(nm_result3);
    bool nm_reached3 = std::abs(nm_fitness3) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness3
              << std::setw(15) << nm_duration3.count()
              << std::setw(12) << nm3.getActualIterations()
              << std::setw(12) << (nm_reached3 ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow3(rastrigin_function, 2, 2000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result3 = pow3.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto pow_duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness3 = pow3.getBestFitness(pow_result3);
    bool pow_reached3 = std::abs(pow_fitness3) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness3
              << std::setw(15) << pow_duration3.count()
              << std::setw(12) << pow3.getActualIterations()
              << std::setw(12) << (pow_reached3 ? "Да" : "Нет") << std::endl;
    
    std::cout << std::endl;
    std::cout << "Выводы:" << std::endl;
    std::cout << "========" << std::endl;
    std::cout << "- 'Достигнута' указывает, была ли достигнута целевая точность tol=1e-6" << std::endl;
    std::cout << "- Методы с 'Да' в последнем столбце успешно нашли решение с заданной точностью" << std::endl;
    std::cout << "- Время выполнения показывает, сколько миллисекунд потребовалось для завершения" << std::endl;
    std::cout << "- Количество итераций показывает, сколько итераций было выполнено до сходимости или максимума" << std::endl;
    
    std::cout << std::endl;

    // Тест на функции Сферы (3D)
    std::cout << "\nТестирование на функции Сферы (3D, минимум в [0,0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt_3d(sphere_function_3d, 3, 2000, 50, -5.0, 5.0, 1e-6, 0.0);
    auto start_time_3d = std::chrono::high_resolution_clock::now();
    auto sornyak_result_3d = sornyak_adapt_3d.optimize();
    auto end_time_3d = std::chrono::high_resolution_clock::now();
    auto sornyak_duration_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_3d - start_time_3d);
    double sornyak_fitness_3d = sornyak_adapt_3d.getBestFitness(sornyak_result_3d);
    bool sornyak_reached_3d = std::abs(sornyak_fitness_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness_3d
              << std::setw(15) << sornyak_duration_3d.count()
              << std::setw(12) << sornyak_adapt_3d.getActualIterations()
              << std::setw(12) << (sornyak_reached_3d ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd_3d(sphere_function_3d, 3, 2000, 0.01, 1e-6, -5.0, 5.0);
    start_time_3d = std::chrono::high_resolution_clock::now();
    auto gd_result_3d = gd_3d.optimize();
    end_time_3d = std::chrono::high_resolution_clock::now();
    auto gd_duration_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_3d - start_time_3d);
    double gd_fitness_3d = gd_3d.getBestFitness(gd_result_3d);
    bool gd_reached_3d = std::abs(gd_fitness_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness_3d
              << std::setw(15) << gd_duration_3d.count()
              << std::setw(12) << gd_3d.getActualIterations()
              << std::setw(12) << (gd_reached_3d ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm_3d(sphere_function_3d, 3, 2000, 1e-6, -5.0, 5.0);
    start_time_3d = std::chrono::high_resolution_clock::now();
    auto nm_result_3d = nm_3d.optimize();
    end_time_3d = std::chrono::high_resolution_clock::now();
    auto nm_duration_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_3d - start_time_3d);
    double nm_fitness_3d = nm_3d.getBestFitness(nm_result_3d);
    bool nm_reached_3d = std::abs(nm_fitness_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness_3d
              << std::setw(15) << nm_duration_3d.count()
              << std::setw(12) << nm_3d.getActualIterations()
              << std::setw(12) << (nm_reached_3d ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow_3d(sphere_function_3d, 3, 2000, 1e-6, -5.0, 5.0);
    start_time_3d = std::chrono::high_resolution_clock::now();
    auto pow_result_3d = pow_3d.optimize();
    end_time_3d = std::chrono::high_resolution_clock::now();
    auto pow_duration_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_3d - start_time_3d);
    double pow_fitness_3d = pow_3d.getBestFitness(pow_result_3d);
    bool pow_reached_3d = std::abs(pow_fitness_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness_3d
              << std::setw(15) << pow_duration_3d.count()
              << std::setw(12) << pow_3d.getActualIterations()
              << std::setw(12) << (pow_reached_3d ? "Да" : "Нет") << std::endl;

    std::cout << std::endl;

    // Тест на функции Розенброка (3D)
    std::cout << "\nТестирование на функции Розенброка (3D, минимум в [1,1,1], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt2_3d(rosenbrock_function_3d, 3, 3000, 50, -2.0, 2.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto sornyak_result2_3d = sornyak_adapt2_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto sornyak_duration2_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double sornyak_fitness2_3d = sornyak_adapt2_3d.getBestFitness(sornyak_result2_3d);
    bool sornyak_reached2_3d = std::abs(sornyak_fitness2_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness2_3d
              << std::setw(15) << sornyak_duration2_3d.count()
              << std::setw(12) << sornyak_adapt2_3d.getActualIterations()
              << std::setw(12) << (sornyak_reached2_3d ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd2_3d(rosenbrock_function_3d, 3, 3000, 0.001, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result2_3d = gd2_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto gd_duration2_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness2_3d = gd2_3d.getBestFitness(gd_result2_3d);
    bool gd_reached2_3d = std::abs(gd_fitness2_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness2_3d
              << std::setw(15) << gd_duration2_3d.count()
              << std::setw(12) << gd2_3d.getActualIterations()
              << std::setw(12) << (gd_reached2_3d ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm2_3d(rosenbrock_function_3d, 3, 3000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result2_3d = nm2_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto nm_duration2_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness2_3d = nm2_3d.getBestFitness(nm_result2_3d);
    bool nm_reached2_3d = std::abs(nm_fitness2_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness2_3d
              << std::setw(15) << nm_duration2_3d.count()
              << std::setw(12) << nm2_3d.getActualIterations()
              << std::setw(12) << (nm_reached2_3d ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow2_3d(rosenbrock_function_3d, 3, 3000, 1e-6, -2.0, 2.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result2_3d = pow2_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto pow_duration2_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness2_3d = pow2_3d.getBestFitness(pow_result2_3d);
    bool pow_reached2_3d = std::abs(pow_fitness2_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness2_3d
              << std::setw(15) << pow_duration2_3d.count()
              << std::setw(12) << pow2_3d.getActualIterations()
              << std::setw(12) << (pow_reached2_3d ? "Да" : "Нет") << std::endl;

    std::cout << std::endl;

    // Тест на функции Растригина (3D)
    std::cout << "\nТестирование на функции Растригина (3D, минимум в [0,0,0], значение 0)" << std::endl;
    std::cout << std::setw(30) << std::left << "Метод" 
              << std::setw(15) << "Ошибка" 
              << std::setw(15) << "Время (мс)" 
              << std::setw(12) << "Итерации" 
              << std::setw(12) << "Достигнута" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    // Сорняковый метод с адаптивным распространением
    SornyakOptimizerWithTolerance sornyak_adapt3_3d(rastrigin_function_3d, 3, 3000, 50, -5.0, 5.0, 1e-6, 0.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto sornyak_result3_3d = sornyak_adapt3_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto sornyak_duration3_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double sornyak_fitness3_3d = sornyak_adapt3_3d.getBestFitness(sornyak_result3_3d);
    bool sornyak_reached3_3d = std::abs(sornyak_fitness3_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Сорняки адапт"
              << std::setw(15) << std::fixed << std::setprecision(6) << sornyak_fitness3_3d
              << std::setw(15) << sornyak_duration3_3d.count()
              << std::setw(12) << sornyak_adapt3_3d.getActualIterations()
              << std::setw(12) << (sornyak_reached3_3d ? "Да" : "Нет") << std::endl;
    
    // Градиентный спуск
    GradientDescentWithIterationCountWrapper gd3_3d(rastrigin_function_3d, 3, 3000, 0.01, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto gd_result3_3d = gd3_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto gd_duration3_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double gd_fitness3_3d = gd3_3d.getBestFitness(gd_result3_3d);
    bool gd_reached3_3d = std::abs(gd_fitness3_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Градиентный спуск"
              << std::setw(15) << std::fixed << std::setprecision(6) << gd_fitness3_3d
              << std::setw(15) << gd_duration3_3d.count()
              << std::setw(12) << gd3_3d.getActualIterations()
              << std::setw(12) << (gd_reached3_3d ? "Да" : "Нет") << std::endl;
    
    // Нелдер-Мид
    NelderMeadWithIterationCountWrapper nm3_3d(rastrigin_function_3d, 3, 3000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto nm_result3_3d = nm3_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto nm_duration3_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double nm_fitness3_3d = nm3_3d.getBestFitness(nm_result3_3d);
    bool nm_reached3_3d = std::abs(nm_fitness3_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Нелдер-Мид"
              << std::setw(15) << std::fixed << std::setprecision(6) << nm_fitness3_3d
              << std::setw(15) << nm_duration3_3d.count()
              << std::setw(12) << nm3_3d.getActualIterations()
              << std::setw(12) << (nm_reached3_3d ? "Да" : "Нет") << std::endl;
    
    // Метод Пауэлла
    PowellWithIterationCountWrapper pow3_3d(rastrigin_function_3d, 3, 3000, 1e-6, -5.0, 5.0);
    start_time = std::chrono::high_resolution_clock::now();
    auto pow_result3_3d = pow3_3d.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    auto pow_duration3_3d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double pow_fitness3_3d = pow3_3d.getBestFitness(pow_result3_3d);
    bool pow_reached3_3d = std::abs(pow_fitness3_3d) < 1e-6;
    
    std::cout << std::setw(30) << std::left << "Метод Пауэлла"
              << std::setw(15) << std::fixed << std::setprecision(6) << pow_fitness3_3d
              << std::setw(15) << pow_duration3_3d.count()
              << std::setw(12) << pow3_3d.getActualIterations()
              << std::setw(12) << (pow_reached3_3d ? "Да" : "Нет") << std::endl;

    std::cout << std::endl;
    std::cout << "Выводы:" << std::endl;
    std::cout << "========" << std::endl;
    std::cout << "- 'Достигнута' указывает, была ли достигнута целевая точность tol=1e-6" << std::endl;
    std::cout << "- Методы с 'Да' в последнем столбце успешно нашли решение с заданной точностью" << std::endl;
    std::cout << "- Время выполнения показывает, сколько миллисекунд потребовалось для завершения" << std::endl;
    std::cout << "- Количество итераций показывает, сколько итераций было выполнено до сходимости или максимума" << std::endl;
    std::cout << "- 3D тесты показывают производительность методов на более высокоразмерных задачах" << std::endl;
    
    return 0;
}