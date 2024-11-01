#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main() {
    // Генерируем случайные данные для примера
    MatrixXd X(100, 2); // Матрица признаков размером 100x2
    VectorXd y(100); // Вектор меток классов

    // Инициализация данных (замените этот блок кода на ваши данные)
    X.setRandom();
    y = (X.array().sum(1) > 0).cast<double>();

    // Добавляем столбец с единицами для учёта свободного члена
    MatrixXd X_augmented(X.rows(), X.cols() + 1);
    X_augmented << MatrixXd::Ones(X.rows(), 1), X;

    // Инициализация параметров модели
    VectorXd theta = VectorXd::Zero(X_augmented.cols());

    // Обучение модели логистической регрессии с помощью градиентного спуска
    const double learning_rate = 0.01;
    const int num_iterations = 1000;
    
    for (int i = 0; i < num_iterations; ++i) {
        VectorXd predictions = 1 / (1 + exp(-X_augmented * theta));
        VectorXd errors = y - predictions;
        theta += learning_rate * X_augmented.transpose() * errors;
    }

    // Вывод полученных параметров
    std::cout << "Параметры модели: " << std::endl << theta << std::endl;

    return 0;
}