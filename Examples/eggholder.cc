#include <Eigen/Eigen>
#include <iostream>

#include <radam/radam.hpp>

double eggholderFunction(const Eigen::VectorXd& X, Eigen::VectorXd& grad)
{
    const double x = X(0);
    const double y = X(1);

    const double energy = -(y + 47) * std::sin(std::sqrt(std::abs(0.5 * x + y + 47))) -
           x * std::sin(std::sqrt(std::abs(x - y - 47)));

    grad[0] = -std::sin(std::sqrt(std::abs(x - y - 47))) -
              (x * (x - y - 47) * std::cos(std::sqrt(std::abs(x - y - 47)))) /
                  (2. * std::pow(std::abs(x - y - 47), 1.5)) -
              (0.25 * (y + 47) * (0.5 * x + y + 47) *
               std::cos(std::sqrt(std::abs(0.5 * x + y + 47)))) /
                  std::pow(std::abs(0.5 * x + y + 47), 1.5);

    grad[1] = -std::sin(std::sqrt(std::abs(0.5 * x + y + 47))) +
              (x * (x - y - 47) * std::cos(std::sqrt(std::abs(x - y - 47))) /
               (2 * std::pow(std::abs(x - y - 47), 1.5))) -
              ((y + 47) * (0.5 * x + y + 47) *
               std::cos(std::sqrt(std::abs(0.5 * x + y + 47)))) /
                  (2 * std::pow(std::abs(0.5 * x + y + 47), 1.5));

    return energy;
}

int main()
{
    RAdamParams param;
    RAdam solver(param);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x[0] = 500;
    x[1] = 400;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(2);

    double energy = eggholderFunction(x, grad);
    int step;
    for (step = 0; energy > -959.64; ++step)
    {
        energy = solver.Step(x, eggholderFunction);
        std::cout << "x: " << x[0] << " " << x[1] << " energy: " << energy
                  << std::endl;
    }

    const double finalEnergy = eggholderFunction(x, grad);
    std::cout << "x: " << x[0] << " " << x[1] << " energy: " << finalEnergy
              << std::endl;
    std::cout << "# total step = " << step << std::endl;
}