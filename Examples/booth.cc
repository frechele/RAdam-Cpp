#include <Eigen/Eigen>
#include <iostream>

#include <radam/radam.hpp>

double boothFunction(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
{
    const double left = (x[0] + 2 * x[1] - 7);
    const double right = (2 * x[0] + x[1] - 5);
    const double energy = left * left + right * right;

    grad[0] = 2 * left + 4 * right;
    grad[1] = 4 * left + 2 * right;

    return energy;
}

int main()
{
    RAdamParams param;
    RAdam solver(param);

    Eigen::VectorXd x = Eigen::VectorXd::Random(2);

    for (int i = 0; i < 5000; ++i)
    {
        const double energy = solver.Step(x, boothFunction);
        std::cout << "x: " << x[0] << " " << x[1] << " energy: " << energy
                  << std::endl;
    }

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(2);
    const double finalEnergy = boothFunction(x, grad);
    std::cout << "x: " << x[0] << " " << x[1] << " energy: " << finalEnergy
              << std::endl;
}
