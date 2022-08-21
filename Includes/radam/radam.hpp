#pragma once

#include <Eigen/Eigen>
#include <cmath>

#include <radam/params.hpp>

class RAdam final
{
 public:
    RAdam(RAdamParams params) : params_(params)
    {
        params_.CheckValidity();
    }

    template <typename Func>
    double Step(Eigen::VectorXd& x, Func&& func)
    {
        if (step_ == 0)
        {
            expAvg_ = Eigen::VectorXd::Zero(x.size());
            expAvgSq_ = Eigen::VectorXd::Zero(x.size());
        }

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
        const double initEnergy = func(x, grad);

        expAvg_ = params_.beta_0 * expAvg_ + (1 - params_.beta_0) * grad;
        expAvgSq_ =
            params_.beta_1 * expAvgSq_.array() + (1 - params_.beta_1) * grad.array() * grad.array();
        ++step_;

        const double beta0_t = std::pow<double>(params_.beta_0, step_);
        const double beta1_t = std::pow<double>(params_.beta_1, step_);
        const double rho_inf = 2 / (1 - params_.beta_1) - 1;
        const double rho_t = rho_inf * 2 * step_ * beta1_t * (1 - beta1_t);

        if (rho_t > 4)
        {
            const double stepSize =
                params_.lr *
                std::sqrt((1 - beta1_t) * (rho_t - 4) * (rho_t - 2) * rho_inf /
                          (rho_inf - 4) / (rho_inf - 2) / rho_t) /
                (1 - beta0_t);
            const Eigen::VectorXd denom =
                expAvgSq_.cwiseSqrt() +
                params_.epsilon * Eigen::VectorXd::Ones(expAvgSq_.size());

            x = x.array() - stepSize * expAvg_.array() / denom.array();
        }
        else
        {
            const double stepSize = params_.lr / (1 - beta0_t);

            x = x - stepSize * expAvg_;
        }

        return initEnergy;
    }

 private:
    RAdamParams params_;

    Eigen::VectorXd expAvg_, expAvgSq_;
    int step_{ 0 };
};
