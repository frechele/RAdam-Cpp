#pragma once

#include <cassert>

struct RAdamParams final
{
    double lr = 1e-3;
    double beta_0 = 0.9;
    double beta_1 = 0.99;
    double epsilon = 1e-10;

    void CheckValidity() const
    {
        assert(lr >= 0);
        assert(beta_0 >= 0 && beta_0 < 1);
        assert(beta_1 >= 0 && beta_1 < 1);
        assert(epsilon >= 0);
    }
};
