#ifndef KAZOO_OPTIM_H
#define KAZOO_OPTIM_H

#include "common.h"

float minimize_grid_search (const Function & f, float x0, float dx);

float minimize_bisection_search (
    const Function & f,
    float x0,
    float x1,
    float dx);

#endif // KAZOO_OPTIM_H
