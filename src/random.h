
#ifndef KAZOO_RANDOM_H
#define KAZOO_RANDOM_H

#include "common.h"
#include "vectors.h"

// most simple random_samplers are already in common.h

// This uses a comb-in-bin particle resampling method.
// The results are identically distributed as multinomial,
// but are not indepenedent.
void random_multinomials (
    const Vector<float> & weights_in,
    Vector<uint32_t> & indices_out);

#endif // KAZOO_RANDOM_H

