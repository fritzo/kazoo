#ifndef KAZOO_CLOUD_FLOW_H
#define KAZOO_CLOUD_FLOW_H

#include "cloud_points.h"

namespace Cloud
{

void estimate_flow (
    const PointSet & points,
    MatrixXf & flow,
    VideoSequence & seq,
    float tol);

} // namespace Cloud

#endif // KAZOO_CLOUD_FLOW_H
