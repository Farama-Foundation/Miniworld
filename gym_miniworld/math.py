import math

import numpy as np

# X, Y, Z-axis vectors
X_VEC = np.array([1, 0, 0])
Y_VEC = np.array([0, 1, 0])
Z_VEC = np.array([0, 0, 1])


def gen_rot_matrix(axis, angle):
    """
    Rotation matrix for a counterclockwise rotation around a given axis
    The angle should be in radians
    """

    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def intersect_circle_segs(point, radius, segs):
    """
    Test if a circle intersects with any wall segments
    """

    # Ignore Y coordinate
    px, _, pz = point
    point = np.array([px, 0, pz])

    a = segs[:, 0, :]
    b = segs[:, 1, :]
    ab = b - a
    ap = point - a

    dotAPAB = np.sum(ap * ab, axis=1)
    dotABAB = np.sum(ab * ab, axis=1)

    proj_dist = dotAPAB / dotABAB
    proj_dist = np.clip(proj_dist, 0, 1)
    proj_dist = np.expand_dims(proj_dist, axis=1)

    # Compute the closest point on the segment
    c = a + proj_dist * ab

    # Check if any distances are within the radius
    dist = np.linalg.norm(c - point, axis=1)
    dist_lt_rad = np.less(dist, radius)

    if np.any(dist_lt_rad):
        return True

    # No intersection
    return None
