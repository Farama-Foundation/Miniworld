import math
import numpy as np
#import pybullet

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
