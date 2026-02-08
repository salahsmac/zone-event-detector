
# Small geometry helpers we use everywhere.

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


def bbox_center_xyxy(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Center point of a bounding box in (x1, y1, x2, y2)."""
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """
    Ray-casting point-in-polygon test.
    Works for convex/concave polygons (as long as it's not self-intersecting).

    Returns True if the point is inside the polygon.
    """
    x, y = point
    inside = False
    n = len(polygon)

    if n < 3:
        return False  # not a polygon

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Does the edge (j->i) cross the horizontal ray to the right of the point?
        crosses = (yi > y) != (yj > y)
        if crosses:
            # Solve for x coordinate where the edge crosses the horizontal line at y
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            if x < x_cross:
                inside = not inside

        j = i

    return inside
