import math
import sys
from typing import List
from typing import Tuple
from itertools import cycle
from statistics import median

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]


def y_intercept(p1: Point, p2: Point, x: int) -> float:
    """
    Given two points, p1 and p2, an x coordinate from a vertical line,
    compute and return the the y-intercept of the line segment p1->p2
    with the vertical line passing through x.
    """
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) else 100_000
    return y1 + (x - x1) * slope


def triangle_area(a: Point, b: Point, c: Point) -> float:
    """
    Given three points a,b,c,
    computes and returns the area defined by the triangle a,b,c.
    Note that this area will be negative if a,b,c represents a clockwise sequence,
    positive if it is counter-clockwise,
    and zero if the points are collinear.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return ((cx - bx) * (by - ay) - (bx - ax) * (cy - by)) / 2


def is_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a clockwise sequence
    (subject to floating-point precision)
    """
    return triangle_area(a, b, c) < -EPSILON


def is_counter_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a counter-clockwise sequence
    (subject to floating-point precision)
    """
    return triangle_area(a, b, c) > EPSILON


def collinear(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c are collinear
    (subject to floating-point precision)
    """
    return abs(triangle_area(a, b, c)) <= EPSILON


def sort_clockwise(points: List[Point]):
    """
    Sorts `points` by ascending clockwise angle from +x about the centroid,
    breaking ties first by ascending x value and then by ascending y value.

    The order of equal points is not modified

    Note: This function modifies its argument
    """
    # Trivial cases don't need sorting, and this dodges divide-by-zero errors
    if len(points) < 2:
        return

    # Compute the centroid
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)

    # Sort by ascending clockwise angle from +x, breaking ties with ^x then ^y
    def sort_key(point: Point):
        angle = math.atan2(point[1] - centroid_y, point[0] - centroid_x)
        normalized_angle = (angle + math.tau) % math.tau
        return (normalized_angle, point[0], point[1])

    # Sort the points
    points.sort(key=sort_key)


def base_case_hull(points: List[Point]) -> List[Point]:
    """Base case of the recursive algorithm."""
    # Naive: for each pair of points, draw a line that connects them.
    # Iff the remaining points are on the same side of that line, add the
    # pair to the hull.

    # How to handle duplicates? Maybe add points to set then sort clockwise?

    # Handle trivial cases
    if len(points) <= 3:
        sort_clockwise(points)
        return points

    # Set allows for automatic deduplication
    hull = set()

    # O(n)
    for a in points:
        # O(n^2)
        for b in points:
            if a == b or (a in hull and b in hull):
                continue

            clockwise = []
            counterclockwise = []
            collinear_ = []

            # O(n^3) iterations, worst case
            for x in points:
                if x in [a, b]:
                    continue

                # Imagine a line drawn between a and b.
                # If we then make the sequence [a, x, b], then whether this sequence is
                # clockwise or counterclockwise determines which "side" of the line
                # x lies on.

                sequence = [a, x, b]

                if is_clockwise(*sequence):
                    clockwise.append(x)
                elif is_counter_clockwise(*sequence):
                    counterclockwise.append(x)
                elif collinear(*sequence):
                    collinear_.append(x)
                else:
                    raise RuntimeError(
                        f"Points {a}, {x}, {b} are not clockwise, counterclockwise, or colliniear. You should never get here!"
                    )

            # If ALL other points are either clockwise or counterclockwise,
            # then both a and b must be on the hull.
            # We also add any collinear points for completeness.

            if not (clockwise and counterclockwise) and (
                clockwise or counterclockwise or collinear_
            ):
                hull.add(a)
                hull.add(b)
                hull.update(collinear_)

    # Convert set back to list for sorting
    hull = list(hull)
    sort_clockwise(hull)
    return hull


def divide_and_conquer_hull(points: List[Point]) -> List[Point]:
    """Primary computation steps for the divide and conquer hull algorithm."""

    # Termination:
    # Recursion stops when the "naive" hull algorithm can be used without a significant amount of slowdown.
    # The hull is then returned, consisting only of the exterior points in clockwise order.

    # Handle base case
    if len(points) <= 6:
        return base_case_hull(points)

    # Initialization:
    # We start with n points, which is less than or equal to the total number of points supplied to the program.
    # These points are sorted by their x-value.

    # Divide
    # median_x = median([point[0] for point in points])
    # left_points = [point for point in points if point[0] >= median_x]
    # right_points = [point for point in points if point[0] < median_x]
    n = len(points)
    left_points = points[: n // 2]
    right_points = points[n // 2 :]

    # Maintenance:
    # The left hull consists of the leftmost n/2 points from the initial set.
    # Likewise with the right hull.

    # Conquer
    left_hull = divide_and_conquer_hull(left_points)
    right_hull = divide_and_conquer_hull(right_points)

    # Merge
    return merge(left_hull, right_hull)


def compute_hull(points: List[Point]) -> List[Point]:
    """
    Given a list of points, computes the convex hull around those points
    and returns only the points that are on the hull *in clockwise order*.
    """

    # sort points by x value
    points = sorted(points, key=lambda point: point[0])
    return divide_and_conquer_hull(points)


def find_upper_tangent(
    left_hull: List[Point], right_hull: List[Point]
) -> Tuple[Point, Point]:
    """Find the upper tangent between two hulls."""
    # Rightmost point of left hull
    l_point = max(left_hull, key=lambda point: point[0])

    # Leftmost point of right hull
    r_point = min(right_hull, key=lambda point: point[0])

    x_separator = (l_point[0] + r_point[0]) / 2
    # x_separator = median(point[0] for point in left_hull + right_hull)

    # Cycles for circular iteration
    l_cycle = cycle(left_hull[::-1])  # Reverse for counterclockwise iteration
    r_cycle = cycle(right_hull)

    # Advance L and R iterators to rightmost and leftmost points, respectively
    while next(l_cycle) != l_point:
        pass
    while next(r_cycle) != r_point:
        pass

    previous_l = None
    previous_r = None

    highest_y_intercept = 0

    while True:
        previous_l = l_point
        previous_r = r_point

        # Walk counterclockwise on left hull until tangent height no longer improves
        next_l = next(l_cycle)
        while y_intercept(next_l, r_point, x_separator) > highest_y_intercept:
            l_point = next_l
            highest_y_intercept = y_intercept(l_point, r_point, x_separator)

        # Walk clockwise on right hull until tangent height no longer improves
        next_r = next(r_cycle)
        while y_intercept(l_point, next_r, x_separator) > highest_y_intercept:
            r_point = next_r
            highest_y_intercept = y_intercept(l_point, r_point, x_separator)

        # Return if neither the next L or R point will result in a better height
        if l_point == previous_l and r_point == previous_r:
            return (l_point, r_point)


def find_lower_tangent(
    left_hull: List[Point], right_hull: List[Point]
) -> Tuple[Point, Point]:
    """Find the lower tangent between two hulls."""
    # Rightmost point of left hull
    l_point = max(left_hull, key=lambda point: point[0])

    # Leftmost point of right hull
    r_point = min(right_hull, key=lambda point: point[0])

    x_separator = (l_point[0] + r_point[0]) / 2

    # Cycles for circular iteration
    l_cycle = cycle(left_hull)
    r_cycle = cycle(right_hull[::-1])  # Reverse for counterclockwise iteration

    # Advance L and R iterators to rightmost and leftmost points, respectively
    while next(l_cycle) != l_point:
        pass
    while next(r_cycle) != r_point:
        pass

    previous_l = None
    previous_r = None

    lowest_y_intercept = 0

    while True:
        previous_l = l_point
        previous_r = r_point

        # Walk counterclockwise on left hull until tangent height no longer improves
        next_l = next(l_cycle)
        while y_intercept(next_l, r_point, x_separator) < lowest_y_intercept:
            l_point = next_l
            lowest_y_intercept = y_intercept(l_point, r_point, x_separator)

        # Walk clockwise on right hull until tangent height no longer improves
        next_r = next(r_cycle)
        while y_intercept(l_point, next_r, x_separator) < lowest_y_intercept:
            r_point = next_r
            lowest_y_intercept = y_intercept(l_point, r_point, x_separator)

        # Return if neither the next L or R point will result in a better height
        if l_point == previous_l and r_point == previous_r:
            return (l_point, r_point)


def merge(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    """Merge two hulls, dropping any interior points.
    Returns the new hull in clockwise order."""
    hull = left_hull + right_hull

    upper_tangent = find_upper_tangent(left_hull, right_hull)
    lower_tangent = find_lower_tangent(left_hull, right_hull)

    l_cycle = cycle(left_hull)
    r_cycle = cycle(right_hull[::-1])

    # Advance L and R iterators to points on upper tangent
    while next(l_cycle) != upper_tangent[0]:
        pass
    while next(r_cycle) != upper_tangent[1]:
        pass

    l_point = next(l_cycle)
    r_point = next(r_cycle)

    while l_point != lower_tangent[0]:
        hull.remove(l_point)
        l_point = next(l_cycle)
    while r_point != lower_tangent[1]:
        hull.remove(r_point)
        r_point = next(r_cycle)

    sort_clockwise(hull)
    return hull


def find_dividing_line(left_hull: List[Point], right_hull: List[Point]) -> int:
    """Find the x-value of the line that separates two hulls."""
    # x-value of rightmost point on left hull
    x1 = max([point[0] for point in left_hull])
    # x-value of leftmost point on right hull
    x2 = min([point[0] for point in right_hull])
    return (x1 + x2) // 2
