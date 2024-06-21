from typing import List
from typing import Optional

import rubik

GODS_NUMBER = 14
# If either BFS goes deeper than this without finding a common midpoint, there is no solution
MAX_BFS_DEPTH = GODS_NUMBER // 2


def shortest_path(
    start: rubik.Position,
    end: rubik.Position,
) -> Optional[List[rubik.Permutation]]:
    """
    Using 2-way BFS, finds the shortest path from start to end.
    Returns a list of Permutations representing that shortest path.
    If there is no path to be found, return None instead of a list.

    You can use the rubik.quarter_twists move 6-tuple.
    Each move can be applied using rubik.perm_apply.
    """

    l_states_seen = {start}
    r_states_seen = {end}
    l_parent = {start: None}
    r_parent = {end: None}
    l_depth = 0
    r_depth = 0
    l_frontier = [start]
    r_frontier = [end]

    path_found = False

    # Alternate between searching from the start and end positions,
    # stopping if a midpoint is found or if there is no solution

    # Invariant: at the start of each loop, l_frontier and r_frontier
    # contain mutually exclusive positions that have not yet been searched.

    # Initialization: l_frontier contains only the start position and
    # r_frontier contains only the end position. Disregarding a trivial
    # case (no moves needed to solve), these lists are mutually exclusive.

    # Maintenance: for each level of depth we search, the frontiers are updated accordingly.
    # Each position seen so far is recorded for future reference.
    # If a position has already been searched, it is not added to the frontier.

    # Termination: if any common elements are found between these frontiers, the loop stops.
    # The loop will also stop if the search space has been exhausted.

    while (
        l_frontier
        and r_frontier
        and l_depth < MAX_BFS_DEPTH
        and r_depth < MAX_BFS_DEPTH
    ):
        l_next = []
        
        # Invariant: 'l_frontier' contains all untraversed positions of depth 'l_depth'
        # Initialization: 'l_frontier' contains the child positions of all positions of the previous depth
        # Maitenance: each possible child move not yet seen is added to 'next'
        # Termination: 'next' is copied to 'l_frontier' and 'l_depth' increases by one
        
        for state in l_frontier:
            # Calculate all twists we can perform
            possible_moves = [
                rubik.perm_apply(twist, state) for twist in rubik.quarter_twists
            ]
            
            # Invariant: only moves not yet seen are eligible to be searched
            # Initialization: 'possible_moves' contains all possible moves from the current state
            # Maintenance: if a move has already been seen, it is skipped
            # Termination: all possible moves have been made eligible for searching or discarded

            for move in possible_moves:
                # Discard positions we've already searched
                if move not in l_states_seen:
                    l_states_seen.add(move)
                    l_parent[move] = state
                    l_next.append(move)
        # Move to next level in the tree
        l_frontier = l_next
        l_depth += 1

        # Check for shared midpoint
        if l_states_seen.intersection(r_states_seen):
            path_found = True
            break

        r_next = []
        # See above invariants
        for state in r_frontier:
            # Calculate all twists we can perform
            possible_moves = [
                rubik.perm_apply(twist, state) for twist in rubik.quarter_twists
            ]
            # See above invariants
            for move in possible_moves:
                # Discard positions we've already searched
                if move not in r_states_seen:
                    r_states_seen.add(move)
                    r_parent[move] = state
                    r_next.append(move)
        # Move to next level in the tree
        r_frontier = r_next
        r_depth += 1

        # Check for shared midpoint
        if l_states_seen.intersection(r_states_seen):
            path_found = True
            break

    if path_found:
        common_state = l_states_seen.intersection(r_states_seen).pop()
        l_path = build_path(common_state, l_parent)
        # Reverse to account for "midpoint-out" path building
        l_path.reverse()
        r_path = build_path(common_state, r_parent)
        # Invert because we want to go from the midpoint towards the end (child -> parent), not the other way around
        r_path = list(map(rubik.perm_inverse, r_path))

        # Remove duplicate midpoint, if applicable
        if l_path and r_path and l_path[-1] == r_path[0]:
            return l_path + r_path[1:]

        return l_path + r_path

    return None


def build_path(
    start: rubik.Position,
    parent_map: dict[rubik.Position, Optional[rubik.Position]],
) -> List[rubik.Permutation]:
    """
    Returns the sequence of permutations needed to get from a Position
    to the root of its respective BFS tree.
    """
    child = start
    parent = parent_map[start]
    path = []

    # Terminates when `parent` is None, in other words when `child` is the root
    while parent:
        # Find the permutation needed to get from parent to child
        possible_children = [
            rubik.perm_apply(twist, parent) for twist in rubik.quarter_twists
        ]
        desired_permutation_index = possible_children.index(child)
        desired_permutation = rubik.quarter_twists[desired_permutation_index]

        # Add to path
        path.append(desired_permutation)

        # Move up one level towards root
        child = parent
        parent = parent_map[parent]

    return path
