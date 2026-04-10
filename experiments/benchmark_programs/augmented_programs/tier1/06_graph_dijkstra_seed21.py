"""
weighted_directed_graph.py

A self-contained module implementing a weighted directed graph using only
the Python standard library. Supports BFS, DFS, Dijkstra's shortest path,
and path reconstruction.

Example
-------
>>> g = Graph()
>>> g.add_edge("A", "B", 2.0)
>>> g.add_edge("B", "C", 5.0)
>>> g.dijkstra("A")
{'A': 0.0, 'B': 2.0, 'C': 7.0}
"""

from collections import deque
import heapq
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s")
logger = logging.getLogger("DijkstraGraph")


class Graph:
    """A weighted directed graph implemented using an adjacency-list representation.

    Nodes can be any hashable type (int, str, tuple, etc.).  Edges are
    directional: an edge from *u* to *v* does **not** imply an edge from
    *v* to *u*.  Edge weights are floating-point values and must be
    non-negative (required by Dijkstra's algorithm).

    Internally the graph is stored as a dict-of-dicts::

        adjacency = {
            node_A: {neighbour_1: weight, neighbour_2: weight, ...},
            node_B: {...},
        }

    which gives O(1) average-time access to the weight of any stored edge.
    """

    def __init__(self) -> None:
        """Initialise an empty graph with no nodes or edges."""
        self.adjacency: dict[Any, dict[Any, float]] = {}
        logger.debug("Initialised new empty Graph")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_node(self, node: Any) -> None:
        """Add a node to the graph if it does not already exist.

        Parameters
        ----------
        node : Any
            A hashable object representing the node.
        """
        if node not in self.adjacency:
            self.adjacency[node] = {}
            logger.debug("Node %r added to graph", node)

    def add_edge(self, u: Any, v: Any, weight: float) -> None:
        """Add a directed edge from node ``u`` to node ``v`` with the given weight.

        If either node does not yet exist in the graph it is automatically
        created.  If an edge ``u → v`` already exists its weight is silently
        overwritten with the new value.

        Parameters
        ----------
        u : Any
            The source node (must be hashable).
        v : Any
            The destination node (must be hashable).
        weight : float
            The non-negative weight of the edge.

        Raises
        ------
        ValueError
            If *weight* is negative.
        """
        if weight < 0:
            raise ValueError(
                "Negative edge weights are not supported "
                "(Dijkstra's algorithm requires non-negative weights)."
            )
        self.add_node(u)
        self.add_node(v)
        self.adjacency[u][v] = weight
        logger.debug("Edge added: %r -> %r (weight: %.2f)", u, v, weight)

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def nodes(self) -> list[Any]:
        """Return a list of all nodes present in the graph."""
        return list(self.adjacency.keys())

    def neighbors(self, node: Any) -> dict[Any, float]:
        """Return a ``{neighbour: weight}`` dict for the given node.

        Returns an empty dict if the node has no outgoing edges or does not
        exist in the graph.
        """
        return self.adjacency.get(node, {})

    # ------------------------------------------------------------------
    # Traversal algorithms
    # ------------------------------------------------------------------

    def bfs(self, start: Any) -> list[Any]:
        """Breadth-First Search traversal starting from *start*.

        Explores the graph level-by-level: all neighbours of the current node
        are visited before any of their neighbours.  Edge weights are ignored
        for traversal ordering – only the graph topology matters.

        Parameters
        ----------
        start : Any
            The node from which to begin traversal.

        Returns
        -------
        list[Any]
            Nodes in the order they were first discovered (BFS order).

        Raises
        ------
        KeyError
            If *start* is not a node in the graph.
        """
        if start not in self.adjacency:
            logger.error("BFS start node %r not found", start)
            raise KeyError(f"Node {start!r} not found in graph.")

        logger.debug("Starting BFS traversal from %r", start)
        visited_order: list[Any] = []
        visited_set: set[Any] = set()
        queue: deque[Any] = deque()

        queue.append(start)
        visited_set.add(start)

        while queue:
            node = queue.popleft()
            visited_order.append(node)

            for neighbour in self.adjacency[node]:
                if neighbour not in visited_set:
                    visited_set.add(neighbour)
                    queue.append(neighbour)

        return visited_order

    def dfs(self, start: Any) -> list[Any]:
        """Depth-First Search traversal starting from *start*.

        Explores as far as possible along each branch before backtracking.
        Implemented iteratively with an explicit stack so that no recursion-
        depth limits are hit on large graphs.  Edge weights are ignored for
        traversal ordering.

        Parameters
        ----------
        start : Any
            The node from which to begin traversal.

        Returns
        -------
        list[Any]
            Nodes in the order they were first discovered (DFS order).

        Raises
        ------
        KeyError
            If *start* is not a node in the graph.
        """
        if start not in self.adjacency:
            logger.error("DFS start node %r not found", start)
            raise KeyError(f"Node {start!r} not found in graph.")

        logger.debug("Starting DFS traversal from %r", start)
        visited_order: list[Any] = []
        visited_set: set[Any] = set()
        stack: list[Any] = [start]

        while stack:
            node = stack.pop()
            if node in visited_set:
                continue

            visited_set.add(node)
            visited_order.append(node)

            # Push neighbours in reverse so that the left-most neighbour
            # (the one that appeared first in insertion order) is processed
            # first, yielding a natural "left-to-right" DFS.
            for neighbour in reversed(list(self.adjacency[node].keys())):
                if neighbour not in visited_set:
                    stack.append(neighbour)

        return visited_order

    # ------------------------------------------------------------------
    # Shortest-path algorithms
    # ------------------------------------------------------------------

    def dijkstra(self, start: Any) -> dict[Any, float]:
        """Compute single-source shortest-path distances (Dijkstra's algorithm).

        Uses a binary min-heap (:mod:`heapq`) for efficient extraction of the
        node with the smallest tentative distance.  All edge weights must be
        non-negative.

        Parameters
        ----------
        start : Any
            The source node from which distances are computed.

        Returns
        -------
        dict[Any, float]
            Mapping of each **reachable** node to its shortest distance from
            *start*.  The start node itself always has distance ``0.0``.
            Nodes that cannot be reached from *start* are omitted.

        Raises
        ------
        KeyError
            If *start* is not a node in the graph.
        """
        if start not in self.adjacency:
            logger.error("Dijkstra start node %r not found", start)
            raise KeyError(f"Node {start!r} not found in graph.")

        logger.debug("Computing Dijkstra shortest path from %r", start)

        # Initialise every node to +∞ except the source.
        distances: dict[Any, float] = {node: float("inf") for node in self.adjacency}
        distances[start] = 0.0

        # Priority queue stores (tentative_distance, node).
        heap: list[tuple[float, Any]] = [(0.0, start)]
        visited: set[Any] = set()

        while heap:
            current_dist, node = heapq.heappop(heap)

            if node in visited:
                continue
            visited.add(node)

            for neighbour, weight in self.adjacency[node].items():
                new_dist = current_dist + weight
                if new_dist < distances[neighbour]:
                    distances[neighbour] = new_dist
                    heapq.heappush(heap, (new_dist, neighbour))

        # Discard unreachable nodes (distance still +∞).
        return {node: dist for node, dist in distances.items() if dist != float("inf")}

    def shortest_path(self, start: Any, end: Any) -> list[Any]:
        """Reconstruct the shortest path from *start* to *end*.

        Runs Dijkstra's algorithm while maintaining a *predecessor* map so
        that the actual node sequence can be back-traced once the destination
        is settled.

        Parameters
        ----------
        start : Any
            The source node.
        end : Any
            The destination node.

        Returns
        -------
        list[Any]
            Ordered list of nodes from *start* to *end* (inclusive) along
            the shortest-weight path.  Returns an empty list if *end* is
            unreachable from *start*.  Returns ``[start]`` when *start*
            equals *end*.

        Raises
        ------
        KeyError
            If *start* or *end* is not a node in the graph.
        """
        if start not in self.adjacency:
            raise KeyError(f"Node {start!r} not found in graph.")
        if end not in self.adjacency:
            raise KeyError(f"Node {end!r} not found in graph.")

        logger.debug("Reconstructing shortest path from %r to %r", start, end)
        if start == end:
            return [start]

        # --- Dijkstra with predecessor tracking --------------------------------
        distances: dict[Any, float] = {node: float("inf") for node in self.adjacency}
        distances[start] = 0.0
        predecessors: dict[Any, Any] = {node: None for node in self.adjacency}

        heap: list[tuple[float, Any]] = [(0.0, start)]
        visited: set[Any] = set()

        while heap:
            current_dist, node = heapq.heappop(heap)

            if node in visited:
                continue
            visited.add(node)

            # Early exit: we have settled the destination with its optimal
            # distance, so no further relaxation can improve it.
            if node == end:
                break

            for neighbour, weight in self.adjacency[node].items():
                new_dist = current_dist + weight
                if new_dist < distances[neighbour]:
                    distances[neighbour] = new_dist
                    predecessors[neighbour] = node
                    heapq.heappush(heap, (new_dist, neighbour))

        # --- Path reconstruction ------------------------------------------------
        if distances[end] == float("inf"):
            return []  # No path exists.

        path: list[Any] = []
        current: Any = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        return path


# ======================================================================
# Demo: construct a small city road network and exercise every method
# ======================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  Graph topology (7 intersections, one-way streets with distances)
    #
    #            2 km          5 km
    #       A ──────► B ──────► C
    #       │         │         │
    #  4 km │    1 km │    3 km │
    #       ▼         ▼         ▼
    #       D ──────► E ──────► F
    #            6 km      1 km │
    #                             │ 7 km
    #                             ▼
    #                             G
    #
    #  Plus a shortcut E → C (2 km) and D → B (3 km).
    # ------------------------------------------------------------------

    g = Graph()

    # Top row: A → B → C
    g.add_edge("A", "B", 2.0)
    g.add_edge("B", "C", 5.0)

    # Left column: A → D
    g.add_edge("A", "D", 4.0)

    # Middle column: B → E
    g.add_edge("B", "E", 1.0)

    # Right column: C → F
    g.add_edge("C", "F", 3.0)

    # Bottom row: D → E → F
    g.add_edge("D", "E", 6.0)
    g.add_edge("E", "F", 1.0)

    # F → G
    g.add_edge("F", "G", 7.0)

    # Cross-edges (shortcuts)
    g.add_edge("E", "C", 2.0)
    g.add_edge("D", "B", 3.0)

    # ── Printing the graph ────────────────────────────────────────────
    logger.info("  WEIGHTED DIRECTED GRAPH — Small City Road Network")

    logger.info("  Nodes (%d): %s", len(g.nodes()), ", ".join(sorted(g.nodes())))
    logger.info("  Edges (source → destination : weight):")
    for src in sorted(g.nodes()):
        for dst, w in g.neighbors(src).items():
            logger.info("    %s → %s : %.1f km", src, dst, w)

    source = "A"

    # ── BFS ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  BFS Traversal from '{source}'")
    print(f"{'─' * 62}")
    bfs_result = g.bfs(source)
    print(f"  Visit order : {' → '.join(bfs_result)}")
    print(f"  Nodes visited: {len(bfs_result)}")

    # ── DFS ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  DFS Traversal from '{source}'")
    print(f"{'─' * 62}")
    dfs_result = g.dfs(source)
    print(f"  Visit order : {' → '.join(dfs_result)}")
    print(f"  Nodes visited: {len(dfs_result)}")

    # ── Dijkstra ──────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  Dijkstra's Shortest Distances from '{source}'")
    print(f"{'─' * 62}")
    distances = g.dijkstra(source)
    for node in sorted(distances):
        print(f"    {source} → {node} : {distances[node]:.1f} km")

    # ── Shortest paths ────────────────────────────────────────────────
    destinations = ["B", "C", "E", "F", "G"]
    print(f"\n{'─' * 62}")
    print(f"  Shortest Paths from '{source}'")
    print(f"{'─' * 62}")
    for dest in destinations:
        path = g.shortest_path(source, dest)
        if path:
            total = distances[dest]
            print(f"    {source} → {dest} : {' → '.join(path)}  (total: {total:.1f} km)")
        else:
            print(f"    {source} → {dest} : No path exists")

    # ── Unreachable-node demo ─────────────────────────────────────────
    g.add_node("X")  # Isolated – no incoming or outgoing edges
    path_to_x = g.shortest_path("A", "X")
    logger.info("  [Demo] Shortest path A → X (isolated node): %s", path_to_x if path_to_x else 'No path exists')