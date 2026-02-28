from copy import deepcopy
import time
import random
import matplotlib.pyplot as plt

class MST:
    """A class with the methods necessary to produce a MST from the adjacency
    matrix of a weighted undirected graph. The class implements two techniques.
    First a barebone Boruvka method that runs in about $O(n^4)$ and next
    Kruska's technique that runs in about $O(n^2)$.
    """

    def __init__(self, G: list[list[int]]):
        """Initializes the MST object with the input graph G. Users are0
        expected to provide a symmetric square matrix with the diagonal
        values set to whatever semantic for no edge; this can be 0 or
        float('inf') which are the usual values; as long as the diagonal
        values are the same we're fine. Those values, found elsewere in
        the matrix will be viewed as no edge signals."""
        # Validate that the input graph is a non-empty square matrix
        if len(G) == 0 or len(G) != len(G[0]):
            raise ValueError("Input graph must be a non-empty square matrix")
        # Validate that the input graph is symmetric
        for i in range(len(G)):
            for j in range(len(G)):
                if G[i][j] != G[j][i]:
                    raise ValueError("Input graph must be undirected (symmetric)")
        # Validate that the input graph has no self-loops
        for i in range(len(G)):
            if G[i][i] != G[0][0]:
                raise ValueError("Input graph must have no self-loops")

        # Local copy of the input graph's adjacency matrix, so that we don't
        # accidentally modify the input graph. Also this local copy is
        # accessible to all methods of this class.
        self.G: list[list[int]] = deepcopy(G)
        # Shorthand notation for number of vertices, so that we don'try have to
        # keep writing len(self.G) all the time.
        self.n: int = len(self.G)
        # Shorthand of no-edge sentinel from input graph. Typicall we expect a 0
        # or an infinity as the no-edge sentinel, but we don't assume anything
        # here. We just take whatever is in G[0][0] as the no-edge sentinel.
        self.no_edge = G[0][0]
        # Adjacency matrix for MST. Initially it has no edges. This will be
        # populated as the algorithm progresses and will be the output
        # of the algorithm.
        self.T: list[list[int]] = [
            [self.no_edge for i in range(self.n)] for j in range(self.n)
        ]

        # Disjoint set data structures used by Kruskal. They live here because
        # we want them accessible to the private helper methods below.
        self._parent: list[int] = [i for i in range(self.n)]
        self._rank: list[int] = [0 for _ in range(self.n)]

    def _reachabilty(self, starting: int, graph: list[list[int]]) -> list[int]:
        """Determines the vertices of the input graph reachable from the
        starting vertex."""
        # List of reachable vertices
        reach: list[int] = []
        # List of vertices to explore next, primed with the starting vertex
        visit_next: list[int] = [starting]
        # Process every vertex in the visit_next list until it's empty.
        while len(visit_next) > 0:
            # Take a vertex out of the visit_next list. This can be the first
            # vertex in the list (operating it as a queue), the last vertex
            # (operating it as a stack), or any vertex with damn well please
            # (operating it as we like). All things being equal, stack is fine.
            vertex: int = visit_next.pop()
            if vertex not in reach:
                # Add this vertex to the reachable list
                reach.append(vertex)
                # Plan to visit this vertex's neighbors next
                for neighbor in range(self.n):
                    if graph[neighbor][vertex] != self.no_edge:
                        visit_next.append(neighbor)
        return reach  # Done!

    def _count_and_label(self):
        """Counts the components of the candidate MST (self.T) and labels its
        vertices with the component they belong to. For every vertex in the graph,
        track its component. The component label is the count value for each component
        when it was discovered. For example, if there are 3 components in the graph,
        the vertices in the first component will be labeled 1, those in the second
        component will be labeled 2, and those in the third component will be labeled 3.
        """
        # Count of components in the candidate MST
        count_of_components: int = 0
        # The component label for each vertex. Initialized to -1, meaning
        # that no vertex has been labeled yet.
        component_label: list[int] = [-1] * self.n
        # Remember the vertices we've seen so far. This is to avoid
        # double counting components.
        visited: list[int] = []
        # Process every vertex. If we haven't seen it before, it means
        # we've just found a new component.
        for vertex in range(self.n):
            if vertex not in visited:
                # We just found a new component, update the count. This count value
                # becomes the label for all vertices in this component.
                count_of_components += 1
                # Find everything we can reach from this vertex, because they'll be
                # in the same component
                reachable_from_vertex = self._reachabilty(vertex, self.T)
                # Add these reachable vertices to those visited, because
                # they are all in the same component and we don't want to
                # process them in the future.
                visited.extend(reachable_from_vertex)
                # The current component count becomes the label of all
                # these vertices
                for v in reachable_from_vertex:
                    component_label[v] = count_of_components
        return count_of_components, component_label  # Done

    # ------------------------------------------------------------------
    # Disjoint set operations for Kruskal. No imports. No nonsense.
    # ------------------------------------------------------------------

    def _ds_reset(self) -> None:
        """Resets the disjoint set structure so that every vertex is alone
        in its own little component, with zero baggage from previous runs."""
        self._parent = [i for i in range(self.n)]
        self._rank = [0 for _ in range(self.n)]

    def _ds_find(self, x: int) -> int:
        """Finds the representative of the set that contains x. We also do
        path compression, because we like our trees short and our finds fast."""
        while self._parent[x] != x:
            # Point x directly to its grandparent. This flattens the structure
            # over time and makes future finds cheaper.
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x  # Done

    def _ds_union(self, a: int, b: int) -> bool:
        """Unions the sets of a and b. Returns True if we actually merged two
        different sets. Returns False if a and b were already in the same set."""
        ra = self._ds_find(a)
        rb = self._ds_find(b)
        merged = False
        if ra != rb:
            # Union by rank: attach the smaller tree under the bigger tree.
            if self._rank[ra] < self._rank[rb]:
                self._parent[ra] = rb
            elif self._rank[ra] > self._rank[rb]:
                self._parent[rb] = ra
            else:
                self._parent[rb] = ra
                self._rank[ra] += 1
            merged = True
        return merged  # Done

    def _edge_list(self) -> list[list[int]]:
        """Extracts the list of edges from the input graph. For an undirected
        graph stored as an adjacency matrix, we only look above the diagonal
        (u < v) so that we don't double-count edges like amateurs."""
        edges: list[list[int]] = []
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if self.G[u][v] != self.no_edge:
                    # Represent an edge as [u, v, w]. Boring, but effective.
                    edges.append([u, v, self.G[u][v]])
        return edges  # Done

    def _sort_edges_by_weight(self, edges: list[list[int]]) -> list[list[int]]:
        """Sorts the edge list by weight, smallest first. This uses Python's
        built-in sort."""
        # The weight lives at index 2, because we said so in _edge_list().
        edges.sort(key=lambda e: e[2])
        return edges  # Done

    def boruvka(self):
        """The algorithm itself!"""
        # Initialize count of components in the candidate MST and also get the
        # labels for each vertex in it.
        count_of_components, component_label = self._count_and_label()
        while count_of_components > 1:
            # Initialize a list to hold the safe edges for the various components.
            # Here lies a YUGE! implementation question. How to represent a safe edge?
            safe_edge = [None] * (self.n + 1)
            # Find any two vertices u, v in different components. The loop examines
            # all pairs of vertices above the main diagonal of the adjacency matrix.
            # There is no reason to look below the main diagonal, because the graph is
            # undirected and the adjacency matrix is symmetric. Also there is no reason
            # to look at the main diagonal, because there are no self-loops in the graph.
            # So we look at all pairs (u,v) such that u < v. For every such pair,
            # if u and v are in different components, we consider edge (u,v) as
            # a candidate for the safe edge of the component of u and also for
            # the safe edge of the component of v.
            for u in range(self.n):
                for v in range(u + 1, self.n):
                    # Is there an edge between u and v in the input graph?
                    if self.G[u][v] != self.no_edge:
                        if component_label[u] != component_label[v]:
                            # Vertices u, v are in different components. Let's figure
                            # out the safe edge for the component of vertex u. We do
                            # things below in a verbose manner for clarity. We'll be
                            # more concise when we do the same for component of vertex v.
                            if safe_edge[component_label[u]] is None:
                                # There is no safe edge for this component yet, so let's
                                # assume that (u,v) is just it. And here is how to represent
                                # the safe edge of a component.
                                safe_edge[component_label[u]] = [u, v]
                            else:
                                # There is currently a safe edge for this vertex, but is
                                # it truly the safest? Let's find out. Is edge (u,v)
                                # safer than the current safe edge for this component?
                                current_component = component_label[u]
                                current_safe_edge = safe_edge[current_component]
                                a = current_safe_edge[0]  # safe edge vertex a
                                b = current_safe_edge[1]  # safe edge vertex b
                                # Look up input graph for weigh of edge (a,b)
                                current_weight = self.G[a][b]
                                # Which of two edges (a,b) and (u,v) is safer, ie,
                                # has the smallest weight?
                                if self.G[u][v] < current_weight:
                                    # If (u,v) is smaller than the existing safe
                                    # edge (a,b) make (u,v) the safe edge for the
                                    # current component.
                                    safe_edge[current_component] = [u, v]
                            # It's component v's turn now. We repeat the same
                            # logic as above, but for component of vertex v. The
                            # code is a bit less verbose this time.
                            if safe_edge[component_label[v]] is None:
                                safe_edge[component_label[v]] = [v, u]
                            else:
                                [a, b] = safe_edge[component_label[v]]
                                if self.G[v][u] < self.G[a][b]:
                                    safe_edge[component_label[v]] = [v, u]
            # Add all the safe edges to the candidate MST
            for edge in safe_edge:
                if edge is not None:
                    u = edge[0]
                    v = edge[1]
                    self.T[u][v] = self.G[u][v]
                    self.T[v][u] = self.G[v][u]
            # Recount the components and re-label the vertices
            count_of_components, component_label = self._count_and_label()
        return self.T  # Done!

    def kruskal(self):
        """Compute the MST using Kruskal's technique. The plan is simple:
        (1) list all edges, (2) sort them by weight, (3) add them in that order
        so long as they don't create a cycle. Disjoint sets do the cycle check.
        """
        # Start with a fresh empty MST matrix. No stale edges from past adventures.
        self.T = [[self.no_edge for i in range(self.n)] for j in range(self.n)]
        # Reset the disjoint set structure: everyone starts alone.
        self._ds_reset()

        # Step 1: Gather all edges from the input graph.
        edges = self._edge_list()
        # Step 2: Sort edges from smallest weight to largest weight.
        edges = self._sort_edges_by_weight(edges)

        # Step 3: Walk the edges in sorted order and take the safe ones.
        edges_added = 0
        i = 0
        while i < len(edges) and edges_added < self.n - 1:
            u, v, w = edges[i]
            # If u and v are in different components, then adding (u,v) is safe.
            if self._ds_union(u, v):
                self.T[u][v] = w
                self.T[v][u] = w
                edges_added += 1
            i += 1

        return self.T  # Done!

graph = [
    [0, 0, 0, 5, 1, 0],
    [0, 0, 20, 5, 0, 10],
    [0, 20, 0, 10, 0, 0],
    [5, 5, 10, 0, 0, 15],
    [1, 0, 0, 0, 0, 20],
    [0, 10, 0, 15, 20, 0],
]

def make_graph(n: int, density: float, max_w: int, seed: int) -> list[list[int]]:
    no_edge: int = 0
    random.seed(seed)

    # Create an empty matrix
    G: list[list[int]] = [[no_edge for _ in range(n)] for _ in range(n)]

    # Make the graphs connected
    for i in range (n - 1):
        w: int = random.randint(1, max_w) # Non negative
        G[i][i + 1] = w
        G[i + 1][i] = w

    # Density
    total_edges: int = int(density * (n * (n - 1) // 2)) # Total edges count
    current_edges: int = n - 1

    while current_edges < total_edges:
        u: int = random.randint(0, n - 1)
        v: int = random.randint(0, n - 1)

        # No self-loops
        if u == v:
            continue

        # Symmetry
        if G[u][v] == no_edge:
            w: int = random.randint(1, max_w)
            G[u][v] = w
            G[v][u] = w
            current_edges += 1

    return G

# Check the weight
def mst_weight(W: list[list[int]], no_edge: int) -> int:
    tw: int = 0
    n: int = len(W)
    for u in range(n):
        for v in range(u + 1, n):
            if W[u][v] != no_edge:
                tw += W[u][v]
    return tw

# Raise an error if the weights are not matching
def check_weight(wb: int, wk: int) -> None:
    if wb != wk:
        raise ValueError("MST weights must be the same.")

# Count the runtime
def time_count(func):
    start = time.time_ns()
    result = func()
    end = time.time_ns()
    return (end - start), result

# Mean formula
def mean(values):
    return sum(values) / len(values)

# Standard deviation formula
def std(values):
    var = sum((x - mean(values)) ** 2 for x in values) / len(values)
    return var ** 0.5

def run_mst(nVal: list[int], trials: int, density: float, max_w: int, seed: int):
    results = []
    no_edge = 0

    for n in nVal:
        G = make_graph(n, density, max_w, seed)

        b_time: list[float] = []
        k_time: list[float] = []

        for t in range(trials):
            # Boruvka
            tb, Tb = time_count(MST(G).boruvka)
            wb = mst_weight(Tb, no_edge)

            # Kruskal
            tk, Tk = time_count(MST(G).kruskal)
            wk = mst_weight(Tk, no_edge)

            check_weight(wb, wk)

            b_time.append(tb)
            k_time.append(tk)

        results.append(
            {
                "n": n,
                "trials": trials,
                "mean_boruvka": mean(b_time),
                "mean_kruskal": mean(k_time),
                "std_boruvka": std(b_time),
                "std_kruskal": std(k_time),
             }
        )

    return results

# Print result's table
def print_results(results) -> None:
    print("n  | trials | mean_boruvka | mean_kruskal | std_boruvka | std_kruskal |")
    for r in results:
        print(
            f"{r['n']:2d}\t{r['trials']:7d}  \t{r['mean_boruvka']:.2f}  \t{r['mean_kruskal']:.2f}      \t{r['std_boruvka']:.2f}     \t{r['std_kruskal']:.2f}"
        )

# Plot the graph
def plot_results(results) -> None:
    nVal = [r["n"] for r in results]
    b = [r["mean_boruvka"] for r in results]
    k = [r["mean_kruskal"] for r in results]

    plt.figure()
    plt.plot(nVal, b, marker="o", label="Boruvka")
    plt.plot(nVal, k, marker="o", label="Kruskal")
    plt.xlabel("n (number of vertices)")
    plt.ylabel("avg runtime (nanoseconds)")
    plt.title("Runtime vs n")
    plt.legend()
    plt.show()

# Main
def main() -> None:
    nVal = [10, 20, 30, 40, 50]
    trials = 10
    density = 0.60
    max_w = 50
    seed = 0

    results = run_mst(nVal, trials, density, max_w, seed)
    print_results(results)
    plot_results(results)

if __name__ == "__main__":
    main()
