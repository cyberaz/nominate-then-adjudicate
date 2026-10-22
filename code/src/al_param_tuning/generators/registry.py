"""
Problem Family Registry

Central registry for all MIP problem families, their metadata,
generator functions, and scale variants. Each family is registered
with a unique ID, natural language description, mathematical model,
and scale variant definitions.
"""

from typing import Dict, List, Callable, Any, Optional
import importlib


# ── Family Metadata ───────────────────────────────────────────────────
PROBLEM_FAMILIES = [
    {
        "id": "set_covering",
        "name": "Set Covering",
        "description": (
            "Given a universe of m elements and a collection of n sets with associated costs, "
            "find the minimum-cost sub-collection of sets whose union covers the entire universe. "
            "Each set covers a random subset of elements. The constraint matrix is sparse (density "
            "parameter controls sparsity). Decision variables are binary: x_j=1 if set j is selected. "
            "Constraints ensure each element is covered by at least one selected set. "
            "This problem has a weak LP relaxation and is a classic NP-hard combinatorial optimization problem."
        ),
        "scale_variants": {
            "small": {"m": 420, "n": 420, "density": 0.10},
            "medium": {"m": 500, "n": 500, "density": 0.05},
            "large": {"m": 2000, "n": 2000, "density": 0.03},
        },
    },
    {
        "id": "facility_location",
        "name": "Capacitated Facility Location",
        "description": (
            "Given a set of potential facility locations and a set of clients, decide which facilities "
            "to open and how to assign clients to open facilities, minimizing total opening and transportation "
            "costs while respecting facility capacity constraints. Binary variables y_j indicate facility "
            "opening; continuous variables x_ij represent the fraction of client i served by facility j. "
            "Constraints: each client must be fully served; open facilities cannot exceed capacity. "
            "The problem has a block structure coupling assignment and location decisions."
        ),
        "scale_variants": {
            "small": {"n_facilities": 40, "n_clients": 190},
            "medium": {"n_facilities": 20, "n_clients": 60},
            "large": {"n_facilities": 40, "n_clients": 120},
        },
    },
    {
        "id": "knapsack",
        "name": "0-1 Knapsack",
        "description": (
            "Given n items each with a weight and value, select a subset to maximize total value "
            "subject to a weight capacity constraint. A single linear constraint and binary variables. "
            "The constraint matrix is trivially sparse (one row). Despite its simplicity, knapsack is "
            "NP-hard and the LP relaxation bound can be tight. Correlation between weights and values "
            "affects difficulty."
        ),
        "scale_variants": {
            "small": {"n": 12000},
            "medium": {"n": 150},
            "large": {"n": 500},
        },
    },
    {
        "id": "bin_packing",
        "name": "Bin Packing",
        "description": (
            "Given n items with sizes and an unlimited supply of bins with fixed capacity, "
            "pack all items into the minimum number of bins. Binary variable x_ij=1 if item i "
            "goes into bin j; binary y_j=1 if bin j is used. Highly symmetric: bins are "
            "interchangeable, creating large symmetry groups that challenge branch-and-bound. "
            "Constraint matrix has a block structure per bin."
        ),
        "scale_variants": {
            "small": {"n_items": 50, "n_bins_ub": 30},
            "medium": {"n_items": 40, "n_bins_ub": 20},
            "large": {"n_items": 80, "n_bins_ub": 40},
        },
    },
    {
        "id": "independent_set",
        "name": "Independent Set",
        "description": (
            "Given an undirected graph, find the largest subset of vertices such that no two "
            "selected vertices are adjacent. Binary variable x_i=1 if vertex i is in the independent "
            "set. For each edge (i,j), constraint x_i + x_j <= 1. The problem is purely combinatorial "
            "with a very weak LP relaxation (optimal LP value is often n/2 while the true answer is "
            "much smaller). Difficulty depends heavily on graph density and structure."
        ),
        "scale_variants": {
            "small": {"n_vertices": 140, "edge_prob": 0.3},
            "medium": {"n_vertices": 100, "edge_prob": 0.2},
            "large": {"n_vertices": 200, "edge_prob": 0.15},
        },
    },
    {
        "id": "graph_coloring",
        "name": "Graph Coloring",
        "description": (
            "Given an undirected graph, assign a color to each vertex such that no two adjacent "
            "vertices share the same color, minimizing the total number of colors used. Binary "
            "variable x_ik=1 if vertex i gets color k; binary w_k=1 if color k is used. "
            "Highly symmetric: colors are interchangeable. Constraints: each vertex gets exactly "
            "one color; adjacent vertices have different colors. The LP relaxation is weak due to "
            "symmetry and integrality gap."
        ),
        "scale_variants": {
            "small": {"n_vertices": 48, "edge_prob": 0.4, "n_colors_ub": 12},
            "medium": {"n_vertices": 50, "edge_prob": 0.25, "n_colors_ub": 15},
            "large": {"n_vertices": 80, "edge_prob": 0.2, "n_colors_ub": 20},
        },
    },
    {
        "id": "vertex_cover",
        "name": "Vertex Cover",
        "description": (
            "Given an undirected graph, find the minimum-size subset of vertices such that every "
            "edge has at least one endpoint in the subset. Binary variable x_i=1 if vertex i is "
            "selected. For each edge (i,j), constraint x_i + x_j >= 1. This is the complement of "
            "Independent Set. The LP relaxation has a well-known 2-approximation property. "
            "Graph structure strongly influences difficulty."
        ),
        "scale_variants": {
            "small": {"n_vertices": 140, "edge_prob": 0.3},
            "medium": {"n_vertices": 100, "edge_prob": 0.2},
            "large": {"n_vertices": 200, "edge_prob": 0.15},
        },
    },
    {
        "id": "multidim_knapsack",
        "name": "Multi-dimensional Knapsack",
        "description": (
            "A generalization of the 0-1 Knapsack with d resource constraints instead of one. "
            "Each item consumes resources across d dimensions. Binary variable x_j=1 if item j "
            "is selected. d linear constraints enforce capacity limits. Much harder than 1D knapsack "
            "because the LP relaxation is weaker and there is no efficient DP approach. "
            "Difficulty grows rapidly with the number of dimensions."
        ),
        "scale_variants": {
            "small": {"n": 350, "d": 15},
            "medium": {"n": 60, "d": 10},
            "large": {"n": 100, "d": 15},
        },
    },
    {
        "id": "job_scheduling",
        "name": "Job Scheduling (Single Machine)",
        "description": (
            "Schedule n jobs on a single machine to minimize total weighted tardiness. Each job "
            "has a processing time, due date, and weight. Decision: ordering of jobs (binary "
            "sequencing variables or start-time variables with big-M constraints). Constraints: "
            "jobs cannot overlap (disjunctive constraints). The problem has a temporal structure "
            "but big-M formulations create weak relaxations. Multiple constraint types: precedence, "
            "time windows, capacity."
        ),
        "scale_variants": {
            "small": {"n_jobs": 6},
            "medium": {"n_jobs": 20},
            "large": {"n_jobs": 40},
        },
    },
    {
        "id": "network_design",
        "name": "Network Design",
        "description": (
            "Given a graph with potential edges and a set of commodity demands, select a subset "
            "of edges to install (with fixed costs) and route all demands through the installed "
            "network, minimizing total cost. Binary variable y_e=1 if edge e is installed; "
            "continuous flow variables f_e^k for each commodity k on edge e. Constraints: flow "
            "conservation at each node for each commodity; capacity on installed edges. "
            "Strong graph/network structure with block-diagonal flow conservation."
        ),
        "scale_variants": {
            "small": {"n_nodes": 16, "n_commodities": 12},
            "medium": {"n_nodes": 12, "n_commodities": 10},
            "large": {"n_nodes": 20, "n_commodities": 15},
        },
    },
    {
        "id": "lot_sizing",
        "name": "Capacitated Lot Sizing",
        "description": (
            "Determine production quantities for a single product over T time periods to minimize "
            "total production, setup, and holding costs while meeting demand. Binary variable y_t=1 "
            "if production occurs in period t; continuous variable x_t for production quantity; "
            "continuous variable s_t for end-of-period inventory. Constraints: inventory balance "
            "(s_t = s_{t-1} + x_t - d_t); production capacity linked to setup (x_t <= M*y_t). "
            "The problem has a staircase/block structure over time."
        ),
        "scale_variants": {
            "small": {"n_periods": 40, "n_products": 8},
            "medium": {"n_periods": 24, "n_products": 5},
            "large": {"n_periods": 48, "n_products": 8},
        },
    },
    {
        "id": "assignment",
        "name": "Generalized Assignment Problem",
        "description": (
            "Assign n jobs to m agents, each agent has a capacity limit. Each job must be assigned "
            "to exactly one agent. Cost and resource consumption depend on both job and agent. "
            "Binary variable x_ij=1 if job j is assigned to agent i. Constraints: each job assigned "
            "once; agent capacity not exceeded. Unlike the classic assignment problem, the generalized "
            "version has different resource consumptions, making it NP-hard. The LP relaxation is "
            "relatively tight due to the assignment structure."
        ),
        "scale_variants": {
            "small": {"n_agents": 6, "n_jobs": 36},
            "medium": {"n_agents": 10, "n_jobs": 40},
            "large": {"n_agents": 15, "n_jobs": 80},
        },
    },
    {
        "id": "tsp",
        "name": "Traveling Salesman Problem (MTZ)",
        "description": (
            "Find the shortest Hamiltonian cycle in a complete graph. Uses the Miller-Tucker-Zemlin "
            "(MTZ) subtour elimination formulation. Binary variable x_ij=1 if edge (i,j) is in the "
            "tour; continuous u_i for subtour elimination. Constraints: each vertex has exactly one "
            "incoming and one outgoing edge; MTZ constraints u_i - u_j + n*x_ij <= n-1. "
            "The MTZ formulation has a weak LP relaxation compared to DFJ, but uses polynomial "
            "constraints. Highly combinatorial with strong variable coupling."
        ),
        "scale_variants": {
            "small": {"n_cities": 16},
            "medium": {"n_cities": 20},
            "large": {"n_cities": 35},
        },
    },
    {
        "id": "crew_scheduling",
        "name": "Crew Scheduling",
        "description": (
            "Assign crew members to a set of tasks (e.g., flights, shifts) over a planning horizon. "
            "Each task has a time window and required skills. Each crew member has availability and "
            "skill constraints. Binary variable x_ij=1 if crew i is assigned to task j. "
            "Constraints: each task covered; crew availability windows respected; consecutive task "
            "compatibility (rest time). Decomposable by crew member or by time period. "
            "Mix of assignment, scheduling, and covering constraints."
        ),
        "scale_variants": {
            "small": {"n_crew": 15, "n_tasks": 50},
            "medium": {"n_crew": 15, "n_tasks": 40},
            "large": {"n_crew": 25, "n_tasks": 80},
        },
    },
    {
        "id": "cutting_stock",
        "name": "Cutting Stock",
        "description": (
            "Given large rolls (or sheets) of fixed length and a set of demand items with specified "
            "lengths and quantities, determine cutting patterns to minimize waste (number of rolls used). "
            "Integer variable x_k for the number of times pattern k is used. Constraints: demand "
            "satisfaction for each item type. The column generation formulation has a natural "
            "decomposition into a knapsack pricing subproblem. Highly symmetric: patterns with "
            "the same item counts are equivalent."
        ),
        "scale_variants": {
            "small": {"roll_length": 300, "n_item_types": 45, "n_patterns": 160},
            "medium": {"roll_length": 150, "n_item_types": 12, "n_patterns": 40},
            "large": {"roll_length": 200, "n_item_types": 20, "n_patterns": 60},
        },
    },
    {
        "id": "combinatorial_auction",
        "name": "Combinatorial Auction (Winner Determination)",
        "description": (
            "A set of indivisible items is auctioned; bidders submit bids on bundles (subsets) "
            "of items, each with an offered price. Decide which bids to accept to maximize total "
            "revenue, subject to each item being allocated to at most one accepted bid. Binary "
            "variable x_b=1 if bid b is accepted. Constraints: for each item, at most one accepted "
            "bid contains it (set-packing). Bundles are generated with skewed item popularity and "
            "super/sub-additive pricing. The problem is a weighted set-packing formulation with a "
            "large LP relaxation gap, weak symmetry, and substantial branch-and-bound search space."
        ),
        "scale_variants": {
            "small": {"n_items": 250, "n_bids": 780},
            "medium": {"n_items": 400, "n_bids": 1200},
            "large": {"n_items": 800, "n_bids": 2400},
        },
    },
]


class ProblemRegistry:
    """
    Registry for MIP problem families.

    Provides:
    - Family lookup by ID
    - Generator function resolution
    - Scale variant management
    - Metadata access
    """

    def __init__(self):
        self._families = {}
        self._generators = {}
        self._register_all()

    def _register_all(self):
        """Register all problem families with their generators."""
        for family_meta in PROBLEM_FAMILIES:
            fid = family_meta["id"]
            self._families[fid] = family_meta

            # Lazy import generator module
            module_path = f"al_param_tuning.generators.{fid}"
            self._generators[fid] = module_path

    def get_family(self, family_id: str) -> Dict:
        """Get family metadata."""
        if family_id not in self._families:
            raise ValueError(f"Unknown family: {family_id}. Available: {list(self._families.keys())}")
        return self._families[family_id]

    def get_generator(self, family_id: str) -> Callable:
        """Get the generator function for a family."""
        if family_id not in self._generators:
            raise ValueError(f"No generator for family: {family_id}")

        module_path = self._generators[family_id]
        module = importlib.import_module(module_path)
        return module.generate_instance

    def generate_instance(
        self,
        family_id: str,
        variant: str = "small",
        seed: Optional[int] = None,
        **override_params,
    ):
        """
        Generate a single MIP instance.

        Args:
            family_id: Problem family ID
            variant: Scale variant ("small", "medium", "large")
            seed: Random seed
            **override_params: Override default variant parameters

        Returns:
            PySCIPOpt Model object
        """
        family = self.get_family(family_id)
        variant_params = family["scale_variants"].get(variant, {})
        params = {**variant_params, **override_params}

        generator = self.get_generator(family_id)
        model = generator(seed=seed, **params)
        return model

    def list_families(self) -> List[str]:
        """List all registered family IDs."""
        return list(self._families.keys())

    def list_variants(self, family_id: str) -> List[str]:
        """List available scale variants for a family."""
        return list(self.get_family(family_id)["scale_variants"].keys())

    def get_description(self, family_id: str) -> str:
        """Get natural language description of a family."""
        return self.get_family(family_id)["description"]

    def get_all_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all families."""
        return {fid: f["description"] for fid, f in self._families.items()}
