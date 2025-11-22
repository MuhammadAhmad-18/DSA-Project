"""
Infectious Disease Spread Simulation in Social Networks
A comprehensive simulation tool for modeling COVID-19-like disease spread
through synthetic social networks with force-directed visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import random
import math
from collections import deque, defaultdict
import json
from datetime import datetime


class SocialNetwork:
    """
    Generates and manages synthetic social networks using various models.
    Implements efficient data structures for network representation.
    """

    def __init__(self, n_nodes, model='small_world', **params):
        """
        Initialize social network.

        Args:
            n_nodes: Number of individuals in the network
            model: 'small_world', 'scale_free', or 'random'
            params: Model-specific parameters
        """
        self.n_nodes = n_nodes
        self.model = model
        self.adjacency_list = defaultdict(set)  # O(1) lookup for neighbors
        self.edges = []
        self.nodes = list(range(n_nodes))

        # Node attributes
        self.positions = {}  # Force-directed layout positions
        self.velocities = {}  # For spring embedder algorithm
        self.status = {}  # 'S' (Susceptible), 'I' (Infected), 'R' (Recovered)
        self.infection_time = {}

        # Generate network based on model
        if model == 'small_world':
            self._generate_small_world(
                params.get('k', 6), params.get('p', 0.1))
        elif model == 'scale_free':
            self._generate_scale_free(params.get('m', 3))
        elif model == 'random':
            self._generate_random(params.get('p', 0.01))

        # Initialize node states
        for node in self.nodes:
            self.status[node] = 'S'
            self.infection_time[node] = -1
            # Random initial positions
            self.positions[node] = [
                random.uniform(-1, 1), random.uniform(-1, 1)]
            self.velocities[node] = [0.0, 0.0]

    def _generate_small_world(self, k, p):
        """
        Watts-Strogatz small-world network model.
        High clustering with short average path lengths.

        Time Complexity: O(n*k)
        Space Complexity: O(n + m) where m is number of edges
        """
        n = self.n_nodes

        # Create ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                self._add_edge(i, neighbor)

        # Rewire edges with probability p
        edges_copy = list(self.edges)
        for u, v in edges_copy:
            if random.random() < p:
                # Remove edge
                self.adjacency_list[u].discard(v)
                self.adjacency_list[v].discard(u)

                # Add new random edge
                new_neighbor = random.randint(0, n - 1)
                while new_neighbor == u or new_neighbor in self.adjacency_list[u]:
                    new_neighbor = random.randint(0, n - 1)

                self._add_edge(u, new_neighbor)

        self.edges = [
            (u, v) for u in self.adjacency_list for v in self.adjacency_list[u] if u < v]

    def _generate_scale_free(self, m):
        """
        Barabási-Albert scale-free network model.
        Preferential attachment: rich get richer.

        Time Complexity: O(n*m)
        Space Complexity: O(n + m)
        """
        # Start with small complete graph
        initial_nodes = min(m + 1, self.n_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                self._add_edge(i, j)

        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, self.n_nodes):
            # Calculate attachment probabilities based on degree
            degrees = [len(self.adjacency_list[node])
                       for node in range(new_node)]
            total_degree = sum(degrees)

            if total_degree == 0:
                probabilities = [1.0 / new_node] * new_node
            else:
                probabilities = [d / total_degree for d in degrees]

            # Select m nodes to connect to
            targets = set()
            while len(targets) < min(m, new_node):
                target = random.choices(
                    range(new_node), weights=probabilities)[0]
                targets.add(target)

            for target in targets:
                self._add_edge(new_node, target)

    def _generate_random(self, p):
        """
        Erdős-Rényi random graph model.

        Time Complexity: O(n^2)
        Space Complexity: O(n + m)
        """
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if random.random() < p:
                    self._add_edge(i, j)

    def _add_edge(self, u, v):
        """Add undirected edge between nodes u and v."""
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, node):
        """Get neighbors of a node in O(1) time."""
        return self.adjacency_list[node]

    def get_degree(self, node):
        """Get degree of a node in O(1) time."""
        return len(self.adjacency_list[node])

    def get_network_stats(self):
        """Calculate network statistics."""
        degrees = [self.get_degree(node) for node in self.nodes]
        return {
            'nodes': self.n_nodes,
            'edges': len(self.edges),
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'clustering': self._calculate_clustering()
        }

    def _calculate_clustering(self):
        """Calculate average clustering coefficient."""
        coefficients = []
        for node in self.nodes:
            neighbors = list(self.get_neighbors(node))
            k = len(neighbors)
            if k < 2:
                continue

            # Count triangles
            triangles = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.get_neighbors(neighbors[i]):
                        triangles += 1

            coeff = 2 * triangles / (k * (k - 1))
            coefficients.append(coeff)

        return np.mean(coefficients) if coefficients else 0.0


class ForceDirectedLayout:
    """
    Implements Spring Embedder algorithm for graph visualization.
    Uses Fruchterman-Reingold force-directed placement.
    """

    def __init__(self, network, width=800, height=600):
        self.network = network
        self.width = width
        self.height = height
        # Optimal distance
        self.k = math.sqrt((width * height) / network.n_nodes)
        self.temperature = width / 10  # Initial temperature for simulated annealing
        self.iterations = 0
        self.max_iterations = 500

    def calculate_repulsive_force(self, dist):
        """Repulsive force between all node pairs (Coulomb's law)."""
        if dist < 0.01:
            dist = 0.01
        return (self.k * self.k) / dist

    def calculate_attractive_force(self, dist):
        """Attractive force between connected nodes (Hooke's law)."""
        return (dist * dist) / self.k

    def iterate(self, iterations=1):
        """
        Perform force-directed layout iterations.

        Time Complexity: O(n^2 + m) per iteration
        Space Complexity: O(n)
        """
        for _ in range(iterations):
            if self.iterations >= self.max_iterations:
                return False

            # Calculate repulsive forces between all pairs
            displacements = {node: [0.0, 0.0] for node in self.network.nodes}

            for i, v in enumerate(self.network.nodes):
                for u in self.network.nodes[i + 1:]:
                    delta_x = self.network.positions[v][0] - \
                        self.network.positions[u][0]
                    delta_y = self.network.positions[v][1] - \
                        self.network.positions[u][1]
                    dist = math.sqrt(delta_x**2 + delta_y**2)

                    if dist > 0:
                        force = self.calculate_repulsive_force(dist)
                        displacements[v][0] += (delta_x / dist) * force
                        displacements[v][1] += (delta_y / dist) * force
                        displacements[u][0] -= (delta_x / dist) * force
                        displacements[u][1] -= (delta_y / dist) * force

            # Calculate attractive forces for edges
            for u, v in self.network.edges:
                delta_x = self.network.positions[v][0] - \
                    self.network.positions[u][0]
                delta_y = self.network.positions[v][1] - \
                    self.network.positions[u][1]
                dist = math.sqrt(delta_x**2 + delta_y**2)

                if dist > 0:
                    force = self.calculate_attractive_force(dist)
                    displacements[v][0] -= (delta_x / dist) * force
                    displacements[v][1] -= (delta_y / dist) * force
                    displacements[u][0] += (delta_x / dist) * force
                    displacements[u][1] += (delta_y / dist) * force

            # Update positions with temperature cooling
            for node in self.network.nodes:
                disp_length = math.sqrt(
                    displacements[node][0]**2 + displacements[node][1]**2)
                if disp_length > 0:
                    self.network.positions[node][0] += (
                        displacements[node][0] / disp_length) * min(disp_length, self.temperature)
                    self.network.positions[node][1] += (
                        displacements[node][1] / disp_length) * min(disp_length, self.temperature)

                # Keep within bounds
                self.network.positions[node][0] = max(-self.width/2, min(
                    self.width/2, self.network.positions[node][0]))
                self.network.positions[node][1] = max(-self.height/2, min(
                    self.height/2, self.network.positions[node][1]))

            # Cool temperature
            self.temperature *= 0.95
            self.iterations += 1

        return True


class DiseaseSpreadSimulator:
    """
    Simulates infectious disease spread using SIR model with social network interactions.
    """

    def __init__(self, network, transmission_prob=0.05, recovery_time=14,
                 initial_infected=5, interaction_model='uniform'):
        """
        Initialize disease spread simulator.

        Args:
            network: SocialNetwork instance
            transmission_prob: Probability of infection per contact (0-1)
            recovery_time: Days until recovery
            initial_infected: Number of initially infected individuals
            interaction_model: 'uniform', 'degree_based', or 'community_based'
        """
        self.network = network
        self.transmission_prob = transmission_prob
        self.recovery_time = recovery_time
        self.interaction_model = interaction_model
        self.current_day = 0

        # Statistics tracking
        self.susceptible_count = [network.n_nodes]
        self.infected_count = [0]
        self.recovered_count = [0]
        self.daily_new_infections = [0]

        # Initialize infections
        self._initialize_infections(initial_infected)

    def _initialize_infections(self, n_infected):
        """
        Initialize patient zero(s).
        Can use random selection or degree-based (superspreaders).
        """
        # Degree-based selection: infect high-degree nodes (superspreaders)
        if self.interaction_model == 'degree_based':
            degrees = [(node, self.network.get_degree(node))
                       for node in self.network.nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            infected_nodes = [node for node, _ in degrees[:n_infected]]
        else:
            # Random selection
            infected_nodes = random.sample(self.network.nodes, n_infected)

        for node in infected_nodes:
            self.network.status[node] = 'I'
            self.network.infection_time[node] = self.current_day

        self.infected_count[0] = n_infected
        self.susceptible_count[0] = self.network.n_nodes - n_infected

    def simulate_day(self):
        """
        Simulate one day of disease spread.

        Algorithm:
        1. For each infected individual, simulate interactions with neighbors
        2. Apply transmission probability for each susceptible contact
        3. Recover individuals who have been infected long enough

        Time Complexity: O(m + n) where m is edges, n is nodes
        Space Complexity: O(n)
        """
        new_infections = []
        nodes_to_recover = []

        # Find currently infected nodes
        infected_nodes = [node for node in self.network.nodes
                          if self.network.status[node] == 'I']

        # Simulate interactions and transmission
        for infected_node in infected_nodes:
            # Check recovery
            if self.current_day - self.network.infection_time[infected_node] >= self.recovery_time:
                nodes_to_recover.append(infected_node)
                continue

            # Simulate interactions with neighbors
            neighbors = list(self.network.get_neighbors(infected_node))

            # Determine number of interactions based on model
            if self.interaction_model == 'uniform':
                n_interactions = len(neighbors)
            elif self.interaction_model == 'degree_based':
                # High-degree nodes interact more
                n_interactions = min(
                    len(neighbors), max(1, len(neighbors) // 2))
            else:
                n_interactions = len(neighbors)

            # Randomly select neighbors to interact with
            if n_interactions < len(neighbors):
                interacting_neighbors = random.sample(
                    neighbors, n_interactions)
            else:
                interacting_neighbors = neighbors

            # Attempt transmission
            for neighbor in interacting_neighbors:
                if self.network.status[neighbor] == 'S':
                    if random.random() < self.transmission_prob:
                        new_infections.append(neighbor)

        # Apply new infections
        for node in new_infections:
            self.network.status[node] = 'I'
            self.network.infection_time[node] = self.current_day

        # Apply recoveries
        for node in nodes_to_recover:
            self.network.status[node] = 'R'

        # Update statistics
        self.current_day += 1
        susceptible = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'S')
        infected = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'I')
        recovered = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'R')

        self.susceptible_count.append(susceptible)
        self.infected_count.append(infected)
        self.recovered_count.append(recovered)
        self.daily_new_infections.append(len(new_infections))

        return len(new_infections), len(nodes_to_recover)

    def get_statistics(self):
        """Get current simulation statistics."""
        return {
            'day': self.current_day,
            'susceptible': self.susceptible_count[-1],
            'infected': self.infected_count[-1],
            'recovered': self.recovered_count[-1],
            'total_infected': self.network.n_nodes - self.susceptible_count[-1],
            'new_infections': self.daily_new_infections[-1],
            'attack_rate': (self.network.n_nodes - self.susceptible_count[-1]) / self.network.n_nodes
        }


class SimulatorGUI:
    """
    Main GUI application for disease spread simulation.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Infectious Disease Spread Simulator")
        self.root.geometry("1400x900")

        self.network = None
        self.simulator = None
        self.layout = None
        self.running = False

        self._create_widgets()

    def _create_widgets(self):
        """Create GUI layout."""
        # Control Panel
        control_frame = ttk.LabelFrame(
            self.root, text="Simulation Controls", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Network Parameters
        ttk.Label(control_frame, text="Network Size:").grid(
            row=0, column=0, sticky='w')
        self.size_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.size_var,
                  width=10).grid(row=0, column=1)

        ttk.Label(control_frame, text="Network Model:").grid(
            row=1, column=0, sticky='w')
        self.model_var = tk.StringVar(value='small_world')
        ttk.Combobox(control_frame, textvariable=self.model_var,
                     values=['small_world', 'scale_free', 'random'],
                     width=12).grid(row=1, column=1)

        # Disease Parameters
        ttk.Label(control_frame, text="Transmission Prob:").grid(
            row=2, column=0, sticky='w')
        self.trans_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.trans_var,
                  width=10).grid(row=2, column=1)

        ttk.Label(control_frame, text="Recovery Time (days):").grid(
            row=3, column=0, sticky='w')
        self.recovery_var = tk.IntVar(value=14)
        ttk.Entry(control_frame, textvariable=self.recovery_var,
                  width=10).grid(row=3, column=1)

        ttk.Label(control_frame, text="Initial Infected:").grid(
            row=4, column=0, sticky='w')
        self.infected_var = tk.IntVar(value=5)
        ttk.Entry(control_frame, textvariable=self.infected_var,
                  width=10).grid(row=4, column=1)

        # Buttons
        ttk.Button(control_frame, text="Generate Network",
                   command=self.generate_network).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Start Simulation",
                   command=self.start_simulation).grid(row=6, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Step",
                   command=self.step_simulation).grid(row=7, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Reset",
                   command=self.reset_simulation).grid(row=8, column=0, columnspan=2, pady=5)

        # Statistics Display
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding=10)
        stats_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.stats_text = scrolledtext.ScrolledText(
            stats_frame, height=15, width=30)
        self.stats_text.pack()

        # Visualization Canvas
        viz_frame = ttk.LabelFrame(
            self.root, text="Network Visualization", padding=10)
        viz_frame.grid(row=0, column=1, rowspan=2,
                       padx=10, pady=10, sticky='nsew')

        self.canvas = tk.Canvas(viz_frame, width=900, height=700, bg='white')
        self.canvas.pack()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def generate_network(self):
        """Generate social network based on parameters."""
        try:
            n = self.size_var.get()
            model = self.model_var.get()

            if n < 10 or n > 5000:
                messagebox.showerror(
                    "Error", "Network size must be between 10 and 5000")
                return

            self.stats_text.insert(
                tk.END, f"Generating {model} network with {n} nodes...\n")
            self.root.update()

            # Generate network
            if model == 'small_world':
                self.network = SocialNetwork(
                    n, model='small_world', k=6, p=0.1)
            elif model == 'scale_free':
                self.network = SocialNetwork(n, model='scale_free', m=3)
            else:
                self.network = SocialNetwork(n, model='random', p=0.01)

            # Initialize layout
            self.layout = ForceDirectedLayout(
                self.network, width=800, height=600)

            # Compute initial layout
            self.stats_text.insert(
                tk.END, "Computing force-directed layout...\n")
            self.root.update()

            for _ in range(50):
                self.layout.iterate(10)
                self.root.update()

            # Display network statistics
            stats = self.network.get_network_stats()
            self.stats_text.insert(tk.END, f"\nNetwork Statistics:\n")
            self.stats_text.insert(tk.END, f"Nodes: {stats['nodes']}\n")
            self.stats_text.insert(tk.END, f"Edges: {stats['edges']}\n")
            self.stats_text.insert(
                tk.END, f"Avg Degree: {stats['avg_degree']:.2f}\n")
            self.stats_text.insert(
                tk.END, f"Clustering: {stats['clustering']:.3f}\n\n")

            self.visualize_network()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to generate network: {str(e)}")

    def start_simulation(self):
        """Initialize and start disease spread simulation."""
        if self.network is None:
            messagebox.showerror("Error", "Please generate network first")
            return

        try:
            trans_prob = self.trans_var.get()
            recovery_time = self.recovery_var.get()
            initial_infected = self.infected_var.get()

            self.simulator = DiseaseSpreadSimulator(
                self.network,
                transmission_prob=trans_prob,
                recovery_time=recovery_time,
                initial_infected=initial_infected,
                interaction_model='degree_based'
            )

            self.stats_text.insert(tk.END, "Simulation started!\n")
            self.visualize_network()
            self.running = True
            self.run_simulation_loop()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to start simulation: {str(e)}")

    def step_simulation(self):
        """Perform one simulation step."""
        if self.simulator is None:
            messagebox.showerror("Error", "Please start simulation first")
            return

        new_inf, new_rec = self.simulator.simulate_day()
        self.update_statistics()
        self.visualize_network()

    def run_simulation_loop(self):
        """Run simulation continuously."""
        if not self.running:
            return

        if self.simulator.infected_count[-1] == 0:
            self.running = False
            self.stats_text.insert(tk.END, "\nSimulation complete!\n")
            return

        self.step_simulation()
        self.root.after(100, self.run_simulation_loop)

    def reset_simulation(self):
        """Reset simulation."""
        self.running = False
        self.simulator = None
        if self.network:
            for node in self.network.nodes:
                self.network.status[node] = 'S'
                self.network.infection_time[node] = -1
            self.visualize_network()
        self.stats_text.delete(1.0, tk.END)

    def visualize_network(self):
        """Draw network on canvas using force-directed layout."""
        self.canvas.delete('all')

        if self.network is None:
            return

        # Transform coordinates to canvas space
        w, h = 900, 700
        margin = 50

        positions = {}
        for node in self.network.nodes:
            x = self.network.positions[node][0]
            y = self.network.positions[node][1]

            # Scale to canvas
            canvas_x = (x / 800) * (w - 2*margin) + w/2
            canvas_y = (y / 600) * (h - 2*margin) + h/2
            positions[node] = (canvas_x, canvas_y)

        # Draw edges
        for u, v in self.network.edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            self.canvas.create_line(x1, y1, x2, y2, fill='lightgray', width=1)

        # Draw nodes
        for node in self.network.nodes:
            x, y = positions[node]

            # Color based on status
            if self.network.status[node] == 'S':
                color = 'lightblue'
            elif self.network.status[node] == 'I':
                color = 'red'
            else:
                color = 'green'

            # Size based on degree
            degree = self.network.get_degree(node)
            size = min(10, 3 + degree / 3)

            self.canvas.create_oval(x-size, y-size, x+size, y+size,
                                    fill=color, outline='black', width=1)

        # Legend
        self.canvas.create_text(100, 30, text="Legend:",
                                font=('Arial', 10, 'bold'))
        self.canvas.create_oval(
            150, 25, 160, 35, fill='lightblue', outline='black')
        self.canvas.create_text(200, 30, text="Susceptible")
        self.canvas.create_oval(250, 25, 260, 35, fill='red', outline='black')
        self.canvas.create_text(290, 30, text="Infected")
        self.canvas.create_oval(
            330, 25, 340, 35, fill='green', outline='black')
        self.canvas.create_text(375, 30, text="Recovered")

    def update_statistics(self):
        """Update statistics display."""
        if self.simulator:
            stats = self.simulator.get_statistics()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Day: {stats['day']}\n")
            self.stats_text.insert(
                tk.END, f"Susceptible: {stats['susceptible']}\n")
            self.stats_text.insert(tk.END, f"Infected: {stats['infected']}\n")
            self.stats_text.insert(
                tk.END, f"Recovered: {stats['recovered']}\n")
            self.stats_text.insert(
                tk.END, f"Total Infected: {stats['total_infected']}\n")
            self.stats_text.insert(
                tk.END, f"New Infections: {stats['new_infections']}\n")
            self.stats_text.insert(
                tk.END, f"Attack Rate: {stats['attack_rate']*100:.1f}%\n")


def main():
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
