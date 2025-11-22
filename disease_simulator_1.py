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
        self._initialize_infections(n_infected)

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
        self.root.title("Epidemic Simulator - Network-Based Disease Spread Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Set theme colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'background': '#f8f9fa'
        }

        self.network = None
        self.simulator = None
        self.layout = None
        self.running = False

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """Configure custom styles for widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Primary.TFrame', background=self.colors['background'])
        style.configure('Secondary.TFrame', background=self.colors['light'])
        style.configure('Card.TFrame', background='white', relief='raised', borderwidth=1)
        
        style.configure('Title.TLabel', 
                       background=self.colors['primary'], 
                       foreground='white',
                       font=('Arial', 12, 'bold'),
                       padding=10)
        
        style.configure('Header.TLabel',
                       background=self.colors['light'],
                       foreground=self.colors['dark'],
                       font=('Arial', 10, 'bold'),
                       padding=5)
        
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Arial', 9, 'bold'),
                       focuscolor='none')
        
        style.map('Accent.TButton',
                 background=[('active', self.colors['success']),
                           ('pressed', self.colors['success'])])
        
        style.configure('Stats.TLabel',
                       background='white',
                       foreground=self.colors['dark'],
                       font=('Arial', 9),
                       padding=2)

    def _create_widgets(self):
        """Create enhanced GUI layout."""
        # Main container
        main_container = ttk.Frame(self.root, style='Primary.TFrame')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Header
        header_frame = ttk.Frame(main_container, style='Secondary.TFrame')
        header_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(header_frame, 
                 text="EPIDEMIC SIMULATOR", 
                 style='Title.TLabel').pack(fill='x')
        
        ttk.Label(header_frame,
                 text="Network-Based Disease Spread Analysis",
                 style='Header.TLabel').pack(fill='x')

        # Content area
        content_frame = ttk.Frame(main_container, style='Primary.TFrame')
        content_frame.pack(fill='both', expand=True)

        # Left panel - Controls and Statistics
        left_panel = ttk.Frame(content_frame, style='Primary.TFrame')
        left_panel.pack(side='left', fill='both', padx=(0, 10))

        # Control Panel Card
        control_card = ttk.LabelFrame(left_panel, 
                                    text="⚙️ SIMULATION CONTROLS", 
                                    padding=15,
                                    style='Card.TFrame')
        control_card.pack(fill='x', pady=(0, 10))

        # Network Parameters Section
        network_section = ttk.Frame(control_card)
        network_section.pack(fill='x', pady=5)
        
        ttk.Label(network_section, 
                 text="Network Parameters", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 8))

        # Network size
        size_frame = ttk.Frame(network_section)
        size_frame.pack(fill='x', pady=2)
        ttk.Label(size_frame, text="Network Size:", width=15, anchor='w').pack(side='left')
        self.size_var = tk.IntVar(value=200)
        size_entry = ttk.Entry(size_frame, textvariable=self.size_var, width=10)
        size_entry.pack(side='right', padx=(5, 0))

        # Network model
        model_frame = ttk.Frame(network_section)
        model_frame.pack(fill='x', pady=2)
        ttk.Label(model_frame, text="Network Model:", width=15, anchor='w').pack(side='left')
        self.model_var = tk.StringVar(value='small_world')
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                 values=['Small World', 'Scale Free', 'Random'],
                                 state='readonly', width=12)
        model_combo.pack(side='right', padx=(5, 0))

        # Disease Parameters Section
        ttk.Label(control_card, 
                 text="Disease Parameters", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(15, 8))

        # Transmission probability
        trans_frame = ttk.Frame(control_card)
        trans_frame.pack(fill='x', pady=2)
        ttk.Label(trans_frame, text="Transmission Rate:", width=15, anchor='w').pack(side='left')
        self.trans_var = tk.DoubleVar(value=0.05)
        trans_entry = ttk.Entry(trans_frame, textvariable=self.trans_var, width=10)
        trans_entry.pack(side='right', padx=(5, 0))

        # Recovery time
        recovery_frame = ttk.Frame(control_card)
        recovery_frame.pack(fill='x', pady=2)
        ttk.Label(recovery_frame, text="Recovery Time:", width=15, anchor='w').pack(side='left')
        self.recovery_var = tk.IntVar(value=14)
        recovery_entry = ttk.Entry(recovery_frame, textvariable=self.recovery_var, width=10)
        recovery_entry.pack(side='right', padx=(5, 0))

        # Initial infected
        infected_frame = ttk.Frame(control_card)
        infected_frame.pack(fill='x', pady=2)
        ttk.Label(infected_frame, text="Initial Infected:", width=15, anchor='w').pack(side='left')
        self.infected_var = tk.IntVar(value=3)
        infected_entry = ttk.Entry(infected_frame, textvariable=self.infected_var, width=10)
        infected_entry.pack(side='right', padx=(5, 0))

        # Control Buttons
        button_frame = ttk.Frame(control_card)
        button_frame.pack(fill='x', pady=(15, 5))

        ttk.Button(button_frame, text="🔄 Generate Network", 
                  command=self.generate_network, style='Accent.TButton').pack(fill='x', pady=2)
        ttk.Button(button_frame, text="▶️ Start Simulation", 
                  command=self.start_simulation, style='Accent.TButton').pack(fill='x', pady=2)
        ttk.Button(button_frame, text="⏭️ Step Forward", 
                  command=self.step_simulation, style='Accent.TButton').pack(fill='x', pady=2)
        ttk.Button(button_frame, text="⏹️ Stop Simulation", 
                  command=self.stop_simulation, style='Accent.TButton').pack(fill='x', pady=2)
        ttk.Button(button_frame, text="🔄 Reset", 
                  command=self.reset_simulation, style='Accent.TButton').pack(fill='x', pady=2)

        # Statistics Panel
        stats_card = ttk.LabelFrame(left_panel, 
                                  text="📊 REAL-TIME STATISTICS", 
                                  padding=15,
                                  style='Card.TFrame')
        stats_card.pack(fill='both', expand=True)

        # Statistics display
        stats_display = ttk.Frame(stats_card)
        stats_display.pack(fill='both', expand=True)

        # Current stats frame
        current_stats_frame = ttk.Frame(stats_display)
        current_stats_frame.pack(fill='x', pady=(0, 10))

        # Stats grid
        stats_grid = ttk.Frame(current_stats_frame)
        stats_grid.pack(fill='x')

        # Day counter
        day_frame = ttk.Frame(stats_grid)
        day_frame.pack(fill='x', pady=2)
        ttk.Label(day_frame, text="Simulation Day:", font=('Arial', 9, 'bold')).pack(side='left')
        self.day_label = ttk.Label(day_frame, text="0", font=('Arial', 9, 'bold'), foreground=self.colors['accent'])
        self.day_label.pack(side='right')

        # SIR counters
        sir_frame = ttk.Frame(stats_grid)
        sir_frame.pack(fill='x', pady=5)

        # Susceptible
        sus_frame = ttk.Frame(sir_frame)
        sus_frame.pack(fill='x', pady=1)
        ttk.Label(sus_frame, text="Susceptible:", width=12, anchor='w').pack(side='left')
        self.sus_label = ttk.Label(sus_frame, text="200", foreground=self.colors['accent'])
        self.sus_label.pack(side='right')

        # Infected
        inf_frame = ttk.Frame(sir_frame)
        inf_frame.pack(fill='x', pady=1)
        ttk.Label(inf_frame, text="Infected:", width=12, anchor='w').pack(side='left')
        self.inf_label = ttk.Label(inf_frame, text="0", foreground=self.colors['danger'])
        self.inf_label.pack(side='right')

        # Recovered
        rec_frame = ttk.Frame(sir_frame)
        rec_frame.pack(fill='x', pady=1)
        ttk.Label(rec_frame, text="Recovered:", width=12, anchor='w').pack(side='left')
        self.rec_label = ttk.Label(rec_frame, text="0", foreground=self.colors['success'])
        self.rec_label.pack(side='right')

        # Additional stats
        ttk.Separator(stats_grid, orient='horizontal').pack(fill='x', pady=10)

        # New infections
        new_inf_frame = ttk.Frame(stats_grid)
        new_inf_frame.pack(fill='x', pady=1)
        ttk.Label(new_inf_frame, text="New Infections:", width=15, anchor='w').pack(side='left')
        self.new_inf_label = ttk.Label(new_inf_frame, text="0", foreground=self.colors['warning'])
        self.new_inf_label.pack(side='right')

        # Attack rate
        attack_frame = ttk.Frame(stats_grid)
        attack_frame.pack(fill='x', pady=1)
        ttk.Label(attack_frame, text="Attack Rate:", width=15, anchor='w').pack(side='left')
        self.attack_label = ttk.Label(attack_frame, text="0.0%", foreground=self.colors['warning'])
        self.attack_label.pack(side='right')

        # Detailed statistics text area
        ttk.Label(stats_display, text="Detailed Log:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        self.stats_text = scrolledtext.ScrolledText(stats_display, height=12, width=35,
                                                   font=('Consolas', 8),
                                                   bg='white', relief='solid', borderwidth=1)
        self.stats_text.pack(fill='both', expand=True)

        # Right panel - Visualization
        right_panel = ttk.Frame(content_frame, style='Primary.TFrame')
        right_panel.pack(side='right', fill='both', expand=True)

        # Visualization Card
        viz_card = ttk.LabelFrame(right_panel, 
                                text="🌐 NETWORK VISUALIZATION", 
                                padding=10,
                                style='Card.TFrame')
        viz_card.pack(fill='both', expand=True)

        # Canvas with border
        canvas_container = ttk.Frame(viz_card, style='Card.TFrame')
        canvas_container.pack(fill='both', expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_container, width=800, height=600, 
                               bg='white', highlightthickness=1, highlightbackground='#ddd')
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)

        # Legend
        legend_frame = ttk.Frame(viz_card)
        legend_frame.pack(fill='x', pady=5)

        ttk.Label(legend_frame, text="Legend:", font=('Arial', 9, 'bold')).pack(side='left', padx=(0, 10))

        # Susceptible legend
        ttk.Label(legend_frame, text="●", foreground=self.colors['accent']).pack(side='left', padx=(0, 5))
        ttk.Label(legend_frame, text="Susceptible").pack(side='left', padx=(0, 15))

        # Infected legend
        ttk.Label(legend_frame, text="●", foreground=self.colors['danger']).pack(side='left', padx=(0, 5))
        ttk.Label(legend_frame, text="Infected").pack(side='left', padx=(0, 15))

        # Recovered legend
        ttk.Label(legend_frame, text="●", foreground=self.colors['success']).pack(side='left', padx=(0, 5))
        ttk.Label(legend_frame, text="Recovered").pack(side='left')

    def generate_network(self):
        """Generate social network based on parameters."""
        try:
            n = self.size_var.get()
            model = self.model_var.get().lower().replace(' ', '_')

            if n < 10 or n > 2000:
                messagebox.showerror(
                    "Error", "Network size must be between 10 and 2000")
                return

            self.stats_text.insert(
                tk.END, f"🔄 Generating {model} network with {n} nodes...\n")
            self.stats_text.see(tk.END)
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
                tk.END, "📐 Computing force-directed layout...\n")
            self.root.update()

            for _ in range(50):
                self.layout.iterate(10)
                self.root.update()

            # Display network statistics
            stats = self.network.get_network_stats()
            self.stats_text.insert(tk.END, f"\n📊 Network Statistics:\n")
            self.stats_text.insert(tk.END, f"• Nodes: {stats['nodes']}\n")
            self.stats_text.insert(tk.END, f"• Edges: {stats['edges']}\n")
            self.stats_text.insert(
                tk.END, f"• Avg Degree: {stats['avg_degree']:.2f}\n")
            self.stats_text.insert(
                tk.END, f"• Clustering: {stats['clustering']:.3f}\n\n")

            self.visualize_network()
            self.update_statistics_display()

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

            self.stats_text.insert(tk.END, "🚀 Simulation started!\n")
            self.stats_text.see(tk.END)
            self.visualize_network()
            self.update_statistics_display()
            self.running = True
            self.run_simulation_loop()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to start simulation: {str(e)}")

    def stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        self.stats_text.insert(tk.END, "⏹️ Simulation stopped.\n")
        self.stats_text.see(tk.END)

    def step_simulation(self):
        """Perform one simulation step."""
        if self.simulator is None:
            messagebox.showerror("Error", "Please start simulation first")
            return

        new_inf, new_rec = self.simulator.simulate_day()
        self.update_statistics_display()
        self.visualize_network()

        # Log step results
        self.stats_text.insert(
            tk.END, f"📅 Day {self.simulator.current_day}: {new_inf} new infections, {new_rec} recovered\n")
        self.stats_text.see(tk.END)

    def run_simulation_loop(self):
        """Run simulation continuously."""
        if not self.running:
            return

        if self.simulator.infected_count[-1] == 0:
            self.running = False
            self.stats_text.insert(tk.END, "\n✅ Simulation complete! Disease has been eradicated.\n")
            self.stats_text.see(tk.END)
            return

        self.step_simulation()
        self.root.after(200, self.run_simulation_loop)  # Slower for better visualization

    def reset_simulation(self):
        """Reset simulation."""
        self.running = False
        self.simulator = None
        if self.network:
            for node in self.network.nodes:
                self.network.status[node] = 'S'
                self.network.infection_time[node] = -1
            self.visualize_network()
            self.update_statistics_display()
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "🔄 Simulation reset.\n")

    def visualize_network(self):
        """Draw network on canvas using force-directed layout."""
        self.canvas.delete('all')

        if self.network is None:
            return

        # Transform coordinates to canvas space
        w = self.canvas.winfo_width() or 800
        h = self.canvas.winfo_height() or 600
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
            self.canvas.create_line(x1, y1, x2, y2, fill='#e0e0e0', width=0.5)

        # Draw nodes
        for node in self.network.nodes:
            x, y = positions[node]

            # Color based on status
            if self.network.status[node] == 'S':
                color = self.colors['accent']  # Light blue
            elif self.network.status[node] == 'I':
                color = self.colors['danger']  # Red
            else:
                color = self.colors['success']  # Green

            # Size based on degree
            degree = self.network.get_degree(node)
            size = min(8, 2 + degree / 4)

            self.canvas.create_oval(x-size, y-size, x+size, y+size,
                                    fill=color, outline=self.colors['dark'], width=1)

        # Update layout if simulation is running
        if self.running and self.layout:
            self.layout.iterate(1)

    def update_statistics_display(self):
        """Update the real-time statistics display."""
        if self.simulator:
            stats = self.simulator.get_statistics()
            
            # Update labels
            self.day_label.config(text=str(stats['day']))
            self.sus_label.config(text=str(stats['susceptible']))
            self.inf_label.config(text=str(stats['infected']))
            self.rec_label.config(text=str(stats['recovered']))
            self.new_inf_label.config(text=str(stats['new_infections']))
            self.attack_label.config(text=f"{stats['attack_rate']*100:.1f}%")
        elif self.network:
            # Initial state
            self.day_label.config(text="0")
            self.sus_label.config(text=str(self.network.n_nodes))
            self.inf_label.config(text="0")
            self.rec_label.config(text="0")
            self.new_inf_label.config(text="0")
            self.attack_label.config(text="0.0%")

    def update_statistics(self):
        """Update statistics display in text widget (legacy method)."""
        if self.simulator:
            stats = self.simulator.get_statistics()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"📅 Day: {stats['day']}\n")
            self.stats_text.insert(
                tk.END, f"💙 Susceptible: {stats['susceptible']}\n")
            self.stats_text.insert(tk.END, f"💔 Infected: {stats['infected']}\n")
            self.stats_text.insert(
                tk.END, f"💚 Recovered: {stats['recovered']}\n")
            self.stats_text.insert(
                tk.END, f"📈 Total Infected: {stats['total_infected']}\n")
            self.stats_text.insert(
                tk.END, f"🆕 New Infections: {stats['new_infections']}\n")
            self.stats_text.insert(
                tk.END, f"🎯 Attack Rate: {stats['attack_rate']*100:.1f}%\n")


def main():
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()