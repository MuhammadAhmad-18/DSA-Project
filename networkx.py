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
import networkx as nx


class SocialNetwork:
    """
    Generates and manages synthetic social networks using NetworkX.
    Implements efficient data structures for network representation.
    """

    def __init__(self, n_nodes, model='small_world', **params):
        """
        Initialize social network.

        Args:
            n_nodes: Number of individuals in the network
            model: 'small_world', 'scale_free', 'random', 'community'
            params: Model-specific parameters
        """
        self.n_nodes = n_nodes
        self.model = model
        self.graph = None
        
        # Node attributes
        self.positions = {}  # Force-directed layout positions
        self.velocities = {}  # For spring embedder algorithm
        self.status = {}  # 'S' (Susceptible), 'I' (Infected), 'R' (Recovered)
        self.infection_time = {}

        # Generate network based on model
        self._generate_network_with_nx(model, params)

        # Initialize node states
        for node in self.graph.nodes():
            self.status[node] = 'S'
            self.infection_time[node] = -1
            # Random initial positions
            self.positions[node] = [
                random.uniform(-1, 1), random.uniform(-1, 1)]
            self.velocities[node] = [0.0, 0.0]

    def _generate_network_with_nx(self, model, params):
        """Generate network using NetworkX library."""
        if model == 'small_world':
            k = params.get('k', 6)
            p = params.get('p', 0.1)
            self.graph = nx.watts_strogatz_graph(self.n_nodes, k, p)
            
        elif model == 'scale_free':
            m = params.get('m', 3)
            self.graph = nx.barabasi_albert_graph(self.n_nodes, m)
            
        elif model == 'random':
            p = params.get('p', 0.01)
            self.graph = nx.erdos_renyi_graph(self.n_nodes, p)
            
        elif model == 'community':
            # Generate network with community structure
            sizes = [self.n_nodes // 3, self.n_nodes // 3, self.n_nodes - 2*(self.n_nodes // 3)]
            p_in = params.get('p_in', 0.1)
            p_out = params.get('p_out', 0.01)
            self.graph = nx.stochastic_block_model(sizes, [[p_in, p_out, p_out], 
                                                          [p_out, p_in, p_out], 
                                                          [p_out, p_out, p_in]])
        else:
            raise ValueError(f"Unknown network model: {model}")

    @property
    def nodes(self):
        """Get list of nodes."""
        return list(self.graph.nodes())

    @property
    def edges(self):
        """Get list of edges."""
        return list(self.graph.edges())

    def get_neighbors(self, node):
        """Get neighbors of a node."""
        return list(self.graph.neighbors(node))

    def get_degree(self, node):
        """Get degree of a node."""
        return self.graph.degree(node)

    def get_network_stats(self):
        """Calculate comprehensive network statistics using NetworkX."""
        degrees = [d for _, d in self.graph.degree()]
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'density': nx.density(self.graph),
            'clustering': nx.average_clustering(self.graph),
        }
        
        # Add model-specific statistics
        if self.model in ['small_world', 'scale_free']:
            try:
                stats['avg_path_length'] = nx.average_shortest_path_length(self.graph)
            except:
                stats['avg_path_length'] = 'Disconnected graph'
                
        if self.model == 'scale_free':
            # Check if degree distribution follows power law
            from collections import Counter
            degree_counts = Counter(degrees)
            stats['degree_distribution'] = dict(degree_counts)
            
        return stats

    def get_connected_components(self):
        """Get connected components of the graph."""
        return list(nx.connected_components(self.graph))

    def get_centrality_measures(self):
        """Calculate various centrality measures."""
        try:
            return {
                'degree_centrality': nx.degree_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph, k=min(100, self.n_nodes)),
                'closeness_centrality': nx.closeness_centrality(self.graph),
                'eigenvector_centrality': nx.eigenvector_centrality(self.graph, max_iter=1000)
            }
        except:
            return {'degree_centrality': nx.degree_centrality(self.graph)}


class ForceDirectedLayout:
    """
    Enhanced force-directed layout with NetworkX integration.
    Can use NetworkX's built-in layouts or custom implementation.
    """

    def __init__(self, network, width=800, height=600, layout_algorithm='spring'):
        self.network = network
        self.width = width
        self.height = height
        self.layout_algorithm = layout_algorithm
        self.iterations = 0
        self.max_iterations = 500

    def compute_layout(self, iterations=50):
        """Compute node positions using various layout algorithms."""
        if self.layout_algorithm == 'spring':
            # Use NetworkX's spring layout
            pos = nx.spring_layout(self.network.graph, iterations=iterations, 
                                 scale=min(self.width, self.height)/2)
            
        elif self.layout_algorithm == 'circular':
            pos = nx.circular_layout(self.network.graph, scale=min(self.width, self.height)/2)
            
        elif self.layout_algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.network.graph, scale=min(self.width, self.height)/2)
            
        elif self.layout_algorithm == 'spectral':
            pos = nx.spectral_layout(self.network.graph, scale=min(self.width, self.height)/2)
            
        else:  # custom force-directed
            pos = self._compute_custom_layout(iterations)
        
        # Update network positions
        for node, (x, y) in pos.items():
            self.network.positions[node] = [x * (self.width/2), y * (self.height/2)]

    def _compute_custom_layout(self, iterations):
        """Custom force-directed layout implementation."""
        # This is your original implementation
        k = math.sqrt((self.width * self.height) / self.network.n_nodes)
        temperature = self.width / 10
        
        # Initialize random positions if not already set
        for node in self.network.nodes:
            if node not in self.network.positions:
                self.network.positions[node] = [
                    random.uniform(-1, 1) * self.width/2, 
                    random.uniform(-1, 1) * self.height/2
                ]

        for iteration in range(iterations):
            displacements = {node: [0.0, 0.0] for node in self.network.nodes}

            # Repulsive forces between all nodes
            nodes = self.network.nodes
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    dx = self.network.positions[u][0] - self.network.positions[v][0]
                    dy = self.network.positions[u][1] - self.network.positions[v][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        repulsive_force = (k * k) / dist
                        displacements[u][0] += (dx / dist) * repulsive_force
                        displacements[u][1] += (dy / dist) * repulsive_force
                        displacements[v][0] -= (dx / dist) * repulsive_force
                        displacements[v][1] -= (dy / dist) * repulsive_force

            # Attractive forces for edges
            for u, v in self.network.edges:
                dx = self.network.positions[v][0] - self.network.positions[u][0]
                dy = self.network.positions[v][1] - self.network.positions[u][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    attractive_force = (dist * dist) / k
                    displacements[v][0] -= (dx / dist) * attractive_force
                    displacements[v][1] -= (dy / dist) * attractive_force
                    displacements[u][0] += (dx / dist) * attractive_force
                    displacements[u][1] += (dy / dist) * attractive_force

            # Update positions
            for node in self.network.nodes:
                dx, dy = displacements[node]
                disp_length = math.sqrt(dx*dx + dy*dy)
                
                if disp_length > 0:
                    self.network.positions[node][0] += (dx / disp_length) * min(disp_length, temperature)
                    self.network.positions[node][1] += (dy / disp_length) * min(disp_length, temperature)

                # Boundary constraints
                self.network.positions[node][0] = max(-self.width/2, min(
                    self.width/2, self.network.positions[node][0]))
                self.network.positions[node][1] = max(-self.height/2, min(
                    self.height/2, self.network.positions[node][1]))

            # Cool down
            temperature *= 0.95

        return self.network.positions


class DiseaseSpreadSimulator:
    """
    Enhanced simulator with NetworkX integration for advanced analysis.
    """

    def __init__(self, network, transmission_prob=0.05, recovery_time=14,
                 initial_infected=5, interaction_model='uniform',
                 infection_strategy='random'):
        """
        Initialize disease spread simulator.

        Args:
            network: SocialNetwork instance
            transmission_prob: Probability of infection per contact (0-1)
            recovery_time: Days until recovery
            initial_infected: Number of initially infected individuals
            interaction_model: 'uniform', 'degree_based', 'centrality_based'
            infection_strategy: 'random', 'superspreader', 'targeted'
        """
        self.network = network
        self.transmission_prob = transmission_prob
        self.recovery_time = recovery_time
        self.interaction_model = interaction_model
        self.infection_strategy = infection_strategy
        self.current_day = 0

        # Statistics tracking
        self.susceptible_count = [network.graph.number_of_nodes()]
        self.infected_count = [0]
        self.recovered_count = [0]
        self.daily_new_infections = [0]
        
        # Advanced statistics
        self.centrality_measures = network.get_centrality_measures()
        self.component_stats = []

        # Initialize infections
        self._initialize_infections(initial_infected)

    def _initialize_infections(self, n_infected):
        """Initialize patient zero(s) with different strategies."""
        if self.infection_strategy == 'superspreader':
            # Infect nodes with highest degree centrality
            degree_centrality = self.centrality_measures['degree_centrality']
            infected_nodes = sorted(degree_centrality.items(), 
                                  key=lambda x: x[1], reverse=True)[:n_infected]
            infected_nodes = [node for node, _ in infected_nodes]
            
        elif self.infection_strategy == 'targeted':
            # Infect nodes with highest betweenness centrality
            betweenness = self.centrality_measures.get('betweenness_centrality', 
                                                     self.centrality_measures['degree_centrality'])
            infected_nodes = sorted(betweenness.items(), 
                                  key=lambda x: x[1], reverse=True)[:n_infected]
            infected_nodes = [node for node, _ in infected_nodes]
            
        else:  # random
            infected_nodes = random.sample(self.network.nodes, n_infected)

        for node in infected_nodes:
            self.network.status[node] = 'I'
            self.network.infection_time[node] = self.current_day

        self.infected_count[0] = n_infected
        self.susceptible_count[0] = self.network.graph.number_of_nodes() - n_infected

    def simulate_day(self):
        """
        Simulate one day of disease spread with enhanced features.
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

            # Get neighbors
            neighbors = list(self.network.get_neighbors(infected_node))
            
            # Determine interaction pattern
            if self.interaction_model == 'degree_based':
                # More interactions for high-degree nodes
                degree = self.network.get_degree(infected_node)
                n_interactions = min(len(neighbors), max(1, degree // 2))
            elif self.interaction_model == 'centrality_based':
                # Use centrality to determine interactions
                centrality = self.centrality_measures['degree_centrality'][infected_node]
                n_interactions = min(len(neighbors), max(1, int(centrality * len(neighbors))))
            else:  # uniform
                n_interactions = len(neighbors)

            # Select neighbors to interact with
            if n_interactions < len(neighbors):
                interacting_neighbors = random.sample(neighbors, n_interactions)
            else:
                interacting_neighbors = neighbors

            # Attempt transmission with probability that can vary by node
            transmission_prob = self.transmission_prob
            for neighbor in interacting_neighbors:
                if self.network.status[neighbor] == 'S':
                    # Optional: transmission probability based on node properties
                    if random.random() < transmission_prob:
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
        susceptible = sum(1 for node in self.network.nodes 
                         if self.network.status[node] == 'S')
        infected = sum(1 for node in self.network.nodes 
                      if self.network.status[node] == 'I')
        recovered = sum(1 for node in self.network.nodes 
                       if self.network.status[node] == 'R')

        self.susceptible_count.append(susceptible)
        self.infected_count.append(infected)
        self.recovered_count.append(recovered)
        self.daily_new_infections.append(len(new_infections))

        # Update component statistics
        self._update_component_stats()

        return len(new_infections), len(nodes_to_recover)

    def _update_component_stats(self):
        """Update statistics about connected components."""
        # Create subgraphs for each health status
        susceptible_nodes = [node for node in self.network.nodes 
                           if self.network.status[node] == 'S']
        infected_nodes = [node for node in self.network.nodes 
                         if self.network.status[node] == 'I']
        
        susceptible_subgraph = self.network.graph.subgraph(susceptible_nodes)
        infected_subgraph = self.network.graph.subgraph(infected_nodes)
        
        self.component_stats.append({
            'day': self.current_day,
            'susceptible_components': nx.number_connected_components(susceptible_subgraph),
            'infected_components': nx.number_connected_components(infected_subgraph),
            'largest_susceptible_component': max((len(c) for c in nx.connected_components(susceptible_subgraph)), default=0),
            'largest_infected_component': max((len(c) for c in nx.connected_components(infected_subgraph)), default=0)
        })

    def get_statistics(self):
        """Get comprehensive simulation statistics."""
        total_infected = self.network.graph.number_of_nodes() - self.susceptible_count[-1]
        
        stats = {
            'day': self.current_day,
            'susceptible': self.susceptible_count[-1],
            'infected': self.infected_count[-1],
            'recovered': self.recovered_count[-1],
            'total_infected': total_infected,
            'new_infections': self.daily_new_infections[-1],
            'attack_rate': total_infected / self.network.graph.number_of_nodes(),
            'peak_infected': max(self.infected_count) if self.infected_count else 0,
            'peak_day': self.infected_count.index(max(self.infected_count)) if self.infected_count else 0
        }
        
        # Add component statistics
        if self.component_stats:
            current_components = self.component_stats[-1]
            stats.update(current_components)
            
        return stats

    def get_network_analysis(self):
        """Get advanced network analysis of current state."""
        # Analyze the subgraph of infected nodes
        infected_nodes = [node for node in self.network.nodes 
                         if self.network.status[node] == 'I']
        
        if infected_nodes:
            infected_subgraph = self.network.graph.subgraph(infected_nodes)
            analysis = {
                'infected_subgraph_nodes': len(infected_nodes),
                'infected_subgraph_edges': infected_subgraph.number_of_edges(),
                'infected_components': nx.number_connected_components(infected_subgraph),
                'avg_degree_infected': sum(d for _, d in infected_subgraph.degree()) / len(infected_nodes) if infected_nodes else 0
            }
        else:
            analysis = {'infected_subgraph_nodes': 0}
            
        return analysis

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


# Update the GUI class to include new NetworkX features
class EnhancedSimulatorGUI(SimulatorGUI):
    """
    Enhanced GUI with NetworkX features.
    """
    
    def _create_widgets(self):
        """Create enhanced GUI layout with NetworkX options."""
        super()._create_widgets()
        
        # Add NetworkX-specific options to control panel
        # You can add dropdowns for:
        # - Additional network models (community)
        # - Layout algorithms (spring, circular, kamada_kawai, spectral)
        # - Infection strategies (random, superspreader, targeted)
        # - Advanced analysis options
        
    def generate_network(self):
        """Generate social network with NetworkX."""
        try:
            n = self.size_var.get()
            model = self.model_var.get().lower().replace(' ', '_')

            if n < 10 or n > 2000:
                messagebox.showerror("Error", "Network size must be between 10 and 2000")
                return

            self.stats_text.insert(tk.END, f"🔄 Generating {model} network with {n} nodes using NetworkX...\n")
            self.stats_text.see(tk.END)
            self.root.update()

            # Generate network with NetworkX
            self.network = SocialNetwork(n, model=model)
            
            # Initialize layout
            self.layout = ForceDirectedLayout(self.network, width=800, height=600, 
                                            layout_algorithm='spring')
            
            # Compute layout
            self.stats_text.insert(tk.END, "📐 Computing network layout...\n")
            self.root.update()
            
            self.layout.compute_layout(iterations=100)

            # Display enhanced network statistics
            stats = self.network.get_network_stats()
            self.stats_text.insert(tk.END, f"\n📊 NetworkX Statistics:\n")
            self.stats_text.insert(tk.END, f"• Nodes: {stats['nodes']}\n")
            self.stats_text.insert(tk.END, f"• Edges: {stats['edges']}\n")
            self.stats_text.insert(tk.END, f"• Density: {stats['density']:.4f}\n")
            self.stats_text.insert(tk.END, f"• Avg Degree: {stats['avg_degree']:.2f}\n")
            self.stats_text.insert(tk.END, f"• Clustering: {stats['clustering']:.3f}\n")
            
            if 'avg_path_length' in stats:
                self.stats_text.insert(tk.END, f"• Avg Path Length: {stats['avg_path_length']:.2f}\n")
            
            # Show connected components
            components = self.network.get_connected_components()
            self.stats_text.insert(tk.END, f"• Connected Components: {len(components)}\n")
            self.stats_text.insert(tk.END, f"• Largest Component: {max(len(c) for c in components) if components else 0}\n\n")

            self.visualize_network()
            self.update_statistics_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate network: {str(e)}")



def main():
    root = tk.Tk()
    app = EnhancedSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()