#!/usr/bin/env python3
"""
covid_network_simulator.py

A Tkinter GUI application that:
 - Loads a dataset (CSV) with columns like User, Timestamp, Hashtags, Country, Text.
 - Generates a social network via multiple methods (hashtags, co-presence, text similarity).
 - Visualizes network using spring_layout embedded in Tkinter (matplotlib).
 - Runs SIR simulation on the network with adjustable parameters.
 - Plots S/I/R time series.
 - Exports graph, results, and generates LaTeX report + Beamer template.

Author: ChatGPT (template)
Date: 2025-11-30
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import io
import math
from datetime import datetime
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Optional: TF-IDF text similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --- UI constants ---
WINDOW_TITLE = "Infectious Disease Spread Simulator — Social Network"
WINDOW_SIZE = "1200x800"

# --- Utility functions ---
def safe_get(df, col, default=None):
    return df[col] if col in df.columns else pd.Series([default]*len(df))

# Simple tooltip
class CreateToolTip(object):
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = y = 0
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)
    def hide(self, event=None):
        tw = self.tipwindow
        if tw:
            tw.destroy()
        self.tipwindow = None

# --- Core application ---
class CovidNetworkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_SIZE)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Data and graph
        self.df = None
        self.G = nx.Graph()
        self.pos = None  # for layout
        self.sim_history = []  # list of dicts with counts over time

        # Simulation params
        self.infection_prob = tk.DoubleVar(value=0.05)
        self.recovery_prob = tk.DoubleVar(value=0.01)
        self.initial_infected = tk.IntVar(value=3)
        self.sim_steps = tk.IntVar(value=50)
        self.running = False

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # Top frame: controls
        top_frame = ttk.Frame(self, padding=(8,8))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Data load controls
        load_btn = ttk.Button(top_frame, text="Load CSV Dataset", command=self.load_csv)
        load_btn.grid(row=0, column=0, padx=4, pady=4, sticky="w")
        CreateToolTip(load_btn, "Load sentimentdataset.csv or any CSV with User, Timestamp, Hashtags, Country columns")

        default_btn = ttk.Button(top_frame, text="Use Sample (first 100 rows)", command=self.load_sample)
        default_btn.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        CreateToolTip(default_btn, "Quick demo using first 100 rows of your loaded dataset")

        export_btn = ttk.Button(top_frame, text="Export Graph (.gpickle)", command=self.export_graph)
        export_btn.grid(row=0, column=2, padx=4, pady=4, sticky="w")

        export_csv_btn = ttk.Button(top_frame, text="Export Simulation CSV", command=self.export_sim_csv)
        export_csv_btn.grid(row=0, column=3, padx=4, pady=4, sticky="w")

        latex_btn = ttk.Button(top_frame, text="Generate LaTeX Report & Beamer", command=self.generate_latex)
        latex_btn.grid(row=0, column=4, padx=4, pady=4, sticky="w")

        # Middle frame: Generation options and simulation controls
        mid = ttk.Frame(self, padding=(8,8))
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # Left side: network generation options
        gen_frame = ttk.Labelframe(mid, text="Network Generation", padding=(8,8))
        gen_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Method selection
        self.method_var = tk.StringVar(value="copresence")
        ttk.Radiobutton(gen_frame, text="Co-presence (Country + Date + Hour)", variable=self.method_var, value="copresence").pack(anchor="w", pady=2)
        ttk.Radiobutton(gen_frame, text="Hashtag-based (projected)", variable=self.method_var, value="hashtag").pack(anchor="w", pady=2)
        tf_text = "Text similarity (TF-IDF; requires scikit-learn)"
        ttk.Radiobutton(gen_frame, text=tf_text, variable=self.method_var, value="text").pack(anchor="w", pady=2)
        CreateToolTip(gen_frame, "Choose method to generate social ties. Co-presence is recommended for COVID simulation.")

        build_btn = ttk.Button(gen_frame, text="Build Network", command=self.build_network_thread)
        build_btn.pack(fill=tk.X, pady=6)
        CreateToolTip(build_btn, "Generates the graph from loaded dataset.")

        # Graph stats
        self.stats_txt = tk.Text(gen_frame, width=36, height=10, state="disabled", wrap="word")
        self.stats_txt.pack(pady=6)

        # Simulation controls
        sim_frame = ttk.Labelframe(mid, text="SIR Simulation Controls", padding=(8,8))
        sim_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        ttk.Label(sim_frame, text="Infection Probability (p)").pack(anchor="w")
        ttk.Scale(sim_frame, from_=0.0, to=1.0, variable=self.infection_prob, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(sim_frame, textvariable=self.infection_prob).pack(anchor="e")

        ttk.Label(sim_frame, text="Recovery Probability (γ)").pack(anchor="w")
        ttk.Scale(sim_frame, from_=0.0, to=1.0, variable=self.recovery_prob, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(sim_frame, textvariable=self.recovery_prob).pack(anchor="e")

        ttk.Label(sim_frame, text="Initial Infected Count").pack(anchor="w")
        ttk.Spinbox(sim_frame, from_=1, to=1000, textvariable=self.initial_infected).pack(fill=tk.X)

        ttk.Label(sim_frame, text="Simulation Steps").pack(anchor="w")
        ttk.Spinbox(sim_frame, from_=1, to=10000, textvariable=self.sim_steps).pack(fill=tk.X)

        run_btn = ttk.Button(sim_frame, text="Run Full Simulation", command=self.run_simulation_thread)
        step_btn = ttk.Button(sim_frame, text="Run 1 Step", command=lambda: self.run_simulation(steps=1))
        reset_btn = ttk.Button(sim_frame, text="Reset States", command=self.reset_states)
        stop_btn = ttk.Button(sim_frame, text="Stop Simulation", command=self.stop_simulation)

        run_btn.pack(fill=tk.X, pady=4)
        step_btn.pack(fill=tk.X, pady=2)
        reset_btn.pack(fill=tk.X, pady=2)
        stop_btn.pack(fill=tk.X, pady=2)

        # Right side: visualization
        vis_frame = ttk.Frame(self, padding=(6,6))
        vis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for network drawing
        self.fig_net, self.ax_net = plt.subplots(figsize=(6,6))
        self.ax_net.axis("off")
        self.canvas_net = FigureCanvasTkAgg(self.fig_net, master=vis_frame)
        self.canvas_net.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel for SIR chart and controls
        right_panel = ttk.Frame(vis_frame, padding=(6,6))
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # SIR plot
        self.fig_sir, self.ax_sir = plt.subplots(figsize=(5,3))
        self.canvas_sir = FigureCanvasTkAgg(self.fig_sir, master=right_panel)
        self.canvas_sir.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info/log
        log_frame = ttk.Labelframe(right_panel, text="Log / Messages", padding=(8,8))
        log_frame.pack(fill=tk.BOTH, expand=False)
        self.log_text = tk.Text(log_frame, width=40, height=12, state="disabled", wrap="word")
        self.log_text.pack(fill=tk.BOTH)

        # Bottom status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Logging helpers
    def log(self, msg, newline=True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(msg if len(msg) < 80 else msg[:77]+"...")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n" if newline else msg)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --- Data Loading ---
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            self.df.columns = [c.strip() for c in self.df.columns]
            # Attempt parse timestamp
            if "Timestamp" in self.df.columns:
                try:
                    self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"], errors="coerce")
                except Exception:
                    pass
            # Ensure expected columns exist
            self.log(f"Loaded dataset: {os.path.basename(path)} with {len(self.df)} rows")
            self.show_stats()
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))
            self.log(f"Error loading CSV: {e}")

    def load_sample(self):
        # Quick sample: use currently loaded df or create minimal demo
        if self.df is None:
            messagebox.showinfo("No dataset", "Load a dataset first (use your sentimentdataset.csv).")
            return
        sample = self.df.head(200).copy()
        self.df = sample
        self.log("Using first 200 rows as sample")
        self.show_stats()

    def show_stats(self):
        self.stats_txt.configure(state="normal")
        self.stats_txt.delete("1.0", "end")
        if self.df is None:
            self.stats_txt.insert("end", "No dataset loaded.\n")
        else:
            cols = list(self.df.columns)
            self.stats_txt.insert("end", f"Rows: {len(self.df)}\nColumns: {len(cols)}\n\n")
            self.stats_txt.insert("end", "Columns detected:\n")
            for c in cols:
                self.stats_txt.insert("end", f" - {c}\n")
            # show sample values for User, Country, Hashtags
            if "User" in self.df.columns:
                users = self.df["User"].astype(str).unique()[:10].tolist()
                self.stats_txt.insert("end", f"\nSample Users: {users}\n")
            if "Country" in self.df.columns:
                countries = self.df["Country"].astype(str).unique()[:10].tolist()
                self.stats_txt.insert("end", f"Sample Countries: {countries}\n")
            if "Hashtags" in self.df.columns:
                self.stats_txt.insert("end", "Hashtags column detected (may need cleaning)\n")
        self.stats_txt.configure(state="disabled")

    # --- Network building ---
    def build_network_thread(self):
        t = threading.Thread(target=self.build_network, daemon=True)
        t.start()

    def build_network(self):
        if self.df is None:
            messagebox.showinfo("No dataset", "Please load the CSV dataset first.")
            return
        method = self.method_var.get()
        self.log(f"Building network using method: {method}")
        self.status_var.set("Building network...")
        # reset
        G = nx.Graph()
        df = self.df.copy()

        # Normalize user column
        if "User" not in df.columns:
            messagebox.showerror("Missing column", "Dataset must contain 'User' column.")
            return
        df["User"] = df["User"].astype(str).str.strip()

        if method == "copresence":
            # Ensure timestamp parsed or use Year/Month/Day/Hour
            if "Timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
                try:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                except Exception:
                    pass
            # create year/month/day/hour from either Timestamp or columns
            if "Timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
                df["Year"] = df["Timestamp"].dt.year
                df["Month"] = df["Timestamp"].dt.month
                df["Day"] = df["Timestamp"].dt.day
                df["Hour"] = df["Timestamp"].dt.hour
            required = ["Country", "Year", "Month", "Day", "Hour"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                self.log(f"Missing columns for co-presence: {missing}. Trying to infer from Timestamp")
            grouped = df.groupby(["Country", "Year", "Month", "Day", "Hour"])
            for key, group in grouped:
                users = group["User"].unique()
                # fully connect users in same place/time
                for u, v in combinations(users, 2):
                    if u != v:
                        G.add_edge(u, v)
            self.log(f"Co-presence graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        elif method == "hashtag":
            # Build bipartite graph user<->hashtag then project
            df["Hashtags"] = df.get("Hashtags", "").astype(str)
            # simple splitting by whitespace and '#'
            bip = nx.Graph()
            for _, row in df.iterrows():
                user = row["User"]
                tags_raw = row["Hashtags"]
                # Normalize tags
                if pd.isna(tags_raw) or tags_raw == "nan":
                    tags = []
                else:
                    # flexible parsing: split on spaces and commas and remove words without #
                    tokens = [tok.strip().strip(",") for tok in tags_raw.split() if tok.strip()]
                    # keep tokens that start with # or treat all as tags
                    tags = []
                    for tok in tokens:
                        if tok.startswith("#"):
                            tags.append(tok.lower())
                        else:
                            # maybe comma-separated list like "#a,#b"
                            for t in tok.split(","):
                                t = t.strip()
                                if t.startswith("#"):
                                    tags.append(t.lower())
                for tag in tags:
                    bip.add_node(user, bipartite=0)
                    bip.add_node(tag, bipartite=1)
                    bip.add_edge(user, tag)
            # project to users
            users = [n for n, d in bip.nodes(data=True) if d.get("bipartite")==0]
            if len(users) == 0:
                self.log("Warning: No hashtags found; hashtag graph empty.")
            else:
                G = nx.bipartite.projected_graph(bip, users)
            self.log(f"Hashtag-projected graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        elif method == "text":
            if not SKLEARN_AVAILABLE:
                messagebox.showerror("Dependency missing", "Text similarity requires scikit-learn. Install it with `pip install scikit-learn`")
                return
            # Use TF-IDF on 'Text' column
            if "Text" not in df.columns:
                messagebox.showerror("Missing column", "Dataset must contain 'Text' column for text-similarity method.")
                return
            texts = df["Text"].fillna("").astype(str).tolist()
            users = df["User"].astype(str).tolist()
            self.log("Computing TF-IDF matrix...")
            vec = TfidfVectorizer(stop_words="english", max_features=2000)
            X = vec.fit_transform(texts)
            sim = cosine_similarity(X)  # might be large
            # build graph with threshold
            threshold = 0.45
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    if sim[i,j] > threshold:
                        G.add_edge(users[i], users[j])
            self.log(f"Text-similarity graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()} (threshold {threshold})")
        else:
            self.log("Unknown method selected")

        # Finalize graph
        if G.number_of_nodes() == 0:
            messagebox.showwarning("Empty graph", "Generated graph is empty. Check your dataset and method.")
        else:
            # remove isolates optionally
            isolates = list(nx.isolates(G))
            G.remove_nodes_from(isolates)
            # store graph
            self.G = G
            # compute layout (spring) - potentially slow, do with limited iterations
            self.status_var.set("Computing layout (spring)...")
            # try to set k proportional to sqrt(area/n)
            try:
                k = None
                if self.G.number_of_nodes() > 0:
                    k = 1.0 / math.sqrt(max(1, self.G.number_of_nodes()))
                self.pos = nx.spring_layout(self.G, k=k, iterations=80)
            except Exception as e:
                self.pos = nx.spring_layout(self.G)
            self.plot_graph()
            self.show_graph_stats()
            self.status_var.set("Graph built.")
            self.log(f"Graph built: nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}")

    def show_graph_stats(self):
        # show on stats panel
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()
        comps = nx.number_connected_components(self.G) if n>0 else 0
        degs = [d for _,d in self.G.degree()] if n>0 else []
        avg_deg = sum(degs)/len(degs) if degs else 0
        txt = f"Graph statistics\nNodes: {n}\nEdges: {m}\nConnected components: {comps}\nAverage degree: {avg_deg:.2f}\n"
        # top degree nodes
        if n > 0:
            topk = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)[:10]
            txt += "\nTop degree nodes:\n"
            for node,deg in topk:
                txt += f" - {node}: {deg}\n"
        self.stats_txt.configure(state="normal")
        self.stats_txt.delete("1.0","end")
        self.stats_txt.insert("end", txt)
        self.stats_txt.configure(state="disabled")

    # --- Plotting ---
    def plot_graph(self):
        self.ax_net.clear()
        self.ax_net.axis("off")
        if self.G is None or self.G.number_of_nodes() == 0:
            self.ax_net.text(0.5,0.5,"No graph to display", ha="center", va="center")
            self.canvas_net.draw()
            return
        # Color nodes by state (if present)
        color_map = []
        for n in self.G.nodes():
            state = self.G.nodes[n].get("state","S")
            if state == "S": color_map.append("#7fb3d5")  # susceptible blue
            elif state == "I": color_map.append("#e74c3c")  # infected red
            elif state == "R": color_map.append("#2ecc71")  # recovered green
            else: color_map.append("#95a5a6")
        # fallback layout
        pos = self.pos if self.pos else nx.spring_layout(self.G, iterations=50)
        nx.draw_networkx_edges(self.G, pos=pos, ax=self.ax_net, alpha=0.2, width=0.6)
        nx.draw_networkx_nodes(self.G, pos=pos, ax=self.ax_net, node_color=color_map, node_size=40)
        # remove axes and render
        self.fig_net.tight_layout()
        self.canvas_net.draw()

    def plot_sir(self):
        self.ax_sir.clear()
        if not self.sim_history:
            self.ax_sir.text(0.5,0.5,"No simulation run yet", ha="center", va="center")
            self.canvas_sir.draw()
            return
        t = [h["step"] for h in self.sim_history]
        S = [h["S"] for h in self.sim_history]
        I = [h["I"] for h in self.sim_history]
        R = [h["R"] for h in self.sim_history]
        self.ax_sir.plot(t, S, label="Susceptible")
        self.ax_sir.plot(t, I, label="Infected")
        self.ax_sir.plot(t, R, label="Recovered")
        self.ax_sir.set_xlabel("Step")
        self.ax_sir.set_ylabel("Count")
        self.ax_sir.legend()
        self.fig_sir.tight_layout()
        self.canvas_sir.draw()

    # --- SIR simulation ---
    def run_simulation_thread(self):
        t = threading.Thread(target=self.run_simulation, daemon=True)
        t.start()

    def run_simulation(self, steps=None):
        if self.G is None or self.G.number_of_nodes() == 0:
            messagebox.showinfo("No graph", "Generate the network first.")
            return
        if steps is None:
            steps = int(self.sim_steps.get())

        # initialize states if not already
        if not any(self.G.nodes[n].get("state") == "I" for n in self.G.nodes()):
            self.reset_states()

        # run steps in background
        self.running = True
        self.log(f"Simulation started for {steps} steps.")
        self.status_var.set("Simulation running...")
        for step in range(steps):
            if not self.running:
                self.log("Simulation stopped by user")
                break
            self._simulate_one_step(step+1)
            # update visualizations
            self.plot_graph()
            self.plot_sir()
            time.sleep(0.02)  # small pause to allow screen updates
        self.running = False
        self.status_var.set("Simulation finished" if self.running==False else "Stopped")
        self.log("Simulation completed.")

    def stop_simulation(self):
        self.running = False
        self.status_var.set("Stopping...")

    def reset_states(self):
        # set all Susceptible
        for n in self.G.nodes():
            self.G.nodes[n]["state"] = "S"
        # choose initial infected randomly
        k = max(1, int(self.initial_infected.get()))
        nodes = list(self.G.nodes())
        if k >= len(nodes):
            k = max(1, len(nodes)//10)
        rng = np.random.RandomState(seed=int(time.time())%100000)
        infected = rng.choice(nodes, size=k, replace=False).tolist()
        for u in infected:
            self.G.nodes[u]["state"] = "I"
        # clear history and add initial counts
        self.sim_history = []
        self._record_history(step=0)
        self.plot_graph()
        self.plot_sir()
        self.log(f"States reset. Initial infected: {len(infected)}")

    def _simulate_one_step(self, step):
        p = float(self.infection_prob.get())
        gamma = float(self.recovery_prob.get())
        to_infect = set()
        to_recover = set()
        # Infectious interactions
        for node in list(self.G.nodes()):
            if self.G.nodes[node].get("state") == "I":
                for nbr in self.G.neighbors(node):
                    if self.G.nodes[nbr].get("state") == "S":
                        if np.random.rand() < p:
                            to_infect.add(nbr)
                # recovery
                if np.random.rand() < gamma:
                    to_recover.add(node)
        # apply state updates
        for n in to_infect:
            self.G.nodes[n]["state"] = "I"
        for r in to_recover:
            self.G.nodes[r]["state"] = "R"
        self._record_history(step=step)
        self.log(f"Step {step}: S={self.sim_history[-1]['S']} I={self.sim_history[-1]['I']} R={self.sim_history[-1]['R']}")

    def _record_history(self, step):
        s = sum(1 for n in self.G.nodes() if self.G.nodes[n].get("state")=="S")
        i = sum(1 for n in self.G.nodes() if self.G.nodes[n].get("state")=="I")
        r = sum(1 for n in self.G.nodes() if self.G.nodes[n].get("state")=="R")
        self.sim_history.append({"step": step, "S": s, "I": i, "R": r})

    # --- Export functions ---
    def export_graph(self):
        if self.G is None or self.G.number_of_nodes() == 0:
            messagebox.showinfo("No graph", "Build the network first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".gpickle", filetypes=[("NetworkX gpickle","*.gpickle")])
        if not path:
            return
        nx.write_gpickle(self.G, path)
        self.log(f"Graph exported: {path}")

    def export_sim_csv(self):
        if not self.sim_history:
            messagebox.showinfo("No data", "No simulation history to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path:
            return
        df = pd.DataFrame(self.sim_history)
        df.to_csv(path, index=False)
        self.log(f"Simulation history exported: {path}")

    # --- LaTeX report generation ---
    def generate_latex(self):
        if self.G is None or self.G.number_of_nodes()==0:
            messagebox.showinfo("No graph", "Build the network first to include data in the report.")
            return
        folder = filedialog.askdirectory(title="Select folder to save LaTeX files")
        if not folder:
            return
        # Create simple report.tex and beamer.tex
        report_tex = os.path.join(folder, "report.tex")
        beamer_tex = os.path.join(folder, "presentation.tex")
        # basic stats
        nodes = self.G.number_of_nodes()
        edges = self.G.number_of_edges()
        avg_deg = sum(d for _,d in self.G.degree()) / nodes if nodes>0 else 0
        # Save a small graph image
        img_path = os.path.join(folder, "network_snapshot.png")
        try:
            plt.figure(figsize=(6,6))
            pos = self.pos if self.pos else nx.spring_layout(self.G, iterations=50)
            nx.draw_networkx_nodes(self.G, pos, node_size=30)
            nx.draw_networkx_edges(self.G, pos, alpha=0.2)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
        except Exception:
            img_path = ""

        report_content = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{hyperref}
\title{Simulating Infectious Disease Spread in a Social Network}
\author{Team}
\date{\today}
\begin{document}
\maketitle
\section*{Summary}
This report was generated automatically by the simulator.
\section*{Network Statistics}
\begin{itemize}
\item Nodes: %d
\item Edges: %d
\item Average degree: %.2f
\end{itemize}
\section*{Method}
Network generation method: %s

\section*{Simulation (SIR)}
Initial infected: %d \\
Infection probability: %.3f \\
Recovery probability: %.3f \\

\section*{Network snapshot}
\begin{figure}[h]
\centering
\includegraphics[width=0.6\textwidth]{%s}
\caption{Network snapshot}
\end{figure}

\end{document}
""" % (nodes, edges, avg_deg, self.method_var.get(), int(self.initial_infected.get()), float(self.infection_prob.get()), float(self.recovery_prob.get()), os.path.basename(img_path))

        beamer_content = r"""\documentclass{beamer}
\usetheme{Madrid}
\title{Simulating Infectious Disease Spread}
\author{Team}
\date{\today}
\begin{document}
\begin{frame}
\titlepage
\end{frame}
\begin{frame}{Network Stats}
\begin{itemize}
\item Nodes: %d
\item Edges: %d
\item Average degree: %.2f
\end{itemize}
\end{frame}
\begin{frame}{Simulation Parameters}
\begin{itemize}
\item Initial infected: %d
\item Infection prob: %.3f
\item Recovery prob: %.3f
\end{itemize}
\end{frame}
\end{document}
""" % (nodes, edges, avg_deg, int(self.initial_infected.get()), float(self.infection_prob.get()), float(self.recovery_prob.get()))

        with open(report_tex, "w", encoding="utf-8") as f:
            f.write(report_content)
        with open(beamer_tex, "w", encoding="utf-8") as f:
            f.write(beamer_content)
        self.log(f"LaTeX report and Beamer saved to {folder}")
        messagebox.showinfo("LaTeX generated", f"report.tex and presentation.tex saved to:\n{folder}\n(You can compile them with pdflatex)")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            self.destroy()

# --- Run app ---
if __name__ == "__main__":
    app = CovidNetworkApp()
    app.mainloop()
