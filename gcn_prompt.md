# Graph Convolutional Network (GCN) Prompt for Harness Bridge SHM Dataset

Act as an expert Senior Machine Learning Engineer and Structural Health Monitoring (SHM) researcher. I need you to write a complete, production-ready Python script using PyTorch, PyTorch Geometric, and NetworkX to implement a Graph Convolutional Network (GCN) for non-colocated damage detection on a double-deck bridge model, using our real experimental acceleration dataset.

### 1. Structural Graph Topology & Spatial Coordinates Setup
The bridge consists of a lower deck (uninstrumented), an upper deck (where the sensors are placed), support columns going to the ground, and columns connecting the two decks.

1. Lower Deck: Has 11 joints (Nodes 1 to 11) at vertical level y = 0.0 with coordinates:
   - Node 1: 0.00 (Fixed Support at deck level | Fixity ID: 2)
   - Node 2: 1.82 (Free Node | Fixity ID: 0)
   - Node 3: 3.64 (Free Node | Fixity ID: 0)
   - Node 4: 5.46 (Zero Point Ground Column Connection | Fixity ID: 1)
   - Node 5: 7.46 (Free Node | Fixity ID: 0)
   - Node 6: 9.46 (Free Node | Fixity ID: 0)
   - Node 7: 11.46 (Free Node | Fixity ID: 0)
   - Node 8: 13.46 (Zero Point Ground Column Connection | Fixity ID: 1)
   - Node 9: 14.71 (Free Node | Fixity ID: 0)
   - Node 10: 15.96 (Free Node | Fixity ID: 0)
   - Node 11: 17.21 (Fixed Support at deck level | Fixity ID: 2)

2. Upper Deck: Situated at y = 1.5. It has 11 joints (Nodes 12 to 22) directly above the corresponding lower deck nodes:
   - Node 12 sits above Node 1, Node 13 above Node 2, ..., up to Node 22 above Node 11.
   - An upper deck connects these nodes sequentially: 12-13, 13-14, ..., 21-22.
   - All upper deck nodes are Free Nodes (Fixity ID: 0).

3. Deck-to-Deck Columns: Vertical columns are erected upwards from the lower deck at Joints 1, 4, 5, 6, 8, and 11 to the upper deck.
   - These form vertical column members connecting: 1-12, 4-15, 5-16, 6-17, 8-19, and 11-22.

4. Ground Columns: Vertical columns extend down to the ground (y = -1.5) at Node 4 (Joint 4) and Node 8 (Joint 8). 
   - Define ground foundation nodes: Node 23 (below Node 4 | Fixity ID: 1) and Node 24 (below Node 8 | Fixity ID: 1).

### 2. Edge Connectivity & Edge Weights
Construct the physical edge index tensor (`edge_index`) representing an undirected graph:
- Lower Deck Members: (1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10-11)
- Upper Deck Members: (12-13, 13-14, 14-15, 15-16, 16-17, 17-18, 18-19, 19-20, 20-21, 21-22)
- Deck-to-Deck Columns: (1-12, 4-15, 5-16, 6-17, 8-19, 11-22)
- Columns-to-Ground: (4-23, 8-24)

**Edge Weights / Attributes**:
- For each edge connecting Node $i$ and Node $j$, calculate the physical Euclidean distance:
  $$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
- Define edge weights as the reciprocal of the distance ($w_{ij} = 1.0 / d_{ij}$) to feed into the GCN convolution layers as spatial weights.

### 3. Data Loading & Feature Construction for Our Dataset
The script must load node features using our clean wired accelerometer dataset:
- File: `Harness Bridge Data Clean.csv` (Skip the first 25 metadata lines).
- Sample Rate: 128.0 Hz.
- Active Sensor Columns: The CSV contains 7 sensor columns that map sequentially to the upper deck nodes: **[Node 13, Node 14, Node 16, Node 17, Node 18, Node 20, Node 21]** (corresponding to x-coordinates of lower deck Joints 2, 3, 5, 6, 7, 9, and 10).

**Physics-Informed Node Feature Construction**:
For each of the 24 nodes, build a compound input feature representation:
1. **Temporal Features**: 
   - Instrumented nodes receive windowed chunks of real acceleration.
   - Uninstrumented nodes receive zero vectors of same window length.
2. **Spatial Coordinate Features**: Concatenate the absolute $(x, y)$ coordinates of each node directly to its feature vector.
3. **Fixity Boundary Conditions**: Map the Fixity IDs (`0` = Free, `1` = Zero Point Support, `2` = Fixed Support) through a learnable `torch.nn.Embedding(3, embedding_dim)` and concatenate it to the node features.

### 4. Non-Colocated Damage Scenario Simulation
Simulate structural damage occurring on an uninstrumented member, such as a ground column (e.g. column 4-23) or a deck-to-deck column (e.g. column 4-15):
- Introduce a simulated localized phase shift, 5% amplitude increase, and frequency modification to the real time-series measurements of the closest upper-deck sensor nodes (Node 13 and Node 14) to represent the physical signature of the damage.

### 5. GCN Architecture Requirements
Implement a PyTorch Geometric model called `BridgeSHMNet`:
- **Input Encoder**: Separately project:
  - Temporal window series using a 1D Convolution or Linear layer.
  - Absolute $(x,y)$ coordinates using a Linear layer.
  - Fixity category using an Embedding layer.
  - Concatenate these three embeddings to form the initial node representation `H_0` of shape `[24, embedding_dim]`.
- **Convolutions**: 3 layers of `GCNConv` (or `GATConv` with attention weights) using the calculated spatial `edge_weight` values to propagate message-passing information through the columns and decks.
- **Explainability**: Return or expose learned edge attention weights or node importance heatmaps.
- **Output Head**: Binary classification (Healthy vs. Damaged) or Autoencoder reconstruction error per node.

### 6. Expected Output
Provide clean, object-oriented, fully commented Python code that:
1. Loads the `Harness Bridge Data Clean.csv` dataset.
2. Builds the NetworkX graph and visualizes the bridge layout, color-coding nodes by category (Upper Deck Sensors, Lower/Upper Uninstrumented Nodes, Supports, Ground Bases).
3. Sets up training pairs and defines the training loop.
4. Plots an "Explainable AI Heatmap" demonstrating how prediction error or attention drops highlight the uninstrumented members that were damaged.
