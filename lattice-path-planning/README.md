# Lattice-Path-Planning (Continuous Printing Optimized)

This project implements advanced path planning algorithms for **continuous 3D printing** (also known as One-Stroke or Spiral Vase mode on steroids). It is specifically optimized for materials and processes that benefit from non-stop extrusion, such as **Ceramic/Clay 3D Printing** and high-speed FDM.

## Key Innovations

### 1. Global Spiral Mode (One-Stroke Printing)
Unlike traditional slicers that lift the nozzle between layers or paths, this planner generates a **single continuous path** from the very first layer to the last.
- **No Retractions**: Eliminates stops and starts, crucial for paste extruders (clay/concrete).
- **Seamless Layer Transitions**: The Z-height increases continuously or uses a smooth transition, avoiding layer seams.
- **M82 Absolute Extrusion**: Ensures precise extrusion tracking to prevent cumulative errors.

### 2. "Wavy" Organic Infill
Replaces standard geometric infills with a **Sinusoidal Lattice**.
- **Structural Integrity**: Wavy patterns provide better self-supporting properties and layer adhesion.
- **Aesthetics**: Creates an organic, woven texture suitable for visible internal structures.
- **Algorithm**: Uses a hexagonal-based grid perturbed by sine waves, connected via `cKDTree` for robust graph generation.

### 3. Adaptive Z-Modulation
Implements a non-planar slicing approach where the nozzle moves up and down within a layer to prevent collision and improve adhesion.
- **Smooth Wave**: Uses a `(1 - cos(x))` function based on global distance.
- **Anti-Digging**: Ensures the nozzle always moves relative to the previous layer's wave, preventing it from plowing into deposited material.

### 4. Robust Connection & Gap Handling
- **Smart Bridging**: If a gap is too large to travel, the planner creates a slow-extrusion bridge instead of an empty travel move.
- **Flow Rate Control**: Integrated `flow_rate` parameter to fine-tune extrusion for different materials.
- **Hole Support**: Correctly handles internal holes in polygons using Shapely's containment logic.

## Usage

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `matplotlib`, `scipy`, `networkx`, `shapely`, `trimesh`

### Running the Planner
The main entry point is `main.py`. It loads an STL file, slices it, generates the path, and exports G-code.

```bash
python main.py
```

### Configuration
Configuration is managed in `main.py` (parameters dict) and `printer.json` (machine limits).

#### Key Parameters (`main.py`)
- `infill`: Set to `"wavy"` for the organic lattice.
- `infill_size`: Controls the density of the lattice (e.g., `1.5` mm).
- `global_spiral`: Set to `True` for one-stroke continuous printing.
- `flow_rate`: Extrusion multiplier (default `1.0`).
- `z_follow_amp`: Amplitude of the Z-modulation wave.

## File Structure
- `main.py`: Entry point. Sets parameters, loads STL, and orchestrates the slicing/planning.
- `slicer.py`: Handles STL slicing, polygon generation, and lattice graph creation (contains `generate_wavy_lattice`).
- `planner.py`: Core path planning logic. Implements the beam search, Eulerian path generation, and G-code export.
- `printer.json`: Printer-specific constraints (bed size, max speeds).

## Troubleshooting
- **Empty Objects**: Ensure `infill` is set to `"wavy"` and `infill_size` is appropriate for your model size.
- **Extrusion Issues**: Adjust `flow_rate` in `main.py` if under/over-extruding.
- **Connection Gaps**: The planner automatically attempts to bridge gaps; check `line_width` settings if gaps persist.
