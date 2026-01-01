"""
File to contain Slicer class which takes a stl model and generates cross sections with different
infill geometry
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List
import trimesh 
from matplotlib.patches import Polygon
import lattpy as lp
from lattpy import Lattice
import networkx as nx
from shapely.geometry import LineString
import shapely as shp
from collections import defaultdict
import pickle
import os
import math
from scipy.spatial import cKDTree

from planner import Planner

class Slicer:
    def __init__(self, params: Dict) -> None:
        """
        Initialize Slicer object with parameters dictionary
        
        Params:
            params: dict - A dictionary that contains the following set values: 
                "layer_height": float, "infill": "square" | "triangle" | "hexagon" | "wavy" | "none" | "contour", and
                "line_width": float
        """
        self.params = params

    def generate_wavy_lattice(self, polygon) -> nx.Graph:
        """
        Generates a 2D lattice graph restricted to the polygon with sinusoidal distortion.
        Innovation: "Organic/Wavy" Infill for better layer adhesion and aesthetics.
        """
        size = self.params["infill_size"]
        minx, miny, maxx, maxy = polygon.bounds
        
        # Expand bounds
        minx -= size
        miny -= size
        maxx += size
        maxy += size
        
        # Grid Parameters (Hexagonal-like)
        dy = size * math.sqrt(3) / 2
        dx = size
        
        rows = int((maxy - miny) / dy) + 2
        cols = int((maxx - minx) / dx) + 2
        
        points = []
        # Generate Grid
        for r in range(rows):
            y_base = miny + r * dy
            offset = (r % 2) * (dx / 2)
            for c in range(cols):
                x_base = minx + c * dx + offset
                
                # Apply Distortion (Wavy/Organic)
                # Amplitudes and Frequencies
                amp = 0.2 * size
                freq = 0.5 / size
                
                # Perturb
                x_new = x_base + amp * math.sin(y_base * freq * 2 * math.pi)
                y_new = y_base + amp * math.cos(x_base * freq * 2 * math.pi)
                
                points.append((x_new, y_new))
        
        # Filter points inside polygon
        from matplotlib.path import Path
        poly_path = Path(list(polygon.exterior.coords))
        points_arr = np.array(points)
        
        if len(points_arr) == 0:
            return nx.Graph()
            
        mask = poly_path.contains_points(points_arr)
        valid_points = points_arr[mask]
        
        if len(valid_points) == 0:
            return nx.Graph()
            
        # Build Graph
        G = nx.Graph()
        for i, pt in enumerate(valid_points):
            G.add_node(i, x=pt[0], y=pt[1])
            
        # Connect Neighbors using KDTree
        tree = cKDTree(valid_points)
        # Search radius: slightly larger than max grid distance (size)
        # In hex grid, max dist is size. With distortion, maybe 1.4*size.
        search_radius = size * 1.4
        pairs = tree.query_pairs(r=search_radius)
        
        for i, j in pairs:
            G.add_edge(i, j)
            
        # print(f"DEBUG: Generated Wavy Lattice with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            
        return G


    def slice(self, path: str, debug_mode = False) -> None:
        """
        Take an input path str to a 3D object file. Take the file and slice it according go given
        parameters, and export path to a gcode file. The slicer first converts model into a set of
        layers, then it converts each layer into an undirected graph. It then creates a tree over
        the graph, and performs a DFS traversal to create a single, non-intersecting path.

        Params:
            path: str - a string that leads to the 3D model file
            debug_mode: bool - an option that causes the slicer to save different
                variables as pickle file during the running of the code. When debugging the code, it
                saves time re-running the same section of code over and over

        Returns:
            None
        """

        # if we are in debug mode, jump to debug function
        if debug_mode:
            return self.slice_debug(path)
        
        # if we are not in debug mode, continue slicing as normal

        # load the mesh and slice and perform a quick generation of the internal lattice 
        self.load_part(path)
        self.layer_edges = self.create_raw_slices()

        # convert each slice of the 3D model into a convenient polygon object
        self.layer_polygons = self.slice_to_polly()

        # special mode: contour — generate continuous inward offsets without lattice
        if self.params.get("infill") == "contour":
            self.layer_paths = self.generate_contour_paths()
            # initialize a planner for gcode generation
            self.planner = Planner(self.params, self.layer_polygons, [])
        else:
            # normal flow: build lattice and graphs
            # self.lattice = self.generate_lattice() # Removed: using generate_wavy_lattice per layer
            self.layer_graphs = self.generate_layer_graphs()
            self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)
            self.layer_paths = self.planner.generate_layer_paths()

        # generate a gcode file
        self.planner.generate_gcode(self.layer_paths)

        # plot the finals paths
        self.plot_final_paths()

    def generate_contour_paths(self) -> List[List[np.ndarray]]:
        """
        Generate continuous inward-offset contour paths for each polygon region per layer.
        Produces a single stitched path per layer to minimize retractions.
        """
        layer_paths = []
        # Use 0.8 * line_width for overlap to prevent gaps
        step = float(self.params.get("line_width", 0.4)) * 0.8
        for layer_idx in range(self.n_layers):
            stitched = []
            regions = self.layer_polygons[layer_idx]
            region_paths = []
            for region in regions:
                # Handle Shapely Polygon directly
                if isinstance(region, shp.Polygon):
                    base_poly = region
                else:
                    base_poly = shp.Polygon(region.get_xy())
                
                # create inward offsets until polygon vanishes
                cur = base_poly
                while True:
                    cur = cur.buffer(-step)
                    if cur.is_empty:
                        break
                    geoms = list(cur.geoms) if isinstance(cur, shp.MultiPolygon) else [cur]
                    for g in geoms:
                        coords = np.array(g.exterior.coords)
                        if coords.shape[0] >= 2:
                            region_paths.append(coords)
            # stitch all region paths into one continuous path
            if region_paths:
                out = region_paths.pop(0)
                while region_paths:
                    end = out[-1]
                    best_idx = 0
                    best_rev = False
                    best_dist = float("inf")
                    for i, cand in enumerate(region_paths):
                        d1 = np.linalg.norm(cand[0] - end)
                        d2 = np.linalg.norm(cand[-1] - end)
                        if d1 < best_dist:
                            best_dist = d1
                            best_idx = i
                            best_rev = False
                        if d2 < best_dist:
                            best_dist = d2
                            best_idx = i
                            best_rev = True
                    next_path = region_paths.pop(best_idx)
                    if best_rev:
                        next_path = next_path[::-1]
                    connector = np.vstack([end, next_path[0]])
                    out = np.vstack([out, connector, next_path])
                if out.shape[0] > 0 and (out[0] != out[-1]).any():
                    out = np.vstack([out, out[0]])
                stitched.append(out)
            else:
                stitched.append(np.zeros((0, 2)))
            layer_paths.append(stitched)
        return layer_paths

    def slice_debug(self, path: str) -> None:
        """
        Perform the exact same operations as the main slice function, but in order to save time in
        debugging and developing code, save major variables as pickle files and load them instead of
        re-running the whole code. 
        """

        # load the mesh and slice and perform a quick generation of the internal lattice 
        self.load_part(path)
        self.lattice = self.generate_lattice()
        self.layer_edges = self.create_raw_slices()

        # base directory for all pickled variables
        base_dir = "pickled-vars/"

        # extra set of variables to force any section to re-run, even if pickle files exist
        force_slice_to_poly = False
        force_generate_layer_graphs = False
        force_generate_layer_paths = False

        # check to see if layer polygons have already been created
        layer_poly_file = base_dir + "layer_polygons.pckl"
        if os.path.isfile(layer_poly_file) and not force_slice_to_poly:
            with open(layer_poly_file, 'rb') as f:
                self.layer_polygons = pickle.load(f)
        else:
            self.layer_polygons = self.slice_to_polly()
            with open(layer_poly_file, 'wb') as f:
                pickle.dump(self.layer_polygons, f)

        # check to see if layer graphs have already been created
        layer_graphs_file = base_dir + "layer_graphs.pckl" 
        if os.path.isfile(layer_graphs_file) and not force_generate_layer_graphs:
            with open(layer_graphs_file, 'rb') as f:
                self.layer_graphs = pickle.load(f)
        else:
            self.layer_graphs = self.generate_layer_graphs()
            with open(layer_graphs_file, 'wb') as f:
                pickle.dump(self.layer_graphs, f)
        
        # pull in the separate planner object to do the final path planning
        self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        # check to see if layer_paths have already been created
        layer_paths_file = base_dir + "layer_paths.pckl"
        if os.path.isfile(layer_paths_file) and not force_generate_layer_paths:
            with open(layer_paths_file, 'rb') as f:
                self.layer_paths = pickle.load(f)
        else:
            self.layer_paths = self.planner.generate_layer_paths()            
            with open(layer_paths_file, 'wb') as f:
                pickle.dump(self.layer_paths, f)

        # generate a gcode file
        self.planner.generate_gcode(self.layer_paths)
        
        # plot final paths
        self.plot_final_paths()        
        
    def load_part(self, path: str) -> None:
        """
        Preform all the necessary initializations when loading a model from a file
        """
        self.mesh = trimesh.load(path)
        self.mesh.rezero()
        
        # Center the model if printable_area is provided
        printable_area = self.params.get("printable_area")
        if printable_area and isinstance(printable_area, list) and len(printable_area) > 0:
            try:
                # Parse printable area to find center
                xs = []
                ys = []
                for point in printable_area:
                    parts = point.split('x')
                    if len(parts) == 2:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
                
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    center_x = (min_x + max_x) / 2.0
                    center_y = (min_y + max_y) / 2.0
                    
                    # Current mesh bounds after rezero
                    current_bounds = self.mesh.bounds
                    current_min = current_bounds[0]
                    current_max = current_bounds[1]
                    
                    current_center_x = (current_min[0] + current_max[0]) / 2.0
                    current_center_y = (current_min[1] + current_max[1]) / 2.0
                    
                    # Calculate translation
                    trans_x = center_x - current_center_x
                    trans_y = center_y - current_center_y
                    
                    print(f"Centering model: Target ({center_x}, {center_y}), Current ({current_center_x:.2f}, {current_center_y:.2f})")
                    
                    self.mesh.apply_translation([trans_x, trans_y, 0])
                    
                    new_bounds = self.mesh.bounds
                    new_center_x = (new_bounds[0][0] + new_bounds[1][0]) / 2.0
                    new_center_y = (new_bounds[0][1] + new_bounds[1][1]) / 2.0
                    print(f"Model centered to ({new_center_x:.2f}, {new_center_y:.2f})")
                    
            except Exception as e:
                print(f"Warning: Failed to center model: {e}")
        else:
            print("No printable_area provided, skipping centering.")

        self.bound = self.mesh.bounds
        self.x_range = self.bound[:,0]
        self.y_range = self.bound[:,1]
        self.z_range = self.bound[:,2]


    def create_raw_slices(self) -> np.ndarray:
        """
        Take model and parameters and slice model uniformly along the xy axis. 
        
        Returns:
            np.ndarray
        """

        # create a vector that contains the height of each layer
        layer_heights = np.arange(0, self.z_range[1], self.params["layer_height"])
        self.n_layers = np.size(layer_heights)

        # create plane origin and normal vector to plane
        origin = np.zeros((3))
        normal = np.array([0,0,1])

        # create slices starting at the origin, in the direction of the normal with
        # layer height specified
        layer_edges, _, _ = trimesh.intersections.mesh_multiplane(self.mesh, origin, 
                                                                  normal, layer_heights)
        
        return layer_edges

    def slice_to_polly(self) -> List[List[shp.Polygon]]:
        """
        Convert the raw edge data into a set of polygons. 
        Handles holes by checking containment and creating Shapely Polygons with interiors.
        """
        slices = []
        for layer in range(self.n_layers):
            # reorder all the edge data, get back a list of distinct regions
            ordered_regions = self.reorder_edges(self.layer_edges[layer])
            
            # Convert rings to Shapely Polygons
            raw_polys = []
            for ring in ordered_regions:
                if len(ring) < 3: continue
                # Ensure ring is closed
                if not np.allclose(ring[0], ring[-1]):
                    ring = np.vstack([ring, ring[0]])
                raw_polys.append(shp.Polygon(ring))

            if not raw_polys:
                slices.append([])
                continue

            # Sort by area (largest first) to handle containment
            raw_polys.sort(key=lambda p: p.area, reverse=True)
            
            # Build hierarchy
            # parent_idx[i] = index of the immediate parent of poly i, or -1 if it's an outer shell
            n = len(raw_polys)
            parent_idx = [-1] * n
            
            # For each polygon, find the smallest polygon that contains it
            for i in range(n):
                # We want the smallest container, so we check from smallest (end of list) to largest
                # But since we are iterating i, we only care about j < i (larger polygons)
                # Actually, if we sort by area descending:
                # 0 is largest.
                # If 1 is inside 0.
                # If 2 is inside 1 (and thus inside 0).
                # We want 2's parent to be 1.
                
                best_parent = -1
                best_parent_area = float('inf')
                
                for j in range(i): # Check all larger polygons
                    if raw_polys[j].contains(raw_polys[i]):
                        # We want the *smallest* valid parent (immediate parent)
                        if raw_polys[j].area < best_parent_area:
                            best_parent = j
                            best_parent_area = raw_polys[j].area
                
                parent_idx[i] = best_parent

            # Construct final polygons with holes
            # finalized_polys map: index in raw_polys -> Shapely Polygon object (potentially with holes)
            # But wait, we need to handle nesting levels. 
            # Even levels (0, 2, 4...) are solids. Odd levels (1, 3...) are holes.
            # Actually, standard rule: Even nesting depth = Solid, Odd = Hole.
            # Root (no parent) is depth 0 (Solid).
            # Child of Root is depth 1 (Hole).
            # Child of Hole is depth 2 (Solid Island).
            
            depths = [0] * n
            for i in range(n):
                if parent_idx[i] != -1:
                    depths[i] = depths[parent_idx[i]] + 1
            
            # We will collect all "Solid" polygons (depth 0, 2, 4...)
            # And assign their immediate children (depth 1, 3, 5...) as holes.
            
            final_polys = []
            
            # Map raw index to the final polygon object (if it's a solid)
            # This is tricky because we construct new Polygon objects.
            
            # Let's group holes by their parent
            holes_by_parent = defaultdict(list)
            for i in range(n):
                p_id = parent_idx[i]
                if p_id != -1:
                    holes_by_parent[p_id].append(raw_polys[i].exterior.coords)
            
            for i in range(n):
                # If depth is even, it's a solid
                if depths[i] % 2 == 0:
                    shell = raw_polys[i].exterior.coords
                    holes = holes_by_parent[i] # These are the direct children (holes)
                    
                    # Create polygon with holes
                    # Note: holes in shapely are list of coordinates
                    poly = shp.Polygon(shell, holes)
                    final_polys.append(poly)
            
            slices.append(final_polys)
            
        return slices

    def reorder_edges(self, coordinates: np.ndarray) -> List[np.ndarray]:
        """
        Iterate through all sets of edges and reorder them. Also split into separate rings if needed
        """

        used_points = []
        
        # first round all the coordinates to 4 digits long. There was a bug in which two points 
        # which should have been the same ones had a very slight different position value due to
        # rounding error. So fix this by rounding everything to 4. also convert numpy array to a 
        # list
        coordinates = np.around(coordinates,4).tolist()

        # create a blank list to layer store ordered regions
        ordered_regions = []
        
        # we are going to iterate over all the coordinates until we have added everything to a 
        # ordered region
        while coordinates:

            # get current edge
            current_edge = coordinates.pop(0)

            # disregard the edge if it has already been used, break out of loop we probably have
            # completed the loop or we have extra edges
            if current_edge[0] in used_points or current_edge[1] in used_points:
                break
                       
            # start a new re-ordered region using the current edge
            reordered_region = [current_edge[0],current_edge[1]]

            while True:

                # see if we can find a point that follows the current reordered region
                next_point_found = False

                for i, edge in enumerate(coordinates):
                    # if we find an edge where the first value is the same as the last value in our
                    # chain, add the second value to the ordered region
                    if edge[0] == reordered_region[-1]:
                        reordered_region.append(edge[1])
                        used_points.append(edge[1])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                    # if we find an edge where the second value is the same as the last value in the
                    # chain, add the first value to the ordered region
                    if edge[1] == reordered_region[-1]:
                        reordered_region.append(edge[0])
                        used_points.append(edge[0])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                
                # if we cannot find the next value, then the loop has been completed
                if not next_point_found:
                    break
            
            # by now, we have a fully defined ordered region, add it to the list of ordered regions
            # per layer
            ordered_regions.append(np.array(reordered_region))
            
        return ordered_regions


    def generate_layer_graphs(self):
        """
        Generates a graph for each layer, populated with the Wavy Lattice.
        Innovation: Generates robust, organic infill per layer.
        """
        self.layer_graphs = []
        
        # Iterate through each layer
        for layer_idx in range(self.n_layers):
            polygons = self.layer_polygons[layer_idx]
            graphs = []
            
            for polygon in polygons:
                # 1. Handle Polygon Buffer
                # Buffer negative to fit inside perimeter
                # Use slightly larger buffer than line_width to ensure clearance
                # But generate_spanning_tree uses -line_width.
                # Let's use -0.6 * line_width to allow some overlap for connection?
                # Actually, if we use -line_width, it matches the perimeter generation.
                offset = -0.6 * self.params["line_width"]
                
                if isinstance(polygon, shp.Polygon):
                    buffer_poly = polygon.buffer(offset)
                else:
                    poly_points = polygon.get_xy()
                    buffer_poly = shp.Polygon(poly_points).buffer(offset)
                
                if buffer_poly.is_empty:
                    continue
                    
                # 2. Handle MultiPolygon (Region splitting)
                regional_polygons = []
                if isinstance(buffer_poly, shp.MultiPolygon):
                    for geom in buffer_poly.geoms:
                        regional_polygons.append(geom)
                else:
                    regional_polygons.append(buffer_poly)
                
                # 3. Generate Graph for each region
                for poly in regional_polygons:
                    # Generate Wavy Lattice inside this polygon
                    graph = self.generate_wavy_lattice(poly)
                    
                    # Store the polygon in the graph for Planner
                    graph.graph["polygon"] = poly
                    
                    # Only add non-empty graphs? 
                    # If empty, Planner skips it? 
                    # If empty, we might want to print at least the perimeter?
                    # Planner.generate_spanning_tree handles empty graphs by printing perimeter.
                    graphs.append(graph)
            
            self.layer_graphs.append(graphs)
        
        return self.layer_graphs
    
    def plot_mesh(self) -> None:
        """
        Plot the model in a 3D pyplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_trisurf(self.mesh.vertices[:, 0],
                        self.mesh.vertices[:,1], 
                        triangles=self.mesh.faces, 
                        Z=self.mesh.vertices[:,2], 
                        alpha=1)
    
    def plot_lattice(self) -> None:
        """
        Plot the lattice
        """
        latt = self.lattice
        latt.plot()

    def plot_layer_edge(self, layer: int) -> None:
        """
        Plot a given layer_edge
        Params: 
            layer: int
        """
        layer_edge = self.layer_edges[layer]

        for idx in range(np.shape(layer_edge)[0]):
            plt.plot(*layer_edge[idx,:,:].T, "-k")

    def plot_layer_graph(self, layer: int) -> None:
        """
        Plot the networkx graph for a given layer
        """
        for G in self.layer_graphs[layer]:
            nx.draw(G, pos=posgen(G), node_size = 1, with_labels=True)

    def plot_final_paths(self) -> None:
        """
        Using the generated self.layer_paths, plot them all on a 3D matplotlib figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for layer_idx, layer_data in enumerate(self.layer_paths):
            for pts in layer_data:
                ax.plot(*pts.T, layer_idx*self.params["layer_height"])

# MISC FUNCTIONS

def dist_btw_graph_nodes(graph: nx.Graph, node_1: int, node_2: int) -> float:
    """
    Given a graph, calculate the distance between two nodes
    """
    x1 = graph.nodes[node_1]["x"]
    y1 = graph.nodes[node_1]["y"]
    x2 = graph.nodes[node_2]["x"]
    y2 = graph.nodes[node_2]["y"]

    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Return true if line segments AB and CD intersect
def intersect(A: List[List], B: List[List], C: List[List], D: List[List]):
    """
    Return true if line segments AB and CD intersect

    params:
        A,B,C,D: List[List] of coordinate points
    """
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def posgen(G: nx.Graph):
    """
    Returns a dictionary where for each key which is node in graph, we get the coordinate value of
    its position
    """
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
