"""
File for taking the graph and crating path and gcode
"""

import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
import math
from shapely.geometry import LineString, Polygon
from shapely import buffer
import shapely as shp
import sys

sys.setrecursionlimit(10000)

class TreeNode(object):
    """
    Tree style object for when creating a spanning tree over the graph. 
    """
    def __init__(self, pos: Tuple[float, float] = None, node: int = None, parent = None, children: List = None) -> None:
        """
        Initializes a single tree node

        Params:
            pos: Tuple[float,float], default None - position of the node in the tree
            node: int, default None - the networkx.Graph node number associated with this tree node
            parent: TreeNode, default None - the parent of the current node
            children : List[TreeNode], default [] - a list of children from the current node

        Returns: 
            None
        """
        self.node = node
        self.pos = pos
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children
    
    def __hash__(self) -> int:
        """
        Creating a hashing for the TreeNode object so that we can put TreeNode into a dictionary
        """
        return hash(self.pos[0]) + hash(self.pos[1])

    def __eq__(self, other):
         """
         Create a way to compare two TreeNode objects to see if they are the same. Also so we can
         put them in a dictionary
         """
         return (
             self.__class__ == other.__class__ and
             self.pos == other.pos and
             self.parent == other.parent
        )
 

class Planner:
    """
    A class that takes the general structure of each layer as defined by the layer polygons and the
    lattice graph and generates a real path and gcode
    """
    def __init__(self, params: Dict, layer_polygons: List[List[Polygon]], layer_graphs: List[List[nx.Graph]]) -> None:
        """
        Initialize planner

        Params:
            params: Dict - a dictionary that contains all the necessary slicing and planning 
                parameters
            layer_polygons: List[List[Polygon]] - a list of lists of polygons that are the bounding
                regions of each layer
            layer_graphs: List[List[nx.Graphs]] - a list of lists of graphs that are the internal
                lattice of each region on each layer
        """
        self.layer_graphs = layer_graphs
        self.params = params
        self.layer_polygons = layer_polygons
        self.n_layers = len(layer_polygons)
    
    def generate_layer_paths(self) -> List[List[np.ndarray]]:
        """
        For all layers, generate spanning tree, then generate the final path

        Returns:
            layer_paths: List[List[np.ndarray]] - a list of list of numpy arrays that contain the xy
                coordinates of a valid path to print a layer
        """

        # create a list to hold the pathing data for each layer
        layer_paths = []

        for layer_idx in range(self.n_layers):

            # create a list to hold the pathing data for each subregion in the layer
            region_paths = []
            
            # enumerate over each separate region in the layer
            for region_idx, graph in enumerate(self.layer_graphs[layer_idx]):
                # Retrieve polygon associated with this graph (from new Slicer logic)
                polygon = graph.graph.get("polygon")
                contour_paths = graph.graph.get("contour_paths")
                
                # Fallback for compatibility or safety
                if polygon is None:
                    try:
                        polygon = self.layer_polygons[layer_idx][region_idx]
                    except IndexError:
                        print(f"Warning: No polygon found for graph {region_idx} in layer {layer_idx}")
                        continue

                if contour_paths:
                    for arr in contour_paths:
                        region_paths.append(arr)
                    continue
                trees = self.generate_spanning_tree(polygon, graph)

                # if a tree can cover this region, generate a path
                if trees:
                    for root in trees:
                        # using the tree, traverse over the tree using DFS to create a "point path"
                        point_path = self.generate_path(root)
                    
                        # create a line string from this point path
                        line = LineString(point_path)
                        
                        # create an offset from the LineString to create the final path
                        # Use 0.6 * line_width to create slightly wider struts (overlap) to fill gaps
                        offset_path = np.array(line.buffer(.6*self.params["line_width"]).exterior.coords)

                        # save the final path into our regional_paths list
                        region_paths.append(offset_path)

            layer_paths.append(region_paths)

        
        # return the layer paths for all layers
        return layer_paths

    def generate_gcode(self, layer_paths: List[List[np.ndarray]]):
        """
        Generate a gcode file that a 3D printer could use to print the input geometry.
        
        params:
            layer_paths: List[List[np.ndarray]] - a list of list of numpy arrays that contain the xy
                coordinates of a valid path to print a layer
        """

        # keep a global counter of how much we have extruded and our current layer number
        extrusion = 0
        layer_num = 0 

        def generate_gcode_for_layer(layer_loops: List[np.ndarray], extrusion: float, layer_num: int, feed_rate=100, retract_distance=.08, lift_distance=2, perimeters: List[np.ndarray] = None) -> Tuple[List, float, int]:
            """
            Each layer is made up of several loops. For each individual loop on a layer, write gcode
            to print that loop, then retract filament, move up, travel to the next loop, and repeat
            """

            # create a list which will contain all the gcode statements
            gcode = []
            gcode.append(f";LAYER:{layer_num}")
            print_rate = int(self.params.get("print_feed_rate", feed_rate))
            travel_rate = int(self.params.get("travel_feed_rate", max(print_rate, 6000)))
            gcode.append(f"G0 F{travel_rate}")
            gcode.append(f"G1 F{print_rate}")

            # print perimeters first if provided
            if perimeters:
                for pts in perimeters:
                    if pts.shape[0] < 2:
                        continue
                    gcode.append(f"G1 F2100 E{extrusion + .08}")
                    for idx, point in enumerate(pts):
                        x, y, z = point
                        if idx == 0:
                            gcode.append(f"G0 X{x} Y{y}")
                            gcode.append(f"G0 Z{z}")
                        else:
                            # Use simple dist for perimeters (assuming flat)
                            distmm = math.sqrt((x - old_x)**2 + (y - old_y)**2)
                            ratio = self.params["layer_height"]*self.params["line_width"] / ((1.75/2)**2*3.14)
                            extrusion += distmm * ratio
                            gcode.append(f"G1 X{x} Y{y} E{extrusion}")
                        old_x = x
                        old_y = y
                    gcode.append(f"G1 F2100 E{extrusion - retract_distance}")

            # iterate through each of the loops
            for points in layer_loops:
                # Set feed rates
                gcode.append(f"G1 F2100 E{extrusion + .08}")
                gcode.append(f"G1 F{print_rate}")  # Set X, Y, and Z feed rate
                
                # Generate G-code for loop
                mode_local = self.params.get("z_follow_mode", "constant")
                start_ramp_mm = float(self.params.get("start_ramp_mm", 0.0))
                coast_mm = float(self.params.get("coast_mm", 0.0))
                wipe_mm = float(self.params.get("wipe_mm", 0.0))
                
                # Calculate total length for ramp/coast
                total_len = 0.0
                seglens = []
                for i in range(1, len(points)):
                    x0, y0, _ = points[i-1]
                    x1, y1, _ = points[i]
                    sl = math.sqrt((x1-x0)**2 + (y1-y0)**2)
                    seglens.append(sl)
                    total_len += sl
                
                printed = 0.0
                for idx, point in enumerate(points):
                    x, y, z = point

                    # go to start point and drop to level
                    if idx == 0:
                        gcode.append(f"G0 X{x} Y{y}")
                        gcode.append(f"G0 Z{z}")

                    else:
                        # calculate the length of filament needed to print a single mm length section
                        filament_area = 3.14159 * (1.75/2)**2
                        path_area = self.params["layer_height"] * self.params["line_width"]
                        flow_rate = float(self.params.get("flow_rate", 1.0))
                        ratio = (path_area / filament_area) * flow_rate

                        # calculate the filament needed to extrude from the current position to the next spot
                        distmm = math.sqrt((x - old_x)**2 + (y - old_y)**2)
                        
                        ramp_scale = 1.0
                        if start_ramp_mm > 0.0:
                            ramp_scale = min(1.0, (printed + distmm) / start_ramp_mm)
                        remain = max(0.0, total_len - printed)
                        
                        if coast_mm > 0.0 and remain <= coast_mm:
                            gcode.append(f"G1 X{x} Y{y} Z{z}") # Move without extrusion
                        else:
                            extrusion += distmm * ratio * ramp_scale
                            gcode.append(f"G1 X{x} Y{y} Z{z} E{extrusion}")
                        printed += distmm

                    old_x = x
                    old_y = y
                
                # retract filament a slight amount and move up Z 
                if wipe_mm > 0.0 and len(points) >= 2 and not (mode_local == "spiral"):
                    x2, y2, _ = points[-1]
                    x1, y1, _ = points[-2]
                    vx = x2 - x1
                    vy = y2 - y1
                    vlen = math.sqrt(vx*vx + vy*vy) or 1.0
                    wx = x2 - (vx / vlen) * wipe_mm
                    wy = y2 - (vy / vlen) * wipe_mm
                    gcode.append(f"G1 X{wx:.5f} Y{wy:.5f}")
                    
                if not (mode_local == "spiral"):
                    gcode.append(f"G1 F2100 E{extrusion - retract_distance}")
                    gcode.append(f"G0 F{travel_rate} Z{z+lift_distance*self.params['layer_height']}")


            layer_num = layer_num + 1
                    
            return gcode, extrusion, layer_num          

        def generate_gcode_for_layers(layer_data: List[List[np.ndarray]], extrusion: float, layer_num: int, feed_rate=100):
            # global extrusion
            gcode = []
            
            # Initialize
            gcode.append("M82") # set absolute extrusion mode
            gcode.append("G21")  # Set units to millimeters
            gcode.append("G90")  # Set to absolute positioning
            gcode.append("G92 E0")  # Zero the extruder
            gcode.append("M82") # set absolute extrusion mode

            # set temps
            gcode.append("M104 S220") # nozzle
            gcode.append("M140 S60") # bed
            gcode.append("M190 S60") # bed
            gcode.append("M109 S220") # nozzle
            
            # Home
            gcode.append("G28")  # Home all axes
            gcode.append("G92 E0.0")
            gcode.append(f"G1 F2100 E{extrusion - .08}")
            
            # Set feed rates
            print_rate = int(self.params.get("print_feed_rate", feed_rate))
            travel_rate = int(self.params.get("travel_feed_rate", max(print_rate, 6000)))
            gcode.append(f"G0 F{travel_rate}")  # Travel feed rate
            gcode.append(f"G1 F{print_rate}")  # Print feed rate
            
            # Generate G-code for each loop
            gcode.append(f";LAYER_COUNT:{self.n_layers - 1}")
            
            # build perimeters data
            perimeters_by_layer = []
            inner_factor = float(self.params.get("perimeter_inner_offset_factor", 0.0))
            for layer_idx in range(self.n_layers):
                base_z = layer_idx * self.params["layer_height"]
                perims = []
                for reg in self.layer_polygons[layer_idx]:
                    if isinstance(reg, shp.Polygon):
                        poly = reg
                    elif hasattr(reg, "get_xy"):
                        poly = shp.Polygon(reg.get_xy())
                    else:
                        continue
                    offset = -inner_factor * self.params["line_width"]
                    poly2 = poly.buffer(offset) if offset != 0.0 else poly
                    geoms = list(poly2.geoms) if isinstance(poly2, shp.MultiPolygon) else [poly2]
                    for g in geoms:
                        coords = np.array(g.exterior.coords)
                        if coords.shape[0] >= 2:
                            zcol = np.full((coords.shape[0], 1), base_z)
                            perims.append(np.hstack((coords, zcol)))
                perimeters_by_layer.append(perims)

            for layer_idx, layer_loops in enumerate(layer_data):
                perims = perimeters_by_layer[layer_idx] if layer_idx < len(perimeters_by_layer) else None
                new_code, extrusion, layer_num = generate_gcode_for_layer(layer_loops, extrusion, layer_num, perimeters=perims)
                gcode.extend(new_code)
                # pbar.update(1)
                
            
            # End of program
            gcode.append("M140 S0")
            gcode.append("M107")
            gcode.append("M104 S0")
            gcode.append("M140 S0")
            gcode.append("M107")
            gcode.append("M84")
            gcode.append("M82")
            gcode.append("M30") 
            
            return "\n".join(gcode)
        
        layer_data = []

        for layer_idx, layer in enumerate(layer_paths):
            layer_loops = []
            base_z = layer_idx * self.params["layer_height"]
            mode = self.params.get("z_follow_mode", "constant")
            amp = float(self.params.get("z_follow_amp", 1.0))
            fn = self.params.get("z_nonlinear_fn", "smoothstep")
            
            for array in layer:
                n = array.shape[0]
                if n < 2:
                    continue
                
                # Adaptive: Randomize Start
                rnd = bool(self.params.get("randomize_start", False))
                if rnd and n > 2 and self.params.get("z_follow_mode", "constant") != "spiral":
                    stride = int(self.params.get("layer_offset_stride", 7))
                    offset = (layer_idx * stride) % n
                    array = np.vstack([array[offset:], array[:offset]])
                
                # Adaptive: Alternate Direction
                if bool(self.params.get("alternate_direction", True)) and (layer_idx % 2 == 1):
                    array = array[::-1]

                if mode == "spiral":
                    t = np.linspace(0, 1, n).reshape((-1, 1))
                    new_column = base_z + t * self.params["layer_height"]
                elif mode == "nonlinear":
                    t = np.linspace(0, 1, n).reshape((-1, 1))
                    if fn == "sin":
                        f = 0.5 * (1 - np.cos(np.pi * t))
                    elif fn == "quad":
                        f = t ** 2
                    else:
                        f = t * t * (3 - 2 * t)
                    new_column = base_z + f * (self.params["layer_height"] * amp)
                else:
                    new_column = np.full((n, 1), base_z)
                new_arr = np.hstack((array, new_column))
                layer_loops.append(new_arr)
            
            layer_data.append(layer_loops)
            

        # Generate G-code
        mode_global = self.params.get("z_follow_mode", "constant")
        if mode_global == "spiral" and bool(self.params.get("global_spiral", False)):
            sorted_layer_data = []
            current_pos = (0, 0)
            for layer_idx, layer_loops in enumerate(layer_data):
                if not layer_loops:
                    sorted_layer_data.append([])
                    continue
                sorted_loops = []
                remaining_loops = layer_loops[:]
                while remaining_loops:
                    best_idx = -1
                    best_dist = float("inf")
                    should_reverse = False
                    for i, loop in enumerate(remaining_loops):
                        d_start = (loop[0][0] - current_pos[0])**2 + (loop[0][1] - current_pos[1])**2
                        if d_start < best_dist:
                            best_dist = d_start
                            best_idx = i
                            should_reverse = False
                        d_end = (loop[-1][0] - current_pos[0])**2 + (loop[-1][1] - current_pos[1])**2
                        if d_end < best_dist:
                            best_dist = d_end
                            best_idx = i
                            should_reverse = True
                    loop = remaining_loops.pop(best_idx)
                    if should_reverse:
                        loop = loop[::-1]
                    sorted_loops.append(loop)
                    current_pos = loop[-1][:2]
                sorted_layer_data.append(sorted_loops)

            # --- Generation: Global Spiral G-code ---
            gcode = []
            
            # Debug prints for path verification
            total_path_len = 0.0
            z_min = float('inf')
            z_max = float('-inf')
            for layer in sorted_layer_data:
                for loop in layer:
                    if len(loop) == 0: continue
                    for i in range(1, len(loop)):
                        d = math.sqrt((loop[i][0]-loop[i-1][0])**2 + (loop[i][1]-loop[i-1][1])**2)
                        total_path_len += d
                    for p in loop:
                        if len(p) > 2:
                            z_min = min(z_min, p[2])
                            z_max = max(z_max, p[2])

            pass

            if "start_gcode" in self.params:
                gcode.append(self.params["start_gcode"])
            else:
                gcode.append("M82") # Absolute Extrusion
                gcode.append("G21") # Metric
                gcode.append("G90") # Absolute Positioning
                gcode.append("M140 S60") # Set Bed Temp
                gcode.append("M104 S220") # Set Nozzle Temp
                gcode.append("M190 S60") # Wait for Bed
                gcode.append("M109 S220") # Wait for Nozzle
                gcode.append("G28") # Home
                
                # Standard Prime Line (Safe & Effective)
                z_speed = self.params.get("max_z_speed", 600)
                
                safe_y_start = 20.0
                safe_y_end = 150.0 
                safe_x = 2.0 
                
                gcode.append("G92 E0")
                gcode.append(f"G1 Z2.0 F{z_speed}") # Move Z up safely
                gcode.append(f"G1 X{safe_x} Y{safe_y_start} Z0.0 F5000.0") # Move to start position
                gcode.append(f"G1 X{safe_x} Y{safe_y_end} Z0.0 F1500.0 E15") # Draw the first line
                gcode.append(f"G1 X{safe_x+0.3} Y{safe_y_end} Z0.0 F5000.0") # Move to side
                gcode.append(f"G1 X{safe_x+0.3} Y{safe_y_start} Z0.0 F1500.0 E30") # Draw the second line (Accumulated E=30)
                gcode.append("G92 E0") # Reset Extruder
                gcode.append(f"G1 Z2.0 F{z_speed}") # Move Z up safely
                gcode.append("G1 F2400 E-0.5") # Small retraction 
            
            # Ensure Absolute Extrusion Mode
            gcode.append("M82") 
            gcode.append("G92 E0")
            
            print_rate = int(self.params.get("print_feed_rate", 1800))
            travel_rate = int(self.params.get("travel_feed_rate", 6000))
            flow_rate = float(self.params.get("flow_rate", 1.0))
            
            # --- Generate Per-Layer Continuous Path ---
            
            # Calculate E-ratio
            filament_area = 3.14159 * (1.75/2)**2
            path_area = self.params["line_width"] * self.params["layer_height"]
            ratio = (path_area / filament_area) * flow_rate

            max_height = (self.n_layers - 1) * self.params["layer_height"]
            
            # Start G-code
            total_extrusion = 0.0
            global_dist = 0.0
            last_pos = (0.0, 0.0)
            
            if sorted_layer_data and sorted_layer_data[0]:
                x0, y0, _ = sorted_layer_data[0][0][0]
                # Start at Z=0 for spiral mode to ensure continuous extrusion from bed
                gcode.append(f"G0 X{x0:.5f} Y{y0:.5f} Z0.0 F{travel_rate}")
                gcode.append("G1 F2400 E0.0") # Unretract (back to 0 from -0.5)
                last_pos = (x0, y0)
            
            # Z-Wave Parameters
            z_wave_amp = 0.05 
            z_wave_freq = 0.2 

            for layer_idx, layer_loops in enumerate(sorted_layer_data):
                # Layer Control
                if layer_idx == 0:
                    current_speed = int(print_rate * 0.4) 
                    gcode.append("M107") 
                elif layer_idx == 1:
                    current_speed = print_rate
                    gcode.append("M106 S255") 
                else:
                    current_speed = print_rate

                # Calculate total distance for this layer to determine Z-slope (Spiralize)
                layer_total_dist = 0.0
                temp_last_pos = last_pos 
                
                # Pre-calculate layer length and optimize start points
                for loop_idx in range(len(layer_loops)):
                    loop = layer_loops[loop_idx]
                    if len(loop) == 0: continue
                    
                    # Optimization: Nearest Start
                    best_start_idx = 0
                    min_dist = float('inf')
                    for i, p in enumerate(loop):
                        d = (p[0] - temp_last_pos[0])**2 + (p[1] - temp_last_pos[1])**2
                        if d < min_dist:
                            min_dist = d
                            best_start_idx = i
                    
                    if best_start_idx != 0:
                        if np.allclose(loop[0], loop[-1]):
                             unique_points = loop[:-1]
                             rotated_unique = np.vstack([unique_points[best_start_idx:], unique_points[:best_start_idx]])
                             loop = np.vstack([rotated_unique, rotated_unique[0:1]])
                        else:
                             loop = np.vstack([loop[best_start_idx:], loop[:best_start_idx]])
                        layer_loops[loop_idx] = loop

                    temp_last_pos = (loop[-1][0], loop[-1][1])
                    
                    # Dist calc
                    sx, sy, _ = loop[0]
                    dist = math.sqrt((sx - temp_last_pos[0])**2 + (sy - temp_last_pos[1])**2)
                    layer_total_dist += dist
                    temp_last_pos = (sx, sy)
                    
                    for i in range(1, len(loop)):
                        p1 = loop[i-1]
                        p2 = loop[i]
                        seg = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        layer_total_dist += seg
                        temp_last_pos = (p2[0], p2[1])

                if layer_total_dist <= 0:
                    layer_total_dist = 1.0 

                current_layer_dist = 0.0
                
                # Fade-in amplitude
                amp_factor = 1.0
                if layer_idx < 3:
                    amp_factor = layer_idx / 3.0

                for loop_idx, loop in enumerate(layer_loops):
                    if len(loop) == 0: continue
                    
                    # 1. Travel/Connect to start of loop (Gap Handling)
                    sx, sy, _ = loop[0]
                    dist = math.sqrt((sx - last_pos[0])**2 + (sy - last_pos[1])**2)
                    
                    if dist > 0.0001: 
                        current_layer_dist += dist
                        global_dist += dist
                        
                        # Calculate Z Base (Spiral)
                        # Use continuous spiral formula for all layers including layer 0
                        progress = current_layer_dist / layer_total_dist
                        z_base = (layer_idx * self.params["layer_height"]) + (progress * self.params["layer_height"])
                        
                        # Apply Z-Wave Modulation (Distance Based + Smooth)
                        z_mod = 0.0
                        if layer_idx > 0:
                            # Use global_dist for continuous wave across layers
                            z_mod = (z_wave_amp * amp_factor) * (1 - math.cos(global_dist * z_wave_freq))
                        
                        z = z_base + z_mod

                        # Gap Handling - Bridge vs Travel
                        # User Request: "Global spiral connection: gap large -> slow extrusion bridge"
                        # Threshold: 2mm?
                        if dist < 2.0 * self.params["line_width"]:
                             # Tiny gap: Just bridge with normal flow (One Stroke)
                             e_inc = dist * ratio
                             total_extrusion += e_inc
                             gcode.append(f"G1 X{sx:.5f} Y{sy:.5f} Z{z:.5f} E{total_extrusion:.5f} F{current_speed}")
                        else:
                             # Large gap: Slow extrusion bridge to maintain continuity
                             bridge_speed = int(current_speed * 0.5)
                             # Reduced flow for bridging to avoid heavy stringing but keep connection
                             bridge_ratio = ratio * 0.5 
                             e_inc = dist * bridge_ratio
                             total_extrusion += e_inc
                             gcode.append(f"G1 X{sx:.5f} Y{sy:.5f} Z{z:.5f} E{total_extrusion:.5f} F{bridge_speed} ; Bridge")
                             # Note: No retraction, purely additive to avoid "jumping"

                    last_pos = (sx, sy)

                    # 2. Print Loop
                    for i in range(1, len(loop)):
                        p = loop[i]
                        x, y = p[0], p[1]
                        seg_dist = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                        
                        if seg_dist > 0:
                            current_layer_dist += seg_dist
                            global_dist += seg_dist
                            
                            # For layer 0, we also want to spiral up from 0 to layer_height
                            progress = current_layer_dist / layer_total_dist
                            z_base = (layer_idx * self.params["layer_height"]) + (progress * self.params["layer_height"])

                            # Z-Modulation
                            z_mod = 0.0
                            if layer_idx > 0:
                                z_mod = (z_wave_amp * amp_factor) * (1 - math.cos(global_dist * z_wave_freq))
                            
                            z = z_base + z_mod
                            
                            # Calculate E amount
                            layer_ratio = ratio
                            if layer_idx == 0:
                                layer_ratio = ratio * 1.2 # 120% flow for first layer
                            
                            e_amount = seg_dist * layer_ratio
                            total_extrusion += e_amount
                            
                            gcode.append(f"G1 X{x:.5f} Y{y:.5f} Z{z:.5f} E{total_extrusion:.5f} F{current_speed}")
                            last_pos = (x, y)

            if "end_gcode" in self.params:
                gcode.append(self.params["end_gcode"])
            else:
                gcode.append("M140 S0")
                gcode.append("M107")
                gcode.append("M104 S0")
                gcode.append("M84")
                gcode.append("M30")
            
            gcode_content = "\n".join(gcode)
            
            # Cleanup PrusaSlicer conditionals
            if "{if max_layer_z < max_print_height}" in gcode_content:
                    safe_z = max_height + 10.0
                    import re
                    gcode_content = re.sub(r"\{if max_layer_z.*?\}\s*Z.*?\{endif\}", f"Z{safe_z:.2f}", gcode_content)

        else:
            gcode_content = generate_gcode_for_layers(layer_data, extrusion, layer_num)

        # Save G-code to a file
        with open("output.gcode", "w") as file:
            file.write(gcode_content)



    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph) -> List[TreeNode]:
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside

        Params:
            polygon: Polygon - a bounding polygon for a region in a layer
            graph: nx.Graph - a lattice graph for a region in a layer

        Returns:
            roots: Lit[TreeNode] - a list of tree root nodes that fully cover a whole layer 
        """


        """ ---------------------------------------------
        Step 1) Create a buffered polygon exterior
        -------------------------------------------------"""


        # get buffer polygon that is 1x width smaller than original polygon, this is when we print
        # perimeter of our part, our part remains dimensionally accurate
        
        # Handle input polygon type
        if isinstance(polygon, shp.Polygon):
             # Use shapely buffer directly on the polygon to handle holes correctly if present
             buffer_poly = polygon.buffer(-self.params["line_width"])
        elif hasattr(polygon, "get_xy"):
             poly_points = polygon.get_xy()
             buffer_poly = shp.Polygon(poly_points).buffer(-self.params["line_width"])
        else:
             raise ValueError(f"Unknown polygon type: {type(polygon)}")

        # now we have to deal with the possible edge case that if we buffer the whole exterior by a 
        # line width, it is possible that our single polygons splits into multiple polygons, so now
        # we create a list of polygons to deal with, if our polygon is still only one region, our
        # list will only have one item in it.
        
        poly_points_array = []
        if isinstance(buffer_poly, shp.MultiPolygon):
            for geom in buffer_poly.geoms:
                poly_points_array.append(np.array(geom.exterior.coords))
        else:
            poly_points_array.append(np.array(buffer_poly.exterior.coords))

        # if any of the regions are null, return empty list
        # NOTE: I think this might be legacy code and isn't needed anymore
        for poly_points in poly_points_array:
            if not np.any(poly_points):
                return []
            

  
        """ ---------------------------------------------
        Step 2) Connect the lattice to the outside buffered polygon
        -------------------------------------------------"""

                    
        # get list of connected graphs
        connected_components = list(nx.connected_components(graph))

        # for each subgraph, find an end piece
        end_nodes = [None] * len(connected_components)

        # for each end node, add an edge that connects them to the outside
        new_end_nodes = [None] * len(connected_components)
        insert_idxs = [(-1,-1) for _ in connected_components]

        def find_connection(region_idx: int, node: int):
            """
            Given a node in a graph, find the best way to connect it to the outside bounding polygon
            """

            prev_node = next(graph.neighbors(node))
            
            # get line that extends outwards
            line_1 = [(graph.nodes[prev_node]["x"],graph.nodes[prev_node]["y"]), 
                      (graph.nodes[node]["x"],graph.nodes[node]["y"])]
            
            best_dist = math.inf
            best_point = (None,None)
            best_idx = None
            idx_total = 0

            # iterate through all edges in all polygon regions
            for poly_points in poly_points_array:
                for idx in range(len(poly_points) - 1): 

                    # get the edge of polygon as line 2   
                    line_2 = [tuple(poly_points[idx]), tuple(poly_points[idx+1])]

                    # find best intersection point between line 1 and line 2
                    point = intersection_point(line_1, line_2)
                    x,y = point

                    # compare this connection to other connections already found
                    dist = math.sqrt((line_1[1][0] - x)**2 + (line_1[1][1] - y)**2)
                    if dist < best_dist:
                        best_point = point
                        best_dist = dist

                        # if this is the best we have found so far, save the idexes
                        # NOTE: I don't really recall what the following logic is for, but it has to
                        # do with some edge cases handling a split regions
                        if idx_total + idx+1 == len(poly_points) - 1:
                            best_idx = (idx+idx_total, idx_total)
                        else:
                            best_idx = (idx+idx_total, idx_total + idx+1)
                idx_total += idx

            # if we really cannot find a way to connect, return None
            if best_point[0] is None:
                return (None, None), None

            return best_point, best_idx
            
        # for each section of the graph, find the connection point and add it to the new end nodes list
        for region_idx, sub_graph in enumerate(connected_components):
            for node in sub_graph:
                if graph.degree(node) == 1:
                    connection_pt, connection_idx = find_connection(region_idx, node)
                    
                    # if the connection is note (None, None), save point and break
                    if not connection_pt[0] is None:
                        end_nodes[region_idx] = node
                        new_end_nodes[region_idx] = connection_pt
                        insert_idxs[region_idx] = connection_idx
                        break

        # get the highest node number in the graph or zero if there is no graph/lattice
        if connected_components:
            graph_n_nodes = max(graph.nodes)
        else:
            graph_n_nodes = 0


        # iterate through all the points in all the possible polygon regions and add their points
        # to the list new_points
        idx = 0
        new_points = []
        for poly_points in poly_points_array:            
            for p_idx in range(len(poly_points)-1):
                x1 = poly_points[p_idx][0]
                y1 = poly_points[p_idx][1]
                new_points.append((idx + graph_n_nodes + 1,{'x':x1, 'y':y1}))
                idx += 1

        # also add in the new end nodes that connect the internal lattice to the outside
        for new_end_node_idx, new_end_node in enumerate(new_end_nodes):
            if new_end_node is None:
                continue
            x1, y1 = new_end_node
            new_points.append((idx + graph_n_nodes + new_end_node_idx + 1, {"x":x1, "y":y1}))
            
        # add all the new points to the graph
        graph.add_nodes_from(new_points)
        
        # create a list to keep track of the node numbers for all exterior nodes in the graph
        outside_pts = []

        # create a list that will store tupes of node idx that will be edges in the graph
        new_edges = []

        vert_idx = 0
        sub_offset = 0
        # iterate through  the possible polygon regions
        for poly_points in poly_points_array:
            regional_outside_pts = []
            # iterate through all but the last two points (because the last one is a duplicate of the first)
            for edge_idx in range(len(poly_points) - 2):
                # calculate the node number the points that form the edge
                node_1_number = vert_idx+graph_n_nodes+1+sub_offset
                node_2_number = vert_idx+graph_n_nodes+2+sub_offset

                # add the node numbers to the regional outside pts list
                regional_outside_pts.append(node_1_number)
                regional_outside_pts.append(node_2_number)  

                # if the current edge is not where we are connecting the internal lattice to, just
                # add the edge to the new_edges list         
                if (vert_idx , vert_idx+1) not in insert_idxs and (vert_idx + 1, vert_idx) not in insert_idxs:
                    new_edges.append((node_1_number, node_2_number))

                # if this is where we are connecting the lattice to the exterior, we need to do a
                # of work to get everything sorted
                else:
                    sub_graph_idx = insert_idxs.index((vert_idx, vert_idx+1))
                    # calculate the node number for the new connection node
                    node_3_number = graph_n_nodes + idx + sub_graph_idx + 1     
                    new_edges.append((node_1_number, node_3_number))
                    new_edges.append((node_2_number, node_3_number))
                    new_edges.append((end_nodes[sub_graph_idx], node_3_number))
                    regional_outside_pts.append(node_3_number)
                vert_idx +=1
            
            # again do the same process for the very last edge that wasn't covered by our enumeration

            # again re-calculate our node numbers
            node_1_number = vert_idx+graph_n_nodes+1+sub_offset
            node_2_number = vert_idx+graph_n_nodes+3+sub_offset - len(poly_points)
            # calculate to see if we are performing a connection on the last edge or not
            last_edge = ( vert_idx, vert_idx + 2 - len(poly_points))
            if last_edge not in insert_idxs and (last_edge[1],last_edge[0]) not in insert_idxs:
                # if not, just add the edge as normal
                new_edges.append((node_1_number, node_2_number))
            else:
                # if we are performing a connection, do so
                sub_graph_idx = insert_idxs.index(last_edge)
                node_3_number = graph_n_nodes + idx + sub_graph_idx + 1
                new_edges.append((node_1_number, node_3_number))
                new_edges.append((node_3_number, node_2_number))
                new_edges.append((node_3_number, end_nodes[sub_graph_idx]))

            sub_offset += 1
            outside_pts.append(regional_outside_pts)

        # add all the new edges
        graph.add_edges_from(new_edges)


        """ ---------------------------------------------
        Step 3) Create a small gap on the outside of the polygon
        -------------------------------------------------"""

        # we are creating a small gap on the outside of the polygon to create an "opening" that will
        # serve as the entry point for the tree. We are going to first try looking at the spots 
        # were we connected the lattice to the outside as a starting point
       
        # creating a list of points we've already looked over incase we try to delete the same point
        # twice. NOTE: might be unnecessary, could maybe delete
        used_pts = []

        # gets a valid starting point to start deleting edges. Needs to be on outside and connect to
        # inside
        def get_valid_start_pt(outside_pts):
            for idx in outside_pts:
                if graph.degree(idx) == 3:
                    return idx
            return idx
        
        potential_start_idxs = []

        potential_start_idx = graph_n_nodes
        # iterate through each region in the layer
        for region_idx, poly_points in enumerate(poly_points_array):
            # get initial starting point
            potential_start_idx = get_valid_start_pt(outside_pts[region_idx])

            # get the next node on the outside of the graph
            del_node = next(graph.neighbors(potential_start_idx))
            while del_node not in outside_pts[region_idx]:
                del_node = next(graph.neighbors(potential_start_idx))
            
            cur_node = potential_start_idx
            used_pts.append(cur_node)

            # keep track of the original starting point
            x_1 = graph.nodes[potential_start_idx]["x"]
            y_1 = graph.nodes[potential_start_idx]["y"]
            
            # iteratively delete edges until we reach the break condition
            while True:

                # get the location of teh current end point
                x_2 = graph.nodes[del_node]["x"]
                y_2 = graph.nodes[del_node]["y"]
                

                # Delete edges
                graph.remove_edges_from([(del_node, cur_node)])

                # if we ever create a gap that is greater than 1.1x the line_width, stop, and fix the 
                # gap so that it is exactly 1.1x the line width gap
                dist = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                if dist > 1.1*self.params["line_width"]:
                    
                    # calculate the position of a new point to add to the exterior edge
                    vec_a = np.array([x_1,y_1]) - np.array([x_2, y_2])
                    # Fix: Reduce gap size to 1.1 * line_width to ensure better connection
                    vec_a = vec_a/np.linalg.norm(vec_a) * (dist - 1.1*self.params["line_width"])
                    new_point = (x_2 + vec_a[0], y_2 + vec_a[1])
                    
                    # add the new node and edge to the graph
                    graph.add_nodes_from([(max(graph.nodes) + 1, {"x": new_point[0], "y":new_point[1]})])
                    graph.add_edges_from([(del_node, max(graph.nodes))])
                    
                    break

                # if we did not reach the break case, keep moving along the edge
                cur_node = del_node

                # break case, we have run out of things to delete
                if not list(graph.neighbors(cur_node)):
                    break

                # find the next del node. Make sure it's an exterior point and not already used
                del_node = next(graph.neighbors(cur_node))
                while del_node not in outside_pts[region_idx] or del_node in used_pts:
                    if del_node in used_pts and graph.degree(cur_node) == 1:
                        break
                    del_node = next(graph.neighbors(cur_node))


            # save our start index into a list. NOTE: the name "potential_start_idxs" is a slight 
            # misnomer since I don't think it's potential anymore, its the actual start idx. But I 
            # don't want to change that now.
            potential_start_idxs.append(potential_start_idx)

            # clean up any remaining junk in the graph
            graph.remove_nodes_from(list(nx.isolates(graph)))


        """ ---------------------------------------------
        Step 4) DFS through the graph to make a tree
        -------------------------------------------------"""

        # The way we are doing DFS through the graph to make a tree is to first DFS to create a list
        # of edges. The list of edges just tells us how things are connected, but does not create a 
        # tree yet. Then after we create this list of edges, we iterate through the edges and form
        # the tree. It's an unintuitive 2 step process that can probably be simplified into a single
        # step. But that's a later problem.

        # work through lattice and create a spanning tree vis DPS
        roots = []
        for start_idx in potential_start_idxs:
            if start_idx in list(graph.nodes):
                visited = set()  # Set to keep track of visited nodes
                tree_edges = []  # List to store edges of the spanning tree

                def dfs(node, parent):
                    # mark the current not as having been visited
                    visited.add(node)

                    # iterate through each connected node
                    for neighbor in graph.neighbors(node):
                        if neighbor != parent:

                            # found a node that hasn't been visited, so continue down path
                            if neighbor not in visited:
                                tree_edges.append((node, neighbor))
                                dfs(neighbor, node)

                dfs(start_idx, -1)
                
                # Step 5) Convert edges to TreeNode structure
                # We need to reconstruct the tree from edges starting from root
                
                # Map node index to TreeNode
                node_map = {}
                
                # Create root
                root_node = TreeNode(
                    pos=(graph.nodes[start_idx]["x"], graph.nodes[start_idx]["y"]), 
                    node=start_idx
                )
                node_map[start_idx] = root_node
                
                # Process edges to build the tree structure
                # Since tree_edges was built via DFS, we can iterate and link
                # But we need to make sure parents exist. 
                # A queue based approach ensures we process parents before children if we traverse.
                # However, tree_edges list order from DFS is: (root, child1), (child1, grandchild1)...
                # So we can just iterate.
                
                # Let's use a queue to be robust
                queue = [start_idx]
                processed_edges = set()
                
                while queue:
                    u = queue.pop(0)
                    u_node = node_map[u]
                    
                    # Find all children of u in tree_edges
                    for edge in tree_edges:
                        if edge in processed_edges:
                            continue
                            
                        # Edge is (parent, child)
                        if edge[0] == u:
                            v = edge[1]
                            v_node = TreeNode(
                                pos=(graph.nodes[v]["x"], graph.nodes[v]["y"]),
                                node=v,
                                parent=u_node
                            )
                            u_node.children.append(v_node)
                            node_map[v] = v_node
                            queue.append(v)
                            processed_edges.add(edge)
                        elif edge[1] == u:
                             # Should not happen in (parent, child) list but handle undirected check
                             pass

                roots.append(root_node)
        
        return roots

    def generate_path(self, root: TreeNode) -> List[Tuple[float, float]]:
        """
        Traverse the tree to generate a path
        """
        path = []
        
        def traverse(node):
            if node.pos:
                path.append(node.pos)
            
            for child in node.children:
                traverse(child)
                # Return to parent to ensure continuity
                if node.pos:
                    path.append(node.pos)
        
        traverse(root)
        return path

            


def intersection_point(line1, line2):
    """
    Calculate the intersection point of two lines, or at least the closest point
    """

    p1 = np.array(line2[0])
    p2 = np.array(line2[1])
    p3 = np.array(line1[1])

    vec_1 = p2 - p1
    vec_2 = p3 - p1

    proj_dist =( vec_2 @ vec_1 ) / (vec_1 @ vec_1)

    proj_dist = max(0,proj_dist)

    proj_dist = min(proj_dist, np.linalg.norm(vec_1))

    point = proj_dist*vec_1 + p1

    return point[0], point[1]



