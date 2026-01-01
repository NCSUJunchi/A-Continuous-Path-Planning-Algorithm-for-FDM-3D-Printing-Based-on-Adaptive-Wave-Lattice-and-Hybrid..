from slicer import Slicer
from matplotlib import pyplot as plt
from pathlib import Path
import json
import os
import ssl
from urllib import request, error


if __name__ == "__main__":
    # Load printer configuration
    printer_config_path = Path(__file__).resolve().parent.parent / "printer.json"
    printer_config = {}
    
    # Define default config (fallback)
    default_config = {
        "machine_max_acceleration_extruding": ["1000"],
        "machine_max_acceleration_retracting": ["1000"],
        "machine_max_speed_z": ["10"],
        "machine_start_gcode": "M104 S[first_layer_temperature]\nM140 S[first_layer_bed_temperature]\nG28\nM109 S[first_layer_temperature]\nM190 S[first_layer_bed_temperature]",
        "machine_end_gcode": "M104 S0\nM140 S0\nG28 X0 Y0\nM84",
        "printable_area": []
    }
    
    print(f"Loading printer config from: {printer_config_path}")
    if printer_config_path.exists():
        try:
            with open(printer_config_path, "r", encoding="utf-8") as f:
                printer_config = json.load(f)
            print("Printer config loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load printer.json: {e}")
            printer_config = default_config
    else:
        print("Warning: printer.json not found at expected path.")
        # Try absolute path fallback
        alt_path = Path("d:/path planning/printer.json")
        if alt_path.exists():
             print(f"Found config at alternate path: {alt_path}")
             try:
                with open(alt_path, "r", encoding="utf-8") as f:
                    printer_config = json.load(f)
             except Exception as e:
                print(f"Warning: Failed to load alt printer.json: {e}")
                printer_config = default_config
        else:
             print("Using default printer configuration.")
             printer_config = default_config

    path = str((Path(__file__).resolve().parent / "models" / "bunny.stl"))

    # Process Printer Config (G-code substitutions)
    start_gcode = printer_config.get("machine_start_gcode", "")
    end_gcode = printer_config.get("machine_end_gcode", "")
    
    # Simple substitutions for placeholders
    # Note: In a full slicer, these would be dynamic. Here we use standard defaults for PLA.
    subs = {
        "[machine_max_acceleration_extruding]": printer_config.get("machine_max_acceleration_extruding", ["1000"])[0],
        "[machine_max_acceleration_retracting]": printer_config.get("machine_max_acceleration_retracting", ["1000"])[0],
        "[first_layer_temperature]": "220",
        "[first_layer_bed_temperature]": "60",
        "[input_shaping_gcode]": "",
        "[motion_sequence]": "",
        "[layer_z]": "0.2", # Initial layer
    }
    
    for key, val in subs.items():
        start_gcode = start_gcode.replace(key, str(val))
        end_gcode = end_gcode.replace(key, str(val))

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "wavy", # Changed to wavy (organic sinusoidal lattice)
        "infill_size": 1.5, # Increased to 1.5mm to visualize the wavy pattern
        "line_width": 0.4,
        "flow_rate": 1.0,   # Added flow rate control
        "planner_mode": "beam", # Core Algorithm: Topology-Aware Adaptive Beam Search
        "z_follow_mode": "spiral",
        "z_nonlinear_fn": "smoothstep",
        "z_follow_amp": 1.0,
        "randomize_start": True,
        "start_ramp_mm": 2.0,
        "coast_mm": 0.6,
        "wipe_mm": 1.5,
        "global_spiral": True,
        "seam_preference": "max_y",
        "perimeter_inner_offset_factor": 0.0,
        "print_feed_rate": 1800, # Reduced from 3600 (60mm/s) to 30mm/s for better adhesion/quality
        "travel_feed_rate": 6000,
        # "start_gcode": start_gcode, # Disable printer.json start gcode as it may be unsafe (X205 > 200)
        "end_gcode": end_gcode,
        # --- Beam Search Hyperparameters ---
        "angle_weight": 1.5,        # Penalize sharp turns
        "dist_weight": 0.05,        # Penalize long travel
        "intersect_penalty": 2000.0, # Hard constraint against self-intersection
        "angle_power": 2.0,
        "beam_width": 5,            # Base beam width
        "lookahead_depth": 2,       # Base lookahead depth
        # --- Innovation: Adaptive Mechanism ---
        "adaptive_beam": True,      # Enable adaptive beam width/depth based on local density
        "adaptive_near_radius": 1.6,
        # --- Innovation: Topology Awareness ---
        "accessibility_weight": 2.0, # Reward nodes with higher future connectivity (Degree Centrality)
        "printable_area": printer_config.get("printable_area", []),
        "max_z_speed": float(printer_config.get("machine_max_speed_z", ["10"])[0]) * 60, # Convert mm/s to mm/min
    }
    # Allow overriding planner mode via environment variable
    env_mode = os.environ.get("PLANNER_MODE")
    if env_mode in ("dfs", "beam"):
        params["planner_mode"] = env_mode
    print(f"Planner mode: {params['planner_mode']}")

    slicer = Slicer(params)

    try:
        slicer.slice(path, debug_mode=False)
    except Exception as e:
        print(f"Error during slicing/planning: {e}")
        import traceback
        traceback.print_exc()

    # plt.show()
