"""
Converts a finished kRRT tree (px) to a JSON file (m/rad/y-up as well as positioning the root 0,0) to import into ROS
"""

import json
import math

def export_tree(nodes, root_index=0, scale_px_per_m=100, file_name="tree.json"):
    root = nodes[root_index]
    adjusted_tree = []

    for i, n in enumerate(nodes):
        # Save the entry of each section, y needs to be negative because pygame has negative y, and -(-y) = +y
        entry = {"id": i, "parent": n.parent, "x": (n.position[0] - root.position[0]) / scale_px_per_m, "y": -(n.position[1] - root.position[1]) / scale_px_per_m, "yaw": math.radians(n.orientation), "cost": n.cost / scale_px_per_m }

        if n.control_input is not None:
            v, w, t = n.control_input
            entry["control"] = {"v": v / scale_px_per_m, "w": w, "t": t}
        else:
            entry["control"] = None # Root does not hold a control input
        adjusted_tree.append(entry)

    # Create the json file for ROS2
    with open(file_name,"w") as file:
        json.dump({"scale_px_per_m": scale_px_per_m, "nodes": adjusted_tree}, file, indent=1) 
    
    print(f"[EXPORT] Wrote {len(adjusted_tree)} nodes to {file_name}")