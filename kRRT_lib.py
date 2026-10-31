"""
Holds the functions for environment set-up, tree growth and reversal evaluation.
An importable version of kRRT to be run quickly for extensive sampling.
"""

import math
import os
import random
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
MAP_DIMENSIONS = [420,420]
import environment as environment_module
import robot as robot_module
from node import Node
from RRT import Graph
ROBOT_IMAGES = ["images/initial.png", "images/goal.png", "images/robot.png"]
import csv
import argparse 
from tqdm import tqdm 


# make_environment: Makes the environment
def make_environment(x_init, has_obstacles=False):
    pygame.init()
    env = environment_module.Environment(dimensions=MAP_DIMENSIONS, show_tree_building=False, has_obstacles=has_obstacles)
    if has_obstacles:
        env.make_obstacles()
    obstacles = [side for obstacle in env.obstacles for side in obstacle] if has_obstacles else []
    robot = robot_module.Robot(start=x_init, robot_img=ROBOT_IMAGES, length=0.01)
    return env, robot, obstacles

# grow_tree: Grows one tree, returns (graph, nodes, metrics)
def grow_tree(sample_budget, seed, scene, x_init=(210, 210, 0), x_goal=(340, 300, 0), delta=0.05, position_boundary=14, 
              orientation_boundary=math.pi, v_magnitude=10.0, k_value=0.5, w_mode="kwt", anti_drive_bias=0.0, 
              anti_drive_mode="reverse", rewire=True, rewire_heading_boundary=180.0, rewire_search_radius=25.0, rewire_heading_weight=0.0, 
              shot_theta_weight=0.0, rewire_shot_exit_radius=5.0,):
    env, robot, obstacles = scene
    random.seed(seed)

    graph = Graph(start=x_init, goal=x_goal, map_dimensions=MAP_DIMENSIONS)
    nodes = [Node(position=x_init[:2], orientation=math.degrees(x_init[2]), parent=0, cost=0.0)]
    graph.position_boundary = position_boundary
    graph.orientation_boundary = orientation_boundary
    graph.show_tree_building = False
    graph.delta = delta
    graph.anti_drive_bias = anti_drive_bias
    graph.anti_drive_mode = anti_drive_mode
    graph.number_of_nodes = len(nodes)
    graph.shot_theta_weight = shot_theta_weight
    graph.verbose = False
    graph.v_magnitude = v_magnitude
    graph.k_value = k_value
    graph.w_mode = w_mode
    graph.rewire_shot_exit_radius = rewire_shot_exit_radius
    graph.shoot_rng = random.Random(seed)

    propagations = 0
    while propagations + graph.total_rewire_shots < sample_budget:

        # Estimated Voronoi bias searching
        x_rand = random.uniform(0, MAP_DIMENSIONS[0])
        y_rand = random.uniform(0, MAP_DIMENSIONS[1])

        best_d, parent_idx = None, None
        for i, node in enumerate(nodes):
            d = (node.position[0] - x_rand) ** 2 + (node.position[1] - y_rand) ** 2 # To compare squared distances
            if best_d is None or d < best_d:
                best_d, parent_idx = d, i # Find the best distance and thus, best parent index

        result = graph.new_state(parent_idx, robot, env, nodes, obstacles)
        propagations += 1

        if result is not None:
            x_new, theta_new, trajectory_samples = result
            new_node = Node(position=x_new, orientation=theta_new, parent=parent_idx, 
                            cost=nodes[parent_idx].cost + graph.edge_time_duration * abs(graph.v), 
                            control_input=(graph.v, graph.w, graph.edge_time_duration), trajectory_samples=trajectory_samples,)
            nodes.append(new_node)
            graph.children_count.append(0)
            graph.children_count[parent_idx] += 1
            graph.number_of_nodes = len(nodes)

            # Rewire the node
            if rewire:
                graph.rewire_node(nodes=nodes, rewire_search_radius=rewire_search_radius, delta=delta, heading_boundary=rewire_heading_boundary, 
                                  robot=robot, rewire_heading_weight=rewire_heading_weight,)

    graph.nodes = nodes
    rewire_calls = graph.total_rewire_shots
    metrics = {"sample_budget": sample_budget, "nodes_grown": len(nodes) - 1, "sample_complexity": propagations + rewire_calls, "growth_samples": propagations, "rewire_samples": rewire_calls, "promotions": graph.promotions}
    return graph, nodes, metrics

# Bi-drectional growth
def grow_two_trees(sample_budget, seed, scene, x_init=(210, 210, 0), **growth_kwargs):
    # Grows two trees in opposite directions
    half_budget = sample_budget // 2
    east_pose = (x_init[0], x_init[1], x_init[2]) # East Tree (A)
    west_pose = (x_init[0], x_init[1], x_init[2] + math.pi) # West Tree (B)
    graphA, nodesA, metricsA = grow_tree(half_budget, seed, scene, x_init=east_pose, **growth_kwargs)
    graphB, nodesB, metricsB = grow_tree(half_budget, seed+22, scene, x_init=west_pose, **growth_kwargs) # Different seed so it is not a mirror copy

    # Offset = amount the nodes need to be pushed around
    offset = len(nodesA)

    # For each Node B
    for j, node in enumerate(nodesB):
        # Move the n.parent in the list
        node.parent = 0 if j == 0 else node.parent + offset # Push all the nodeB's by the offset

    # Merge the two trees into one tree
    nodes = nodesA + nodesB
    graphA.nodes = nodes
    graphA.rewire_errors = graphA.rewire_errors + graphB.rewire_errors
    metrics = {"sample_budget": sample_budget}
    for key in ("nodes_grown", "sample_complexity", "growth_samples", "rewire_samples", "promotions"):
        metrics[key] = metricsA[key] + metricsB[key] # The sample count of both trees count towards one total budget
    return graphA, nodes, metrics

# evaluate_reversal: Runs a batch of reversibility trials against an already grown tree
def evaluate_reversal(graph, nodes, scene, seed, position_boundary=14, reverse_heading_boundary=30.0, reverse_search_radius=25.0, 
                      reverse_heading_weight=3.0, reverse_extension_multiplier=5.0, reverse_gear_change_penalty=50.0, reverse_use_samples=False, 
                      use_visited=True, reversibility_trials=20, reverse_spawn_mode="tree_box", max_reverse_steps=150, per_trial_timeout_s=3.0, 
                      fixed_spawns=None, show_progress=False,):
    env, robot, obstacles = scene
    random.seed(seed)
    reverse_selection_mode = "cost" # Hard-coded, other options kinda suck

    # Other attributes living in graph
    graph.heading_weight = reverse_heading_weight
    graph.reverse_extension_multiplier = reverse_extension_multiplier
    graph.gear_change_penalty = reverse_gear_change_penalty
    graph.use_samples = reverse_use_samples
    graph.max_steps = max_reverse_steps
    graph.position_weight = 0.0 # Unused parameter

    successes = 0
    hops, driven_costs, pos_drifts, all_hops = [], [], [], []
    failure_reasons = {}
    trial_records = []

    if fixed_spawns is not None:
        spawn_iter = list(fixed_spawns)
    else:
        spawn_iter = None

    n_trials = len(fixed_spawns) if fixed_spawns is not None else reversibility_trials

    for trial_i in tqdm(range(n_trials), desc=f"Reversing rvhb={reverse_heading_boundary}, rvrad={reverse_search_radius}", disable=not show_progress):
        if spawn_iter is not None:
            sx, sy, sth = spawn_iter[trial_i]
        else:
            while True:
                if reverse_spawn_mode == "tree_box":
                    xs = [n.position[0] for n in nodes]
                    ys = [n.position[1] for n in nodes]
                    sx, sy = random.uniform(min(xs), max(xs)), random.uniform(min(ys), max(ys))
                else: # It's 'map'
                    sx, sy = random.uniform(0, MAP_DIMENSIONS[0]), random.uniform(0, MAP_DIMENSIONS[1])
                sth = random.uniform(0, 360) # Not dependent on reverse spawn mode
                if abs(sx - nodes[0].position[0]) > position_boundary or abs(sy - nodes[0].position[1]) > position_boundary:
                    break # Inside goal

        result = graph.drive_backwards(nodes=nodes, robot=robot, start_x=sx, start_y=sy, start_theta_deg=sth, reverse_search_radius=reverse_search_radius, 
                                       reverse_heading_boundary=reverse_heading_boundary, position_threshold=position_boundary, 
                                       selection_mode=reverse_selection_mode, use_visited=use_visited, use_samples=reverse_use_samples, 
                                       max_steps=max_reverse_steps, max_seconds=per_trial_timeout_s)

        success, fx, fy, fth, n_hops, cost, _avg_t, reason, _pose_path, _node_path = result
        all_hops.append(n_hops)
        trial_records.append({"trial": trial_i, "success": success, "reason": reason, "n_hops": n_hops, "path_cost": cost, "final_x": fx, "final_y": fy, "final_theta_deg": fth}) 
        if success:
            successes += 1
            hops.append(n_hops)
            driven_costs.append(cost)
            pos_drifts.append(math.hypot(fx - nodes[0].position[0], fy - nodes[0].position[1]))
        else:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "success_rate": successes / n_trials if n_trials else None,
        "avg_hops_on_success": sum(hops) / len(hops) if hops else None, 
        "avg_hops_all_trials": sum(all_hops) / len(all_hops) if all_hops else None,
        "avg_driven_cost": sum(driven_costs) / len(driven_costs) if driven_costs else None, # cost of driving back to center
        "avg_position_drift": sum(pos_drifts) / len(pos_drifts) if pos_drifts else None,
        "failure_reasons": failure_reasons,
        "trials": trial_records,
    }


def run_lhs_point(
    folder,
    sample_budget,
    runs=3,
    trials=30,
    master_folder="MASTER_LHS",
    position_boundary=14,
    v_magnitude=10.0,
    k_value=0.5,
    w_mode="kwt",
    anti_drive_bias=0.0,
    anti_drive_mode="reverse",
    rewire=True,
    rewire_heading_boundary=180.0,
    rewire_search_radius=25.0,
    rewire_heading_weight=0.0,
    shot_theta_weight=0.0,
    rewire_shot_exit_radius=5.0,
    reverse_heading_boundary=30.0,
    reverse_search_radius=25.0,
    reverse_heading_weight=3.0,
    reverse_extension_multiplier=5.0,
    reverse_gear_change_penalty=50.0,
    reverse_spawn_mode="tree_box",
    max_reverse_steps=150,
    per_trial_timeout_s=3.0,
    seed_offset=0,
    two_trees=False):
    out_dir = os.path.join(master_folder, folder)
    os.makedirs(out_dir, exist_ok=True)
    scene = make_environment(x_init=(210, 210, 0), has_obstacles=False)

    rvhw, rvhb, rvrad = reverse_heading_weight, reverse_heading_boundary, reverse_search_radius
    rwhw, rwhb, rwrad = rewire_heading_weight, rewire_heading_boundary, rewire_search_radius
    adb, pb, cwe = anti_drive_bias, position_boundary, 0.0 # cwe isn't used anymore but is kept so that the csv names match
    file_name = os.path.join(out_dir, f"Modecost__RVHW{rvhw}_RVHB{rvhb}_RVRad{rvrad}_RWHW{rwhw}_RWHBd{rwhb}_RWRads{rwrad}_CWE{cwe}_ADB{adb}_k{k_value}_PB{pb}__Vmag{v_magnitude}_MaxSteps{max_reverse_steps}_RVExt{reverse_extension_multiplier}.csv")
    
    for s in range(seed_offset, seed_offset + runs):
        grow = grow_two_trees if two_trees else grow_tree
        graph, nodes, growth = grow(
            sample_budget=sample_budget, seed=s + 10, scene=scene, position_boundary=pb,
            v_magnitude=v_magnitude, k_value=k_value, w_mode=w_mode, anti_drive_bias=adb, anti_drive_mode=anti_drive_mode,
            rewire=rewire, rewire_heading_boundary=rwhb, rewire_search_radius=rwrad, rewire_heading_weight=rwhw,
            shot_theta_weight=shot_theta_weight, rewire_shot_exit_radius=rewire_shot_exit_radius)
        rev = evaluate_reversal(
            graph, nodes, scene, seed=s, position_boundary=pb,
            reverse_heading_boundary=rvhb, reverse_search_radius=rvrad, reverse_heading_weight=rvhw,
            reverse_extension_multiplier=reverse_extension_multiplier, reverse_gear_change_penalty=reverse_gear_change_penalty,
            reversibility_trials=trials, reverse_spawn_mode=reverse_spawn_mode, max_reverse_steps=max_reverse_steps,
            per_trial_timeout_s=per_trial_timeout_s, show_progress=True)

        successes = 0
        failure_reasons = {}
        hops_for_reached_root_trials = []
        total_path_costs = []
        position_drifts = []
        heading_drifts = []
        failure_position_drifts = []
        spawn_to_root_costs = []
        

        for t in rev["trials"]:
            trial, success, reason, total_path_cost = t["trial"], t["success"], t["reason"], t["path_cost"]
            if t["final_x"] is not None:
                position_drift = math.hypot(t["final_x"] - nodes[0].position[0], t["final_y"] - nodes[0].position[1])
                heading_drift = graph.angle_difference(t["final_theta_deg"], nodes[0].orientation)
            else:
                position_drift = heading_drift = None

            if success:
                    successes += 1
                    hops_for_reached_root_trials.append(t["n_hops"])
                    total_path_costs.append(total_path_cost)
                    position_drifts.append(position_drift)
                    heading_drifts.append(heading_drift)
            else:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                if position_drift is not None:
                    failure_position_drifts.append(position_drift)

            # Cost log the same as kRRT py
            cost_file = file_name.replace('.csv', '_costs.csv')
            cost_exists = os.path.isfile(cost_file)
            with open(cost_file, 'a', newline='') as cf:
                cw = csv.writer(cf)
                if not cost_exists:
                    cw.writerow(["run_number","trial","spawn_node","spawn_to_root_cost", "driven_path_cost","cost_ratio","success","reason","tree_seed","spawn_seed"])
                this_spawn_cost = None # tree_box / map spawns: no spawn node, so no spawn cost or ratio
                ratio = None
                cw.writerow([s+1, trial, None, this_spawn_cost, round(total_path_cost,2) if total_path_cost is not None else None, ratio, success, reason, s+10, s])

        # Record all the mean/max/avg stuff, exact same as kRRT
        avg_failure_position_drift = sum(failure_position_drifts)/len(failure_position_drifts) if failure_position_drifts else None
        avg_heading_drift = sum(heading_drifts)/len(heading_drifts) if heading_drifts else None
        avg_position_drift = sum(position_drifts)/len(position_drifts) if position_drifts else None
        n_trials = len(rev["trials"])
        success_rate = successes/n_trials if n_trials > 0 else None
        avg_hops_on_success = sum(hops_for_reached_root_trials)/len(hops_for_reached_root_trials) if hops_for_reached_root_trials else None
        avg_total_path_cost = sum(total_path_costs)/len(total_path_costs) if total_path_costs else None
        mean_rewire_error = sum(graph.rewire_errors)/len(graph.rewire_errors) if graph.rewire_errors else 0.0
        max_rewire_error = max(graph.rewire_errors) if graph.rewire_errors else 0.0
        avg_spawn_to_root_cost = sum(spawn_to_root_costs)/len(spawn_to_root_costs) if spawn_to_root_costs else None

        stuck_no_candidate_count = failure_reasons.get('stuck_no_candidate', 0)
        max_steps_exceeded_count = failure_reasons.get('max_steps_exceeded', 0)
        max_substeps_exceeded_count = failure_reasons.get('max_substeps_exceeded', 0)

        run_exists = os.path.isfile(file_name)
        with open(file_name, 'a', newline='') as rf:
            rw = csv.writer(rf)
            if not run_exists:
                rw.writerow(["run_number", "reverse_theta_weight", "rewire_heading_weight", "rewire_heading_boundary", "rewire_search_radius", "reverse_heading_boundary", "reverse_search_radius", "success_rate", "avg_position_drift", "avg_heading_drift", "avg_hops_on_success", "robot_avg_path_cost", "baseline_avg_path_cost", "mean_rewire_error", "max_rewire_error", "avg_failure_position_drift", "stuck_no_candidate_count", "max_steps_exceeded_count", "max_substeps_exceeded_count", "nodes_grown", "sample_complexity", "growth_samples", "rewire_samples", "promotions"])
            rw.writerow([s+1, rvhw, rwhw, rwhb, rwrad, rvhb, rvrad, success_rate, avg_position_drift, avg_heading_drift, avg_hops_on_success, avg_total_path_cost, avg_spawn_to_root_cost, mean_rewire_error, max_rewire_error, avg_failure_position_drift, stuck_no_candidate_count, max_steps_exceeded_count, max_substeps_exceeded_count, growth['nodes_grown'], growth['sample_complexity'], growth['growth_samples'], growth['rewire_samples'], growth['promotions']])
            print(f"[RUN {s+1}/{runs} DONE] success_rate={success_rate:.2f}, nodes={growth['nodes_grown']}, sample_complexity={growth['sample_complexity']}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one LHS point with kRRT_lib and save kinodynamic_RRT.py's per-trial and per-run csv", allow_abbrev=False)
    p.add_argument('-fd', '--folder', required=True, help='Point folder name, e.g. lhs_point_007')
    p.add_argument('--budget', type=int, default=130000, help='Max simulated interactions per tree (never exceeded)')
    p.add_argument('--master', default="MASTER_LHS", help='Folder that holds all the point folders')
    p.add_argument('-rpn', '--repeat_program_num', type=int, default=3, help='Runs (trees) per point')
    p.add_argument('-rvt', '--reversibility_trials', type=int, default=20, help='Reversal trials per tree')
    # Tree growth (same as kinodynamic_RRT.py)
    p.add_argument('-rw', '--rewire', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('-rwhb', '--rewire_heading_boundary', type=float, default=180)
    p.add_argument('-rwrad', '--rewire_search_radius', type=float, default=25)
    p.add_argument('-rwhw', '--rewire_heading_weight', type=float, default=0.0)
    p.add_argument('-stw', '--shot_theta_weight', type=float, default=0.0)
    p.add_argument('-rwexit', '--rewire_shot_exit_radius', type=float, default=5.0)
    p.add_argument('-k', '--k_value', type=float, default=0.5)
    p.add_argument('-wmode', '--w_mode', type=str, default='kwt')
    p.add_argument('-adb', '--anti_drive_bias', type=float, default=0)
    p.add_argument('-adm', '--anti_drive_mode', type=str, default='reverse')
    p.add_argument('-vmag', '--v_magnitude', type=float, default=10)
    p.add_argument('-pb', '--position_boundary', type=int, default=14)
    # Reversal (same names as kinodynamic_RRT.py)
    p.add_argument('-rvhb', '--reverse_heading_boundary', type=float, default=30)
    p.add_argument('-rvrad', '--reverse_search_radius', type=float, default=25)
    p.add_argument('-rvhw', '--reverse_heading_weight', type=float, default=3.0)
    p.add_argument('-rvgcp', '--reverse_gear_change_penalty', type=float, default=50)
    p.add_argument('-rvext', '--reverse_extension_multiplier', type=float, default=6.0) # TODO: Change?
    p.add_argument('-ms', '--max_steps', type=int, default=500) # TODO: Change?
    p.add_argument('-rvspawn', '--reverse_spawn_mode', type=str, default='map', choices=['tree_box', 'map'])
    p.add_argument('--timeout', type=float, default=10, help='Wall-clock seconds per reversal trial') # TODO: Change? Just bc so many are getting evaluated
    p.add_argument('--seed_offset', type=int, default=0, help='Added to every seed; 0 = same seeds as the sweep')
    p.add_argument('--two_trees', action='store_true', help='Grow two trees from G (east + west) and half the budget each')

    # accepted but dont actually mean anything in krrt_lib, just so that previous kRRT args still work 
    p.add_argument('-rp', action='store_true') # Overridden by -rpn
    p.add_argument('-png', action='store_true') # Never saves img
    p.add_argument('-cl', action='store_true') # Always true
    p.add_argument('-srr') # Never saves img/mp4
    p.add_argument('-sd') # Doesnt accept seed
    p.add_argument('-rvsd') # Ditto

    args = p.parse_args()
    
    run_lhs_point(
       folder=args.folder, sample_budget=args.budget, runs=args.repeat_program_num, trials=args.reversibility_trials,
       master_folder=args.master, position_boundary=args.position_boundary, v_magnitude=args.v_magnitude,
       k_value=args.k_value, w_mode=args.w_mode, anti_drive_bias=args.anti_drive_bias, anti_drive_mode=args.anti_drive_mode,
       rewire=args.rewire, rewire_heading_boundary=args.rewire_heading_boundary, rewire_search_radius=args.rewire_search_radius,
       rewire_heading_weight=args.rewire_heading_weight, shot_theta_weight=args.shot_theta_weight,
       rewire_shot_exit_radius=args.rewire_shot_exit_radius, reverse_heading_boundary=args.reverse_heading_boundary,
       reverse_search_radius=args.reverse_search_radius, reverse_heading_weight=args.reverse_heading_weight,
       reverse_extension_multiplier=args.reverse_extension_multiplier, reverse_gear_change_penalty=args.reverse_gear_change_penalty,
       reverse_spawn_mode=args.reverse_spawn_mode, max_reverse_steps=args.max_steps, per_trial_timeout_s=args.timeout, seed_offset=args.seed_offset, two_trees=args.two_trees)