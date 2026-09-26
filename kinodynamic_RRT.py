import pygame
import environment
import robot
import RRT
import math
import random
import argparse
import sys
import csv
import os 
import itertools

from tqdm import tqdm
from export_tree import export_tree
from node import Node

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Tree building
parser = argparse.ArgumentParser(description='kRRT algorithm for a car-like robot.')
parser.add_argument('-o', '--obstacles', type=bool, action=argparse.BooleanOptionalAction,
	metavar='', required=False, help='Obstacles on the map')
parser.add_argument('-n', '--nodes', type=int, metavar='', required=False, default=10000,
	help='Max number of nodes')
parser.add_argument('-dt', '--delta', type=float, metavar='', required=False, default=0.05,
	help='Fixed steps between frames for the visualization')
parser.add_argument('-init', '--x_init', nargs='+', type=float, metavar='', required=False,
	default=(210, 210, 0), help='Initial node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-goal', '--x_goal', nargs='+', type=float, metavar='', required=False,
	default=(340, 300, 0), help='Goal node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-snc', '--show_new_configurations', type=bool,
 	action=argparse.BooleanOptionalAction, metavar='', required=False,
 	help='Show new config on screen')
parser.add_argument('-ptg', '--path_to_goal', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=False, help='Draws the milestones from path to goal')
parser.add_argument('-si', '--show_interpolation', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=False, help='Draws the configurations needed to reach the \
	 	goal')
parser.add_argument('-mr', '--move_robot', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=False,
	help='Shows the movements of the robot from start to end')
parser.add_argument('-pb', '--position_boundary', type=int, metavar='', required=False, default=14,
	help='Allowed position region of the pixels for the robot to reach the goal')
parser.add_argument('-ob', '--orientation_boundary', type=float, metavar='', required=False,
 	default=math.pi, help='Allowed orientation region of the angle for the robot to reach the goal')
parser.add_argument('-stb', '--show_tree_building', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=False,
	help='Shows the simulation forward in time')
parser.add_argument('-sf', '--show_final', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=False,
	help='Shows the final simulation')
parser.add_argument('-sd', '--seed', type=int, metavar='', required=False, default=None,
	help='Seed of the random numbers taken.')
parser.add_argument('-vmag', '--v_magnitude', type=float, metavar='', required=False, default=10,
	help='Constant forward speed magnitude.')
parser.add_argument('-cwe', '--children_weight_exponent', type=float, metavar='', required=False, default=0.0,
	help='Exponent on children weighting when sampling a parent to extend (Exponent on the inverse of the number of childrem, 0 = NO bias)')
parser.add_argument('-cwel', '--children_weight_exponent_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of cwe values to sweep automatically: e.g. -cwel 1 5 10 15')
parser.add_argument('-s3d', '--show_3d', action=argparse.BooleanOptionalAction, default=False,
    help='Show the matplotlib 3D tree at the end of tree generation.')
parser.add_argument('-k', '--k_value', type=float, metavar='', required=False, default=0.5,
	help='The k value in w = k/t.')

# Anti-drive
parser.add_argument('-adb', '--anti_drive_bias', type=float, metavar='', required=False, default=0,
	help='Percent change (0-100) an edge uses zero/reverse velocity instead of forward driving.')
parser.add_argument('-adm', '--anti_drive_mode', type=str, choices=['stop', 'reverse', 'both'], 
    metavar='', required=False, default='reverse', help='Which behavior triggers when anti_bias engages, v=0(stop), v=-v(reverse), or both (randomly picked).')

# Logging / Data collection
parser.add_argument('-rp', '--repeat_program', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Repeats the program multiple times.')
parser.add_argument('-rpn', '--repeat_program_num', type=int, metavar='', required=False, default=3, help='Number of times to repeat the program.')
parser.add_argument('-gep', '--goal_end_program', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Stop growing the tree as soon as the goal is reached.')
parser.add_argument('-png', '--save_png', type=str, nargs='?', const='final_scene.png', metavar='', required=False, default=None, help='Save a 2D and 3D png of the tree on EVERY run.') 
parser.add_argument('-cl', '--cost_log', action=argparse.BooleanOptionalAction, default=False, help='Write a per-trial cost-comparison CSV alongside the summary file.')
parser.add_argument('-fd', '--folder', type=str, metavar='', required=False, default=None,
    help='Folder name to place every image saved during this run (created if it does not exist), e.g. --folder heading_bound_sweep')
parser.add_argument('-srr', '--save_reversibility_record', type=str, choices=['off', 'all', 'failures', 'successes'], metavar='', required=False, default='off', help='Save a PNG and MP4 of each reversal trial.')
# Rewire configurations
parser.add_argument('-rw', '--rewire', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=False,
	help='Rewires the tree nodes.')
parser.add_argument('-rwhb', '--rewire_heading_boundary', type=float, metavar='', required=False,
	default=180, help='The boundary of the heading difference which is used in rewiring.')
parser.add_argument('-rwhbl', '--rewire_heading_boundary_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of heading_boundary values to sweep automatically: e.g. -rwhbl 15 20 25 30')
parser.add_argument('-rwrad', '--rewire_search_radius', type=float, metavar='', required=False,
	default=25, help='The search radius of the robot when deciding which node to rewire.')
parser.add_argument('-rwradl', '--rewire_search_radius_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of search_radius values to sweep automatically: e.g. -radl 1 5 10 15')
parser.add_argument('-stw', '--shot_theta_weight', type=float, metavar='', required=False, default=0.0,
    help='Weight applied to heading mismatch when picking the best shot, 0 = position-only (original behavior).')
parser.add_argument('-pbl', '--position_boundary_list', nargs='+', type=int, metavar='', required=False,
    default=None, help='List of position_boundary values to sweep: e.g. -pbl 10 20 30 50')
parser.add_argument('-rvhbl', '--reverse_heading_boundary_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of reverse_heading_boundary values to sweep: e.g. -rvhbl 30 60 90 180')
parser.add_argument('-rwhw', '--rewire_heading_weight', type=float, metavar='', required=False, default=0.0,
	help='Weight (px per degree) applied to heading mismatch in rewire candidate scoring. 0 = pure pass/fail filter (original behavior).')
parser.add_argument('-rwhwl', '--rewire_heading_weight_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of rewire_heading_weight values to sweep: e.g. -rwhwl 0 0.5 1.0 1.5 2.0')

# Reversibility configurations
parser.add_argument('-rvhb', '--reverse_heading_boundary', type=float, metavar='', required=False,
	default=30, help='The boundary of the heading difference which is used in reversing.')
parser.add_argument('-rvradl', '--reverse_search_radius_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of reverse_search_radius values to sweep: e.g. -rvradl 10 20 30 50')
parser.add_argument('-rvrad', '--reverse_search_radius', type=float, metavar='', required=False,
	default=25, help='The search radius of the robot when deciding which node to reverse.')
parser.add_argument('-rvt', '--reversibility_trials', type=int, metavar='', required=False, default=20,
	help='Number of reversibility trials to do.')
parser.add_argument('-rvsd', '--reversibility_seed', type=int, metavar='', required=False, default=None,
	help='Seed for selecting which nodes get tested in the reversibility.')
parser.add_argument('-rvspawn', '--reverse_spawn_mode', type=str, choices=['node', 'map', 'tree_box', 'file'], 
    metavar='', required=False, default='tree_box', help='Where the robot spawns at the start of each reversibility trial: node (existing tree node), map (anywhere on the map), tree_box (random point within a trees bounding box)')
parser.add_argument('-rvmode', '--reverse_selection_mode', type=str, choices=['root', 'robot', 'cost'],
    metavar='', required=False, default='cost', help='Candidate selection rule during reversal: cost (lowest cost-to-come), root (closest to root) or robot (closest to the robot itself).')
parser.add_argument('-uv', '--use_visited', action=argparse.BooleanOptionalAction, default=True, help='Exclude already-selected nodes from future searches during reversal.')
parser.add_argument('-ms', '--max_steps', type=int, metavar='', required=False,
	default=1000, help='The max steps taken in reversing.')
parser.add_argument('-srvt', '--show_reversal_trials', type=str, choices=['off', 'all', 'failures', 'successes'], metavar='', required=False, default='off', help='Draw the reversal trials after running.')
parser.add_argument('-rvhw', '--reverse_heading_weight', type=float, metavar='', required=False, default=3.0,
	help='Weight (px per degree) applied to heading mismatch in reversal candidate scoring. 0 = pure pass/fail filter (original behavior).')
parser.add_argument('-rvhwl', '--reverse_heading_weight_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of reverse_heading_weight values to sweep: e.g. -rvhwl 0 0.5 1.0 1.5 2.0')
parser.add_argument('-rvs', '--reverse_use_samples', action=argparse.BooleanOptionalAction, default=False,
    help='During reversal, search every trajectory sample along each edge instead of only node endpoints.')
parser.add_argument('-kl', '--k_value_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of k values to sweep: e.g. -kl 10 20 30 50')
parser.add_argument('-wmode', '--w_mode', type=str, choices=['random', 'kwt2', 'kwt'], metavar='', required=False, default='kwt',
	help='How angular velocity w is generated: random (uniform w in [-0.5, 0.5]) or kwt (w = turn_direction * k_value / edge_time_duration**2)')
parser.add_argument('-adbl', '--anti_drive_bias_list', nargs='+', type=float, metavar='', required=False,
    default=None, help='List of anti_drive_bias values to sweep: e.g. -adbl 0 10 25 50')
parser.add_argument('-rvspawnfile', '--reverse_spawn_file', type=str, metavar='', required=False, default=None,
    help='CSV from generate_shared_spawns.py, used when -rvspawn file is set.')
parser.add_argument('-rvext', '--reverse_extension_multiplier', type=float, metavar='', required=False, default=1000,
    help="Multiple of a held control's own t to keep driving it before giving up and picking fresh.")
parser.add_argument('-rvpw', '--reverse_position_weight', type=float, metavar='', required=False, default=0,
	help='Weight applied to distance (robot to candidate) in cost-mode reversal scoring 0 = distance has no effect (original behavior).')
parser.add_argument('-rvgcp', '--reverse_gear_change_penalty', type=float, metavar='', required=False, default=50,
   help='Extra cost penalty applied when a reversal candidate would flip driving direction relative to the currently-held control. 0 = no penalty.')
parser.add_argument('-rwsn', '--rewire_shot_count', type=int, metavar='', required=False, default=1000)
parser.add_argument('-rwexit', '--rewire_shot_exit_radius', type=float, metavar='', required=False, default=5.0,
    help='Early-exit landing radius (px) for rewire shooting. Shot loop stops once a shot lands within this AND improves cost.')

# Quality of life
parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False, help='Show per-step debug tqdm.writes. Turn off for progress bars during sweeps.')

args = parser.parse_args()

if args.folder is not None:
    os.makedirs(args.folder, exist_ok=True)

def in_output_folder(file_name):
    return os.path.join(args.folder, file_name) if args.folder is not None else file_name

if args.save_png:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    args.show_tree_building = False

# Initialization
pygame.init()

#assert not (args.save_png is not None and args.show_reversal_trials != 'off'), "--save_png is not compatible with --show_reversal_trials"
# Constants
ROBOT_INIT_PATH = 'images/initial.png'
ROBOT_GOAL_PATH = 'images/goal.png'
ROBOT_IMG_PATH = 'images/robot.png'
MAP_DIMENSIONS = 420,420 # used to be 680, 520, and used to be 640,480. Had to be changed for QL
robot_images = [ROBOT_INIT_PATH, ROBOT_GOAL_PATH, ROBOT_IMG_PATH]

# Initial and final robot configs (px, px, rad)
x_init = args.x_init
x_goal = args.x_goal

# Creating environment/robot/RRT
environment = environment.Environment(dimensions=MAP_DIMENSIONS, show_tree_building=args.show_tree_building, has_obstacles=args.obstacles)
robot = robot.Robot(start=args.x_init, robot_img=robot_images, length=0.01)
def main():
    rewire_heading_boundaries_to_test = args.rewire_heading_boundary_list if args.rewire_heading_boundary_list is not None else [args.rewire_heading_boundary]
    rewire_search_radii_to_test = args.rewire_search_radius_list if args.rewire_search_radius_list is not None else [args.rewire_search_radius]
    reverse_search_radii_to_test = args.reverse_search_radius_list if args.reverse_search_radius_list is not None else [args.reverse_search_radius]
    position_boundaries_to_test = args.position_boundary_list if args.position_boundary_list is not None else [args.position_boundary]
    reverse_heading_boundary_to_test = args.reverse_heading_boundary_list if args.reverse_heading_boundary_list is not None else [args.reverse_heading_boundary]
    reverse_heading_weights_to_test = args.reverse_heading_weight_list if args.reverse_heading_weight_list is not None else [args.reverse_heading_weight]
    k_values_to_test = args.k_value_list if args.k_value_list is not None else [args.k_value]
    cwe_values_to_test = args.children_weight_exponent_list if args.children_weight_exponent_list is not None else [args.children_weight_exponent]
    rewire_heading_weights_to_test = args.rewire_heading_weight_list if args.rewire_heading_weight_list is not None else [args.rewire_heading_weight]
    adb_values_to_test = args.anti_drive_bias_list if args.anti_drive_bias_list is not None else [args.anti_drive_bias]

    combos = list(itertools.product(
        rewire_heading_boundaries_to_test, rewire_search_radii_to_test, reverse_search_radii_to_test,
        position_boundaries_to_test, reverse_heading_boundary_to_test, reverse_heading_weights_to_test, 
        k_values_to_test, cwe_values_to_test, rewire_heading_weights_to_test, adb_values_to_test
    ))

    repeats = args.repeat_program_num if args.repeat_program else 1
    progress_bar_overall = tqdm(total=len(combos)*repeats, desc="Overall sweep", position=0)

    for rwhb, rwrad, rvrad, pb, rvhb, rvhw, k_value, cwe, rwhw, adb in combos:
        s = 0
        while True:
            graph = RRT.Graph(start=x_init, goal=x_goal, map_dimensions=MAP_DIMENSIONS)
            if args.seed is not None:
                if args.repeat_program is True:
                    random.seed(s+10)
                else:
                    random.seed(args.seed)
            run = True
            clock = pygame.time.Clock()

            nodes = [Node(position=x_init[:2], orientation=math.degrees(x_init[2]), parent=0, cost=0.0)]

            iteration = 0
            k = 0 
            # All the graph declarations (It's alot)
            graph.position_boundary = pb
            graph.orientation_boundary = args.orientation_boundary
            graph.show_tree_building = args.show_tree_building
            graph.delta = args.delta 
            graph.anti_drive_bias = adb
            graph.anti_drive_mode = args.anti_drive_mode
            graph.number_of_nodes = len(nodes)
            graph.shot_theta_weight = args.shot_theta_weight
            graph.verbose = args.verbose
            graph.v_magnitude = args.v_magnitude
            graph.max_steps = args.max_steps
            graph.use_samples = args.reverse_use_samples
            graph.heading_weight = rvhw
            graph.k_value = k_value
            graph.w_mode = args.w_mode
            graph.reverse_extension_multiplier = args.reverse_extension_multiplier 
            graph.position_weight = args.reverse_position_weight
            graph.gear_change_penalty = args.reverse_gear_change_penalty
            graph.rewire_shot_count = args.rewire_shot_count
            graph.rewire_shot_exit_radius = args.rewire_shot_exit_radius
            graph.shoot_rng = random.Random(args.seed + s) # Note: Added s so the seeds differ per trials but are still repeatable
            
            environment.make_obstacles()

            if not args.show_tree_building and args.verbose:
                tqdm.write('[EVENT] Simulating the system... Please wait.')

            progress_bar_tree = tqdm(total=args.nodes, desc=f"Growing rwhb={rwhb}, rwrad={rwrad}", position=1, leave=False)
            # Main growth loop
            while run and k < args.nodes:
                if args.show_tree_building:
                    clock.tick(environment.FPS)
            
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                # Draw all initial states
                if args.show_tree_building:
                    obstacles = environment.draw_obstacles() if args.obstacles else []
                    graph.draw_initial_node(map=environment.map)
                    graph.draw_goal_node(map=environment.map)
                    graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
                    graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
                else:
                    obstacles = [side for obstacle in environment.obstacles for side in obstacle] if args.obstacles else []
                if args.show_tree_building:
                    environment.draw_trajectory_trail()
                
                # If the goal is not found
                if not graph.is_goal_found or not args.goal_end_program:
                    # NOTE: OLD CWE HERE:
                    # weights = [1.0/((1+graph.children_count[i])**cwe) for i in range(len(nodes))] # = (1/children_count)*^exp for each node
                    # weighted_random_number = random.choices(range(len(nodes)), weights=weights, k=1)
                    # random_parent_index = weighted_random_number[0]
                    # NOTE: NEW OLD VORONOI ESTIMATION HERE:
                    x_rand = random.uniform(0, MAP_DIMENSIONS[0])
                    y_rand = random.uniform(0, MAP_DIMENSIONS[1])
                    old_closest_distance = None
                    random_parent_index = None
                    for i in range(len(nodes)):
                        node_x = nodes[i].position[0]
                        node_y = nodes[i].position[1]
                        squared_distance = (node_x - x_rand)**2 + (node_y - y_rand)**2
                        if old_closest_distance is None or squared_distance < old_closest_distance:
                            old_closest_distance = squared_distance
                            random_parent_index = i

                    # The result of propgation with random control function from a random node
                    result = graph.new_state(random_parent_index, robot, environment, nodes, obstacles)
                    
                    # If the path did not collide with something... 
                    if result is not None:
                        x_new, theta_new, trajectory_samples = result

                        # Create new node
                        new_node = Node(
                            position=x_new,
                            orientation=theta_new,
                            parent=random_parent_index,
                            cost=nodes[random_parent_index].cost + graph.edge_time_duration*abs(graph.v),
                            control_input=(graph.v, graph.w, graph.edge_time_duration),
                            trajectory_samples=trajectory_samples
                        )
                        nodes.append(new_node)
                        new_index = len(nodes) -1
                        graph.children_count.append(0) # Add a new node with 0 children
                        graph.children_count[random_parent_index] += 1 

                        assert new_node.cost >= nodes[new_node.parent].cost, f"Cost did not increase from parent ({nodes[new_node.parent].cost}) to child ({new_node.cost})"
                        
                        # Update sampled config (k)
                        graph.number_of_nodes = len(nodes)
                        k += 1
                        progress_bar_tree.update(1)
                        if args.verbose:
                            tqdm.write(f"[DEBUG] k={k}, len(tree)={len(nodes)}")

                        if args.rewire: 
                            graph.rewire_node(nodes=nodes, rewire_search_radius=rwrad, delta=args.delta, heading_boundary=rwhb, robot=robot, rewire_heading_weight=rwhw)
                    
                    if args.show_new_configurations and args.show_tree_building:
                        # Draws a small rotated robot image at every node position added to the tree 
                        graph.draw_new_robot_configuration(robot_img=robot.img, environment=environment)
                        # Draws a small brown circle at every node position
                        graph.draw_new_node(map=environment.map)

                    # if goal is found
                    if graph.is_goal_found and args.goal_end_program:
                        break # Stop growing 

                    iteration += 1
                    if args.show_tree_building:
                        pygame.display.update()
                        environment.map.fill(environment.WHITE)

            progress_bar_tree.close()
            none_count = sum(1 for n in nodes if n.control_input is None and n is not nodes[0]) 
            if args.verbose:
                tqdm.write(f"[EVENT] Loop ended. Goal found: {graph.is_goal_found}, Total nodes: {len(nodes)}, k={k}.")
                tqdm.write(f"[STATS] heading_boundary of {rwhb} blocked {graph.candidates_blocked_by_heading}/{graph.candidates_in_radius} candidates.")
                tqdm.write(f"[STATS] nodes with control_input=None (excluding root): {none_count}")

            export_tree(nodes, root_index=0, scale_px_per_m=100.0, file_name="tree.json")
            
            # Handling a shared spawn set between kRRT and Qlearning
            graph.nodes = nodes 
            shared_spawns = None
            if args.reverse_spawn_mode == 'file':
                with open(args.reverse_spawn_file, newline='') as _file:
                    shared_spawns = [(float(r["x"]), float(r["y"]), float(r["theta_deg"])) for r in csv.DictReader(_file)]
                assert len(shared_spawns) >= args.reversibility_trials, ("Shared spawn file has {len(shared_spawns)} poses but -rvt asked for {args.reversibility_trials} trials.")

            # --------- Reversibility Testing Start ---------

            # Handle seed reversibility
            if args.reversibility_seed is not None:
                    if args.repeat_program is True:
                        random.seed(s if args.repeat_program else args.reversibility_seed)

            successes = 0
            failure_reasons = {}
            hops_for_reached_root_trials = []
            total_path_costs = []
            position_drifts = []
            heading_drifts = []
            failure_position_drifts = []
            average_ts = []
            spawn_to_root_costs = []
            file_name = in_output_folder(f"Mode{args.reverse_selection_mode}__RVHW{rvhw}_RVHB{rvhb}_RVRad{rvrad}_RWHW{rwhw}_RWHBd{rwhb}_RWRads{rwrad}_CWE{cwe}_ADB{adb}_k{k_value}_PB{pb}__Vmag{args.v_magnitude}_MaxSteps{args.max_steps}.csv")
            for trial in tqdm(range(args.reversibility_trials), desc=f"Reversing rvhb={rvhb}, rvrad={rvrad}", position=1, leave=False): # Test drift for MULTIPLE nodes on the SAME tree
                # Get the robot's starting position
                while True: 
                    if args.reverse_spawn_mode == 'node':
                        random_node_index = random.randint(1, len(nodes)-1)
                        start_node = nodes[random_node_index]
                        start_x, start_y, start_theta_deg = start_node.position[0], start_node.position[1], start_node.orientation
                    elif args.reverse_spawn_mode == 'map':
                        start_x = random.uniform(0, MAP_DIMENSIONS[0])
                        start_y = random.uniform(0, MAP_DIMENSIONS[1])
                        start_theta_deg = random.uniform(0, 360)
                    elif args.reverse_spawn_mode == 'tree_box': # tree box
                        xs = [n.position[0] for n in nodes]
                        ys = [n.position[1] for n in nodes]
                        start_x = random.uniform(min(xs), max(xs))
                        start_y = random.uniform(min(ys), max(ys))
                        start_theta_deg = random.uniform(0,360)
                    elif args.reverse_spawn_mode == 'file':
                        start_x, start_y, start_theta_deg = shared_spawns[trial]
                        if abs(start_x - nodes[0].position[0]) <= pb and abs(start_y - nodes[0].position[1]) <= pb:
                            break # fail safe

                    if abs(start_x - nodes[0].position[0]) > pb or abs(start_y - nodes[0].position[1]) > pb:  # Don't count "lucky" spawns
                        break    
                
                this_spawn_cost = nodes[random_node_index].cost if args.reverse_spawn_mode == 'node' else None
                if this_spawn_cost is not None: 
                    spawn_to_root_costs.append(this_spawn_cost)   
                
                success, final_x, final_y, final_theta, n_hops, total_path_cost, average_t, reason, pose_path, node_path = graph.drive_backwards(nodes=nodes, robot=robot, start_x=start_x, start_y=start_y, start_theta_deg=start_theta_deg, reverse_search_radius=rvrad, reverse_heading_boundary=rvhb, position_threshold=pb, selection_mode=args.reverse_selection_mode, use_visited=args.use_visited, use_samples=args.reverse_use_samples, max_steps=args.max_steps)  
                average_ts.append(average_t)
                position_drift = math.hypot(final_x - nodes[0].position[0], final_y - nodes[0].position[1])
                heading_drift = graph.angle_difference(final_theta, nodes[0].orientation)
                
                if args.verbose:
                    tqdm.write(f"[REVERSE] trial={trial}, spawn_mode={args.reverse_spawn_mode}, success={success}, reason={reason}, hops={n_hops}, cost={total_path_cost}, drift={position_drift}")

                if success:
                    successes += 1
                    hops_for_reached_root_trials.append(n_hops)
                    total_path_costs.append(total_path_cost)
                    position_drifts.append(position_drift)
                    heading_drifts.append(heading_drift)
                else:
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    failure_position_drifts.append(position_drift)

                # Record a cost log
                if args.cost_log:
                    cost_file = file_name.replace('.csv', '_costs.csv')
                    cost_exists = os.path.isfile(cost_file)
                    with open(cost_file, 'a', newline='') as cf:
                        cw = csv.writer(cf)
                        if not cost_exists:
                            cw.writerow(["run_number","trial","spawn_node","spawn_to_root_cost", "driven_path_cost","cost_ratio","success","reason"])
                        ratio = (total_path_cost/this_spawn_cost) if (success and this_spawn_cost) else None
                        cw.writerow([s+1, trial, random_node_index if args.reverse_spawn_mode=='node' else None, round(this_spawn_cost,2) if this_spawn_cost else None, round(total_path_cost,2), round(ratio,3) if ratio else None, success, reason])
                
                record_this = (args.save_reversibility_record == 'all' or (args.save_reversibility_record == 'failures' and not success) or (args.save_reversibility_record == 'successes' and success))
                if record_this:
                    rec_base = in_output_folder(f"reversal_{args.reverse_selection_mode}_rvhb{rvhb}_rvrad{rvrad}_rvhw{rvhw}_pb{pb}_rwhw{rwhw}_adb{adb}_run{s+1}_trial{trial}_{'success' if success else 'fail'}")
                    graph.save_reversal_record(robot=robot, environment=environment, nodes=nodes, pose_path=pose_path, position_threshold=pb,node_path=node_path, success=success, start_pose=(start_x, start_y), search_radius=rvrad, heading_boundary=rvhb, base_path=rec_base)  
                    
                # Check argument passed
                show_this = (args.show_reversal_trials == 'all' or (args.show_reversal_trials == 'failures' and not success) or (args.show_reversal_trials == 'successes' and success))
                if show_this and not args.repeat_program:
                    # Show the screen
                    if not args.show_tree_building:
                        environment.show_screen()
                    print("[EVENT] Showing reversal trial")
                    graph.draw_reversal_trials(robot=robot, environment=environment, nodes=nodes, pose_path=pose_path, position_threshold=pb, node_path=node_path, success=success, start_pose=(start_x, start_y), search_radius=rvrad, heading_boundary=rvhb) 

            avg_failure_position_drift = sum(failure_position_drifts)/len(failure_position_drifts) if failure_position_drifts else None
            avg_heading_drift = sum(heading_drifts)/len(heading_drifts) if heading_drifts else None
            avg_position_drift = sum(position_drifts)/len(position_drifts) if position_drifts else None
            success_rate = successes/args.reversibility_trials if args.reversibility_trials > 0 else None
            avg_hops_on_success = sum(hops_for_reached_root_trials)/len(hops_for_reached_root_trials) if hops_for_reached_root_trials else None
            avg_total_path_cost = sum(total_path_costs)/len(total_path_costs) if total_path_costs else None
            # mean_average_t = sum(average_ts)/len(average_ts) if average_ts else 0.0
            mean_rewire_error = sum(graph.rewire_errors)/len(graph.rewire_errors) if graph.rewire_errors else 0.0
            max_rewire_error = max(graph.rewire_errors) if graph.rewire_errors else 0.0
            avg_spawn_to_root_cost = sum(spawn_to_root_costs)/len(spawn_to_root_costs) if spawn_to_root_costs else None

            stuck_no_candidate_count = failure_reasons.get('stuck_no_candidate', 0)
            max_steps_exceeded_count = failure_reasons.get('max_steps_exceeded', 0)
            max_substeps_exceeded_count = failure_reasons.get('max_substeps_exceeded', 0)

            if args.verbose:
                if success_rate is not None:
                    tqdm.write(f"[REVERSE] success_rate={success_rate:.2%}, avg_position_drift={avg_position_drift}, avg_hops_on_success={avg_hops_on_success}, avg_total_path_cost={avg_total_path_cost}")
                else:
                    tqdm.write(f"[REVERSE] success_rate=None, avg_position_drift={avg_position_drift}, avg_hops_on_success=None, avg_total_path_cost={avg_total_path_cost}")
                tqdm.write("[REVERSE] failure breakdown:")
                for reason, count in failure_reasons.items():
                    tqdm.write(f"    {reason}: {count}")

            # --------- Reversibility Testing End --------- 

            if graph.is_goal_found:
                goal_indices = []
                for i, n in enumerate(nodes):
                    if graph.is_goal_reached(n.position, n.orientation):
                        goal_indices.append(i)

            if goal_indices:
                min_cost = None
                min_index = None
                for i in goal_indices:
                    if min_cost is None or nodes[i].cost < min_cost:
                        min_cost = nodes[i].cost
                        min_index = i
                
                graph.goal_configuration = min_index    
                graph.nodes = nodes # Attaches growing node list to graph.node
                # Get the list of node IDs of the winning route
                graph.path_to_goal() 
                # Use node IDs to make (x,y) coordinate pairs and recover step-by-step movement data
                graph.get_path_coordinates()

                flat_thetas = []
                for segment in graph.theta_final_path[1:]:
                    flat_thetas.extend(segment)
                
                jumps = []
                for i in range(len(flat_thetas)-1):
                    current_theta = flat_thetas[i]
                    next_theta = flat_thetas[i+1]

                    jump = graph.angle_difference(current_theta, next_theta)
                    jumps.append(jump)

                max_jump = max(jumps)
                total_variation = sum(jumps)
                avg_jump = total_variation/len(jumps)
                physical_ceiling = math.degrees(0.5*args.delta)
                if args.verbose:
                    tqdm.write(f"[STATS] producing a final_path_cost={nodes[graph.goal_configuration].cost:.2f}")
                    tqdm.write(f"[STATS] max_jump={max_jump:.2f} deg, avg_jump={avg_jump:.2f} deg, total_variation={total_variation:.2f} (physical_ceiling={physical_ceiling:.2f}) of {len(flat_thetas)} samples.")
                    tqdm.write(f"The total number of shots made during rewiring were = {graph.total_rewire_shots:,}")
                    tqdm.write(f"(successful rewires: {len(graph.rewire_errors)})")
                final_cost = nodes[graph.goal_configuration].cost
            else:
                if args.verbose:
                    tqdm.write("[STATS] Goal not found this run.")
                final_cost = None

            
            # file_name = f"Mode{args.reverse_selection_mode}__RWHeadingBound{rwhb}_RWRadius{rwrad}_RVHB{rvhb}_RVRad{rvrad}__RVThetaWeight{args.reverse_heading_weight}_PosBound{pb}__Vmag{args.v_magnitude}_CWE{args.children_weight_exponent}_MaxSteps{args.max_steps}.csv"
            graph.log_results(file_name=file_name, run_number=s+1, theta_weight=rvhw, rewire_heading_weight=rwhw, rewire_heading_boundary=rwhb, rewire_search_radius=rwrad, reverse_heading_boundary=rvhb, reverse_search_radius=rvrad, success_rate=success_rate, avg_position_drift=avg_position_drift, avg_heading_drift=avg_heading_drift, avg_hops_on_success=avg_hops_on_success, avg_total_path_cost=avg_total_path_cost, cost=avg_spawn_to_root_cost, mean_rewire_error=mean_rewire_error, max_rewire_error=max_rewire_error, avg_failure_position_drift=avg_failure_position_drift, stuck_no_candidate_count=stuck_no_candidate_count, max_steps_exceeded_count=max_steps_exceeded_count, max_substeps_exceeded_count=max_substeps_exceeded_count)
            
            if args.save_png:
                if args.w_mode == 'random':
                    base_name = f"tree_randomw_cwe{cwe}_sd{args.seed if args.seed is not None else 0}_adb{adb}"
                elif args.w_mode == 'kwt':
                    base_name = f"tree_k{k_value}_cwe{cwe}_sd{args.seed if args.seed is not None else 0}_adb{adb}"
                else: #' kwt2'
                    base_name = f"tree_k{k_value}tsquared_cwe{cwe}_sd{args.seed if args.seed is not None else 0}_adb{adb}"
                
                base_name += f"_rwhb{rwhb}_rwrad{rwrad}_rwhw{rwhw}_run{s+1}"

                # Save 2D Graph
                png_name = in_output_folder(base_name+"_2D.png")
                environment.clear_screen()
                if args.obstacles:
                    environment.draw_obstacles()
                # graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
                # graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
                graph.draw_final_backdrop_2d(environment, nodes)
                if graph.is_goal_found and args.path_to_goal:
                    graph.draw_path_to_goal(map=environment.map)
                root_x, root_y = nodes[0].position 

                half = args.position_boundary
                root_rect = pygame.Rect(int(root_x - half), int(root_y - half), 2 * half, 2 * half)
                pygame.draw.rect(environment.map, (34, 182, 101), root_rect)
                pygame.draw.rect(environment.map, (0, 0, 0), root_rect, width=2)
                g_font = pygame.font.SysFont(None, 30)
                g_surf = g_font.render("G", True, (255, 255, 255))
                environment.map.blit(g_surf, g_surf.get_rect(center=(int(root_x), int(root_y))))
                
                pygame.image.save(environment.map, png_name)
                tqdm.write(f"[EVENT] Saved 2D tree to {os.path.abspath(png_name)}")

                # Save 3D Graph
                png_3d = in_output_folder(base_name + "_3d.png")
                graph.save_3d_map(nodes=nodes, init_point=args.x_init, goal_point=args.x_goal, file_name=png_3d, title=base_name)
                tqdm.write(f"[EVENT] Saved 3D tree to {os.path.abspath(png_3d)}")


            progress_bar_overall.update(1)

            if not(args.repeat_program and s+1 < args.repeat_program_num):
                break
            s+=1
        
    progress_bar_overall.close()

    if args.repeat_program == False:

        if not args.show_tree_building and args.show_final:
            environment.show_screen()
            print("[EVENT] Showing the screen")
            graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
            graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
            graph.draw_final_backdrop_2d(environment, nodes)
            pygame.display.update()

            print("[EVENT] Waiting for user input")
            waiting_for_input = True
            while waiting_for_input:
                print("[DEBUG] I am in the while loop")
                if args.save_png is not None:
                        print("[EVENT] No running simultion, quitting pygame.")
                        waiting_for_input = False
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        print("[EVENT] Recieved input, quitting pygame.")
                        waiting_for_input = False
                pygame.display.update()

        if graph.is_goal_found:
            # Display such and such
            if args.path_to_goal:
                graph.draw_path_to_goal(map=environment.map)
            if args.move_robot:
                graph.draw_trajectory(robot=robot, environment=environment)
            if args.show_interpolation:
                graph.draw_interpolation(robot=robot, environment=environment)

        # Show 3D map  
        pygame.quit()
        if args.show_3d:
            print("[EVENT] Showing 3d graph now")
            graph.show_3d_map(nodes=nodes, init_point=args.x_init, goal_point=args.x_goal)

    sys.exit()

if __name__ == '__main__':
    main()