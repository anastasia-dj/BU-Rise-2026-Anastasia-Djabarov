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

from node import Node

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Command line inputs 
parser = argparse.ArgumentParser(description='kRRT algorithm for a car-like robot.')
parser.add_argument('-o', '--obstacles', type=bool, action=argparse.BooleanOptionalAction,
	metavar='', required=False, help='Obstacles on the map')
parser.add_argument('-n', '--nodes', type=int, metavar='', required=False, default=5000,
	help='Max number of nodes')
parser.add_argument('-dt', '--delta', type=float, metavar='', required=False, default=0.05,
	help='Fixed steps between frames for the visualization')
parser.add_argument('-init', '--x_init', nargs='+', type=float, metavar='', required=False,
	default=(320, 240, 0), help='Initial node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-goal', '--x_goal', nargs='+', type=float, metavar='', required=False,
	default=(540, 320, 0), help='Goal node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-snc', '--show_new_configurations', type=bool,
 	action=argparse.BooleanOptionalAction, metavar='', required=False,
 	help='Show new config on screen')
parser.add_argument('-ptg', '--path_to_goal', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=False, help='Draws the milestones from path to goal')
parser.add_argument('-si', '--show_interpolation', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=True, help='Draws the configurations needed to reach the \
	 	goal')
parser.add_argument('-mr', '--move_robot', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=True,
	help='Shows the movements of the robot from start to end')
parser.add_argument('-pb', '--position_boundary', type=int, metavar='', required=False, default=10,
	help='Allowed position region of the pixels for the robot to reach the goal')
parser.add_argument('-ob', '--orientation_boundary', type=float, metavar='', required=False,
 	default=math.pi, help='Allowed orientation region of the angle for the robot to reach the goal')
parser.add_argument('-stb', '--show_tree_building', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=True,
	help='Shows the simulation forward in time')
parser.add_argument('-rad', '--search_radius', type=float, metavar='', required=False,
	default=10, help='The search radius of the robot when deciding which node to rewire.')
parser.add_argument('-hb', '--heading_boundary', type=float, metavar='', required=False,
	default=180, help='The boundary of the heading difference which is used in rewiring..')
parser.add_argument('-rw', '--rewire', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=True,
	help='Rewires the tree nodes.')
parser.add_argument('-sd', '--seed', type=int, metavar='', required=False, default=None,
	help='Seed of the random numbers taken.')
parser.add_argument('-rsd', '--reversibility_seed', type=int, metavar='', required=False, default=None,
	help='Seed for selecting which nodes get tested in the reversibility.')
parser.add_argument('-adb', '--anti_drive_bias', type=float, metavar='', required=False, default=0,
	help='Percent change (0-100) an edge uses zero/reverse velocity instead of forward driving.')
parser.add_argument('-adm', '--anti_drive_mode', type=str, choices=['stop', 'reverse', 'both'], 
    metavar='', required=False, default='both', help='Which behavior triggers when anti_bias engages, v=0(stop), v=-v(reverse), or both (randomly picked).')
parser.add_argument('-gep', '--goal_end_program', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=True, help='Stop growing the tree as soon as the goal is reached.')
parser.add_argument('-rt', '--reversibility_trials', type=int, metavar='', required=False, default=30,
	help='Number of reversibility trials to do..')
parser.add_argument('-rp', '--repeat_program', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Repeats the program multiple times.')
parser.add_argument('-rpn', '--repeat_program_num', type=int, metavar='', required=False, default=3,
	help='Number of times to repeat the program.')
args = parser.parse_args()

# Initialization
pygame.init()

# Constants
ROBOT_INIT_PATH = 'images/initial.png'
ROBOT_GOAL_PATH = 'images/goal.png'
ROBOT_IMG_PATH = 'images/robot.png'
MAP_DIMENSIONS = 640,480
robot_images = [ROBOT_INIT_PATH, ROBOT_GOAL_PATH, ROBOT_IMG_PATH]

# Initial and final robot configs (px, px, rad)
x_init = args.x_init
x_goal = args.x_goal

# Creating environment/robot/RRT
environment = environment.Environment(dimensions=MAP_DIMENSIONS, show_tree_building=args.show_tree_building, has_obstacles=args.obstacles)
robot = robot.Robot(start=args.x_init, robot_img=robot_images, length=0.01)
def main():
    s = 0
    while True:
        graph = RRT.Graph(start=x_init, goal=x_goal, map_dimensions=MAP_DIMENSIONS)
        if args.seed is not None:
            if args.repeat_program is True:
                random.seed(s)
            else:
                random.seed(args.seed)
        run = True
        clock = pygame.time.Clock()

        nodes = [Node(position=x_init[:2], orientation=math.degrees(x_init[2]), parent=0, cost=0.0)]

        iteration = 0
        k = 0 
        graph.position_boundary = args.position_boundary
        graph.orientation_boundary = args.orientation_boundary
        graph.show_tree_building = args.show_tree_building
        graph.delta = args.delta 
        graph.anti_drive_bias = args.anti_drive_bias
        graph.anti_drive_mode = args.anti_drive_mode
        graph.number_of_nodes = len(nodes)
        environment.make_obstacles()

        if not args.show_tree_building:
            print('[EVENT] Simulating the system... Please wait.')

        # Main growth loop
        while run and k < args.nodes:
            clock.tick(environment.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Draw all initial states
            obstacles = environment.draw_obstacles() if args.obstacles else []
            graph.draw_initial_node(map=environment.map)
            graph.draw_goal_node(map=environment.map)
            graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
            graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)

            if args.show_tree_building:
                environment.draw_trajectory_trail()
            
            # If the goal is not found
            if not graph.is_goal_found or not args.goal_end_program:
                # Find a random node
                random_parent_index = random.randint(0, len(nodes)-1)

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

                    assert new_node.cost >= nodes[new_node.parent].cost, f"cost did not increase from parent ({nodes[new_node.parent].cost}) to child ({new_node.cost})"

                    # Update sampled config (k)
                    graph.number_of_nodes = len(nodes)
                    k += 1
                    print(f"[DEBUG] k={k}, len(tree)={len(nodes)}")

                    if args.rewire: 
                        graph.rewire_node(nodes=nodes, search_radius=args.search_radius, delta=args.delta, heading_boundary=args.heading_boundary)
                
                if args.show_new_configurations and args.show_tree_building:
                    # Draws a small rotated robot image at every node position added to the tree 
                    graph.draw_new_robot_configuration(robot_img=robot.img, environment=environment)
                    # Draws a small brown circle at every node position
                    graph.draw_new_node(map=environment.map)

                # if goal is found
                if graph.is_goal_found and args.goal_end_program:
                    break # Stop growing 

                iteration += 1
                pygame.display.update()
                environment.map.fill(environment.WHITE)
                            
        print(f"[EVENT] Loop ended. Goal found: {graph.is_goal_found}, Total nodes: {len(nodes)}, k={k}.")
        print(f"[STATS] heading_boundary of {args.heading_boundary} blocked {graph.candidates_blocked_by_heading}/{graph.candidates_in_radius} candidates.")

        # --- Reversibility test --- 
        graph.nodes = nodes # For path_from_nodes
        if args.reversibility_seed is not None:
                if args.repeat_program is True:
                    random.seed(s)
                else:
                    random.seed(args.reversibility_seed)
        
        shot_distances = 0.0
        position_drifts = []
        heading_drifts = []
        for trial in range(args.reversibility_trials): # Test drift for MULTIPLE nodes on the SAME tree
            random_node_index = random.randint(1, len(nodes)-1)
            final_x,final_y,final_theta, shot_distance, n_shot_segments = graph.drive_backwards(random_node_index, robot, nodes)
            position_drift = math.hypot(final_x - nodes[0].position[0], final_y - nodes[0].position[1])
            heading_drift = graph.angle_difference(final_theta, nodes[0].orientation)
            position_drifts.append(position_drift)
            heading_drifts.append(heading_drift)
            print(f"[REVERSE] trial={trial}, start_node={random_node_index}, heading_drift={heading_drift:.6f}px\n"
                f"position_drift = {position_drift}, shot_distance={shot_distance}px, n_shot_segments={n_shot_segments}")
            shot_distances += shot_distance
        
        avg_shot_distance = shot_distances/args.reversibility_trials
        avg_position_drift = sum(position_drifts)/len(position_drifts)
        avg_heading_drift = sum(heading_drifts)/len(heading_drifts)
        print(f"[REVERSE] POSITION - averages over {len(position_drifts)} trials: drift_from_final_goal={avg_position_drift:.6f}, shot_distance_from_goal={avg_shot_distance}"
              f"[REVERSE] HEADING - averages over {len(heading_drifts)} trials: drift_from_final_goal={avg_heading_drift:.6f}, shot_distance_from_goal={avg_shot_distance}")

        if graph.is_goal_found:
            goal_indices = []
            for i, n in enumerate(nodes):
                if graph.is_goal_reached(n.position, n.orientation):
                    goal_indices.append(i)
            
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
            print(f"[STATS] producing a final_path_cost={nodes[graph.goal_configuration].cost:.2f}")
            print(f"[STATS] max_jump={max_jump:.2f} deg, avg_jump={avg_jump:.2f} deg, total_variation={total_variation:.2f} (physical_ceiling={physical_ceiling:.2f}) of {len(flat_thetas)} samples.")

            file_name = f"HeadingBound{args.heading_boundary}.csv"
            graph.log_results(file_name=file_name, heading_boundary=args.heading_boundary, run_number=s, avg_position_drift=avg_position_drift, avg_heading_drift=avg_heading_drift, cost=nodes[graph.goal_configuration].cost)
            
        if not(args.repeat_program and s+1 < args.repeat_program_num):
            break
        s+=1

    if args.repeat_program == False:
        if not args.show_tree_building:
            environment.show_screen()
            graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
            graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
            graph.draw_final_backdrop_2d(environment, nodes)
            pygame.display.update()

            waiting_for_input = True
            while waiting_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
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
        graph.show_3d_map(nodes=nodes, init_point=args.x_init, goal_point=args.x_goal)

    sys.exit()

if __name__ == '__main__':
    main()