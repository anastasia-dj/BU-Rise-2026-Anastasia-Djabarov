import pygame
import environment
import robot
import RRT
import math
import random
import argparse
import unittest
import sys

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
graph = RRT.Graph(start=x_init, goal=x_goal, map_dimensions=MAP_DIMENSIONS)

def main():
    run = True;
    clock = pygame.time.Clock()

    nodes = [Node(position=x_init[:2], orientation=math.degrees(x_init[2]), parent=0, cost=0.0)]

    iteration = 0
    k = 0 
    graph.position_boundary = args.position_boundary
    graph.orientation_boundary = args.orientation_boundary
    graph.show_tree_building = args.show_tree_building
    graph.delta = args.delta 
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
        if not graph.is_goal_found:
            # Find a random node
            random_parent_index = random.randint(0, len(nodes)-1)

            # The result of propgation with random control function from a random node
            result = graph.new_state(random_parent_index, robot, environment, nodes, obstacles)
            
            # If the path did not collide with something... #TODO: Sample points across the collision
            if result is not None:
                x_new, theta_new, trajectory_samples = result

                # Create new node
                new_node = Node(
                    position=x_new,
                    orientation=theta_new,
                    parent=random_parent_index,
                    cost=nodes[random_parent_index].cost + graph.edge_time_duration*graph.v,
                    trajectory_samples=trajectory_samples
                )
                nodes.append(new_node)
                new_index = len(nodes) -1

                assert new_node.cost > nodes[new_node.parent].cost, f"cost did not increase from parent ({nodes[new_node.parent].cost}) to child ({new_node.cost})"


                # Update sampled config (k)
                
                graph.number_of_nodes = len(nodes)
                k += 1
                print(f"[DEBUG] k={k}, len(tree)={len(nodes)}")

                graph.rewire_node(nodes=nodes, search_radius=args.search_radius, delta=args.delta)
            
            if args.show_new_configurations and args.show_tree_building:
                # Draws a small rotated robot image at every node position added to the tree 
                graph.draw_new_robot_configuration(robot_img=robot.img, environment=environment)
                # Draws a small brown circle at every node position
                graph.draw_new_node(map=environment.map)

            # if goal is found
            if graph.is_goal_found:
                break # Stop growing 

            iteration += 1
            pygame.display.update()
            environment.map.fill(environment.WHITE)
                        
    print(f"[EVENT] Loop ended. Goal found: {graph.is_goal_found}, Total nodes: {len(nodes)}, k={k}")

    if graph.is_goal_found:
        graph.nodes = nodes # Attaches growing node list to graph.node
        # Get the list of node IDs of the winning route
        graph.path_to_goal() 
        # Use node IDs to make (x,y) coordinate pairs and recover step-by-step movement data
        graph.get_path_coordinates()
        
    if not args.show_tree_building:
        environment.show_screen()
        graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
        graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
        environment.draw_trajectory_trail()
        pygame.display.update()
        pygame.time.delay(500)

    if graph.is_goal_found:
        # Display such and such
        if args.path_to_goal:
            graph.draw_path_to_goal(map=environment.map)
        if args.move_robot:
            graph.draw_trajectory(robot=robot, environment=environment)
        if args.show_interpolation:
            graph.draw_interpolation(robot=robot, environment=environment)
        
    pygame.quit()
    graph.show_3d_map(nodes=nodes, init_point=args.x_init, goal_point=args.x_goal)
    

    sys.exit()

if __name__ == '__main__':
    main()