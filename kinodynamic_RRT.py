import pygame
import environment
import robot
import RRT
import math
import argparse
import sys

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
	default=(50, 50, 0), help='Initial node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-goal', '--x_goal', nargs='+', type=float, metavar='', required=False,
	default=(540, 380, 0), help='Goal node config in X, Y (pixels) and theta (radians)')
parser.add_argument('-src', '--show_random_configurations', type=bool,
	action=argparse.BooleanOptionalAction, metavar='', required=False, 
	help='Show random config on screen')
parser.add_argument('-snc', '--show_new_configurations', type=bool,
 	action=argparse.BooleanOptionalAction, metavar='', required=False,
 	help='Show new config on screen')
#TODO: Delete the bias when the "goal" position is deleted, so the tree expands in all directions equally.
parser.add_argument('-bp', '--bias_percentage', type=int, metavar='', required=False, default=50,
	help='Amount of bias the RRT from 1-100')
parser.add_argument('-ptg', '--path_to_goal', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=False, help='Draws the milestones from path to goal')
parser.add_argument('-si', '--show_interpolation', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=True, help='Draws the configurations needed to reach the \
	 	goal')
parser.add_argument('-mr', '--move_robot', type=bool, action=argparse.BooleanOptionalAction, 
	metavar='', required=False, default=True,
	help='Shows the movements of the robot from start to end')
parser.add_argument('-pb', '--position_boundary', type=int, metavar='', required=False, default=25,
	help='Allowed position region of the pixels for the robot to reach the goal')
parser.add_argument('-ob', '--orientation_boundary', type=float, metavar='', required=False,
 	default=math.pi, help='Allowed orientation region of the angle for the robot to reach the goal')
parser.add_argument('-stb', '--show_tree_building', type=bool, 
	action=argparse.BooleanOptionalAction,	metavar='', required=False, default=True,
	help='Shows the simulation forward in time')
parser.add_argument('-rad', '--search_radius', nargs='+', type=float, metavar='', required=False,
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
    tree = [] # Pos in px
    orientation_tree = [] # Orientation in degrees
    parent = []
    cost = [0.0] # Cost to get to a node from x_init
    values = []
    tree.append(x_init[:2]) # Append init node configuration
    orientation_tree.append(x_init[2]) # Append init node orientation
    # Add ID to the initial node
    parent.append(0)
    values.append(0)
    node_value = 0
    iteration = 0
    k = 0 
    graph.position_boundary = args.position_boundary
    graph.orientation_boundary = args.orientation_boundary
    graph.show_tree_building = args.show_tree_building
    graph.delta = args.delta 
    environment.make_obstacles()

    if not args.show_tree_building:
        print('Simulating the system... Please wait.')

    while run and k < args.nodes:
        # Set loop run to 60 FPS
        clock.tick(environment.FPS)
        print(f"top of while loop, is_fwd_finished={graph.is_forward_simulation_finished}")
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
            # If the forward simulation is not finished
            if graph.is_forward_simulation_finished:
                # Generate a random node and check for collision
                x_rand = graph.generate_random_node(bias=args.bias_percentage)
                random_rect = graph.draw_random_robot_configuration(robot_img=robot.img, environment=environment)
                rand_collision_free = graph.is_free(configuration=random_rect, obstacles=obstacles)

            if rand_collision_free:
                # Propagate with random control function 
                x_new = graph.new_state(x_rand, robot, event, environment, obstacles)
                
                # If the path did not collide with something...
                if x_new is not None:
                    tree.append(x_new[0]) # Returns a tuple of configuration
                    orientation_tree.append(x_new[1]) # Returns orientation

                    # Update node ID list and parent list with x_new
                    values.append(node_value)
                    parent = graph.generate_parents(values, parent)
                    parent_index = parent[-1] # Check which index the parents are on
                    edge_time = graph.edge_time_duration 
                    cost.append(cost[parent_index] + edge_time) # Find cost of the present node by adding the parents 

                    # Update node list, number of free collisions and sampled config (K)
                    node_value+=1
                    graph.number_of_nodes = len(tree)
                    graph.num_free_collision = 0	
                    k += 1

                    graph.rewire_node(tree=tree, orientation_tree=orientation_tree, parent=parent, cost=cost, 
                                      search_radius=args.search_radius, delta=args.delta, x_new=x_new)
                                    
                # if goal is found
                if graph.is_goal_found:
                    print(f"Goal is found!")
                    graph.parent = parent # Attaches growing parent list to graph.parent
                    graph.tree = tree # Attaches growing tree list to graph.tree
                    graph.orientation_tree = orientation_tree # Attaches growing orientation tree list to graph.orientation_tree
                    # Get the list of node IDs of the winning route
                    graph.path_to_goal() 
                    # Use node IDs to make (x,y) coordinate pairs and recover step-by-step movement data
                    graph.get_path_coordinates()
                    

        # For this to run, both need to be explicitly ENABLED
        if args.show_new_configurations and args.show_tree_building:
            # Draws a small rotated robot image at every node position added to the tree 
            graph.draw_new_robot_configuration(robot_img=robot.img, environment=environment)
            # Draws a small brown circle at every node position
            graph.draw_new_node(map=environment.map)

            # Goal is found but show_tree_building is off
        if graph.is_goal_found:
            if not args.show_tree_building:
                environment.show_screen()
                graph.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
                graph.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
                environment.draw_trajectory_trail()
                pygame.display.update()
                pygame.time.delay(500)

            # Display such and such
            if args.path_to_goal:
                graph.draw_path_to_goal(map=environment.map)
            if args.move_robot:
                graph.draw_trajectory(robot=robot, environment=environment)
            if args.show_interpolation:
                graph.draw_interpolation(robot=robot, environment=environment)
            
            pygame.quit()
            graph.show_3d_map(init_point=args.x_init, goal_point=args.x_goal)

        if rand_collision_free and args.show_random_configurations and args.show_tree_building:
            random_rect = graph.draw_random_robot_configuration(robot_img=robot.img, environment=environment)
            graph.draw_random_node(map=environment.map)

        iteration +=1

        if not graph.is_goal_found:
            pygame.display.update()
            environment.map.fill(environment.WHITE)

    sys.exit()

if __name__ == '__main__':
    main()