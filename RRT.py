import pygame
import random
import math
import numpy as np

class Graph():
    # Class for RRT
    # start : tuple (init x, y, theta)
    # goal : tuple (final x, y, theta)
    # map_dimensions : tuple (width and height)

    def __init__(self, start, goal, map_dimensions):
        self.x_init = start
        self.x_goal = goal
        self.WIDTH, self.HEIGHT = map_dimensions

        self.max_simulation_time = 4 # Max fwd simulation time
        self.robot_last_position = []
        self.robot_last_orientation = []
        self.controls = []
        self.is_forward_simulation_finished = True
        self.is_goal_found = False
        self.iteration = 0

        self.robot_last_position.append(self.x_init[:2]) # List accumulating history
        self.robot_last_orientation.append(math.degrees(self.x_init[2]))
        
        self.last_position = self.x_init[:2] # Single value holding most recent position
        self.last_orientation = self.x_init[2] #TODO: Change to degrees
        self.u1 = random.uniform(-0.4, 0.4) #TODO: Verify numbers 
        self.u2 = random.uniform(-0.5, 0.5)
        self.u = [self.u1, self.u2]

        # Stored list of positions and orientations
        self.x_positions = []
        self.y_positions = []
        self.theta_orientations = []

        # Present position and orientation
        self.x_position = []
        self.y_position = []
        self.theta_orientation = []

        #TODO: Come back here and add comments
        self.x_interpolation = []
        self.y_interpolation = []
        self.theta_interpolation = []

        self.interpolation_factor = 10

        self.x_interpolation.append(self.x_init[0])
        self.y_interpolation.append(self.x_init[1])
        self.theta_interpolation.append(self.x_init[2])

        # Store added states (i.e. multiple u_new)
        self.u_news = []
        self.theta_news = []

        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (105, 105, 105)
        self.BLACK = (0, 0, 0)
        self.FUCSIA = (255, 0, 255)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BROWN = (189, 154, 122)

    #TODO: Checks the ending config. if the ending config is free, it may go through a wall
    def is_free(self, configuration, obstacles):
        # Check if a config will collide with an obstacle
        # PARAMETERS:
            # configuration : pygame.Rect
            # obstacles : pygame.Rect
        # RETURNS:
            # bool

        for obstacle in obstacles:
            if obstacle.colliderect(configuration):
                return False
        return True

    def generate_random_node(self, bias):
        # Generates a random configuratin
        # PARAMETERS:
            # bias : int (Goal bias percentage)
        # RETURNS: 
            # tuple : coordinate of random node

        #Sample a robot config in (x, y, theta)
        self.x_rand = int(random.uniform(0, self.WIDTH)), int(random.uniform(0, self.HEIGHT)), math.radians(random.uniform(0, 360))

        # Likelihood of goal bias 
        bias_percentage = 10 - bias//10 if bias != 100 else 1
        if self.iteration%bias_percentage == 0:
            self.x_rand = self.x_goal
        
        return self.x_rand
    
    def metric(self, p1, p2):
        # Definition of the metric for a car (C = R^2 x S)
        # --> How far are p1 and p2? <--
        # PARAMETERS
            # p1 : int (Start configuration)
            # p2 : int (End configuration)
        # RETURNS
            # gloat (R^2 x S metric)

        # Angle wraparound
        theta = self.x_rand[2] #in rad
        alpha = min(abs(self.last_orientation - theta), 2*math.pi - abs(self.last_orientation - theta))
        
        # Metric simplification through Euclidean formula (c^2 = a^2 + b^2)
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + alpha**2)
    
    def nearest_neighbor(self, tree, x_rand):
        # Returns the index of the nearest neighbor to randomly generated config (x_rand)
        # PARAMETERS
            # tree : list (Tree containing ALL configurations)
            # x_rand : tuple (x, y, and theta of random node generated)
        # RETURNS
            # tuple (Nearest node to random node generated)
        distances = []

        # Collect all the distances between each state and x-rand
        for state in tree:
            distance = self.metric(state, x_rand)
            distances.append(distance)

        # Index of the minimum distance to x_rand
        self.min_distance = np.argmin(distances)
        x_near = tree[self.min_distance]

        return x_near
    
    def simulate(self, robot, event, environment, obstacles):
        # It simulates the robot configuration from u (control input)
        # PARAMETERS
            # robot : robot object (Where all control inputs are simulated)
            # event : Event (All events happening on screen)
            # environment : environment object (Environment to draw trails & simulations fwd)
        # RETURNS
            # list : Collection of positions the robot has passed through

        time = robot.dt

        self.is_forward_simulation_finished = False

        if time <= self.max_simulation_time:
            # Draw map/robot/trail is max sim time is not reached & show_tree_building
            if self.show_tree_building:
                robot.draw(map=environment.map) # Draw robot at each fwd in time simulation
            robot.move(event=event, u=self.u)
            environment.trail(position=(robot.x, robot.y), draw_trail=self.show_tree_building)
            self.x_positions.append(robot.x)
            self.y_positions.append(robot.y)
            self.theta_orientations.append(robot.theta)
        else: 
            # Simulation time restarted
            self.is_forward_simulation_finished = True
            self.max_simulation_time = 4
            robot.dt = 0
            time = robot.dt

            # TODO: Change these numbers (Sampled?)
            self.u1 = random.uniform(-0.4, 0.4)
            self.u2 = random.uniform(-0.5, 0.5)
            self.u = [self.u1, self.u2]

            # Check collisions
            collision_free = self.is_free(configuration=robot.rect, obstacles=obstacles)

            if collision_free:
                # Store the last trails of the robot, empty piecewise trail list
                environment.store_trails()
                environment.empty_trails()

                # Store the last config of the robot
                self.store_configuration(position=(robot.x, robot.y), orientation=robot.theta)
                self.store_controls(controls=self.u)

                # Store the list of positions
                self.x_position.append(self.x_positions)
                self.y_position.append(self.y_positions)
                self.theta_orientation.append(self.theta_orientations)

                # Reset the list of positions
                self.x_positions = []
                self.y_positions = []
                self.theta_orientations = []

                return self.robot_last_position
            else:
                # Empty the trail list
                environment.empty_trails()

                self.x_positions = []
                self.y_positions = []
                self.theta_orientations = []

                # Rollback robot position and orientation
                robot.x, robot.y = self.last_position[:2]
                robot.theta = self.last_orientation

                return None
            
    def new_state(self, x_rand, robot, event, environment, obstacles=None):
        # Propagates a small step towards a random node (Given fwd simulations, selects the more suitable to advance toward x_rand)
        # PARAMETERS
            # x_rand : tuple (Coordinates of random generated node)
            # robot: robot object (Robot w/ simulated control inputs)
            # event : Event (All events happening on screen) (Unused?)
        # RETURNS
            # tuple (Best position towards x_rand)
            # float (Best orientation towards x_rand)
        if obstacles is None:
            obstacles = []
            
        
        self.obstacles = obstacles

        # Simulation fwd in time
        simulation = self.simulate(robot=robot, event=event, environment=environment, obstacles=obstacles)

        if simulation is not None:
            self.iteration += 1 # One tree iteration is done

            # Robot pos & orientation with min distance to x_rand
            # TODO: Rename u_new
            self.u_new = self.nearest_neighbor(simulation, x_rand) # New config
            self.theta_new = self.robot_last_orientation[self.min_distance] # Look at angle of nearest node

            # Append new configs to a list
            self.u_news.append(self.u_new)
            self.theta_news.append(self.theta_new)

            if self.is_goal_reached():
                self.goal_configuration = self.number_of_nodes-1
                self.is_goal_found = True
            
            #Store the configs as latest configs
            self.last_position = self.u_new
            self.last_orientation = self.theta_new
            robot.x, robot.y = self.last_position[:2]
            robot.theta = self.last_orientation

            return self.u_new, self.theta_new
        
    def generate_parents(self, values, parent):
        # Generates a list of parents given value of node and nearest neighbor
        # PARAMETERS
            # values : list (Node IDs)
            # parent : list (Previous collection of parent nodes until the new is processed)
        # RETURNS
            # list (Ordered collection of the parents)

        parent_value = values[self.min_distance] 
        parent_index = len(parent) 
        parent.insert(parent_index, parent_value) # Add parent to the end of parent list

        if self.is_goal_reached():
            # If goal is reached, add a final index +1 
            parent.insert(parent_index+1, values[-1]+1)

        return parent

    def store_configuration(self, position, orientation):
        # Stores the current robot configuration to an ongoing list for later use
        # PARAMETERS
            # position : tuple (X,Y in C-space)
            # orientation : tuple (theta in C-space)
        # RETURNS
            # None
        self.robot_last_position.append(position)
        self.robot_last_orientation.append(orientation)

    def path_to_goal(self):
        # Given x_goal node, it searches next parent until it reaches the x_init node
        self.path = []
        self.path.append(self.goal_configuration)
        new_configuration = self.parent[self.goal_configuration]

        while new_configuration != 0:
            # Append the parent of parent and update config
            self.path.append(new_configuration)
            new_configuration = self.parent[new_configuration]

        # Append parent 0 (x_init)
        self.path.append(0)

    def get_path_coordinates(self):
        # Collects correspondant coordinates
        # PARAMETERS
            # None
        # RETURNS
            # None

        self.path_coordinates = []

        for node in self.path:
            x, y = self.tree[node]
            self.path_coordinates.append((x,y))
        
        for i in range(len(self.x_position)):
            for j in range(len(self.path_coordinates)):
                x_matches = self.x_position[i][-1] == self.path_coordinates[j][0]
                y_matches = self.y_position[i][-1] == self.path_coordinates[j][1]
                if x_matches and y_matches:
                    self.x_interpolation.append(self.x_position[i])
                    self.y_interpolation.append(self.y_position[i])
                    self.theta_interpolation.append(self.theta_orientation[i])

        return self.path_coordinates
    
    def is_goal_reached(self):
        # Check if the tree reached the goal
        goal_position, goal_orientation = self.x_goal[:2], self.x_goal[2]
        last_position_x, last_position_y = self.last_position[0], self.last_position[1]
        goal_position_x, goal_position_y = goal_position[0], goal_position[1]
        last_orientation = math.radians(self.last_orientation)

        # Provide allowed offsets ("Good enough")
        upper_offset_x = goal_position_x+self.position_boundary
        lower_offset_x = goal_position_x-self.position_boundary
        upper_offset_y = goal_position_y+self.position_boundary
        lower_offset_y = goal_position_y-self.position_boundary
        upper_offset_theta = goal_orientation+self.orientation_boundary
        lower_offset_theta = goal_orientation-self.orientation_boundary
        
        # Angle wrapping
        if last_orientation >= upper_offset_theta:
            last_orientation = -2*math.pi + last_orientation
        elif last_orientation <= lower_offset_theta:
            last_orientation = 2*math.pi + last_orientation

        # Checking allowed position/orientation regions
        upper_condition_x = last_position_x <= upper_offset_x
        lower_condition_x = last_position_x >= lower_offset_x
        upper_condition_y = last_position_y <= upper_offset_y
        lower_condition_y = last_position_y >= lower_offset_y
        upper_condition_theta = last_orientation <= upper_offset_theta
        lower_condition_theta = last_orientation >= lower_offset_theta

        position_conditions = upper_condition_x and lower_condition_x and upper_condition_y and lower_condition_y
        orientation_conditions = upper_condition_theta and lower_condition_theta

        if position_conditions and orientation_conditions:
            return True
    
        return False
    
    def store_controls(self, controls):
        # Stores the controls
        self.controls.append(controls)

    def draw_initial_node(self, map):
        # Draws the x_init node on the map
        pygame.draw.circle(surface=map, color=self.BLUE, center=self.x_init[:2], radius=3)
    
    def draw_goal_node(self, map):
        # Draw the x_goal node on the map
        pygame.draw.circle(surface=map, color=self.RED, center=self.x_goal[:2], radius=3)

    def draw_random_node(self, map):
        # Draw the x_rand node on the map
        pygame.draw.circle(surface=map, color=self.GREEN, center=self.x_rand[:2], radius=3)

    def draw_new_node(self, map):
        # Draw the x_new node on the map\
        for i in range(len(self.u_news)):
            pygame.draw.circle(surface=map, color=self.BROWN, center=self.u_news[i], radius=3)
    
    def draw_initial_robot_configuration(self, robot_img, environment):
        # Draw the robot's x_init config 
        position = self.x_init[:2]
        orientation = math.degrees(self.x_init[2])

        self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

    def draw_goal_robot_configuration(self, robot_img, environment):
        # Draw the robot's x_goal configuration
        position = self.x_goal[:2]
        orientation = math.degrees(self.x_init[2])

        self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

    def draw_random_robot_configuration(self, robot_img, environment):
        # Draw the robot's x_rand configuration
        position = self.x_rand[:2]
        orientation = math.degrees(self.x_rand[2])
        
        # Needs to be rect to check for collision
        rect = self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

        return rect;

    def draw_new_robot_configuration(self, robot_img, environment):
        # Draw a robot's configuration at every new node
        for i in range(len(self.u_news)):
            position = self.u_news[i]
            orientation = self.theta_news[i]
            self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

    def draw_path_to_goal(self, map):
        # Draw the path from x_init node to x_goal node
        for i in range(len(self.path_coordinates)-1):
            pygame.draw.circle(surface=map, color=self.BLACK, center=self.path_coordinates[i],radius=3)

    def draw_interpolation(self, robot, environment):
        # Shows the robot driving from starting pos to goal
        x_interpolation = self.x_interpolation[1:] # x interpolation holds path coordinates
        y_interpolation = self.y_interpolation[1:]
        theta_interpolation = self.theta_interpolation[1:]

        for x,y,theta in zip(x_interpolation, y_interpolation, theta_interpolation):
            for interpolation, (i,j,k) in enumerate(zip(x,y,theta)):
                if self.obstacles != []:
                    environment.draw_obstacles()

                # Frame rate of the robot drawing, in a way        
                if interpolation%self.interpolation_factor == 0:
                    self.draw_robot_configuration(image=robot.img, position=(i,j), orientation=k, environment=environment)
        
        # Draw the initial and goal points
        self.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
        self.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)

        pygame.display.update()
        pygame.time.delay(5000)
        environment.map.fill(self.WHITE)

    def draw_trajectory(self, robot, environment):
        # Draw the interpolation from the init config to the goal config
        x_interpolation = self.x_interpolation[1:]
        y_interpolation = self.y_interpolation[1:]
        theta_interpolation = self.theta_interpolation[1:]

        for x,y,theta in zip(x_interpolation, y_interpolation, theta_interpolation):
            for interpolation, (i,j,k) in enumerate(zip(x,y,theta)):
                if self.obstacles != []:
                    environment.draw_obstacles()
                
                self.draw_initial_robot_configuration(robot.init, environment=environment)
                self.draw_initial_goal_configuration(robot.goal, environment=environment)
                self.draw_robot_configuration(image=robot.img, position=(i,j), orientation=k, environment=environment)

                # Refresh the screen
                pygame.display.update()
                pygame.time.delay(3)
                environment.map.fill(self.WHITE)
    
    def draw_robot_configuration(self, image, position, orientation, environment):
        # Create the translation & rotation animation
        rotated = pygame.transform.rotozoom(surface=image, angle=orientation, scale=1)
        rect = rotated.get_rect(center=position)
        environment.map.blit(source=rotated, dest=rect)

        return rect