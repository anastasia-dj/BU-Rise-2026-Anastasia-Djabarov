import pygame
import random
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Graph():
    """
    Class for RRT

    start : tuple (init x, y, theta)
    goal : tuple (final x, y, theta)
    map_dimensions : tuple (width and height)
    """

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
        self.last_orientation = self.x_init[2] 

        # Present position and orientation
        self.x_position = []
        self.y_position = []
        self.theta_orientation = []

        # Holds path coordinates for later animation (i.e. in draw_interpolation() and draw_trajectory())
        self.x_final_path = []
        self.y_final_path = []
        self.theta_final_path = []

        self.interpolation_factor = 10

        self.x_final_path.append(self.x_init[0])
        self.y_final_path.append(self.x_init[1])
        self.theta_final_path.append(self.x_init[2])

        # Store added states (i.e. multiple x_new)
        self.x_news = []
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

    def is_free(self, configuration, obstacles):
        """
        Check if a config will collide with an obstacle
        PARAMETERS:
            configuration : pygame.Rect
            obstacles : pygame.Rect
        RETURNS:
            bool (If configuration is free of obstacles)
            """

        for obstacle in obstacles:
            if obstacle.colliderect(configuration):
                return False
        return True

    def generate_random_node(self, bias):
        """
        Generates a random configuratin
        PARAMETERS:
            bias : int (Goal bias percentage)
        RETURNS: 
            tuple : coordinate of random node
        """

        #Sample a robot config in (x, y, theta)
        self.x_rand = int(random.uniform(0, self.WIDTH)), int(random.uniform(0, self.HEIGHT)), math.radians(random.uniform(0, 360))

        # Likelihood of goal bias 
        bias_percentage = 10 - bias//10 if bias != 100 else 1

        if self.iteration%bias_percentage == 0:
            self.x_rand = self.x_goal
        
        return self.x_rand
    
    def metric(self, p1, p2):
        """
        Definition of the metric for a car (C = R^2 x S)
        In other words: How far are p1 and p2?
        PARAMETERS
            p1 : int (Start configuration)
            p2 : int (End configuration)
        RETURNS
            float (R^2 x S metric)
        """

        # Angle wraparound
        theta = self.x_rand[2] #in rad
        alpha = min(abs(self.last_orientation - theta), 2*math.pi - abs(self.last_orientation - theta))
        
        # Metric simplification through Euclidean formula (c^2 = a^2 + b^2)
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + alpha**2)
    
    def nearest_neighbor(self, tree, x_rand):
        """
        Returns the index of the nearest neighbor to randomly generated config (x_rand)
        PARAMETERS
            tree : list (Tree containing ALL configurations)
            x_rand : tuple (x, y, and theta of random node generated)
        RETURNS
            tuple (Nearest node to random node generated)
        """
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
        """
        It simulates the robot configuration from u (control input)
        PARAMETERS
            robot : robot object (Where all control inputs are simulated)
            event : Event (All events happening on screen)
            environment : environment object (Environment to draw trails & simulations fwd)
        RETURNS
            list : Collection of positions the robot has passed through
        """

        # Changes: This function used to be called once per frame to add a step under a Euler assumption.
        # Given the simplicity of the system and its geometry, a closed form assumption can be used instead.
        # The new version of simulate() runs once per edge.

        x_init, y_init = self.last_position
        theta_init = math.radians(self.last_orientation)

        self.v = 1; # Linear velocity
        self.w = random.uniform(-0.5, 0.5) # Angular velocity
        self.edge_time_duration = random.uniform(0.5, 4)
        self.controls_vw = [self.v, self.w] 
        
        # Check if collision free
        x_final, y_final, theta_final = robot.closed_form_step(x_init, y_init, theta_init, self.v, self.w, self.edge_time_duration)
        theta_final_deg = math.degrees(theta_final) # pygame wants degrees within update_visual
        robot.update_visual(x_final, y_final, theta_final_deg)
        collision_free = self.is_free(configuration=robot.rect, obstacles=obstacles)

        if not collision_free:
            # Put the robot's visual back to where it still is and drop the path
            robot.update_visual(self.last_position[0], self.last_position[1], self.last_orientation)
            return None
        
        # Now that the path is confirmed to be free, build the full trajectory
        xs_sample, ys_sample, thetas_rad = robot.generate_arc_samples(x_init, y_init, theta_init, self.v, self.w, self.edge_time_duration, self.delta)
        thetas_deg = []
        for t in thetas_rad:
            degrees_theta_value = math.degrees(t)
            thetas_deg.append(degrees_theta_value)
        
        # Store the edge's last configuration and controls
        self.store_configuration(position=(robot.x, robot.y), orientation=robot.theta)
        self.store_controls(controls=self.controls_vw)

        # Store the edge's trajectory
        self.x_position.append(xs_sample)
        self.y_position.append(ys_sample)
        self.theta_orientation.append(thetas_deg)

        # Play the edge's motion back if requested
        if self.show_tree_building:
            for x, y, theta in zip(xs_sample, ys_sample, thetas_deg):
                robot.update_visual(x, y, theta)

                self.draw_initial_robot_configuration(robot_img=robot.init, environment=environment)
                self.draw_goal_robot_configuration(robot_img=robot.goal, environment=environment)
                environment.draw_trajectory_trail()
                environment.trail(position=(x,y), draw_trail=True)
                robot.draw(map=environment.map)
                pygame.display.update()
                pygame.time.delay(3)
                environment.map.fill(environment.WHITE)

            environment.store_trails()
            environment.empty_trails()

        # TODO: Check
        return self.robot_last_position
            
    def new_state(self, x_rand, robot, event, environment, obstacles=None):
        """
        Propagates a small step towards a random node (Given fwd simulations, selects the more suitable to advance toward x_rand)
        PARAMETERS
            x_rand : tuple (Coordinates of random generated node)
            robot: robot object (Robot w/ simulated control inputs)
            event : Event (All events happening on screen) (Unused?)
        RETURNS
            tuple (Best position towards x_rand)
            float (Best orientation towards x_rand)
        """
        if obstacles is None:
            obstacles = [] 
        self.obstacles = obstacles

        # Simulation fwd in time
        simulation = self.simulate(robot=robot, event=event, environment=environment, obstacles=obstacles) # Returns self.robot_last_position (ENTIRE tree history)

        if simulation is not None:
            self.iteration += 1 # One tree iteration is done

            # Robot pos & orientation with min distance to x_rand
            self.x_new = self.nearest_neighbor(simulation, x_rand) # New config
            self.theta_new = self.robot_last_orientation[self.min_distance] # Look at angle of nearest node

            # Append new configs to a list
            self.x_news.append(self.x_new)
            self.theta_news.append(self.theta_new)

            if self.is_goal_reached():
                self.goal_configuration = self.number_of_nodes-1
                self.is_goal_found = True
            
            #Store the configs as latest configs
            self.last_position = self.x_new
            self.last_orientation = self.theta_new
            robot.x, robot.y = self.last_position[:2]
            robot.theta = self.last_orientation

            return self.x_new, self.theta_new
        
    def generate_parents(self, values, parent):
        """
        Generates a list of parents given value of node and nearest neighbor
        PARAMETERS
            values : list (Node IDs)
            parent : list (Previous collection of parent nodes until the new is processed)
        RETURNS
            list (Ordered collection of the parents)
        """

        parent_value = values[self.min_distance] 
        parent_index = len(parent) 
        parent.insert(parent_index, parent_value) # Add parent to the end of parent list

        if self.is_goal_reached():
            # If goal is reached, add a final index +1 
            parent.insert(parent_index+1, values[-1]+1)

        return parent
    
    def rewire_node(self, tree, orientation_tree, parent, cost, search_radius, delta, x_new):
        """
        Within a search radius, it finds the position (not strictly a node) with the lowest cost and rewires the node.
        PARAMETERS
            tree : tree
            parent : list
            orientation_tree : list
            cost : float
            search_radius : float (take in arg)
            delta : float (take in arg)
            x_new : list with tuple for (x,y)
        RETURNS
            None
        """
        # Check costs
        best_cost = cost[-1]
        best_point = None

        for i in range(len(self.x_position)):
            node_index = i+1 # Consider that [0] = x_init, so edge i created node i+1 
            if node_index == len(tree)-1: # Do not consider the new node as a parent to itself
                continue
            for j in range(len(self.x_position[i])): # Check each individual sample
                x_candidate = [self.x_position[i][j], self.y_position[i][j]]
                new_x, new_y = x_new[0] # Unpack the tuple
                distance = math.hypot(x_candidate[0] - new_x, x_candidate[1] - new_y)
                if distance <= search_radius:
                    cost_to_candidate = cost[parent[node_index]] + (j+1)*delta # Cost to reach the candidate from the root
                    candidate_cost = cost_to_candidate + distance
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_point = (i,j)
        
            if best_point is not None: # A better route was found
                edge_index, point_index = best_point
                last_point_index = len(self.x_position[edge_index])-1
                
                if last_point_index == point_index: # Check if candidate is already a node
                    node_index = edge_index+1
                    parent[-1] = node_index
                    cost[-1] = best_cost
                else: # Candidate is not a node (Consider: Node A is being rewired, Node B is being promoted, Node C is Node B's parent)
                    # Save Node A's index first
                    original_new_node_index = len(tree)-1
                    print(f"len(tree)={len(tree)}, len(parent)={len(parent)}, original_new_node_index={original_new_node_index}")

                    # Pull out Node B's data
                    point_x = self.x_position[edge_index][point_index]
                    point_y = self.y_position[edge_index][point_index]
                    point_theta = self.theta_orientation[edge_index][point_index]

                    # Find the parent of the edge Node B is on (Node C)
                    edge_start_index = parent[edge_index+1] 

                    # Create Node B 
                    tree.append((point_x, point_y)) 
                    orientation_tree.append(point_theta)
                    parent.append(edge_start_index)
                    cost.append(cost[edge_start_index] + (point_index+1) * delta) # Append the cost  of B to the list

                    # Find Node B's index
                    new_node_index = len(tree)-1

                    # Rewire Node A to point to Node B
                    parent[original_new_node_index] = new_node_index
                    cost[original_new_node_index] = best_cost

                    print(f"Rewired node {original_new_node_index} to parent {parent[original_new_node_index]}, cost {best_cost:.3f}")

    def store_configuration(self, position, orientation):
        """
        Stores the current robot configuration to an ongoing list for later use
        PARAMETERS
            position : tuple (X,Y in C-space)
            orientation : tuple (theta in C-space)
        RETURNS
            None
        """
        self.robot_last_position.append(position)
        self.robot_last_orientation.append(orientation)

    def path_to_goal(self):
        """
        Given x_goal node, it searches next parent until it reaches the x_init node
        PARAMETERS
            none
        RETURNS
            none
        In other words, it finds the path from x_init to x_goal.
        """
        self.path = []
        self.path.append(self.goal_configuration)
        new_configuration = self.parent[self.goal_configuration]

        while new_configuration != 0:
            # Append the parent of parent and update config
            self.path.append(new_configuration)
            new_configuration = self.parent[new_configuration]

        # Append parent 0 (x_init)
        self.path.append(0)

    # TODO: Issue with path generations
    def get_path_coordinates(self):
        """
        Collects correspondant coordinates
        PARAMETERS
            None
        RETURNS
            Coordinates for successful path
        """

        self.path_coordinates = []

        for node in self.path:
            x, y = self.tree[node]
            self.path_coordinates.append((x,y))
        
        for i in range(len(self.x_position)):
            for j in range(len(self.path_coordinates)):
                x_match = self.x_position[i][-1] == self.path_coordinates[j][0]
                y_match = self.y_position[i][-1] == self.path_coordinates[j][1]
                if x_match and y_match:
                    self.x_final_path.append(self.x_position[i])
                    self.y_final_path.append(self.y_position[i])
                    self.theta_final_path.append(self.theta_orientation[i])
        
        print(f"path_coordinates: {len(self.path_coordinates)}, x_final_path: {len(self.x_final_path)}")
        return self.path_coordinates

    def is_goal_reached(self):
        """
        Check if the tree reached the goal
        PARAMETERS
            none
        RETURNS 
            bool
        """

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
        """
        Stores the controls
        PARAMETERS
            controls : list
        RETURNS
            None
        """
        self.controls.append(controls)

    def draw_initial_node(self, map):
        """
        Draws the x_init node on the map
        PARAMETER
            map : Environment object
        RETURN
            None  
        """
        pygame.draw.circle(surface=map, color=self.BLUE, center=self.x_init[:2], radius=3)
    
    def draw_goal_node(self, map):
        """
        Draw the x_goal node on the map
        PARAMETER
            map : Environment object
        RETURN
            None
        """
        pygame.draw.circle(surface=map, color=self.RED, center=self.x_goal[:2], radius=3)

    def draw_random_node(self, map):
        """
        Draw the x_rand node on the map
        PARAMETER
            map : Environment object
        RETURN
            None
        """
        pygame.draw.circle(surface=map, color=self.GREEN, center=self.x_rand[:2], radius=3)

    def draw_new_node(self, map):
        """
        Draw the x_new node on the map
        PARAMETER
            map : Environment object
        RETURN
            None
        """
        for i in range(len(self.x_news)):
            pygame.draw.circle(surface=map, color=self.BROWN, center=self.x_news[i], radius=3)
    
    def draw_initial_robot_configuration(self, robot_img, environment):
        """
        Draw the robot's x_init config 
        PARAMETER
            robot_img : String
            environment : Environment object
        RETURN
            None
        """
        position = self.x_init[:2]
        orientation = math.degrees(self.x_init[2])

        self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

    def draw_goal_robot_configuration(self, robot_img, environment):
        """
        Draw the robot's x_init config 
        PARAMETER
            map : Environment object
        RETURN
            None
        """
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
        for i in range(len(self.x_news)):
            position = self.x_news[i]
            orientation = self.theta_news[i]
            self.draw_robot_configuration(image=robot_img, position=position, orientation=orientation, environment=environment)

    def draw_path_to_goal(self, map):
        # Draw the path from x_init node to x_goal node
        for i in range(len(self.path_coordinates)-1):
            pygame.draw.circle(surface=map, color=self.BLACK, center=self.path_coordinates[i],radius=3)

    def draw_interpolation(self, robot, environment):
        # Shows the robot driving from starting pos to goal
        x_interpolation = self.x_final_path[1:] # x interpolation holds path coordinates
        y_interpolation = self.y_final_path[1:]
        theta_interpolation = self.theta_final_path[1:]

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
        x_interpolation = self.x_final_path[1:]
        y_interpolation = self.y_final_path[1:]
        theta_interpolation = self.theta_final_path[1:]

        for x,y,theta in zip(x_interpolation, y_interpolation, theta_interpolation):
            for interpolation, (i,j,k) in enumerate(zip(x,y,theta)):
                if self.obstacles != []:
                    environment.draw_obstacles()
                
                self.draw_initial_robot_configuration(robot.init, environment=environment)
                self.draw_goal_robot_configuration(robot.goal, environment=environment)
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

    def show_3d_map(self, init_point, goal_point):
        """Initialize figure in the 3D axis
            PARAMETERS
                init_point : list (px, px, theta)
            RETURNS
                None
        """
        print("Reached plotting now")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Graph points
        for i in range(len(self.x_final_path)):
            ax.scatter(self.x_final_path[i], self.y_final_path[i], self.theta_final_path[i], color='brown', s=5)
        
        # Initial point
        ax.scatter(init_point[0], init_point[1], init_point[2], color='red', s=20)
        ax.text(init_point[0], init_point[1], init_point[2], 'Initial Point', ha='left', va='center', fontsize=12)
        
        # Final point
        ax.scatter(goal_point[0], goal_point[1], goal_point[2], color='green', s=20)
        ax.text(goal_point[0], goal_point[1], goal_point[2], 'Goal Point', ha='left', va='center', fontsize=12)

        # Label axes and window
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Theta (deg)')
        fig.canvas.manager.set_window_title('kRRT Car-Like Robot Path')
        
        plt.show()

