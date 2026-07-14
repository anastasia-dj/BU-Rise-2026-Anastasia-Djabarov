import pygame
import random
import math
import csv
import os 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from node import Node



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
        
        self.candidates_in_radius = 0
        self.candidates_blocked_by_heading = 0
        self.v_magnitude = 10

        self.robot_position_history = []
        self.robot_orientation_history = []
        self.controls = []
        self.is_goal_found = False
        self.iteration = 0

        self.robot_position_history.append(self.x_init[:2]) # List accumulating history
        self.robot_orientation_history.append(math.degrees(self.x_init[2]))
        
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
        self.theta_final_path.append(math.degrees(self.x_init[2]))

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
        self.LIGHT_BROWN = (222, 184, 135)

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
    
    def simulate(self, robot, environment, obstacles, random_parent_index):
        """
        It simulates the robot's configuration from the initial configuration.
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

        print(f"[DEBUG] simulate() starting from {self.last_position} with chosen parent tree[{random_parent_index}]")

        if random.uniform(0,100) < self.anti_drive_bias:
            if self.anti_drive_mode == 'stop':
                self.v = 0
            elif self.anti_drive_mode == 'reverse':
                self.v = -self.v_magnitude
            else: 
                self.v = random.choice([0, -self.v_magnitude]) # 'both
        else:
            self.v = self.v_magnitude # Linear velocity
        self.w = random.uniform(-0.5, 0.5) # Angular velocity
        self.edge_time_duration = random.uniform(0.5, 4)
        self.controls_vw = [self.v, self.w] 
        
        # Check if collision free
        x_final, y_final, theta_final = robot.closed_form_step(x_init, y_init, theta_init, self.v, self.w, self.edge_time_duration)
        theta_final_deg = math.degrees(theta_final) # Pygame wants degrees within update_visual
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
        
        # Check if the POINTS are collision free
        for x,y,theta in zip(xs_sample, ys_sample, thetas_deg):
            robot.update_visual(x,y,theta)
            if not self.is_free(configuration=robot.rect, obstacles=obstacles):
                robot.update_visual(self.last_position[0], self.last_position[1], self.last_orientation)
                return None
        
        # Store the edge's last controls
        self.store_controls(controls=self.controls_vw)

        # Play the edge's motion back if requested
        if self.show_tree_building:
            for x, y, theta in zip(xs_sample, ys_sample, thetas_deg):
                robot.update_visual(x, y, theta)

                if self.obstacles != []:
                    environment.draw_obstacles()

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

        return (x_final, y_final), theta_final_deg, (xs_sample, ys_sample, thetas_deg)

    def new_state(self, random_parent_index, robot, environment, nodes, obstacles=None):
        """
        Given a random node, propagates forward with a random control input (Constant v=1, randomized w and t)
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

        self.last_position = nodes[random_parent_index].position
        self.last_orientation = nodes[random_parent_index].orientation

        # Simulation fwd in time
        simulation = self.simulate(robot=robot, environment=environment, obstacles=obstacles, random_parent_index=random_parent_index) # Returns x_final, y_final, theta_final_deg

        if simulation is not None:
            self.iteration += 1 # One tree iteration is done

            # Save new configurations and trajectory sample
            self.x_new, self.theta_new, trajectory_samples = simulation
            
            # Append new configs to a list
            self.x_news.append(self.x_new)
            self.theta_news.append(self.theta_new)

            if self.is_goal_reached(self.x_new, self.theta_new):
                self.goal_configuration = self.number_of_nodes
                self.is_goal_found = True

            return self.x_new, self.theta_new, trajectory_samples
    
    def rewire_node(self, nodes, search_radius, delta, heading_boundary, robot):
        """
        Within a search radius, it finds the position (not strictly a node) with the lowest cost and copies the position's control input and cost to the new node.
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
        new_node_index = len(nodes)-1
        new_node = nodes[new_node_index]
        best_cost = new_node.cost
        best_choice = None # (candidate_node_index, sample_index_j, candidate_v, candidate_w)

        for i in range(len(nodes)):
            if i == new_node_index: 
                continue # Do not consider the new node as a parent to itself
            candidate_node = nodes[i]
            if candidate_node.trajectory_samples is None or candidate_node.control_input is None:
                continue # Root has no trajectory/control_input to search
            xs, ys, thetas = candidate_node.trajectory_samples
            candidate_v, candidate_w, candidate_t = candidate_node.control_input
            for j in range(len(xs)): # Check each individual sample
                distance = math.hypot(xs[j] - new_node.position[0], ys[j] - new_node.position[1])
                if distance <= search_radius:
                    self.candidates_in_radius += 1
                    if self.angle_difference(thetas[j], new_node.orientation) <= heading_boundary:
                        # Cost as if new_node were sample j of candidate_node's edge
                        candidate_cost = nodes[candidate_node.parent].cost + (j+1)*delta*abs(candidate_v) 
                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_choice = (i,j, candidate_v, candidate_w)
                    else:
                        self.candidates_blocked_by_heading +=1 ;
        
        if best_choice is None: # A better route was NOT found
            return 
        
        # Record old parent and old cost before reassignment (DEBUGGING)
        old_parent = new_node.parent
        old_cost = new_node.cost

        candidate_index, j, candidate_v, candidate_w = best_choice
        candidate_node = nodes[candidate_index]
        candidate_t = (j+1)*delta # Time to the position

        new_node.parent = candidate_node.parent # Make new_node the sibling of the parent
        new_node.control_input = (candidate_v, candidate_w, candidate_t)
        new_node.cost = best_cost

        assert new_node.cost >= nodes[new_node.parent].cost, "[ERROR] Post-wire cost inconsistency with child cost ({new_node.cost}) >= parent cost ({nodes[new_node.parent].cost})"

        parent_node = nodes[new_node.parent]
        xs,ys,thetas_rad = robot.generate_arc_samples(parent_node.position[0], parent_node.position[1], math.radians(parent_node.orientation), candidate_v, candidate_w, candidate_t, delta)
        new_node.trajectory_samples = (xs,ys,[math.degrees(t) for t in thetas_rad])

        print(f"[REWIRE] node {new_node_index}: parent {old_parent} --> {new_node.parent}, cost {old_cost:.3f} --> {new_node.cost:.3f}")

    # NOTE: May be retired soon. 
    def interpolate_connection(self, start_x, start_y, start_theta, end_x, end_y, end_theta, delta):
        """
        Interpolates a connection assuming a straight line between start_position and end_position. Purely for visualization.
        PARAMETERS
            start_x/y/theta : float (position: px/deg)
            end_x/y/theta : float (position: px/deg)
            delta : float
        RETURNS xs, ys, thetas
        """
        
        distance = math.hypot(end_x - start_x, end_y - start_y)
        spacing = self.v_magnitude*delta # Px distance between points
        steps = max(1, round(distance/spacing))
        theta_difference = ((end_theta - start_theta) + 180) % 360 - 180 # Shortest angular path

        xs,ys,thetas = [],[],[]

        for i in range(1, steps+1):
            frac = i/steps
            xs.append(start_x + frac * (end_x - start_x))
            ys.append(start_y + frac * (end_y - start_y))
            thetas.append(start_theta + frac * theta_difference)

        # Make sure the final sample matches the endpoint node
        xs[-1],ys[-1],thetas[-1] = end_x, end_y, end_theta

        return xs,ys,thetas


    def store_configuration(self, position, orientation):
        """
        Stores the current robot configuration to an ongoing list for later use
        PARAMETERS
            position : tuple (X,Y in C-space)
            orientation : tuple (theta in C-space)
        RETURNS
            None
        """
        self.robot_position_history.append(position)
        self.robot_orientation_history.append(orientation)

    def path_to_goal(self):
        """
        Given x_goal node, it searches next parent until it reaches the x_init node
        PARAMETERS
            none
        RETURNS
            none
        In other words, it finds the path from x_init to x_goal.
        """
        self.path = [self.goal_configuration]
        current_parent = self.nodes[self.goal_configuration].parent
        while current_parent != 0:
            # Append the parent of parent and update config
            self.path.append(current_parent)
            current_parent = self.nodes[current_parent].parent
        # Append parent 0 (x_init)
        self.path.append(0)

    def get_path_coordinates(self):
        """
        Collects correspondant coordinates
        PARAMETERS
            None
        RETURNS
            Coordinates for successful path
        """

        # Building path_coordinates (x, y)
        self.path_coordinates = []
        for index in self.path:
            node_object = self.nodes[index]
            position = node_object.position
            self.path_coordinates.append(position)
        
        for index in reversed(self.path[:-1]): # Go through all path backwards (root --> goal) but skip root
            node_object = self.nodes[index]
            if node_object.trajectory_samples is not None:
                xs, ys, thetas = node_object.trajectory_samples
                self.x_final_path.append(xs)
                self.y_final_path.append(ys)
                self.theta_final_path.append(thetas)
        
        print(f"[DEBUG] path_coordinates: {len(self.path_coordinates)}, x_final_path: {len(self.x_final_path)}")
        return self.path_coordinates

    def is_goal_reached(self, position, orientation_deg):
        """
        Check if the tree reached the goal
        PARAMETERS
            position : list
            orientation_deg : float (deg)
        RETURNS 
            bool
        """

        goal_position, goal_orientation = self.x_goal[:2], self.x_goal[2]
        last_position_x, last_position_y = position[0], position[1]
        goal_position_x, goal_position_y = goal_position[0], goal_position[1]
        last_orientation = math.radians(orientation_deg)

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
        orientation = math.degrees(self.x_goal[2])

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

    def show_3d_map(self, nodes, init_point, goal_point):
        """Initialize figure in the 3D axis
            PARAMETERS
                init_point : list (px, px, theta)
            RETURNS
                None
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        branch_line = None

        if self.is_goal_found:
            # Graph successful points
            for i in range(len(self.x_final_path)):
                ax.plot(self.x_final_path[i], self.y_final_path[i], self.theta_final_path[i], color='red', marker='.', markersize=2)

            # Build parent -> child branches as one line, with None to break segments
            branch_x, branch_y, branch_theta = [], [], []
            for i, n in enumerate(nodes):
                if i == 0:
                    continue # The root's parent is itself so skip this 
                parent = nodes[n.parent]
                branch_x += [n.position[0], parent.position[0], float('nan')]
                branch_y += [n.position[1], parent.position[1], float('nan')]
                branch_theta += [n.orientation, parent.orientation, float('nan')]
            branch_line, = ax.plot(branch_x, branch_y, branch_theta, alpha=0.7, color='brown', linewidth=0.5, visible=False)

            title = 'kRRT Final Path'
        else:
            # Plot all the nodes
            xs = [n.position[0] for n in nodes]
            ys = [n.position[1] for n in nodes]
            thetas = [n.orientation for n in nodes]
            ax.scatter(xs, ys, thetas, color='brown', s=5)
            title = 'kRRT Tree (Goal Not Yet Reached)'

            # Build parent -> child branches as one line, with None to break segments
            branch_x, branch_y, branch_theta = [], [], []
            for i, n in enumerate(nodes):
                if i == 0:
                    continue # The root's parent is itself so skip this 
                parent = nodes[n.parent]
                branch_x += [n.position[0], parent.position[0], float('nan')]
                branch_y += [n.position[1], parent.position[1], float('nan')]
                branch_theta += [n.orientation, parent.orientation, float('nan')]
            branch_line, = ax.plot(branch_x, branch_y, branch_theta, color='brown', linewidth=0.5, visible=False)
            
        # Initial point
        ax.scatter(init_point[0], init_point[1], math.degrees(init_point[2]), color='red', s=20)
        ax.text(init_point[0], init_point[1], init_point[2], 'Initial Point', ha='left', va='center', fontsize=12)
        
        # Final point
        ax.scatter(goal_point[0], goal_point[1], math.degrees(goal_point[2]), color='green', s=20)
        ax.text(goal_point[0], goal_point[1], math.degrees(goal_point[2]), 'Goal Point', ha='left', va='center', fontsize=12)

        # Label axes and window
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.set_zlabel('Theta (deg)')
        fig.canvas.manager.set_window_title(title)


        # View switching
        default_elevation, default_azimuth = ax.elev, ax.azim

        def on_key(event):
            if event.key == 't':
                ax.view_init(elev=90, azim=-90)
            elif event.key == 'r':
                ax.view_init(elev=default_elevation, azim=default_azimuth)
            elif event.key == 'b' and branch_line is not None:
                branch_line.set_visible(not branch_line.get_visible())
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    def angle_difference(self, theta_start, theta_end):
        #Smallest angular difference between two headings in degrees, accounting for wraparound at +-180. 
        #(e.g. 179 and -179 are 2 degrees apart, not 358).
        
        diff = abs(theta_end - theta_start) % 360
        return min(diff, 360-diff)

    def draw_final_all_branches_2d(self, map, nodes, color, width=1):
        # Draws all the branches faded.
        for n in nodes:
            if n.trajectory_samples is not None:
                xs,ys,thetas = n.trajectory_samples
                points = list(zip(xs, ys))
                if len(points) >= 2:
                    pygame.draw.lines(surface=map, color=color, closed=False, points=points, width=width)

    def draw_final_path_2d(self, map, color, width=1):
        # Draws the succesful branch solid
        for i in range(1, len(self.x_final_path)):
            points = list(zip(self.x_final_path[i],self.y_final_path[i]))
            if len(points) >= 2:
                    pygame.draw.lines(surface=map, color=color, closed=False, points=points, width=width)

    def draw_final_backdrop_2d(self, environment, nodes):
        self.draw_final_all_branches_2d(map=environment.map, nodes=nodes, color=environment.LIGHT_BROWN, width=1)
        if self.is_goal_found:
            self.draw_final_path_2d(map=environment.map, color=environment.RED, width=1)

    # NOTE: RETIRED 
    def path_from_node_to_root(self, start_index):
        """
        Given any node, this function walks back the .parent references until it reaches the root. (start_index --> root) 
        This is essentially a generalized version of path_to_goal().
        PARAMETERS
            self_index : int (Index of the starting node)
        RETURN
            list of node indices, from start_index --> root
        """

        path = [start_index]
        current_parent = self.nodes[start_index].parent
        while current_parent != 0:
            path.append(current_parent)
            current_parent = self.nodes[current_parent].parent
        path.append(0)
        return path

    def drive_backwards(self, start_index, robot, nodes):
        """
        Given any node, this function calculates the reversed control_input and returns the position which the robot
        ended up in after following the reversed control_input. 
        PARAMETERS
            start_index : int (Index of starting node)
            robot : Robot object
            nodes : Node 
        RETURNS
            x, y, theta(deg) : float
            total_shot_distance : float (Total distance shot off from the goal)
            n_shot_segments : int (Number of segments shot)
        """
        path_indices = self.path_from_node_to_root(start_index) # List of start_index -> root
        x = nodes[start_index].position[0]
        y = nodes[start_index].position[1]
        theta = math.radians(nodes[start_index].orientation)
        
        total_shot_distance = 0.0
        n_shot_segments = 0
        shot_misses = []

        for node_index in path_indices:
            if node_index == 0:
                break # Root does not have a control input to reverse
            if nodes[node_index].control_input is not None:
                v,w,t = nodes[node_index].control_input
                x,y,theta = robot.closed_form_step(x,y,theta,-v,-w,t)
            else: 
                parent_index = nodes[node_index].parent
                target_x,target_y = nodes[parent_index].position
                target_theta_deg = nodes[parent_index].orientation
                (x,y,theta), shot_distance = self.shoot_toward_target(robot,x,y,theta,target_x,target_y, target_theta_deg)
                print(f"[SHOT] target=({target_x:.1f},{target_y:.1f}), landed at ({x:.1f},{y:.1f}), miss={shot_distance:.2f}px")
                total_shot_distance += shot_distance
                n_shot_segments +=1
                shot_misses.append(shot_distance)
        
        first_shot_miss = shot_misses[0] if shot_misses else None
        last_shot_miss = shot_misses[-1] if shot_misses else None
        max_shot_miss = max(shot_misses) if shot_misses else None

        return x, y, math.degrees(theta), total_shot_distance, n_shot_segments, first_shot_miss, last_shot_miss, max_shot_miss
        
    # NOTE: WILL BE RETIRED 
    def shoot_toward_target(self, robot, x0, y0, theta0, target_x, target_y, target_theta_deg, n_shots = 10000):
        """
        When given an initial point and target point, this function loops random control inputs for a fixed amount of trials
        and selects the end point which lands closest to the target.
        PARAMETERS
            robot : Robot object
            x0, y0, theta0 : float (Initial position)
            target_x, target_y : float (Goal position)
            n_shots : int (Number of trials)
        RETURNS
            best_configuration : tuple (x1, y1, theta1) of closest ending position
            best_distance : The best distance between the actual end point and target point
        """
        
        best_distance = None
        best_configuration = None
        for _ in range(n_shots):
            v = random.choice([self.v_magnitude, -self.v_magnitude])
            w = random.uniform(-0.5, 0.5)
            t = random.uniform(0.5, 4)

            x1,y1,theta1 = robot.closed_form_step(x0, y0, theta0, v, w, t)
            position_distance = math.hypot(x1-target_x, y1-target_y)
            heading_distance = self.angle_difference(math.degrees(theta1), target_theta_deg)
            distance = position_distance + self.shot_theta_weight * heading_distance
            if best_distance is None or distance < best_distance:
                best_distance, best_configuration = distance, (x1, y1, theta1)
        
        return best_configuration, best_distance
    

    
    def log_results(self, file_name, heading_boundary, search_radius, run_number, avg_position_drift, avg_heading_drift, n_shot_segments, cost, avg_first_shot_miss, avg_last_shot_miss, avg_max_shot_miss, n_multi_shot_trials):
        """
        Append a run's worth of data into a csv file.
        PARAMETERS:
            file_name : str (Name of CSV file)
            heading_boundary : int (Current heading boundary being tested)
            run_number : int (Current run of heading_boundary (NOT inner iteration of reversibility))
            final_drift : float (The final drift average from the goal)
            max_theta_jump : float (The max theta jump)
            average_theta_jump : float (The average theta jump)
            cost : float (The cost between the initial and goal positions)
        """
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["run_number", "heading_boundary", "search_radius", "avg_position_drift", "avg_heading_drift", "avg_n_shot_segments", "cost", "avg_first_shot_miss", "avg_last_shot_miss", "avg_max_shot_miss", "n_multi_shot_trials"])
                file_exists = True
            writer.writerow([run_number, heading_boundary, search_radius, avg_position_drift, avg_heading_drift, n_shot_segments, cost, avg_first_shot_miss, avg_last_shot_miss, avg_max_shot_miss, n_multi_shot_trials])

