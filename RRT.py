import pygame
import random
import math
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

        self.v = 10; # Linear velocity
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
        
    # No longer used in node-sampling version
    # def generate_parents(self, values, parent):
    #     """
    #     Generates a list of parents given value of node and nearest neighbor
    #     PARAMETERS
    #         values : list (Node IDs)
    #         parent : list (Previous collection of parent nodes until the new is processed)
    #     RETURNS
    #         list (Ordered collection of the parents)
    #     """

    #     parent_value = values[self.min_distance] 
    #     parent_index = len(parent) 
    #     parent.insert(parent_index, parent_value) # Add parent to the end of parent list

    #     if self.is_goal_reached():
    #         # If goal is reached, add a final index +1 
    #         parent.insert(parent_index+1, values[-1]+1)

    #     return parent
    
    def rewire_node(self, nodes, search_radius, delta):
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
        new_node_index = len(nodes)-1
        new_node = nodes[new_node_index]
        best_cost = new_node.cost
        best_point = None

        for i in range(len(nodes)):
            if i == new_node_index: 
                continue # Do not consider the new node as a parent to itself
            candidate_node = nodes[i]
            if candidate_node.trajectory_samples is None:
                continue # Root has no trajectory to search
            xs, ys, thetas = candidate_node.trajectory_samples
            for j in range(len(xs)): # Check each individual sample
                distance = math.hypot(xs[j] - new_node.position[0], ys[j] - new_node.position[1])
                if distance <= search_radius:
                    cost_to_candidate = nodes[candidate_node.parent].cost + (j+1)*delta*self.v # Cost to reach the candidate from the root
                    candidate_cost = cost_to_candidate + distance
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_point = (i,j)
        
        if best_point is None: # A better route was NOT found
            return 
        
        # Record old parent and old cost before reassignment (DEBUGGING)
        old_parent = new_node.parent
        old_cost = new_node.cost

        edge_index, point_index = best_point
        edge_node = nodes[edge_index]
        xs, ys, thetas = edge_node.trajectory_samples
        last_point_index = len(xs)-1
        
        if point_index == last_point_index: # Check if candidate is already a node
            new_node.parent = edge_index
            new_node.cost = best_cost

            # Add init and final path point (NEW)
            new_node.trajectory_samples = (
                [xs[point_index], new_node.position[0]], 
                [ys[point_index], new_node.position[1]],
                [thetas[point_index], new_node.orientation]
            )
        else: 
            # Candidate is mid-edge. Split edge_node's edge, promoting a new node M
            point_x, point_y, point_theta = xs[point_index], ys[point_index], thetas[point_index]

            M = Node(
                position=(point_x, point_y),
                orientation=point_theta,
                parent=edge_node.parent,
                cost=nodes[edge_node.parent].cost + (point_index+1)*delta*self.v,
                trajectory_samples=(xs[:point_index+1], ys[:point_index+1], thetas[:point_index+1])
            )
            nodes.append(M)
            m_index = len(nodes)-1

            # edge_node now starts from M instead of its old parent (edge splitting here)
            edge_node.parent = m_index
            edge_node.trajectory_samples = (xs[point_index+1:], ys[point_index+1:], thetas[point_index+1:])
            edge_node.cost = M.cost + len(edge_node.trajectory_samples[0])*delta*self.v

            # Rewire new_node to promoted M
            new_node.parent = m_index
            new_node.cost = best_cost

            # Add init and final path point (NEW)
            new_node.trajectory_samples = (
                [point_x, new_node.position[0]],
                [point_y, new_node.position[1]],
                [point_theta, new_node.orientation]
            )
        
        print(f"[REWIRE] node {new_node_index}: parent {old_parent} --> {new_node.parent}, cost {old_cost:.3f} --> {new_node.cost:.3f}")

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
                ax.plot(self.x_final_path[i], self.y_final_path[i], self.theta_final_path[i], color='brown', marker='.', markersize=2)
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

        