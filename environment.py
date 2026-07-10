import pygame

class Environment():
    """
    Class of the map the robot will be moving around within.

    dimensions : tuple (X and Y window dimensions)
    show_tree_building : bool (Display tree growth, Y/N)
    has_obstacles : bool (Display obstacles, Y/N)
    """

    def __init__ (self, dimensions, show_tree_building, has_obstacles):
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

        self.has_obstacles = has_obstacles
        self.trail_set = [] # Holds tuple positions of one current ongoing trail
        self.trail_sets = [] # Holds the collection of multiple completed trail sets
        self.obstacles = []

        # Map dimensions
        self.WIDTH, self.HEIGHT = dimensions

        # Window settings
        self.FPS = 120
        pygame.display.set_caption("Kinodynamic RRT Car")

        if show_tree_building:
            self.show_screen()
        else:
            self.hide_screen()

    def trail(self, position, draw_trail):
        # Draws a robot's trail. A line is drawn between two different immediate positions.
        # Process is repeated until robot stops. 
        # position : tuple (x and y)
        # draw_trail : bool (Show the trail Y/N)

        for i in range(len(self.trail_set)-1):
            trail_set_start = self.trail_set[i][0], self.trail_set[i][1] 
            trail_set_end = self.trail_set[i+1][0], self.trail_set[i+1][1]
            if draw_trail:
                pygame.draw.line(surface=self.map, color=self.BROWN, start_pos=trail_set_start, end_pos=trail_set_end)

        self.trail_set.append(position)

    def empty_trails(self):
        # Empty the list so it doesn't create a straight line between old and new trail
        self.trail_set = []

    def store_trails(self):
        # Store the trails 
        self.trail_sets.append(self.trail_set)

    #TODO: Possibly, make this only make the main trail red
    def draw_trajectory_trail(self):
        # Makes all the trails red (As of right now)
        for trail in self.trail_sets:
            for i in range(len(trail)-1):
                trail_set_start = trail[i][0], trail[i][1]
                trail_set_end = trail[i+1][0], trail[i+1][1]
                pygame.draw.line(surface=self.map, color=self.RED, start_pos=trail_set_start, end_pos=trail_set_end)


    #TODO: Make box obstacle
    def make_obstacles_T(self, initial_point):
        # Makes a T shaped obstacle given an init point
        # initial_point : tuple (X and Y coordinates)
        # Returns a list (A collection of sides composing the T obstacle)

        x, y = initial_point[0], initial_point[1]
        width, height = 150, 50

        side1 = pygame.Rect(x, y, width, height)
        side2 = pygame.Rect((x+height//2) - width//2, y, width, height)
        
        obstacle = [side1, side2]
        return obstacle
    
    def make_obstacles_L(self, initial_point):
        # Makes a L shaped obstacle given an init point
        # initial_point : tuple (X and Y coordinates)
        # Returns a list (A collection of sides composing the T obstacle)

        x, y = initial_point[0], initial_point[1]
        width, height = 50, 150

        side1 = pygame.Rect(x, y, width, height)
        side2 = pygame.Rect(x, y+height-width, height, width)
        
        obstacle = [side1, side2]
        return obstacle
    
    def make_obstacles(self):
        # Generate/Combine the obstacles 
        obstacle1 = self.make_obstacles_T(initial_point=(350,300))
        obstacle2 = self.make_obstacles_L(initial_point=(150,20))

        self.obstacles.append(obstacle1)
        self.obstacles.append(obstacle2)

        return self.obstacles
    
    def draw_obstacles(self):
        # Draw each side of the obstacles
        obstacles = []

        for obstacle in self.obstacles:
            for side in obstacle:
                pygame.draw.rect(surface=self.map, color=self.GRAY, rect=side)
                obstacles.append(side)
        
        return obstacles
    
    def show_screen(self):
        # Show screen by setting display flag to SHOWN
        self.map = pygame.display.set_mode(size=(self.WIDTH, self.HEIGHT), flags=pygame.SHOWN)
        self.map.fill(self.WHITE)

        if self.has_obstacles:
            self.draw_obstacles()
        
    def hide_screen(self):
        # Hides screen by setting display flag to HIDDEN
        self.map = pygame.display.set_mode(size=(self.WIDTH, self.HEIGHT), flags=pygame.HIDDEN)
        self.map.fill(self.WHITE)

    def clear_screen(self):
        # Clears the screen
        self.map.fill(self.WHITE)
        