import pygame
import math

class Robot():
    """
    The robot drawing, translation, position, orientation.

    start : tuple (x, y, theta)
    robot_img : str (list of paths for robot imgs - init, goal, robot)
    length : float (length b/w rear and front wheels)
    
    """

    def __init__(self, start, robot_img, length):
        self.x = start[0] # Note, values in PIXELS.
        self.y = start[1] # TODO: CHANGE PIXELS -> METERS
        self.theta = math.degrees(start[2]); # Rad heading --> Degrees
        self.phi = 0 # Init steering angle
        self.L = self.meters_to_pixels(length);

        # Graphics
        self.init = pygame.image.load(robot_img[0])
        self.goal = pygame.image.load(robot_img[1])
        self.img = pygame.image.load(robot_img[2])
        self.rotated = self.img # For when the robot rotates
        #TODO: Consider adding a mask here instead of box
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        # Time variant
        self.dt = 0.0 #dt
        self.last_time = pygame.time.get_ticks() # Previous recorded time
        self.x_position = [] # Log of x 
        self.y_position = [] # Log of y
        self.theta_orientation = [] # Log of theta

    def meters_to_pixels(self, meters):
        # Takes in meters and returns pixel value
        return meters*3779.52

    def draw(self, map):
        # Draws the robot on a map.
        # Map: pygame.Surface
        map.blit(source=self.rotated, dest=self.rect)

    #TODO: Get rid of "event"
    def move(self, event=None, u=None):
        # Move the img of the robot with a Euler assumption

        # Establish the control input u
        self.u1, self.u2 = u #u1 linear, u2 angular

        # Car kinematic model
        self.x += self.u1 * math.cos(math.radians(self.theta)) * self.dt # delta X
        self.y -= self.u1 * math.sin(math.radians(self.theta)) * self.dt # delta Y
        self.theta += self.u2 * self.dt

        # Translation and rotation animation
        self.rotated = pygame.transform.rotozoom(surface=self.img, angle=self.theta, scale=1);
        self.rect = self.rotated.get_rect(center=(self.x, self.y)) # Update bounding rectangle

        # Save history of points
        self.x_position.append(self.x)
        self.y_position.append(self.y)
        self.theta_orientation.append(self.theta)
    
    def closed_form_step(self, x_init, y_init, theta_init, v, w, t):
        """
        Takes in the robot's position and (constant) velocities, and calculates position over a duration time with a closed form assumption.
        PARAMETERS
            x_init : float (initial position x, px)
            y_init : float (initial position y, px)
            theta_init : float (initial orientation theta, rad)
            v : float (linear velocity - kept constant, px/s)
            w : float (angular velocity - randomized, rad/s)
            t : float (time)
        RETURNS
            (x_fin, y_fin, theta_fin) : tuple (final position)
        """
        if abs(w) < 1e-3: # Check if the robot is turning at all
            return x_init + v*t*math.cos(theta_init), y_init + v*t*math.sin(theta_init), theta_init # Theta is constant, integration is trivial
        
        r = v/w
        theta_fin = theta_init + w*t # Calculate new theta
        x_fin = x_init + r * (math.sin(theta_fin) - math.sin(theta_init))
        y_fin = y_init - r * (math.cos(theta_fin) - math.cos(theta_init))
        return x_fin, y_fin, theta_fin

    
    def generate_arc_samples(self, x_init, y_init, theta_init, v, w, t_total, delta):
        """
        Samples an arc's trajectory at evenly spaced timesteps (delta).
        PARAMETERS
            x_init : float (initial position x, px)
            y_init : float (initial position y, px)
            theta_init : float (initial orientation theta, rad)
            v : float (linear velocity - kept constant, px/s)
            w : float (angular velocity - randomized, rad/s)
            t_total : float (time)
            delta : float (spacing between samples)
        RETURNS
            x_samples, y_samples, theta_samples : list, list, list (Samples from t = 0 to t = t_tot)
        """

        x_samples = []
        y_samples = []
        theta_samples = []

        steps = int(t_total/delta)

        for k in range(1, steps+1):
            x, y, theta = self.closed_form_step(x_init, y_init, theta_init, v, w, k*delta)
            x_samples.append(x)
            y_samples.append(y)
            theta_samples.append(theta)

        # Add end-point if the end point is missed 
        x_end, y_end, theta_end = self.closed_form_step(x_init, y_init, theta_init, v, w, t_total)
        if x_samples[-1] != x_end:
            x_samples.append(x_end)
            y_samples.append(y_end)
            theta_samples.append(theta_end)
            
        
        print(f"x_samples[-1]: {x_samples[-1]}; x_end: {x_end}")
        print(f"y_samples[-1]: {y_samples[-1]}; y_end: {y_end}")

        return x_samples, y_samples, theta_samples

    def update_visual(self, x, y, theta):
        """
        Update the live visual of the path generation (Translation and rotation generation).
        PARAMETERS
            x : float (px)
            y : float (px)
            theta : float (deg)
        RETURNS
            None
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.rotated = pygame.transform.rotozoom(surface=self.img, angle=self.theta, scale=1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))