import pygame
import math

class Robot():
    """
    The robot; such like drawing, translation, position, orientation.

    start : tuple (x, y, theta)
    robot_img : str (list of paths for robot imgs - init, goal, robot)
    length : float (lenth b/w rear and front wheels)
    
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
        # Move the img of the robot

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