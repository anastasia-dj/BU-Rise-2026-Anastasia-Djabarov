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
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BROWN = (189, 154, 122)

        self.has_obstacles = has_obstacles
        self.trail_set = []
        self.trail_sets = []
        