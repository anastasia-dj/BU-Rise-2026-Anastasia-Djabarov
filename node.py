class Node():
    """
    Represents a node in the kRRT tree.

    position : Configuration of the robot at the node (x, y)
    orientation : Orientation of the robot at the node (theta)
    parent : Reference to the parent node object 
    cost : Cost-to-come for the node

    """
    
    def __init__(self, position, orientation, parent, cost, trajectory_samples=None):
        self.position = position
        self.orientation = orientation
        self.parent = parent
        self.cost = cost
        self.trajectory_samples = trajectory_samples
