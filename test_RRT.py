import math
import unittest 
from RRT import Graph
from node import Node
from robot import Robot

"""
Unit tests for RRT.py.

Covers rewire_node's cost calculation and edge-splitting, as well as is_goal_reached.
"""

class TestRewireNode(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(start=(0,0,0), goal=(100,100,0), map_dimensions=(640,480))
        self.graph.v = 1
        self.robot = Robot.__new__(Robot) # type: ignore

    # Test sibling copy and rewire
    def test_sibling_copy_rewire(self):
        root = Node(position=(0,0), orientation=0, parent=0, cost=0.0, control_input=None, trajectory_samples=None) # Create tree root
        node1 = Node(position=(5,0), orientation=0, parent=0, cost=5.0, control_input=(1, 0.0, 5.0), trajectory_samples=([1,2,3,4,5], [0,0,0,0,0], [0,0,0,0,0]))
        new_node = Node(position=(4,0.5), orientation=0, parent=0, cost=100.0, control_input=(1,0.4,3.0)) # Make fake new node
        nodes = [root, node1, new_node]

        self.graph.rewire_node(nodes, search_radius=1, delta=1, heading_boundary=180, robot=self.robot)

        self.assertEqual(len(nodes),3) # No new nodes were made
        self.assertEqual(new_node.parent, node1.parent) # Siblings
        self.assertEqual(new_node.control_input, (1, 0.0, 4)) # Copied correctly
        self.assertAlmostEqual(new_node.cost, 4.0) # Copied correctly
        self.assertEqual(new_node.position, (4, 0.5)) # Untouched
        self.assertEqual(new_node.orientation, (0)) # Untouched
    
    # Test if there is nothing else on the map
    def test_no_rewire_when_nothing_qualifies(self):
        root = Node(position=(0,0), orientation=0, parent=0, cost=0.0, control_input=None) # Create tree root
        far_node = Node(position=(500,500), orientation=0, parent=0, cost=8.0, control_input=(1, 0.3, 2.0))
        nodes = [root, far_node]
        original_control = far_node.control_input
        self.graph.rewire_node(nodes, search_radius=1, delta=1, heading_boundary=180, robot=self.robot)

        self.assertEqual(far_node.control_input, original_control)
        self.assertEqual(len(nodes), 2)

    def test_no_rewire_when_cost_does_not_improve(self):
        root = Node(position=(0,0), orientation=0, parent=0, cost=0.0, control_input=None) # Create tree root
        node1 = Node(position=(5,0), orientation=0, parent=0, cost=5.0, control_input=(1, 0.0, 5.0), trajectory_samples=([1,2,3,4,5], [0,0,0,0,0], [0,0,0,0,0]))
        not_rewired_node = Node(position=(4,0.5), orientation=0, parent=0, cost=1.0, control_input=(1, 0.2, 2.0))
        nodes = [root, node1, not_rewired_node]
        original_control = not_rewired_node.control_input
        
        self.graph.rewire_node(nodes, search_radius=1, delta=1, heading_boundary=180, robot=self.robot)
        
        self.assertGreater(self.graph.candidates_in_radius, 0) # Confirm the candidate is actuallyseen
        self.assertEqual(not_rewired_node.parent, 0) # Did not rewire
        self.assertEqual(not_rewired_node.control_input, original_control) # Did not copy control input
        self.assertAlmostEqual(not_rewired_node.cost, 1.0) # Did not copy cost
    
    def test_trajectory_samples_regenerated_after_rewire(self):
        root = Node(position=(0,0), orientation=0, parent=0, cost=0.0, control_input=None, trajectory_samples=None)
        node1 = Node(position=(5,0), orientation=0, parent=0, cost=5.0, control_input=(1, 0.0, 5.0),trajectory_samples=([1,2,3,4,5], [0,0,0,0,0], [0,0,0,0,0]))
        new_node = Node(position=(4,0.5), orientation=0, parent=0, cost=100.0, control_input=(1, 0.4, 3.0),trajectory_samples=([1,0.5,2], [0.2,0.3,0.4], [0,0,0]))
        nodes = [root, node1, new_node]
        self.graph.rewire_node(nodes, search_radius=1, delta=1, heading_boundary=180, robot=self.robot)

         # Re-derive independently what the control input should produce 
        v,w,t = new_node.control_input
        parent_node = nodes[new_node.parent]
        expected_end_x, expected_end_y, expected_end_theta = self.robot.closed_form_step(parent_node.position[0], parent_node.position[1], math.radians(parent_node.orientation), v, w, t)

        # Check the actual endings
        end_x = new_node.trajectory_samples[0][-1]
        end_y = new_node.trajectory_samples[1][-1]
        end_theta = new_node.trajectory_samples[2][-1]

        # Compare
        self.assertAlmostEqual(end_x, expected_end_x, places=5)
        self.assertAlmostEqual(end_y, expected_end_y, places=5)
        self.assertAlmostEqual(end_theta, math.degrees(expected_end_theta), places=5)


class TestGoalReached(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(start=(0,0,0), goal=(100,100, math.pi), map_dimensions=(640,480))
        self.graph.position_boundary = 10
        self.graph.orientation_boundary = math.radians(10)

    def test_position_is_reached(self):
        self.assertTrue(self.graph.is_goal_reached((100,100), math.degrees(math.pi)))
    
    def test_position_is_not_reached(self):
        self.assertFalse(self.graph.is_goal_reached((150,150), math.degrees(math.pi)))
    
    def test_orientation_is_not_reached(self):
        self.assertFalse(self.graph.is_goal_reached((100,100), 90))
    
    def test_angle_wraparound(self):
        self.assertTrue(self.graph.is_goal_reached((100,100), -177))

class TestAngleDifference(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(start=(0,0,0), goal=(100,100, math.pi), map_dimensions=(640,480))
        
    def test_no_difference(self):
        self.assertEqual(self.graph.angle_difference(45,45), 0)
    
    def test_simple_difference_no_wraparound(self):
        self.assertEqual(self.graph.angle_difference(0,90), 90)
    
    def test_opposite_headings(self):
        self.assertEqual(self.graph.angle_difference(0,180), 180)

    def test_wraparound(self):
        self.assertEqual(self.graph.angle_difference(179,-179), 2)

    def test_order_does_not_matter(self):
        self.assertEqual(self.graph.angle_difference(30,200), self.graph.angle_difference(200,30))