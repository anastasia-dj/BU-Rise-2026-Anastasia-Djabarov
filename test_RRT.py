import math
import unittest 
from RRT import Graph
from node import Node

"""
Unit tests for RRT.py.

Covers rewire_node's cost calculation and edge-splitting, as well as is_goal_reached.
"""

class TestRewireNode(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(start=(0,0,0), goal=(100,100,0), map_dimensions=(640,480))
        self.graph.v = 1
    
    # Simulate a fake root and edge
    def make_root_and_edge(self):
        root = Node(position=(0,0), orientation=0, parent=0, cost=0.0, trajectory_samples=None) # Create tree root
        node1 = Node(position=(5,0), orientation=0, parent=0, cost=5.0, trajectory_samples=([1,2,3,4,5], [0,0,0,0,0], [0,0,0,0,0]))
        return root, node1

    # Test if node is rewired if cost is cheaper after rewiring
    def test_mid_edge_rewire_and_split(self):
        root, node1 = self.make_root_and_edge()
        rewired_node = Node(position=(4,0.5), orientation=0, parent=0, cost=8.0) # Node added close to node1 with purposefully high cost
        nodes = [root, node1, rewired_node]
        self.graph.rewire_node(nodes, search_radius=1, delta=1) # search_radius = delta = 1 for easy arithmetic

        # M got promoted, check position
        self.assertEqual(len(nodes),4)
        promoted = nodes[3]
        self.assertAlmostEqual(promoted.position[0], 4)
        self.assertAlmostEqual(promoted.position[1], 0)

        # Check that rewired_node was rewired through M
        self.assertEqual(rewired_node.parent, 3) 
        self.assertAlmostEqual(rewired_node.cost, 4.5)

        # Check that node 1 still has the same cost
        self.assertAlmostEqual(node1.cost, 5.0)
    
    # Test if node is rewired if cost is more expensive after rewiring
    def test_no_rewire_when_nothing_improves_cost(self):
        root, node1 = self.make_root_and_edge()
        not_rewired_node = Node(position=(4,0.5), orientation=0, parent=0, cost=1.0) # Node added close to node1 with purposefully low cost
        nodes = [root, node1, not_rewired_node]
        self.graph.rewire_node(nodes, search_radius=1, delta=1)

        # M was NOT promoted
        self.assertEqual(len(nodes), 3)
        self.assertEqual(not_rewired_node.parent, 0)
        self.assertAlmostEqual(not_rewired_node.cost, 1)

    # Test if node is rewired to an existing node if cost is cheaper after rewiring
    def test_rewire_to_existing_node(self):
        root, node1 = self.make_root_and_edge()
        rewired_node = Node(position=(5,0.5), orientation=0, parent=0, cost=8.0) # Node added close to node1 with purposefully high cost
        nodes = [root, node1, rewired_node]
        self.graph.rewire_node(nodes, search_radius=1, delta=1)

        # No node promoted but node was rewired
        self.assertEqual(len(nodes), 3)
        self.assertEqual(rewired_node.parent, 1)
        self.assertAlmostEqual(rewired_node.cost, 5.5)

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