import unittest
from node import Node

"""
Unit tests for Node.py

Tests construction and defaults.
"""

class TestNode(unittest.TestCase):
    def test_constructor_storage(self):
        node = Node(position=(1,2), orientation=45, parent=3, cost=7.5, trajectory_samples=([1,2], [3,4], [5,6]))
        self.assertEqual(node.position, (1,2))
        self.assertEqual(node.orientation, 45)
        self.assertEqual(node.parent, 3)
        self.assertEqual(node.cost, 7.5)
        self.assertEqual(node.trajectory_samples, ([1,2], [3,4], [5,6]))
    
    def test_trajectory_samples_default_to_none(self):
        node = Node(position=(0,0), orientation=0, parent=0, cost=0.0)
        self.assertIsNone(node.trajectory_samples)