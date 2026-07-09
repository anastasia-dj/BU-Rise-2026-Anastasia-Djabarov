import unittest 
from robot import Robot

"""
Unit tests for robot.py.

Covers the kinematics: closed_form_step (arc math) and generate_arc_samples (trajectory sampling, validates Issue #2(b) in the README.)

No drawing/visual methods are tested yet. 
"""

class TestRobot(unittest.TestCase):
    # Create an empty Robot instance
    def setUp(self): 
        self.robot = Robot.__new__(Robot)
    
    # Check if the robot goes straight when w = 0
    def test_straight_line_when_w_is_zero(self):
        x,y,theta = self.robot.closed_form_step(x_init=0, y_init=0, theta_init=0, v=10, w=0, t=2)
        self.assertAlmostEqual(x, 20)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(theta, 0)

    # Check if theta = w * t
    def test_theta_equals_w_times_t(self):
        x,y,theta = self.robot.closed_form_step(x_init=0, y_init=0, theta_init=0, v=10, w=0.3, t=2)
        self.assertAlmostEqual(theta, 0.6)

    # Check if endpoint of generate_arc_samples = closed_form_step
    def test_generate_arc_last_sample_matches_true_endpoint(self):
        xs,ys,thetas = self.robot.generate_arc_samples(x_init=0, y_init=0, theta_init=0, v=10, w=0.2, t_total=1.3, delta=0.05)
        x_end,y_end,theta_end = self.robot.closed_form_step(x_init=0, y_init=0, theta_init=0, v=10, w=0.2, t=1.3)
        self.assertAlmostEqual(xs[-1], x_end)
        self.assertAlmostEqual(ys[-1], y_end)
        self.assertAlmostEqual(thetas[-1], theta_end)
