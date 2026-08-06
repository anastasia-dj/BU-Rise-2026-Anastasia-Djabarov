import argparse
import math
import numpy as np
import qlearning
from robotenv import RobotEnv, replay, evaluate_and_log

# Parser Arguments (CLI)
parser = argparse.ArgumentParser(description='Q-learning for a nonholonomic car reaching a fixed goal from any start. (Aka, the RL mirror of kRRT reversibility trials).')

# QLearning
parser.add_argument('-pb', '--position_boundary', type=int, metavar='', required=False, default=20, help='Allowed distance (px) from the goal.')
parser.add_argument('-goal', '--goal', nargs=2, type=float, metavar='', required=False, default=[210.0, 210.0], help='Goal position in x(px), y(px).')
parser.add_argument('-mp', '--map_dimensions', nargs=2, type=int, metavar='', required=False, default=[420, 420], help='Size of the map.')
#parser.add_argument('-ts', '--t_step', type=float, metavar='', required=False, default=4.0, help='Seconds each action is held for, the equivalent of the edge_length_duration for kRRT.')
parser.add_argument('-sp', '--step_penalty', type=float, metavar='', required=False, default=-1.0, help='The penalty given for every action that does not reach the goal.')
parser.add_argument('-gr', '--goal_reward', type=float, metavar='', required=False, default=100.0, help='Reward given the instant the goal is reached.')
parser.add_argument('-cs', '--cell_size', type=int, metavar='', required=False, default=28, help='The size of the cells in the grid')
parser.add_argument('-tb', '--theta_bins', type=int, metavar='', required=False, default=12, help='Number of theta bins')
parser.add_argument('-vmag', '--v_magnitude', type=float, metavar='', required=False, default=10, help='The magnitude of linear velocity used.')
parser.add_argument('-wmag', '--w_magnitude', type=float, metavar='', required=False, default=0.5, help='The magnitude of angular velocity used. (0 angular velocity is also always an action the robot can take regardless of w_mag)')
parser.add_argument('-rev', '--use_reverse', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=True, help='Whether or not to allow the robot to pursue -v_magnitude')
parser.add_argument('-ms', '--max_steps', type=int, metavar='', required=False, default=200, help='The max_steps taken in one episode.')
parser.add_argument('-nte', '--number_training_episodes', type=int, metavar='', required=False, default=30000, help='The number of training episodes (Or the maximum number when searching for convergence).')
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', required=False, default=0.01, help='The learning rate.')
parser.add_argument('-g', '--gamma', type=float, metavar='', required=False, default=0.9, help='The gamma.')
parser.add_argument('-dr', '--decay_rate', type=float, metavar='', required=False, default=0.0002, help='The decay rate.')
parser.add_argument('-aps', '--actions_per_state', type=int, metavar='', required=False, default=1, help='Number of actions per state.')
parser.add_argument('-sv', '--save_video', type=str, metavar='', required=False, default=None, help='Base path to save the drive animation as .mp4 instead of showing it, like kRRT does.')

# Position startings:
parser.add_argument('-sx', '--start_x', type=float, metavar='', required=False, default=None,
    help='Manual start X position (px). Must be given together with -sy and -stheta.')
parser.add_argument('-sy', '--start_y', type=float, metavar='', required=False, default=None,
    help='Manual start Y position (px). Must be given together with -sx and -stheta.')
parser.add_argument('-stheta', '--start_theta', type=float, metavar='', required=False, default=None,
    help='Manual start heading (deg). 0=facing +x (right), 90=facing up-screen, -90=facing down-screen. '
         'Must be given together with -sx and -sy.')

# Convergence testing (QLearning still)
parser.add_argument('-fec', '--find_episodes_for_convergence', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Train until Q-values converge and record in a csv (no fixed episode count).')
parser.add_argument('-ci', '--checkpoint_interval', type=int, metavar='', required=False, default=500, help='Number of episodes between convergence checkpoints.')
parser.add_argument('-ct', '--convergence_threshold', type=float, metavar='', required=False, default=1, help='Max allowed delta between Q values in the same state before declaring convergence.')

# Logging and data collection
parser.add_argument('-rp', '--repeat_program', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Repeats the program multiple times.')
parser.add_argument('-rpnn', '--repeat_program_new_table_num', type=int, metavar='', required=False, default=0, help='Number of times to repeat the program.')
parser.add_argument('-rpns', '--repeat_program_same_table_num', type=int, metavar='', required=False, default=1, help='Number of times to repeat the program.')
parser.add_argument('-sad', '--show_agent_drive', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Shows the animation of the agent driving.')
parser.add_argument('-sqt', '--save_qtable', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Saves the qtable as a .npy file.')
parser.add_argument('-vrn', '--view_run_number', type=int, metavar='', required=False, default=None, help="Shows the animation of a specific run.")
parser.add_argument('-lqt', '--load_qtable', type=str, metavar='', required=False, default=None, help='Path to a saved .npy qtable.')
parser.add_argument('-sspawn', '--spawn_file', type=str, metavar='', required=False, default=None, help='Evaluation runs exactly one episode per row at the SAME poses kRRT tests with -rvspawn file.')

args = parser.parse_args()

if (args.start_x is None) != (args.start_y is None) or (args.start_x is None) != (args.start_theta is None):
    parser.error('--start_x, --start_y, and --start_theta must all be provided together.')

if args.start_x is not None:
    start_config = (args.start_x, args.start_y, math.radians(args.start_theta))
else:
    start_config = None


# Constants
MAP_DIMENSIONS = args.map_dimensions # Used to be 680, 520, 640 480. Now default is 420x420

# CLI arguments
goal = args.goal
actions_per_state = args.actions_per_state
position_boundary = args.position_boundary # NOTE: Not used anymore
cell_size = args.cell_size # Default 40
theta_bins = args.theta_bins # Default 8
v_magnitude = args.v_magnitude # Default 10
w_step = args.w_magnitude # Default 0.5
include_reverse = args.use_reverse # Default off
t_step = cell_size/(actions_per_state*v_magnitude)
step_penalty = args.step_penalty
goal_reward = args.goal_reward
n_training_episodes = args.number_training_episodes
learning_rate = args.learning_rate
gamma = args.gamma
max_steps = args.max_steps
decay_rate = args.decay_rate

n_eval_episodes = 200
max_epsilon = 1.0
min_epsilon = 0.05

# Sanity check
px_per_action = v_magnitude * t_step
actions_per_cell = cell_size/px_per_action
print(f"[SANITY] One t_step {t_step} moves {px_per_action:.2f}px. {actions_per_cell} actions to cross one cell (Assuming w=0).")
if actions_per_cell > 4:
    print(f"[SANITY] actions_per_cell has been flagged as too high. Shrink cell_size, raise t_step, or raise max_steps per action.")

env = RobotEnv(
    map_dimensions=MAP_DIMENSIONS,
    goal=goal,
    position_boundary=position_boundary,
    cell_size=cell_size,
    theta_bins=theta_bins,
    v_magnitude=v_magnitude,
    t_step=t_step,
    w_step=w_step,
    include_reverse=include_reverse,
    max_steps=max_steps,
    step_penalty=step_penalty,
    goal_reward=goal_reward
)

file_name = f"QL_CellSize{cell_size}_ThetaBins{theta_bins}_NumActions{len(env.actions)}_ActionsPerState{actions_per_state}_StepPenalty{step_penalty}_Reward_{goal_reward}_TStep{t_step}.csv"
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)

spawn_list = None
if args.spawn_file:
    import csv as _csv
    with open(args.spawn_file, newline = '') as _file:
        spawn_list = [(float(r["x"]), float(r["y"]), float(r["theta_deg"])) for r in _csv.DictReader(_file)]
    n_eval_episodes = len(spawn_list)
    print(f"[SPAWNS] Loaded {n_eval_episodes} shared spawns from {args.spawn_file}; n_eval_episodes overridden to match.")

if args.load_qtable:
    Qtable_robot = np.load(args.load_qtable)
    print(f"[LOAD] Loaded Q-table from {args.load_qtable}, shape {Qtable_robot.shape}")
    for repeats in range(args.repeat_program_same_table_num):
        evaluate_and_log(env, Qtable_robot, max_steps=max_steps, n_eval_episodes=n_eval_episodes, file_name=file_name, run_number=repeats+1, spawn_list=spawn_list, cell_size=env.cell_size, theta_bins=env.theta_bins, t_step=env.t_step, w_step=env.w_step, position_boundary=env.position_boundary, step_penalty=env.step_penalty)

if args.find_episodes_for_convergence and not args.load_qtable:
    agent = qlearning.QLearning(env, learning_rate=learning_rate, gamma=gamma)
    #NOTE: n_training_episodes is reused as the maximum number of episodes here to decrease the number of CLI args (because it is alot)
    Qtable_robot, episodes_required = agent.find_episodes_for_convergence(max_episodes=n_training_episodes, min_epsilon=min_epsilon, max_epsilon=max_epsilon, decay_rate=decay_rate, max_steps=max_steps, file_name=f"CONVERGENCE_LR{args.learning_rate}_convergethresh_{args.convergence_threshold}_{file_name}", checkpoint_interval=args.checkpoint_interval, convergence_threshold=args.convergence_threshold)
    
    # CLI arg pass
    if args.save_qtable:
        np.save("robot_qtable.npy", Qtable_robot)

    # Evaluate
    evaluate_and_log(env, Qtable_robot, max_steps=max_steps, n_eval_episodes=n_eval_episodes, file_name=file_name, run_number=1, spawn_list=spawn_list, cell_size=env.cell_size, theta_bins=env.theta_bins, t_step=env.t_step, w_step=env.w_step, position_boundary=env.position_boundary, step_penalty=env.step_penalty)

if args.repeat_program_new_table_num != 0 and not args.load_qtable:
    for repeats in range(args.repeat_program_new_table_num):
        agent = qlearning.QLearning(env, learning_rate=learning_rate, gamma=gamma)
        Qtable_robot = agent.train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)
        if args.save_qtable is True:
            np.save("robot_qtable.npy", Qtable_robot)
        evaluate_and_log(env, Qtable_robot, max_steps=max_steps, n_eval_episodes=n_eval_episodes, file_name=file_name, run_number=repeats+1, spawn_list=spawn_list, cell_size=env.cell_size, theta_bins=env.theta_bins, t_step=env.t_step, w_step=env.w_step, position_boundary=env.position_boundary, step_penalty=env.step_penalty)

elif args.find_episodes_for_convergence is False and not args.load_qtable: # 'repeat_program_same_table_num' is >=1 (Also covers the possibility of wanting 1 run since default=1 for repeat_program_same_table_num)
    agent = qlearning.QLearning(env, learning_rate=learning_rate, gamma=gamma)
    Qtable_robot = agent.train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)
    if args.save_qtable is True:
        np.save("robot_qtable.npy", Qtable_robot)
    for repeats in range(args.repeat_program_same_table_num):
         evaluate_and_log(env, Qtable_robot, max_steps=max_steps, n_eval_episodes=n_eval_episodes, file_name=file_name, run_number=repeats+1, spawn_list=spawn_list, cell_size=env.cell_size, theta_bins=env.theta_bins, t_step=env.t_step, w_step=env.w_step, position_boundary=env.position_boundary, step_penalty=env.step_penalty)

if args.view_run_number:
    robot_images = ['images/initial.png', 'images/goal.png', 'images/robot.png']
    replay(env, robot_images, Qtable_robot, max_steps=200, run_number=args.view_run_number, start=start_config, save_path=args.save_video)

if args.show_agent_drive:
    robot_images = ['images/initial.png', 'images/goal.png', 'images/robot.png']
    replay(env, robot_images, Qtable_robot, max_steps=200, start=start_config, save_path=args.save_video)