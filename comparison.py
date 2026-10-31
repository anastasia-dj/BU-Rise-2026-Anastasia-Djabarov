"""
This file is for the final comparison between kRRT and PPO.

Pseudocode:
# STARTING
#1: Randomize your positions. 50 per heading/location pair.
#2: Initialize 4 PPO success table and PPO path cost table
#3: Initialize 4 kRRT success table and kRRT path cost table

# PPO Section
# Repeat x3:
#1: Create a PPO convergence with 100,000 samples. Preferably with seeds 1, 2, 3
#2: Test the positions initialized in STARTING 1
#3: Fill out a PPO success table and path cost table
#--- After repeated x3
#4. Fill out a master PPO success table and path cost table, taking the average of the 3

# kRRT Section
# Repeat x3:
#1: Grow a kRRT tree with 100,000 samples (two tree, with softmin). Preferably with seeds 1, 2, 3
#2: Test the positions initialized in STARTING 1
#3: Fill out a kRRT success table and path cost table
#--- After repeated x3
#4. Fill out a master kRRT success table and path cost table, taking the average of the 3

# Final Prints
#1. Create 3 CSVS: PPO_TABLES, kRRT_TABLES
#2. In PPO_TABLES, paste the 8 tables. Keep them separated so it is easy to tell which is which.  State the seed (...if that is even applicable here?)
#3. In kRRT_TABLES, paste the 8 tables. Keep them separated so it is easy to tell which is which. State the seed
"""

import argparse
import csv
import math
import os
import random

p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--out_folder', default="compare_krrt_ppo")
p.add_argument('--n', type=int, default=50, help="Start spots per side. Each is tried in 4 directions, so this is also the trials per box")
p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3], help="One PPO model and one kRRT tree per seed")
p.add_argument('--spawn_seed', type=int, default=0, help="Seed for the start spots")
p.add_argument('--ppo_steps', type=int, default=100000, help="PPO training timesteps per seed")
p.add_argument('--only', choices=['ppo', 'krrt'], default=None, help="Run only one of the two methods (default: both)")
 
# Flags for kRRT_lib
p.add_argument('-fd', default="point"); p.add_argument('--budget', type=int, default=100000)
p.add_argument('-rw', action='store_true')
p.add_argument('-rwhb', type=float, default=180); p.add_argument('-rwrad', type=float, default=25)
p.add_argument('-rwhw', type=float, default=0.0); p.add_argument('-stw', type=float, default=0.0)
p.add_argument('-rwexit', type=float, default=5.0)
p.add_argument('-k', type=float, default=0.5); p.add_argument('--w_mode', '-wmode', default='kwt')
p.add_argument('-adb', type=float, default=0); p.add_argument('-adm', default='reverse')
p.add_argument('-vmag', type=float, default=10); p.add_argument('-pb', type=int, default=14)
p.add_argument('-rvhb', type=float, default=30); p.add_argument('-rvrad', type=float, default=25)
p.add_argument('-rvhw', type=float, default=3.0); p.add_argument('-rvgcp', type=float, default=50)
p.add_argument('-rvext', type=float, default=6.0); p.add_argument('-ms', type=int, default=500)
p.add_argument('--timeout', type=float, default=10)
args, _ = p.parse_known_args()
os.makedirs(args.out_folder, exist_ok=True)

############### STARTING POSITIONS AND HEADINGS ###############
#1: Randomize your positions. 50 per heading/location pair.
SIDES = {"west": 180, "east": 0, "north": 90, "south": -90}
DIRECTIONS = {"toward": 0, "away": 180, "ccw": -90, "cw": 90}
random_num_gen = random.Random(args.spawn_seed)
spots = {}
for side, side_angle in SIDES.items():
    spots[side] = []
    while len(spots[side]) < args.n:
        angle, radius = math.radians(side_angle + random_num_gen.uniform(-45, 45)), random_num_gen.uniform(20, 200)
        x, y = 210 + radius * math.cos(angle), 210 - radius * math.sin(angle) # Get a random position
        if 0 <= x <= 420 and 0 <= y <= 420: # If it is within the map...
            spots[side].append((x, y))

starts = []
for side in SIDES: # Each side
    for direction, extra in DIRECTIONS.items(): # Each direction
        for sx, sy in spots[side]:
            heading = (math.degrees(math.atan2(sy - 210, 210 - sx)) + extra + 180) % 360 - 180  # Heading created
            starts.append((side, direction, sx, sy, heading)) # append to starts

ROWS, COLS = list(SIDES) + ["all"], list(DIRECTIONS) + ["all"]

def run_trials(drive, label):
    """
    Runs every start through drive(x, y, heading_deg) -> (outcome, cost_px)
    Returns one row per trial.
    drive : Function of driving
    label : Model (PPO/kRRT)
    """
    rows = []
    for i, (side, direction, sx, sy, heading) in enumerate(starts):
        outcome, cost = drive(sx, sy, heading)
        rows.append({"model": label, "side": side, "direction": direction, "x": round(sx, 1), "y": round(sy, 1), "heading": round(heading, 1), "outcome": outcome, "cost_px": None if cost is None else round(cost, 1)})
        if (i + 1) % 100 == 0:
            print(f"    {label}: {i + 1}/{len(starts)} trials done")
    return rows     

def make_tables(trials):
    """
    Makes the success/cost rows for a table. 
    Success --> list of % that were successful in the [(side, direction)]
    Cost --> List of mean path cost in px (if one exists) for the [(side, direction)] # TODO: Add stdev? Add it here!
    """
    success, cost = {}, {}
    for side in ROWS:
        for direction in COLS:
            considered_trials = [] # Collect the trials
            for trial in trials:
                side_matches = (side == "all") or (trial["side"] == side) # Side matches
                direction_matches = (direction == "all") or (trial["direction"] == direction)
                if side_matches and direction_matches:
                    considered_trials.append(trial)

            # But only consider SUCCESSFUL paths for path cost
            successful_costs = []
            for trial in considered_trials:
                if trial["outcome"] == "success": successful_costs.append(trial["cost_px"])
            n_success = len(successful_costs)

            # Two numbers for table
            success[(side, direction)] = 100.0 * n_success / len(considered_trials)
            if n_success > 0:
                cost[(side, direction)] = sum(successful_costs) / n_success
            else:
                cost[(side, direction)] = None # Nothing worked we all failed whoops

    return success, cost # Each are lists with side, direction. 

def average_tables(tables):
    """
    Averages three seeds into one bigger table
    """
    master = {}
    for side in ROWS:
        for direction in COLS:
            # Average the values in each cell
            cell = (side, direction)
            values = []
            for table in tables:
                if table[cell] is not None:
                    values.append(table[cell])

            if len(values) > 0:
                master[cell] = sum(values)/len(values) # Average
            else:
                master[cell] = None # 100% failed
    return master

def show(title, table):
    print(f"\n{title}")

    header = " " * 7 # Header column
    for directions in COLS:
        header += f"{directions:>9}"
    print(header)

    for side in ROWS:
        line = f"{side:<7}"
        for direction in COLS:
            value = table[(side, direction)]
            if value is None:
                text = "-" # No successes
            else:
                text = f"{value:.1f}"
            line += f"{text:>9}" # Right-aligned formatting stuff
        print(line)

def write_tables(path, titled_tables):
    """
    Writes the tables one under the other with a blank row between them. An empty cell = no successes in that box.
    """
    with open(path, "w", newline="") as file:
        w = csv.writer(file)
        for title, table in titled_tables:
            w.writerow([title] + COLS)
            for side in ROWS:
                w.writerow([side] + ["" if table[(side, c)] is None else f"{table[(side, c)]:.1f}" for c in COLS])
            w.writerow([])

def write_trials(path, rows):
    """
    Prints one row per trial
    """
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

# the method
def method(name, make_function):
    """
    The part of the experiment that is the same for PPO and kRRT. 

    Essentially:
    1. Build the planner
    2. Test the 800 starts from STARTING 
    3. make this seed's success table and cost table
    4. Repeat 1-3, 3 times
    5. Master success table and master cost table = average of the seeds' tables
    6. print some sort of log that indicates the process is done & add to csv files

    name : PPO / kRRT
    make_function : The function that grows PPO / kRRT
    """
    success_table_of = {}
    cost_table_of = {}
    all_trials = [] # every trial of every seed

    # Step 3 & 4
    for seed in args.seeds:
        print(f"\n [START] {name}, seed {seed}")

        # Build the planner
        planner = make_function(seed)

        # Run the 800 trials
        trials = run_trials(planner, f"{name}_seed{seed}")

        # Seed's success & cost tables
        success_table, cost_table = make_tables(trials)
        show(f"{name} seed {seed}: success rate (%), {args.n} trials per box", success_table)
        show(f"{name} seed {seed}: mean path cost (px, successes only)", cost_table)

        # Keep everything for the master tables and the csv files
        success_table_of[seed] = success_table
        cost_table_of[seed] = cost_table
        all_trials += trials

    # Step 5
    seeds_text = ", ".join(str(seed) for seed in args.seeds) # For the titles
    master_success = average_tables([success_table_of[seed] for seed in args.seeds])
    master_cost = average_tables([cost_table_of[seed] for seed in args.seeds])
    master_success_title = f"{name} MASTER (average of seeds {seeds_text}): success rate (%), {args.n} trials per box per seed"
    master_cost_title = f"{name} MASTER (average of seeds {seeds_text}): mean path cost (px, successes only)"
    show(master_success_title, master_success)
    show(master_cost_title, master_cost)

    # Step 6
    # The 8 tables with their titles, in the order they go into the csv (the two master tables & then each seed's pair)
    titled_tables = []
    titled_tables.append((master_success_title, master_success))
    titled_tables.append((master_cost_title, master_cost))
    for seed in args.seeds:
        titled_tables.append((f"{name} seed {seed}: success rate (%), {args.n} trials per box", success_table_of[seed]))
        titled_tables.append((f"{name} seed {seed}: mean path cost (px, successes only)", cost_table_of[seed]))

    tables_file = os.path.join(args.out_folder, f"{name}_TABLES.csv")
    trials_file = os.path.join(args.out_folder, f"{name}_trials.csv")
    write_tables(tables_file, titled_tables)
    write_trials(trials_file, all_trials)
    print(f"\n[LOG] Wrote {tables_file} ({len(titled_tables)} tables) and {trials_file} ({len(all_trials)} trials)")



# PPO Section
# Repeat x3:
#1: Create a PPO convergence with 100,000 samples. Preferably with seeds 1, 2, 3
#2: Test the positions initialized in STARTING 1
#3: Fill out a PPO success table and path cost table
#--- After repeated x3
#4. Fill out a master PPO success table and path cost table, taking the average of the 3

######### PPO SECTION!!!!! #####
if args.only != 'krrt': 
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from robotenv import RobotEnv

    def make_ppo(seed):
        """
        Step 1 for PPO: train one model with this seed, and return its drive function
        """
        env = RobotEnv(map_dimensions=(420, 420), goal=(210.0, 210.0), position_boundary=14, cell_size=28, theta_bins=12, v_magnitude=10.0, t_step=2.8, w_step=0.5, include_reverse=True, max_steps=200, step_penalty=-1.0, goal_reward=100.0, continuous_obs=True)
        # wrap env for training
        training_env = Monitor(env, filename=os.path.join(args.out_folder, f"PPO_seed{seed}_training_log"))

        model = PPO("MlpPolicy", training_env, seed=seed, verbose=0)
        model.learn(total_timesteps=args.ppo_steps, progress_bar=True) # training
        model.save(os.path.join(args.out_folder, f"PPO_seed{seed}"))# save as zip
        print(f"[TRAINED] trained {model.num_timesteps} timesteps (SB3 rounds the budget up to a whole number of 2048-step rollouts)")

        def drive(x, y, heading_deg):
            obs, _ = env.reset_to(x, y, math.radians(heading_deg))
            cost = 0.0
            for _ in range (env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                cost += env.v_magnitude * info["elapsed_t"] # arc length 
                if terminated:
                    return "success", cost
                if truncated:
                    return "max_steps", cost
            return "max_steps", cost
         
        return drive   # run_method will call this 800 times
    method("PPO", make_ppo)

# kRRT Section
# Repeat x3:
#1: Grow a kRRT tree with 100,000 samples (two tree, with softmin). Preferably with seeds 1, 2, 3
#2: Test the positions initialized in STARTING 1
#3: Fill out a kRRT success table and path cost table
#--- After repeated x3
#4. Fill out a master kRRT success table and path cost table, taking the average of the 3

if args.only != 'ppo':
    import kRRT_lib
    scene = kRRT_lib.make_environment(x_init=(210, 210, 0))
    env, robot, _ = scene
    OUTCOMES = {"reached_root": "success", "stuck_no_candidate": "no_cand", "max_steps_exceeded": "max_steps", "max_substeps_exceeded": "max_substeps"}

    def make_krrt(seed):
        """
        same thing as make_ppo but kRRT instead
        """

        # Grow the kRRT
        graph, nodes, growth = kRRT_lib.grow_two_trees(sample_budget=args.budget, seed=seed, scene=scene, rewire=args.rw, rewire_heading_boundary=args.rwhb, rewire_search_radius=args.rwrad, rewire_heading_weight=args.rwhw, shot_theta_weight=args.stw, rewire_shot_exit_radius=args.rwexit, k_value=args.k, w_mode=args.w_mode, anti_drive_bias=args.adb, anti_drive_mode=args.adm, v_magnitude=args.vmag, position_boundary=args.pb)

        # Reversal settings
        graph.heading_weight, graph.gear_change_penalty, graph.reverse_extension_multiplier, graph.max_steps , graph.position_weight, graph.use_samples, graph.use_softmin, graph.softmin_temperature = args.rvhw, args.rvgcp, args.rvext, args.ms, 0.0, False, True, 1.0

        # The budget split
        print(f"[GROWTH] grew: {growth}")
        growth_file = os.path.join(args.out_folder, "kRRT_growth.csv")
        new_file = not os.path.isfile(growth_file)
        with open(growth_file, "a", newline="") as file:
            w = csv.writer(file)
            if new_file:
                w.writerow(["tree_seed"] + list(growth)) # header row, first time only
            w.writerow([seed] + list(growth.values())) # append

        def drive(x, y, heading_deg):
            """
            One trial of krrt
            """
            result = graph.drive_backwards(nodes=nodes, robot=robot, start_x=x, start_y=y, start_theta_deg=heading_deg, reverse_search_radius=args.rvrad, reverse_heading_boundary=args.rvhb, position_threshold=args.pb, selection_mode='cost', use_visited=True, use_samples=False, max_steps=args.ms, max_seconds=args.timeout)
            reason = result[7]
            cost = result[5]
            return OUTCOMES.get(reason, reason), cost # translate

        return drive # method will call this 800 times

    method("kRRT", make_krrt)