# Sampling-Based Motion Planning vs. Model-Free RL

## Overview 
This project's goal is to test sampling-based motion planning as an alternative to model-free reinforcement learning in reversible systems. A kinodynamic Rapidly-exploring Random Tree (kRRT) grows a tree of physically drivable trajectories for a unicycle-like robot (x, y, θ). This study investigates whether a kRRT tree, built once for forward planning, can be reused to guide a system back to its root, serving as a sample-efficient alternative to model-free reinforcement learning when the simulation budget is limited. 

The tree building and reversal process is observable end to end, with per-trial animations of the robot and live candidate qualification, failure-mode classification, and parameter sweeps with per-configuration CSV output. The PPO algorithm is also similarly observable.

Built on the 2D Pygame environment and visualization of [AngeloEspinoza's kinodynamic RRT](https://github.com/AngeloEspinoza/kinodynamic-rrt-car-like-robot), with the original per-frame Euler-integrated motion replaced by a closed-form arc solution.


## Current Results
kRRT and PPO were evaluated under a budget of 100,000 simulated interactions, where one interaction corresponds to a control input applied for a specified duration (a kRRT tree-propagation attempt or rewire shot, or a PPO environment step). They have both been evaluated on 30 seeds, testing 50 positions per heading (Facing towards the goal, facing away from the goal, clockwise, and counterclockwise) and location (North, East, South, West) pair.

### Overall Table
| Algorithm | Sample Budget | Success Rate (%) | Mean Path Cost (px) | Standard Deviation of Success Rate (%) |
| -------- | -------- | -------- | -------- | -------- |
| kRRT  | 100,000  | 95.8 | 199.42 | 1.290 |
| PPO  | 100,000  | 57.1 | N/A* | 39.925 | 

*PPO fails, on average, 42.9% of the time. The path cost is only calculated through successful paths. The mean path cost of PPO is not accurately reflective of its performance at 100k samples.

### Per Heading Direction - kRRT
| Robot Starting Heading | kRRT 100k success (%) | kRRT 100k Cost Average (px) |
| -------- | -------- | -------- |
| Toward goal | 98.6 | 120.4 |
| Away from goal | 98.1 | 121.7 |
| CCW (orthogonal) | 93.4 | 284.8 |
| CW (orthogonal) | 93.1 | 279.5 |

Certain initial robot orientations lead to lower success rates and longer paths for kRRT, as shown above (see [issue 9](#issue-9)).

### Per Seed Breakdown (100,000 Sample Budget, 800 Trials per Seed)
| Seed | kRRT success (%) | kRRT cost (px) | PPO success (%) |
| -------- | -------- | -------- | -------- |
| 1 | 96.4 | 197 | 14.0 |
| 2 | 92.2 | 193 | 98.8 |
| 3 | 97.9 | 185 | 14.0 |
| 4 | 96.0 | 198 | 17.8 |
| 5 | 93.8 | 218 | 22.1 |
| 6 | 94.8 | 200 | 85.9 |
| 7 | 94.0 | 200 | 48.9 |
| 8 | 94.2 | 202 | 98.4 |
| 9 | 96.5 | 195 | 77.0 |
| 10 | 97.4 | 203 | 94.5 |
| 11 | 96.9 | 193 | 75.0 |
| 12 | 96.2 | 203 | 13.6 |
| 13 | 96.4 | 198 | 97.1 |
| 14 | 96.0 | 190 | 8.9 |
| 15 | 96.1 | 190 | 91.4 |
| 16 | 96.9 | 203 | 95.9 |
| 17 | 95.8 | 188 | 11.0 |
| 18 | 98.0 | 198 | 98.0 |
| 19 | 95.9 | 212 | 12.9 |
| 20 | 96.5 | 202 | 97.0 |
| 21 | 95.4 | 204 | 9.4 |
| 22 | 94.5 | 204 | 19.0 |
| 23 | 96.2 | 202 | 86.9 |
| 24 | 94.8 | 193 | 4.5 |
| 25 | 94.8 | 203 | 98.8 |
| 26 | 95.5 | 189 | 99.8 |
| 27 | 96.1 | 207 | 12.5 |
| 28 | 95.6 | 198 | 92.5 |
| 29 | 97.8 | 212 | 99.5 |
| 30 | 95.9 | 200 | 18.9 |

## Previous Results (BU RISE)
kRRT and Q-Learning have been evaluated on the same 50 initial configurations (x, y, theta), with the results:

| Algorithm | Success Rate | Mean Path Cost (px) | Std dev |
| -------- | -------- | -------- | -------- |
| Q-learning  | 100% (50/50)  | 184.20 | 65.996 |
| kRRT Seed 4  | 94% (47/50)  | 198.53 | 70.16 |
| kRRT Seed 45  | 94% (47/50)  | 209.18| 65.188 |

No statistically significant difference was found between any of the values, with p > 0.07 (Welch's T-Test for mean cost difference, F-Test used for cost variance, and McNemar's exact test for success rate). kRRT achieved this performance with ~186x fewer simulated interactions than Q-learning. 

The reversal process was additionally validated on hardware across 30 trials with a physical LIMO, reaching the root in all 30 (100%).

## Status
### kRRT
Current implementations of softmin are undergoing refinement to improve success rate and decrease mean path cost.

## File Organization
### kRRT
- **node.py** - Node class: holds one node's position, orientation, parent, cost, trajectory samples, and control_input.
- **robot.py** - Robot class: kinematics (`closed_form_step`, `generate_arc_samples`), drawing and rotation.
- **environment.py** - Environment class: map setup, obstacles, and trail drawing.
- **RRT.py** - Graph class: tree recording, sampling, `simulate()` (simulates one full edge per call), `rewire_node()` (rewires nodes within a search radius), path extraction, plotting, reversibility testing (`drive_backwards()`, `find_nearest_qualifying_node`) and CSV result logging (`log_results()`).
- **kinodynamic_RRT.py** - Main script: parses command-line arguments, then runs the main loop (sampling -> simulate -> tree recording -> `rewire_node()` -> goal check -> reversibility testing). Now also supports automated parameter sweeps (`-rwhbl`, `-rwradl`, etc.) and repeated trials (`-rp`/`-rpn`).
- **kRRT_lib.py** - kRRT as importable functions: `make_environment()` (environment, robot and obstacles), `grow_two_trees()` (grows two trees facing opposite directions & returns the metrics), `evaluate_reversal()` (runs reversal trials from random spawns and records each trial) and `run_lhs_point()` (runs a specific lhs sweep point given all lhs sweep commands). 
- **export_tree.py** - Exports the tree as a JSON file.
### PPO
- **robotenv.py** - RobotEnv class (Gym environment): kinematics, state discretization, goal/step logic, and `replay_model()` for visualization.
- **ppo_robot.py** - Trains the PPO model (Stable-Baselines3).
### Both
- **comparison.py** - Runs kRRT and PPO on the same starts. 

## Pseudocode
### kRRT
    nodes = [Node(position=x_init, parent=0, cost=0)]  # root's parent is itself

    while propagations + rewire_shots < sample_budget: # one interaction = one extension attempt or one rewire shot
        x_rand = random point on the map # Estimated Voronoi-Bias Sampling
        random_parent_index = nearest existing node to x_rand
        propagations += 1 # Counted within sampling budget, even if it is disregarded later

        # --- Simulate a node and create an arc-like edge ---
        v = constant                                        # Fixed
        w = random(0.5, -0.5)                               # Held through the duration of an edge
        t = random (0.5, 4)                                 # Held through the duration of an edge

        trajectory = generate_arc_samples(nodes[random_parent_index].position, v, w, t, delta)

        if collision(trajectory):
            continue                # Discard edge if any point along it collides

        x_end = trajectory[-1]
        new_node = Node(position=x_end, parent=random_parent_index, cost=nodes[random_parent_index].cost + t*|v|, trajectory_samples=trajectory, control_input = v,w,t)
        append new_node to nodes

        # --- Rewiring ---
        for each sample within rewire_search_radius of new_node (across all edges):
            if heading difference within rewire_heading_boundary:
                candidate_cost = cost[candidate's parent] + (elapsed time to that sample)
                if candidate_cost < cost[new_node]:
                    shoot random (v, w, t) from the sample through 3 rounds refining around the best
                    if cost[cost_to_come through closest and best heading-aligned shot] is cheaper than cost[new_node] and is within rewire search radius:
                        sample becomes a node (if not previously a node)
                        new_node.parent = sample
                        new_node.control_input = (shot_v, shot_w, shot_t)
                        new_node.cost = sample.cost + shot_t
                        new_node.position = where the shot lands

                                
    # --- Reversibility Testing ---
    spawn robot at (x, y, theta)
    #visited = empty set

    while steps < max_steps:
        if within position_boundary of root:
            return success
        candidates = nodes within reverse_search_radius and reverse_heading_boundary of robot (excluding visited , if use_visited is enabled) 
        if no candidates:
            return failure # No candidates in the vicinity
        
        best = candidate with lowest score of argmin OR calculation by softmin 
        #visited.add(best)
        v, w, t = best.control_input
        t = reverse_extension_multiplier * t # A hop may drive longer than a stored edge

        while True:
            for k in 1(t/delta) # Reverse in increments
                step robot to closed_form_step(x,y,theta,-v,-w,k*delta)
                if within position_boundary of root:
                    return success
                better = search again for a better node from current position #(excluding visited, if use_visited is enabled)
                if better is not None and better score is more preferable to best 
                    #visited.add(better)
                    v, w, t = better.control_input # New control overwrites old control
                    break # Restart stepping with new control
            else:
                    break # Ran to completion untouched
        steps += 1

    return failure # max steps exceeded


## Current Design Decisions
The core design decisions (so far) for this implementation:
### kRRT
- **Off-branch reversal** - During reversal the robot searches the whole tree from its current position rather than walking its own .parent chain back to the root. It selects a nearby node, reverses that node's control input, and drives it incrementally, switching to a better node if one comes into range mid-motion. This allows the robot to cross between branches and use only existing control inputs. Currently, softmin is being implemented.
- **Bi-directional tree implementation** - Single tree growth biases most node orientations in the same heading of the goal. For this reason, we have implemented a bi-directional tree growth starting from the root, which grows two trees in opposite directions, each with half of the sample budget, and then merges them.
- **Multi-shot rewiring** - When a cheaper connection is found, the new node shoots 3 batches of increasingly precise randomized-propagations. Within a certain radius, the closest propagation is chosen. Then, the cost between the cost-to-root is compared to the previous, and the cheapest path is chosen.
- **Approximate Voronoi-bias sampling** - A random point is sampled on the map, and the nearest existing node is propagated. 
- **|v| fixed, w and t randomized per edge** - Makes cost calculation easier (cost = v*t) and guarantees each edge is a circular arc with a known closed-form solution.
- **Theta as a third axis** - The 3-dimensional configuration space shows tree structure that a 2D top-down view hides.

### PPO
- **Potential Reward-Shaping** - An implementation of potential reward-shaping from "Policy invariance under reward transformations: Theory and application to reward shaping" (Andrew Y. Ng, Daishi Harada, Stuart Russell) has been added to decrease time to convergence without influencing the optimal policy.


## Known Issues
### kRRT
<a id="issue-1"></a>
1. ~~Sampling used nearest-neighbor search, which frequently duplicated old node coordinates.~~ **Resolved.** Sampling now uses an approximate Voronoi bias, and the new node is the propagation endpoint rather than the sampled point, so coordinates are not duplicated. See kRRT current design decisions. 
<a id="issue-2"></a>
2. ~~Previous path extraction (`get_path_coordinates()`) currently matches path nodes to edges by checking for an exact endpoint match with the successful path. This fails for two reasons: (a) nodes created via mid-edge rewiring aren't stored at the end of an edge, and (b) `generate_arc_samples()` truncates to floor(t/delta) steps while the real node is computed using the full, untruncated edge_time_duration. In both cases, the sample essentially does not match the node's true coordinates for many nodes.~~ **Resolved.** (a) is fixed by the Node refactor. Every node, including ones created via mid-edge rewiring, stores its own trajectory directly, so path extraction reads it off the node instead of searching for an edge endpoint match. (b) was fixed separately by manually having `generate_arc_samples()` append the true, untruncated endpoint as its final sample. 
<a id="issue-3"></a>
3. ~~Rewiring currently uses straight-line distance as an approximate connection cost between two arbitrary points. Since this project is based on forward propagation only (no steering), there is no v,w or t which can be calculated to reach a specific point.~~ **Resolved.** Reversal now only ever uses control inputs that already exist within the tree.
<a id="issue-4"></a>
4. ~~Currently, checking for a collision only includes the initial and final position. This does not account for the possibility that an obstacle may be between the initial and final positions.~~ **Resolved.** Within `simulate()`, the algorithm now cycles through `xs_samples`, `ys_samples`, and `thetas_samples`, checking each sampled pose for collisions to verify that the entire path is collision-free.
<a id="issue-5"></a>
5. ~~Theta oscillates significantly along the final path even as x/y progress steadily toward the goal. Both are sampled with equal per-edge randomness (via w), but only position feeds into cost and rewiring's selection criteria (cost = v*t) to become a successful candidate. This means that arcs that happen to make preferred positional movements are kept during rewiring, while heading has no equivalent selection pressure at all.~~ **Partially resolved.** A `rewire_heading_boundary` parameter was implemented, blocking a rewire from occurring if the heading difference between two configurations exceeds a set threshold. Its effect has not been re-measured since the rewiring and reversal redesign.
<a id="issue-6"></a>
6. ~~The 3D plot for simulations that do reach the goal shows incomplete portions and visible "jumps" between segments of the path. The suspected cause is from a missed update of .trajectory_samples.~~ **Resolved.** In `rewire_node()`​, the `trajectory_samples`​ for the promoted node and for the node on the edge containing it were being updated, but the `trajectory_samples​` of the node being rewired itself were not, resulting in a gap. This was fixed by adding the new parent's configuration and the node's own configuration to `trajectory_samples`,​ and by switching the 3D plot from `ax.scatter​` to `ax.plot`​ so these two points create a continuous line.
<a id="issue-7"></a>
7. ~~Currently, the random control inputs are propagated from nodes which are already present within the tree at v=v_magnitude. This causes for a "fan"-like distribution of nodes, in which one half of the map is extensively covered, while the other half of the map is nearly empty. Since the goal of this project is to be able to use the tree from any resulting position, this node distribution is an issue.*~~ **Resolved.** An approximate Voronoi bias sampling creates a visually even tree consistently.
<a id="issue-8"></a>
8. ~~Rewired nodes keep their original position while adopting a truncated control input, so the adopted control lands near the node rather than on it. Each rewired edge carries a small approximation error bounded by `rewire_search_radius`.~~ **Resolved.** The node now moves to the landing pose of the rewiring shot.
<a id="issue-9"></a>
9. kRRT’s path cost and success rate depend on the initial heading (see the per-heading table above). Originally, we found that having one tree biases the node orientations towards one direction, decreasing node heading diversity. We addressed this by growing a bi-directional tree with the goal as the root. However, there is likely still a lack of nodes holding orientations orthogonal to the goal, which is where the sideways (CW/CCW) starts fall behind.

## Acknowledgements
I greatly acknowledge and thank Professor Tron for his support, patience and time throughout the development of this research project. I additionally extend thanks to PhD students Taha Ondogan and Idris Seidu for their mentorship, and to the EPIC and RASTIC laboratories for their invaluable resources and welcoming research environment. I further acknowledge RISE for the opportunity to begin on this research.