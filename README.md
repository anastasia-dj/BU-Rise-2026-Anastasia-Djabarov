# RISE - Sample-Based Motion Planning vs. Model-Free RL
## Status
The rewiring and reversal logic were both redesigned. Rewiring now copies an existing control input rather than approximating a straight-line connection, and reversal drives control inputs incrementally from the robot's actual position, using the control inputs of nodes with the "best" score. Reversibility testing (`drive_backwards()`) is currently being done to test whether a robot can retrace a path back to the root using only control inputs already stored in the tree. Sweeps of `reverse_search_radius` and `position_boundary` are in progress. Three candidate-selection modes (`cost`, `robot`, `root`) are currently being compared.

## Overview 
This project's goal is to test sampling-based motion planning as an alternative to model-free reinforcement learning in reversible systems. A kinodynamic RRT grows a tree of physically drivable trajectories for a car-like robot (x, y, θ). The central question: can the tree be used in reverse? This project explores how a robot spawned on or near the tree retracing stored control inputs back to the root, with no steering function and no replanning.

**Forward planning**: The tree grows by forward simulation only. A parent node is sampled (optionally weighted against heavily-extended nodes), a random control input is propagated via a closed-form arc solution, and the resulting state becomes a new node. Optional rewiring lets a new node adopt a cheaper nearby trajectory's parent and a truncated copy of its control input. The rewired node keeps its originally sampled state, accepting a bounded mismatch in exchange for preserved coverage.

**Reverse navigation**: Reversibility testing spawns a robot at a random location within the tree, where it repeatedly selects a nearby stored configuration and drives that configuration's control input in reverse, switching branches mid-motion when a better candidate comes into range. Candidate selection weighs path cost, and heading alignment.

The tree building and reversal process is observable end to end, with per-trial animations of the robot and live candidate qualification, failure-mode classification, and parameter sweeps with per-configuration CSV output.

Built on the 2D Pygame environment and visualization of [AngeloEspinoza's kinodynamic RRT](https://github.com/AngeloEspinoza/kinodynamic-rrt-car-like-robot), with the original per-frame Euler-integrated motion replaced by a closed-form arc solution.

## File Organization
- **node.py** - Node class: holds one node's position, orientation, parent, cost, trajectory samples, and control_input.
- **robot.py** - Robot class: kinematics (`closed_form_step`, `generate_arc_samples`), drawing and rotation.
- **environment.py** - Environment class: map setup, obstacles, and trail drawing.
- **RRT.py** - Graph class: tree recording, sampling, `simulate()` (simulates one full edge per call), `rewire_node()` (rewires nodes within a search radius), path extraction, plotting, reversibility testing (`drive_backwards()`, `find_nearest_qualifying_node`) and CSV result logging (`log_results()`).
- **kinodynamic_RRT.py** - Main script: parses command-line arguments, then runs the main loop (sampling -> simulate -> tree recording -> `rewire_node()` -> goal check -> reversibility testing). Now also supports automated parameter sweeps (`-hbl`, `-radl`) and repeated trials (`-rp`/`-rpn`).

## Pseudocode
    nodes = [Node(position=x_init, parent=0, cost=0)]  # root's parent is itself

    while not goal_found and num_nodes < max_nodes:
        random_parent_index = random_existing_node(nodes) # Sample a random node (NOT configuration)

        # --- Simulate a node and create an arc-like edge ---
        v = constant                                        # Fixed
        w = random(0.5, -0.5)                               # Held through the duration of an edge
        t = random (0.5, 4)                                 # Held through the duration of an edge

        trajectory = generate_arc_samples(nodes[random_parent_index].position, v, w, t, delta)

        if collision(trajectory):
            continue                # Discard edge if any point along it collides

        x_end = trajectory[-1]
        new_node = Node(position=x_end, parent=random_parent_index, cost=nodes[random_parent_index].cost + t, trajectory_samples=trajectory)
        append new_node to nodes

        # --- Rewiring ---
        for each sample within rewire_search_radius of new_node (across all edges):
            if heading difference within rewire_heading_boundary:
                candidate_cost = cost[candidate's parent] + (elapsed time to that sample)
                if candidate_cost < cost[new_node]:
                    new_node.parent = candidate.parent
                    new_node.control_input = (candidate_v, candidate_w, truncated_t)
                    new_node.cost = candidate_cost
                    # new_node.position is NOT updated
            
    # --- Reversibility Testing ---
    spawn robot at (x, y, theta)
    #visited = empty set

    while steps < max_steps:
        if within position_boundary of root:
            return success
        candidates = nodes within reverse_search_radius and reverse_heading_boundary of robot #(excluding visited) <-- Latest version no longer excludes visited
        if no candidates:
            return failure # No candidates in the vicinity
        
        best = candidate with lowest score # cost-to-come, distance to robot, or distance to root - depends on mode
        #visited.add(best)
        v, w, t = best.control_input

        while True:
            for k in 1(t/delta) # Reverse in increments
                step robot to closed_form_step(x,y,theta,-v,-w,k*delta)
                if within position_boundary of root:
                    return success
                better = search again for a better node from current position #(excluding visited) <-- Latest version no longer excludes visited
                if better is not None and better score is more preferable to best 
                    #visited.add(better)
                    v, w, t = better.control_input # New control overwrites old control
                    break # Restart stepping with new control
            else:
                    break # Ran to completion untouched
        steps += 1

    return failure # max steps exceeded

## Current Design Decisions
The core design decisions (so far) were suggested by Professor Tron and adopted for this implementation:
- **Off-branch reversal** - During reversal the robot searches the whole tree from its current position rather than walking its own .parent chain back to the root. It selects a nearby node, reverses that node's control input, and drives it incrementally, switching to a better node if one comes into range mid-motion. This allows the robot to cross between branches and use only existing control inputs.
- **Sibling-copy rewiring** - When a cheaper connection is found, the new node adopts the candidate's parent and a truncated copy of the candidate's control input, becoming a child of that parent. Every rewired connection is therefore a real, already-simulated control input rather than a straight-line approximation. The node's position is not updated to the adopted control's true endpoint, so each rewired edge carries a small approximation error - see [Known Issue #8](#issue-8)
- **Sampling by extending from a random existing node** - Rather than sampling a random point on the map and connecting to the nearest existing node (as in standard RRT*), a random node already in the tree is selected and extended directly. This makes each new node's parent known at creation, removing the need to search for one - see [Known Issue #1](#issue-1)
- **v fixed, w and t randomized per edge** - Makes cost calculation easier (cost = v*t) and guarantees each edge is a circular arc with a known closed-form solution
- **Theta as a third axis** - The 3-dimensional configuration space shows tree structure that a 2D top-down view hides



## Known Issues
<a id="issue-1"></a>
1. ~~Sampling used nearest-neighbor search, which frequently duplicated old node coordinates.~~ **Resolved** — sampling now extends directly from a randomly chosen existing node, so parent is known at creation. 
<a id="issue-2"></a>
2. ~~Previous path extraction (`get_path_coordinates()`) currently matches path nodes to edges by checking for an exact endpoint match with the successful path. This fails for two reasons: (a) nodes created via mid-edge rewiring aren't stored at the end of an edge, and (b) `generate_arc_samples()` truncates to floor(t/delta) steps while the real node is computed using the full, untruncated edge_time_duration. In both cases, the sample essentially does not match the node's true coordinates for many nodes.~~ **Resolved.** (a) is fixed by the Node refactor. Every node, including ones created via mid-edge rewiring, stores its own trajectory directly, so path extraction reads it off the node instead of searching for an edge endpoint match. (b) was fixed separately by manually having `generate_arc_samples()` append the true, untruncated endpoint as its final sample. 
<a id="issue-3"></a>
3. ~~Rewiring currently uses straight-line distance as an approximate connection cost between two arbitrary points. Since this project is based onf orward propagation only (no steering), there is no v,w or t which can be calcualted to reach a specific point.~~ **Resolved.** Reversal now only ever uses control inputs that already exist within the tree.
<a id="issue-4"></a>
4. ~~Currently, checking for a collision only includes the initial and final position. This does not account for the possibility that an obstacle may be between the initial and final positions.~~ **Resolved.** Within `simulate()`, the algorithm now cycles through `xs_samples`, `ys_samples`, and `thetas_samples`, checking each sampled pose for collisions to verify that the entire path is collision-free.
<a id="issue-5"></a>
5.~~ Theta oscillates significantly along the final path even as x/y progress steadily toward the goal. Both are sampled with equal per-edge randomness (via w), but only position feeds into cost and rewiring's selection criteria (cost = v*t) to become a successful candidate. This means that arcs that happen to make preferred positional movements are kept during rewiring, while heading has no equivalent selection pressure at all.~~ **Partially resolved.** A `rewire_heading_boundary` parameter was implemented, blocking a rewire from occurring if the heading difference between two configurations exceeds a set threshold. Its effect has not been re-measured since the rewiring and reversal redesign.
<a id="issue-6"></a>
6. ~~The 3D plot for simulations that do reach the goal shows incomplete portions and visible "jumps" between segments of the path. The suspected cause is from a missed update of .trajectory_samples.~~ **Resolved.** In `rewire_node()`​, the `trajectory_samples`​ for the promoted node and for the node on the edge containing it were being updated, but the `trajectory_samples​` of the node being rewired itself were not, resulting in a gap. This was fixed by adding the new parent's configuration and the node's own configuration to `trajectory_samples`,​ and by switching the 3D plot from `ax.scatter​` to `ax.plot`​ so these two points create a continuous line.
<a id="issue-7"></a>
7. Currently, the random control inputs are propagated from nodes which are already present within the tree at v=v_magnitude. This causes for a "fan"-like distribution of nodes, in which one half of the map is extensively covered, while the other half of the map is nearly empty. Since the goal of this project is to be able to use the tree from any resulting position, this node distribution is an issue. **Update:** an `anti_drive_bias` parameter was added, giving an edge a configurable change of driving with `v=0` or `v=-v_magnitude` intended to encourage exploration into underrepresented regions of the map. This has been implemented but not yet rigorously validated. 
<a id="issue-8"></a>
8. Rewired nodes keep their original position while adopting a truncated control input, so the adopted control lands near the node rather than on it. Each rewired edge carries a small approximation error bounded by `rewire_search_radius`.