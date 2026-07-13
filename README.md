# RISE - Sample-Based Motion Planning vs. Model-Free RL
## Status
Six of the original known issues have been resolved or at least in some way addressed. Reversibility testing methods have been added (`drive_backwards()`, `shoot_toward_target()`, `path_from_node_to_root()`) to test whether a robot can retrace a path back to the root using only stored (or best-effort reconstructed for approximated connections) controls. From this, `heading_boundary` is currently found to significantly affect cost but not position drift, and `search_radius` to significantly affect both. An extension of `search_radius` sweeps (where search_radius = 25, 30, 40, 50) is in progress to eventually select a final recommended configuration.

## Overview 
This project currently extends AngeloEspinoza's 2D Pygame kinodynamic RRT implementation of a car-like robot (x, y, theta) in three ways:
1. Replacing the per-frame Euler-integrations with a closed-form arc trajectory solution by holding the linear velocity and angular velocity w constant over an edge's duration.
2. Adding a cost-based rewiring within a search radius of each new node, using arc length as cost. Since v is fixed at 1, cost-to-come is the elapsed time (s = v*t = t)
3. Visualize the tree in 3D space with theta as the z axis. 

## File Organization
- **node.py** - Node class: holds one node's position, orientation, parent, cost, trajectory samples, and control_input.
- **robot.py** - Robot class: kinematics (`closed_form_step`, `generate_arc_samples`), drawing and rotation.
- **environment.py** - Environment class: map setup, obstacles, and trail drawing.
- **RRT.py** - Graph class: tree recording, sampling, `simulate()` (simulates one full edge per call), `rewire_node()` (rewires nodes within a search radius), path extraction, plotting, reversibility testing (`drive_backwards()`, `shoot_toward_target()`, `path_from_node_to_root()`) and CSV result logging (`log_results()`).
- **kinodynamic_RRT.py** - Main script: parses command-line arguments, then runs the main loop (sampling -> simulate -> tree recording -> `rewire_node()` -> goal check). Now also supports automated parameter sweeps (`-hbl`, `-radl`) and repeated trials (`-rp`/`-rpn`).

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
        for each sample within search_radius of new_node (across all edges):
            candidate_cost = cost[that_sample's_edge_start] 
                + (elapsed time to that sample) 
                + (approx. connection cost to new_node)
            cost_to_connect_sample_to_new_node = approximately straight-line distance  # approximation, not a real arc
            if candidate_cost < cost[new_node]:
                rewire new_node's parent to that sample
                (promoting it to its own Node, with its own trajectory slice, if it's mid-edge)
            
        if goal_reached(new_node):
            goal_found = True
            walk .parent references backward from goal --> root
            plot tree + path in 3D (x, y, theta)

## Current Design Decisions
The core design decisions (so far) were suggested by Professor Tron and adopted for this implementation:
- **v fixed, w and t randomized per edge** - Makes cost calculation easier (cost = v*t) and guarantees each edge is a circular arc with a known closed-form solution
- **Theta as a third axis** - The 3-dimensional configuration space shows tree structure that a 2D top-down view hides
- **Rewiring by searching through simulated arc points and tree nodes** - Every candidate point being searched is a real previously-simulated position rather than an arbitrary coordinate. However, the final connection from that candidate to the node being rewired is currently still being approximated by straight-line distance (not a verified arc) - see [Known Issue #3](#issue-3)
- **Sampling by extending from a random existing node** - Rather than sampling a random point on the map and connecting to the nearest existing node (as in standard RRT*), a random node already in the tree is selected and extended directly. This makes each new node's parent known at creation, removing the need to search for one - see [Known Issue #1](#issue-1)


## Known Issues
<a id="issue-1"></a>
1. ~~Sampling used nearest-neighbor search, which frequently duplicated old node coordinates.~~ **Resolved** — sampling now extends directly from a randomly chosen existing node, so parent is known at creation. 
<a id="issue-2"></a>
2. ~~Previous path extraction (`get_path_coordinates()`) currently matches path nodes to edges by checking for an exact endpoint match with the successful path. This fails for two reasons: (a) nodes created via mid-edge rewiring aren't stored at the end of an edge, and (b) `generate_arc_samples()` truncates to floor(t/delta) steps while the real node is computed using the full, untruncated edge_time_duration. In both cases, the sample essentially does not match the node's true coordinates for many nodes.~~ **Resolved.** (a) is fixed by the Node refactor. Every node, including ones created via mid-edge rewiring, stores its own trajectory directly, so path extraction reads it off the node instead of searching for an edge endpoint match. (b) was fixed separately by manually having `generate_arc_samples()` append the true, untruncated endpoint as its final sample. 
<a id="issue-3"></a>
3. ~~Rewiring currently uses straight-line distance as an approximate connection cost between two arbitrary points. Since this project is based onf orward propagation only (no steering), there is no v,w or t which can be calcualted to reach a specific point.~~ **Partially resolved.** During reversibility testing, a lack of control input to reverse becomes an issue. Currently, the algorithm instead samples 1000 random forward controls and keeps whichever lands closest to the target.
<a id="issue-4"></a>
4. ~~Currently, checking for a collision only includes the initial and final position. This does not account for the possibility that an obstacle may be between the initial and final positions.~~ **Resolved.** Within `simulate()`, the algorithm now cycles through `xs_samples`, `ys_samples`, and `thetas_samples`, checking each sampled pose for collisions to verify that the entire path is collision-free.
<a id="issue-5"></a>
5.~~ Theta oscillates significantly along the final path even as x/y progress steadily toward the goal. Both are sampled with equal per-edge randomness (via w), but only position feeds into cost and rewiring's selection criteria (cost = v*t) to become a successful candidate. This means that arcs that happen to make preferred positional movements are kept during rewiring, while heading has no equivalent selection pressure at all.~~ **Partially resolved.** A `heading_boundary` parameter was implemented, blocking a rewire from occurring if the heading difference between two configurations exceeds a set threshold. However, a two-sample independent t-test comparing average position drift across seven tested average `heading_boundary` values (n=30 repeats each) found no statistically significant effect on position drift. 
<a id="issue-6"></a>
6. ~~The 3D plot for simulations that do reach the goal shows incomplete portions and visible "jumps" between segments of the path. The suspected cause is from a missed update of .trajectory_samples.~~ **Resolved.** In `rewire_node()`​, the `trajectory_samples`​ for the promoted node and for the node on the edge containing it were being updated, but the `trajectory_samples​` of the node being rewired itself were not, resulting in a gap. This was fixed by adding the new parent's configuration and the node's own configuration to `trajectory_samples`,​ and by switching the 3D plot from `ax.scatter​` to `ax.plot`​ so these two points create a continuous line.
<a id="issue-7"></a>
7. Currently, the random control inputs are propagated from nodes which are already present within the tree at v=1. This causes for a "fan"-like distribution of nodes, in which one half of the map is extensively covered, while the other half of the map is nearly empty. Since the goal of this project is to be able to use the tree from any resulting position, this node distribution is an issue. **Update:** an `anti_drive_bias` parameter was added, giving an edge a configurable change of driving with `v=0` or `v=-v_magnitude` intended to encourage exploration into underrepresented regions of the map. This has been implemented but not yet rigorously validated. 
