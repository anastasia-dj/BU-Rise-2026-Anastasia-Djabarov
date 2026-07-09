# RISE - Sample-Based Motion Planning vs. Model-Free RL
## Status
Cost-tracking and rewiring still require further verification. The 3D visualization now displays the full tree/path. Rewired nodes lacking an updated trajectory_sample caused visible discontinuities in the plotted path, which has been resolved (see [Known Issue #6](#issue-6)). Collision checking has also been extended to verify the entire path rather than only the initial and final position (see [Known Issue #4](#issue-4)). Currently, theta oscillates significantly along the final path, understood to be caused by a lack of cost penalty on heading and/or heading boundary during rewiring (see [Known Issue #5](#issue-5)). Additionally, the node distribution heavily crowds one half of the map and not the other due to how the nodes propagate themselves through the control input (see [Known Issue #7](#issue-7))

## Overview 
This project currently extends AngeloEspinoza's 2D Pygame kinodynamic RRT implementation of a car-like robot (x, y, theta) in three ways:
1. Replacing the per-frame Euler-integrations with a closed-form arc trajectory solution by holding the linear velocity and angular velocity w constant over an edge's duration.
2. Adding a cost-based rewiring within a search radius of each new node, using arc length as cost. Since v is fixed at 1, cost-to-come is the elapsed time (s = v*t = t)
3. Visualize the tree in 3D space with theta as the z axis. 

## File Organization
- **node.py** - Node class: holds one node's position, orientation, parent, cost, and trajectory samples.
- **robot.py** - Robot class: kinematics (`closed_form_step`, `generate_arc_samples`), drawing and rotation.
- **environment.py** - Environment class: map setup, obstacles, and trail drawing.
- **RRT.py** - Graph class: tree recording, sampling, `simulate()` (simulates one full edge per call), `rewire_node()` (rewires nodes within a search radius), path extraction, and 3D matplotlib plotting.
- **kinodynamic_RRT.py** - Main script: parses command-line arguments, then runs the main loop (sampling -> simulate -> tree recording -> `rewire_node()` -> goal check).

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
1. ~~In `new_state()`, the new node's configuration is currently chosen through `nearest_neighbor(simulation, x_rand)`, which searches through the entire tree's history for whichever existing point is closest to the random sample. At short constant-velocity arcs with thousands of nodes, an old unrelated node is more often than not closer to a fresh random sample than the computed edge, causing new nodes to frequently duplicate old node coordinates. These duplications, I believe, corrupt parent/cost bookkeeping and cause for a final "path" far shorter than physically plausible.~~ **Resolved.** Sampling now picks a random existing node and extends directly from it, so a new node's parent is known at creation rather than derived from a nearest-neighbor search. This removes the possibility of a new node silently duplicating an unrelated old node's coordinates. 
<a id="issue-2"></a>
2. ~~Previous path extraction (`get_path_coordinates()`) currently matches path nodes to edges by checking for an exact endpoint match with the successful path. This fails for two reasons: (a) nodes created via mid-edge rewiring aren't stored at the end of an edge, and (b) `generate_arc_samples()` truncates to floor(t/delta) steps while the real node is computed using the full, untruncated edge_time_duration. In both cases, the sample essentially does not match the node's true coordinates for many nodes.~~ **Resolved.** (a) is fixed by the Node refactor. Every node, including ones created via mid-edge rewiring, stores its own trajectory directly, so path extraction reads it off the node instead of searching for an edge endpoint match. (b) was fixed separately by manually having `generate_arc_samples()` append the true, untruncated endpoint as its final sample. 
<a id="issue-3"></a>
3. Rewiring currently uses straight-line distance as an approximate connection cost between two arbitrary points. Solving for the w and t that would truly connect the two configurations has not yet been implemented, so rewiring decisions are based on an approximation and not currently verified against a real simulated connecting arc. 
<a id="issue-4"></a>
4. ~~Currently, checking for a collision only includes the initial and final position. This does not account for the possibility that an obstacle may be between the initial and final positions.~~ **Resolved.** Within `simulate()`, the algorithm now cycles through `xs_samples`, `ys_samples`, and `thetas_samples`, checking each sampled pose for collisions to verify that the entire path is collision-free.
<a id="issue-5"></a>
5. Theta oscillates significantly along the final path even as x/y progress steadily toward the goal. Both are sampled with equal per-edge randomness (via w), but only position feeds into cost and rewiring's selection criteria (cost = v*t) to become a successful candidate. This means that arcs that happen to make preferred positional movements are kept during rewiring, while heading has no equivalent selection pressure at all.
<a id="issue-6"></a>
6. ~~The 3D plot for simulations that do reach the goal shows incomplete portions and visible "jumps" between segments of the path. The suspected cause is from a missed update of .trajectory_samples.~~ **Resolved.** In `rewire_node()`​, the `trajectory_samples`​ for the promoted node and for the node on the edge containing it were being updated, but the `trajectory_samples​` of the node being rewired itself were not, resulting in a gap. This was fixed by adding the new parent's configuration and the node's own configuration to `trajectory_samples`,​ and by switching the 3D plot from `ax.scatter​` to `ax.plot`​ so these two points create a continuous line.
<a id="issue-7"></a>
7. Currently, the random control inputs are propagated from nodes which are already present within the tree at v=1. This causes for a "fan"-like distribution of nodes, in which one half of the map is extensively covered, while the other half of the map is nearly empty. Since the goal of this project is to be able to use the tree from any resulting position, this node distribution is an issue. 