# RISE - Sample-Based Motion Planning vs. Model-Free RL
## Status
Closed-form kinematics are implemented. Cost-tracking and rewiring are 
also implemented but still require further verification. Additionally, 
the 3D visualization currently displays only the initial and goal points; 
this bug is likely tied to Issues #1/#2. I am still isolating 
the exact cause.

## Overview 
This project currently extends AngeloEspinoza's 2D Pygame kinodynamic RRT implementation of a car-like robot (x, y, theta) in three ways:
1. Replacing the per-frame Euler-integrations with a closed-form arc trajectory solution by holding the linear velocity and angular velocity w constant over an edge's duration.
2. Adding a cost-based rewiring within a search radius of each new node, using arc length as cost. Since v is fixed at 1, cost-to-come is the elapsed time (s = v*t = t)
3. Visualize the tree in 3D space with theta as the z axis. 

## File Organization
- **robot.py** - Robot class: simulated physical model, kinematics (closed_form_step), arc sampling, drawing/rotation.
- **environment.py** - Environment class: map, obstacles, trail drawing
- **RRT.py** - Graph class: tree bookkeeping, sampling, simulate() (which checks one full edge per call), rewire_node() (rewires a node depending on positions in close proximity), path extraction, 3D matplotlib plot
- **kinodynamic_RRT.py** - main script: parse command line args, then runs the main loop: sampling -> simulate() -> tree bookkeeping -> rewire_node() -> check goal

## High-Level Pseudocode
    tree = [x_init];
    parent = [0]; # The root's parent index is itself
    cost = [0];

    while not goal_found and num_nodes < max_nodes:
        x_rand = sample_random_configuration()

        if collision(x_rand):
            continue

        # --- Simulate a node and create an arc-like edge ---
        v = 1                                               # Fixed
        w = random(0.5, -0.5)                               # Held through the duration of an edge
        t = random (0.5, 4)                                 # Held through the duration of an edge
        x_end = closed_form_step(last_position, v, w, t)    # Final position after random w and t

        if collision(x_end):
            continue                # Discard position if collision occurs

        # Save x_end into the ongoing history of ALL nodes ever added (not just this edge)
        history.append(x_end)
        
        trajectory = generate_arc_samples(last_position, v, w, t, delta)
        store trajectory in x_position / y_position / theta_orientation

        # The new_node is whichever point in the ENTIRE history
        #is closest to x_rand. (See Known Issue #1.)

        new_node = nearest_neighbor(history, x_rand)
        append new_node to tree
        parent[new_node] = <ID at that same nearest-neighbor index>
        cost[new_node] = cost[parent[new_node]] + t   # v=1, so cost = arc length = time

        # --- Rewiring ---
        for each sample within search_radius of new_node (across all edges):
            candidate_cost = cost[that_sample's_edge_start] 
                + (elapsed time to that sample) 
                + (approx. connection cost to new_node)
            cost_to_connect_sample_to_new_node = approximately straight-line distance  # approximation, not a real arc
            if candidate_cost < cost[new_node]:
                rewire new_node's parent to that sample
                (promoting it to a tree node if it's mid-edge, not an endpoint)
            
        if goal_reached(new_node):
            goal_found = True
            walk parent[] backward from goal to x_init -> path
            plot tree + path in 3D (x, y, theta)

## Current Design Decisions
The three core design decisions (so far) were suggested by Professor Tron and adopted for this implementation:
- **v fixed, w and t randomized per edge** - makes cost calculation easier (cost = v*t = t) and guarantees each edge is a circular arc with a known closed-form solution
- **Theta as a third axis** - the 3-dimensional configuration space shows tree structure that a 2D top-down view hides
- **Rewiring by searching through simulated arc points and tree nodes** - Every candidate point being searched is a real previously-simulated position rather than an arbitrary coordinate. However, the final connection from that candidate to the node being rewired is currently still being approximated by straight-line distance (not a verified arc) - see Known Issue #3

## Known Issues
1. In new_state(), the new node's configuration is currently chosen through nearest_neighbor(simulation, x_rand), which searches through the entire tree's history for whichever existing point is closest to the random sample. At short constant-velocity arcs with thousands of nodes, an old unrelated node is more often than not closer to a fresh random sample than the computed edge, causing new nodes to frequently duplicate old node coordinates. These duplications, I believe, corrupt parent/cost bookkeeping and cause for a final "path" far shorter than physically plausible.
2. Previous path extraction (get_path_coordinates()) currently matches path nodes to edges by checking for an exact endpoint match with the successful path. This fails for two reasons: (a) nodes created via mid-edge rewiring aren't stored at the end of an edge, and (b) generate_arc_samples() truncates to floor(t/delta) steps while the real node is computed using the full, untruncated edge_time_duration. In both cases, the sample essentially does not match the node's true coordinates for many nodes. 
3. Rewiring currently uses straight-line distance as an approximate connection cost between two arbitrary points. Solving for the w and t that would truly connect the two configurations has not yet been implemented, so rewiring decisions are based on an approximation and not currently verified against a real simulated connecting arc. 