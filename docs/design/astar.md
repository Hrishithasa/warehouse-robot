# A* Path Planning Design

## 1. Overview

The A* (A-Star) Path Planning module is responsible for generating the shortest feasible path for the warehouse robot from its current position to a specified goal within a known warehouse environment. This module performs global path planning by considering the static warehouse layout, including walls, shelves, and other fixed obstacles.

The planner generates an ordered sequence of waypoints that guide the robot toward the destination. If environmental changes make the current path invalid, the planner recomputes a new path using dynamic replanning.

---

## 2. Objectives

The objectives of the A* planner are:

- Generate the shortest collision-free path.
- Minimize travel distance.
- Avoid static obstacles.
- Produce waypoints for the DDQN local navigation agent.
- Support dynamic replanning when necessary.

---

## 3. Why A*

A* was selected because it combines the advantages of Dijkstra's Algorithm and Greedy Best-First Search.

Advantages include:

- Guarantees the shortest path when using an admissible heuristic.
- Efficient exploration of the search space.
- Widely used in robotics and autonomous navigation.
- Suitable for grid-based warehouse environments.
- Easy integration with reinforcement learning.

---

## 4. Grid Representation

The warehouse environment is represented as a configurable 12 × 12 two-dimensional grid.

Each grid cell represents one of the following:

| Value | Meaning |
|--------|---------|
| 0 | Free space |
| 1 | Static obstacle (Shelf/Wall) |
| 2 | Robot |
| 3 | Goal |
| 4 | Dynamic obstacle |

The planner searches only traversable cells.

---

## 5. Node Representation

Each node stores:

- Position (x, y)
- Parent node
- g-cost (distance from start)
- h-cost (estimated distance to goal)
- f-cost = g + h

The parent node is used during path reconstruction.

---

## 6. Manhattan Distance Heuristic

Since the warehouse is represented as a four-directional grid, the Manhattan Distance heuristic is used.

Formula:

h = |x₁ − x₂| + |y₁ − y₂|

This heuristic estimates the remaining distance to the goal while ensuring optimality.

---

## 7. Open Set

The Open Set contains nodes that have been discovered but not yet expanded.

The node with the lowest f-cost is selected for expansion.

If multiple nodes have equal cost, the node with the lower h-cost is preferred.

---

## 8. Closed Set

The Closed Set stores nodes that have already been explored.

Nodes inside this set are not revisited, preventing unnecessary computations.

---

## 9. Path Planning Workflow

1. Insert the start node into the Open Set.
2. Select the node with the lowest f-cost.
3. Move the node to the Closed Set.
4. Generate neighbouring cells.
5. Ignore invalid or blocked cells.
6. Compute g, h and f costs.
7. Insert valid neighbours into the Open Set.
8. Repeat until the goal is reached.
9. Reconstruct the path using parent nodes.

---

## 10. Path Reconstruction

After reaching the goal, the planner reconstructs the shortest path by following parent pointers from the goal node back to the start node.

The resulting path is reversed before being returned.

---

## 11. Dynamic Replanning

Warehouse environments may contain moving obstacles.

Whenever:

- a waypoint becomes blocked,
- the planned path becomes invalid,
- or the environment changes significantly,

the planner performs A* again using the robot's current position as the new starting node.

This ensures the robot can continue navigating safely.

---

## 12. Waypoint Generation

Instead of directly controlling the robot, the planner outputs an ordered list of waypoints.

Example:

Start
↓

(1,1)

↓

(2,1)

↓

(3,1)

↓

Goal

The DDQN navigation agent follows these waypoints while reacting to dynamic obstacles.

---

## 13. API Design

### Inputs

- Warehouse grid
- Robot start position
- Goal position
- Static obstacle map

### Outputs

- Planned path
- Ordered waypoint list
- Planning success status

---

## 14. Planned Classes

```
Node
 ├── position
 ├── parent
 ├── g
 ├── h
 └── f

AStarPlanner
 ├── find_path()
 ├── get_neighbors()
 ├── heuristic()
 ├── reconstruct_path()

DynamicReplanner
 ├── replan()

WaypointGenerator
 ├── generate_waypoints()
```

---

## 15. Expected Integration

The A* planner forms the global planning layer of the hybrid navigation system.

Interaction flow:

Warehouse Environment

↓

A* Planner

↓

Waypoint Generator

↓

DDQN Agent

↓

Robot Movement

The A* module determines where the robot should go, while the DDQN agent determines how the robot should move safely between waypoints.

---

## 16. Deliverables

The A* module will provide:

- Global shortest-path planning.
- Dynamic replanning.
- Waypoint generation.
- Efficient path reconstruction.
- Integration interfaces for the hybrid A* + DDQN navigation system.