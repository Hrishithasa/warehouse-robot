# Day 6 Verification – Dynamic A* Replanning

## Objective

Implement dynamic A* path planning and replanning so that the warehouse robot can avoid temporary obstacles introduced by moving workers and dynamic robots.

## Scope

Day 6 focuses on:

- Dynamic obstacle-aware A* planning
- Temporary blocked-cell support
- Dynamic obstacle-aware diagonal safety
- Path blockage detection
- Dynamic path replanning
- Unit testing of the replanning mechanism

## Files Modified

- `src/astar/astar_planner.py`
- `src/astar/utils.py`

## Files Created

- `src/astar/replanner.py`
- `tests/unit/test_replanner.py`

## Implementation

### 1. Dynamic Obstacle Support in A*

The A* planner was extended to accept:

```python
blocked_cells