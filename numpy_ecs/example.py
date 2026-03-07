from numpy.typing import NDArray
from typing import Any

import numpy as np
from ecs import ECS

ecs = ECS(initial_capacity=4096)

Position = ecs.define_component(np.dtype([("x", "f4"), ("y", "f4")]))
Velocity = ecs.define_component(np.dtype([("vx", "f4"), ("vy", "f4")]))


def move_system(ecs: ECS, eids: np.ndarray, udata):
    pos = ecs.gather(Position, eids)
    vel = ecs.gather(Velocity, eids)
    dt = udata["dt"]
    # Vectorized: updates all matched entities at once
    pos["x"] += vel["vx"] * dt
    pos["y"] += vel["vy"] * dt
    ecs.scatter(Position, eids, pos)
    return 0
""
move_sys = ecs.define_system(
    move_system,
    require=[Position, Velocity],
    category_mask=(1 << 0),
    udata={"dt": 1 / 60},
)

# create 1M movers quickly
N = 1_000_000
eids = np.array([ecs.create() for _ in range(N)], dtype=int)

# add components
for eid in eids[:]:
    ecs.add(eid, Position)
    ecs.add(eid, Velocity)

# initialize data in bulk (faster than per-entity)
pos: NDArray[Any] = ecs.gather(Position, eids)
vel = ecs.gather(Velocity, eids)
pos["x"] = np.random.rand(N).astype("f4")
pos["y"] = np.random.rand(N).astype("f4")
vel["vx"] = 1.0
vel["vy"] = 0.5
ecs.scatter(Position, eids, pos)
ecs.scatter(Velocity, eids, vel)

# run only category 0 systems
ecs.run_systems(run_mask=(1 << 0))
