# numpy_ecs

A NumPy-first Entity Component System (ECS) for Python, with vectorized entity
selection and typed component storage. Inspired by
[pico_ecs](https://github.com/empyreanx/pico_headers).

---

## Overview

numpy_ecs organises simulation state into three concepts:

- **Entities** — plain integer IDs. They carry no data of their own.
- **Components** — typed data arrays (one NumPy array per component type,
  indexed by entity ID). Storage is Structure-of-Arrays (SoA).
- **Systems** — callbacks that receive a batch of matching entity IDs each
  frame and operate on them.

Entity-to-component membership is tracked with a `uint64` bitmask per entity,
so selection (which entities match a system's requirements) is a single
vectorised bitwise expression over all entities — no per-entity Python loops.

---

## Requirements

- Python 3.10+
- NumPy

---

## Quick Start

```python
import numpy as np
from numpy_ecs.ecs import ECS

# 1. Create the world
ecs = ECS()

# 2. Register component types (any NumPy dtype, including structured)
POS = ecs.define_component(np.dtype([("x", np.float32), ("y", np.float32)]))
VEL = ecs.define_component(np.dtype([("dx", np.float32), ("dy", np.float32)]))

# 3. Create entities and add components
player = ecs.create()
pos = ecs.add(player, POS)
pos["x"][0], pos["y"][0] = 0.0, 0.0

vel = ecs.add(player, VEL)
vel["dx"][0], vel["dy"][0] = 1.0, 0.5

# 4. Define a system that moves every entity with Position + Velocity
def movement_system(world, eids, udata):
    positions  = world.gather(POS, eids)
    velocities = world.gather(VEL, eids)
    positions["x"] += velocities["dx"]
    positions["y"] += velocities["dy"]
    world.scatter(POS, eids, positions)
    return 0

ecs.define_system(movement_system, require=[POS, VEL])

# 5. Game loop
for _ in range(60):
    ecs.run_systems(0)
```

---

## Core Concepts

### Entities

```python
eid = ecs.create()   # allocate (recycles destroyed IDs)
ecs.destroy(eid)     # free; calls component destructors
```

Entity IDs are plain `int` values. Destroyed IDs are recycled via a free list,
so they should not be stored beyond their lifetime.

### Components

Components are registered once at startup with a NumPy dtype. The dtype can be
a scalar type (`np.float32`) or a structured type (`np.dtype([("x", np.float32), ...])`).

```python
HP  = ecs.define_component(np.float32)
POS = ecs.define_component(np.dtype([("x", np.float32), ("y", np.float32)]))
```

Up to **64** component types are supported per ECS instance.

#### Adding and removing

```python
view = ecs.add(eid, POS)          # returns a writable 1-element view
view["x"][0] = 10.0               # write in place

ecs.has(eid, POS)                 # True
ecs.remove(eid, POS)              # calls destructor if registered
```

#### Reading and writing per-entity data

```python
val  = ecs.get(eid, POS)                    # read-only scalar / structured scalar
view = ecs.get(eid, POS, writable=True)     # writable 1-element slice
view["x"][0] = 42.0
```

> **Structured dtypes** — always use the slice form (`writable=True` or `add()`
> return value) when writing fields. Plain element access (`arr[eid]`) returns a
> copy for structured dtypes, so mutations would be silently discarded.

#### Constructors and destructors

```python
def pos_ctor(world, view, args):
    x0, y0 = args
    view["x"][0] = x0
    view["y"][0] = y0

def pos_dtor(world, val):
    print(f"position removed: {val}")

POS = ecs.define_component(
    np.dtype([("x", np.float32), ("y", np.float32)]),
    constructor=pos_ctor,
    destructor=pos_dtor,
)

view = ecs.add(eid, POS, args=(3.0, 7.0))
```

Constructor ordering guarantees:

1. Component slot is zero-initialised.
2. Mask bit is committed (so `has(eid, comp_id)` returns `True` inside the constructor).
3. Constructor runs.
4. `on_add` system events fire.

If the constructor raises, the mask bit and slot are **rolled back** atomically
and the exception is re-raised — no partial state is left behind.

### Systems

```python
def my_system(world, eids, udata):
    # eids — int64 array of all matching entity IDs this frame
    # udata — arbitrary user data registered with the system
    ...
    return 0   # nonzero stops run_systems early

sid = ecs.define_system(
    my_system,
    require=[POS, VEL],   # entity must own all of these
    exclude=[TAG],         # entity must own none of these
    udata={"dt": 0.016},
)
```

Systems run in registration order via `run_systems()`. Return a nonzero code
from any system to stop the chain early; `run_systems` returns that code.

```python
ret = ecs.run_systems(run_mask)   # run all systems
ret = ecs.run_system(sid, run_mask)  # run one system
```

#### Category masks

Systems can be grouped into categories with a bitmask. A system only runs if
its `category_mask` shares at least one bit with the `run_mask` passed to
`run_system` / `run_systems`. A `category_mask` of `0` (the default) means
**always run**.

```python
RENDER   = 0b01
PHYSICS  = 0b10

ecs.define_system(render_cb,  require=[POS], category_mask=RENDER)
ecs.define_system(physics_cb, require=[POS, VEL], category_mask=PHYSICS)

ecs.run_systems(PHYSICS)  # only physics runs
ecs.run_systems(RENDER | PHYSICS)  # both run
```

#### Enable / disable

```python
ecs.disable_system(sid)
ecs.enable_system(sid)
ecs.set_system_mask(sid, NEW_MASK)
ecs.get_system_mask(sid)
```

### Batch data access — `gather` and `scatter`

Inside a system callback, use `gather` / `scatter` for efficient batch reads
and writes over all matching entities.

```python
def physics(world, eids, udata):
    pos = world.gather(POS, eids)   # copy (advanced indexing)
    vel = world.gather(VEL, eids)   # copy
    pos["x"] += vel["dx"] * udata["dt"]
    pos["y"] += vel["dy"] * udata["dt"]
    world.scatter(POS, eids, pos)   # write back
    return 0
```

When `eids` is a `slice` (e.g. a contiguous range), `gather` returns a true
NumPy **view** with no copy overhead. When it is an integer array, it returns
a copy; use `scatter` to write back.

---

## Membership Events

Systems can be notified when entities enter or leave their matched set.

```python
def on_enter(world, eids, udata):
    print(f"entities joined: {eids}")

def on_leave(world, eids, udata):
    print(f"entities left: {eids}")

ecs.define_system(
    my_cb,
    require=[POS, VEL],
    on_add=on_enter,
    on_remove=on_leave,
)
```

---

## Capacity and Growth

Internal arrays start at `initial_capacity` (default 1024) and double
automatically when more entities are needed. Existing data is preserved across
growth.

```python
ecs = ECS(initial_capacity=65536)  # pre-allocate for large worlds
```

---

## Design Notes and Limitations

**64-component limit.** Membership is tracked with a single `uint64` bitmask,
so at most 64 distinct component types can be registered per ECS instance.

**No stale-handle detection.** Entity IDs are plain integers. Accessing a
destroyed (and potentially recycled) entity ID is not caught at runtime.

**Numeric components only.** Component data must be expressible as a NumPy
dtype. Arbitrary Python objects, variable-length data, and inter-entity
references are not supported.

---

## API Reference

### `ECS(initial_capacity=1024)`

| Method | Description |
|---|---|
| `create() → EntityId` | Allocate a new entity |
| `destroy(eid)` | Free an entity and run component destructors |
| `define_component(dtype, constructor?, destructor?) → CompId` | Register a component type |
| `has(eid, comp_id) → bool` | Test component ownership |
| `add(eid, comp_id, args?) → view` | Add component; return writable 1-element view |
| `get(eid, comp_id, writable=False) → scalar\|view` | Read (or write) a single entity's component |
| `remove(eid, comp_id)` | Remove component and call destructor |
| `define_system(cb, *, require, exclude, category_mask, on_add, on_remove, udata, active) → SystemId` | Register a system |
| `enable_system(sid)` / `disable_system(sid)` | Toggle system execution |
| `set_system_mask(sid, mask)` / `get_system_mask(sid)` | Read/write category mask |
| `run_system(sid, run_mask) → int` | Run one system |
| `run_systems(run_mask) → int` | Run all active systems in order |
| `gather(comp_id, eids\|slice) → NDArray` | Batch read component data |
| `scatter(comp_id, eids\|slice, data)` | Batch write component data |

### Public attributes

| Attribute | Type | Description |
|---|---|---|
| `entity_active` | `BoolArray` | True for every live entity |
| `entity_ready` | `BoolArray` | True for every entity that can receive components |
| `entity_masks` | `NDArray[uint64]` | Component bitmask per entity |
