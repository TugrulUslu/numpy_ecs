"""
Unit tests for numpy_ecs.py

Run with:  pytest test_numpy_ecs.py -v
"""

from __future__ import annotations

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Import the module under test.
# ---------------------------------------------------------------------------
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from numpy_ecs.ecs import ECS


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

POS_DTYPE  = np.dtype([("x", np.float32), ("y", np.float32)])
VEL_DTYPE  = np.dtype([("dx", np.float32), ("dy", np.float32)])
HP_DTYPE   = np.dtype(np.float32)
TAG_DTYPE  = np.dtype(np.uint8)


@pytest.fixture
def ecs():
    """Fresh ECS with a small initial capacity so resize paths are exercised."""
    return ECS(initial_capacity=4)


@pytest.fixture
def world(ecs):
    """ECS pre-loaded with Position, Velocity, HP, and Tag component types."""
    pos = ecs.define_component(POS_DTYPE)
    vel = ecs.define_component(VEL_DTYPE)
    hp  = ecs.define_component(HP_DTYPE)
    tag = ecs.define_component(TAG_DTYPE)
    return ecs, pos, vel, hp, tag


# ===========================================================================
# 1. ECS construction
# ===========================================================================

class TestConstruction:
    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            ECS(initial_capacity=0)

    def test_initial_state(self, ecs):
        assert ecs._next_entity == 0
        assert ecs._free_list == []
        assert ecs.immediate_events is True


# ===========================================================================
# 2. Entity lifecycle
# ===========================================================================

class TestEntityLifecycle:
    def test_create_returns_unique_ids(self, ecs):
        ids = [ecs.create() for _ in range(8)]
        assert len(set(ids)) == 8

    def test_entity_is_active_after_create(self, ecs):
        eid = ecs.create()
        assert ecs.entity_active[eid]
        assert ecs.entity_ready[eid]

    def test_destroy_deactivates_entity(self, ecs):
        eid = ecs.create()
        ecs.destroy(eid)
        assert not ecs.entity_active[eid]
        assert not ecs.entity_ready[eid]

    def test_destroy_recycles_id(self, ecs):
        eid = ecs.create()
        ecs.destroy(eid)
        eid2 = ecs.create()
        assert eid2 == eid

    def test_double_destroy_is_safe(self, ecs):
        eid = ecs.create()
        ecs.destroy(eid)
        ecs.destroy(eid)  # should not raise or double-append to free list

    def test_capacity_grows_automatically(self, ecs):
        # initial_capacity=4; create more than 4 entities
        ids = [ecs.create() for _ in range(20)]
        assert len(ids) == 20
        assert all(ecs.entity_active[e] for e in ids)


# ===========================================================================
# 3. Component definition
# ===========================================================================

class TestComponentDefinition:
    def test_define_returns_sequential_ids(self, ecs):
        c0 = ecs.define_component(np.float32)
        c1 = ecs.define_component(np.float64)
        assert c0 == 0
        assert c1 == 1

    def test_max_64_components(self, ecs):
        for _ in range(64):
            ecs.define_component(np.uint8)
        with pytest.raises(ValueError):
            ecs.define_component(np.uint8)


# ===========================================================================
# 4. Component add / has / get / remove
# ===========================================================================

class TestComponentOperations:
    def test_add_sets_has(self, world):
        ecs, pos, vel, hp, tag = world
        eid = ecs.create()
        assert not ecs.has(eid, pos)
        ecs.add(eid, pos)
        assert ecs.has(eid, pos)

    def test_add_returns_writable_view(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        view = ecs.add(eid, pos)
        view["x"][0] = 3.0
        view["y"][0] = 7.0
        assert ecs.get(eid, pos)["x"] == pytest.approx(3.0)
        assert ecs.get(eid, pos)["y"] == pytest.approx(7.0)

    def test_add_idempotent_returns_same_view(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        v1 = ecs.add(eid, pos)
        v1["x"][0] = 5.0
        v2 = ecs.add(eid, pos)   # already owned
        assert v2["x"][0] == pytest.approx(5.0)

    def test_get_writable_flag(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        ecs.add(eid, pos)
        w = ecs.get(eid, pos, writable=True)
        w["x"][0] = 42.0
        assert ecs.get(eid, pos)["x"] == pytest.approx(42.0)

    def test_remove_clears_has(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        ecs.add(eid, pos)
        ecs.remove(eid, pos)
        assert not ecs.has(eid, pos)

    def test_remove_zeroes_slot(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        v = ecs.add(eid, pos)
        v["x"][0] = 99.0
        ecs.remove(eid, pos)
        assert ecs.get(eid, pos)["x"] == pytest.approx(0.0)

    def test_remove_nonexistent_is_safe(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        ecs.remove(eid, pos)  # never added; should not raise

    def test_add_to_inactive_entity_raises(self, world):
        ecs, pos, *_ = world
        eid = ecs.create()
        ecs.destroy(eid)
        with pytest.raises(RuntimeError):
            ecs.add(eid, pos)

    def test_destroy_clears_all_components(self, world):
        ecs, pos, vel, hp, tag = world
        eid = ecs.create()
        ecs.add(eid, pos)
        ecs.add(eid, vel)
        ecs.destroy(eid)
        assert ecs.entity_masks[eid] == 0


# ===========================================================================
# 5. Constructor and destructor callbacks
# ===========================================================================

class TestConstructorDestructor:
    def test_constructor_receives_writable_view(self, ecs):
        def ctor(e, view, args):
            view["x"][0] = 10.0
            view["y"][0] = 20.0

        pos = ecs.define_component(POS_DTYPE, constructor=ctor)
        eid = ecs.create()
        ecs.add(eid, pos)
        assert ecs.get(eid, pos)["x"] == pytest.approx(10.0)
        assert ecs.get(eid, pos)["y"] == pytest.approx(20.0)

    def test_constructor_can_call_has(self, ecs):
        """Mask is committed before constructor, so has() must return True."""
        seen = []

        def ctor(e, view, args):
            # e is the ECS instance; args carries the eid and comp_id
            eid_, comp_id_ = args
            seen.append(e.has(eid_, comp_id_))

        pos = ecs.define_component(POS_DTYPE, constructor=ctor)
        eid = ecs.create()
        ecs.add(eid, pos, args=(eid, pos))
        assert seen == [True]

    def test_constructor_args_forwarded(self, ecs):
        received = []

        def ctor(e, view, args):
            received.append(args)

        pos = ecs.define_component(POS_DTYPE, constructor=ctor)
        eid = ecs.create()
        ecs.add(eid, pos, args={"init": 42})
        assert received == [{"init": 42}]

    def test_constructor_exception_rolls_back(self, ecs):
        def bad_ctor(e, view, args):
            raise RuntimeError("boom")

        pos = ecs.define_component(POS_DTYPE, constructor=bad_ctor)
        eid = ecs.create()
        with pytest.raises(RuntimeError, match="boom"):
            ecs.add(eid, pos)

        # Mask must be rolled back
        assert not ecs.has(eid, pos)
        # Slot must be zeroed
        assert ecs.get(eid, pos)["x"] == pytest.approx(0.0)

    def test_destructor_called_on_remove(self, ecs):
        calls = []

        def dtor(e, val):
            calls.append(float(val["x"]))

        pos = ecs.define_component(POS_DTYPE, destructor=dtor)
        eid = ecs.create()
        v = ecs.add(eid, pos)
        v["x"][0] = 7.0
        ecs.remove(eid, pos)
        assert calls == [pytest.approx(7.0)]

    def test_destructor_called_on_destroy(self, ecs):
        calls = []

        def dtor(e, val):
            calls.append(True)

        pos = ecs.define_component(POS_DTYPE, destructor=dtor)
        eid = ecs.create()
        ecs.add(eid, pos)
        ecs.destroy(eid)
        assert calls == [True]


# ===========================================================================
# 6. System selection
# ===========================================================================

class TestSystemSelection:
    def test_system_sees_matching_entities(self, world):
        ecs, pos, vel, hp, tag = world

        seen = []
        def sys_cb(e, eids, ud):
            seen.extend(eids.tolist())
            return 0

        ecs.define_system(sys_cb, require=[pos, vel])

        e1 = ecs.create(); ecs.add(e1, pos); ecs.add(e1, vel)
        e2 = ecs.create(); ecs.add(e2, pos)              # missing vel
        e3 = ecs.create(); ecs.add(e3, pos); ecs.add(e3, vel)

        ecs.run_systems(0)
        assert sorted(seen) == sorted([e1, e3])

    def test_exclude_mask_filters_entities(self, world):
        ecs, pos, vel, hp, tag = world

        seen = []
        def sys_cb(e, eids, ud):
            seen.extend(eids.tolist())
            return 0

        ecs.define_system(sys_cb, require=[pos], exclude=[tag])

        e1 = ecs.create(); ecs.add(e1, pos)
        e2 = ecs.create(); ecs.add(e2, pos); ecs.add(e2, tag)

        ecs.run_systems(0)
        assert seen == [e1]

    def test_category_mask_skips_unmatched_systems(self, world):
        ecs, pos, *_ = world

        calls = []
        def sys_cb(e, eids, ud):
            calls.append(True)
            return 0

        ecs.define_system(sys_cb, require=[pos], category_mask=0b10)
        eid = ecs.create(); ecs.add(eid, pos)

        ecs.run_systems(0b01)   # no overlap with 0b10
        assert calls == []

        ecs.run_systems(0b10)   # matches
        assert calls == [True]

    def test_category_mask_zero_runs_always(self, world):
        ecs, pos, *_ = world

        calls = []
        def sys_cb(e, eids, ud):
            calls.append(True)
            return 0

        ecs.define_system(sys_cb, require=[pos], category_mask=0)
        eid = ecs.create(); ecs.add(eid, pos)

        ecs.run_systems(0b1111)
        ecs.run_systems(0)
        assert len(calls) == 2

    def test_run_systems_stops_on_nonzero_return(self, world):
        ecs, pos, *_ = world

        order = []
        def sys_a(e, eids, ud): order.append("A"); return 1
        def sys_b(e, eids, ud): order.append("B"); return 0

        ecs.define_system(sys_a, require=[pos])
        ecs.define_system(sys_b, require=[pos])
        eid = ecs.create(); ecs.add(eid, pos)

        ret = ecs.run_systems(0)
        assert ret == 1
        assert order == ["A"]   # B was never called


# ===========================================================================
# 7. System enable / disable
# ===========================================================================

class TestSystemEnableDisable:
    def test_disabled_system_does_not_run(self, world):
        ecs, pos, *_ = world

        calls = []
        def cb(e, eids, ud): calls.append(True); return 0

        sid = ecs.define_system(cb, require=[pos])
        ecs.disable_system(sid)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.run_systems(0)
        assert calls == []

    def test_re_enabled_system_runs(self, world):
        ecs, pos, *_ = world

        calls = []
        def cb(e, eids, ud): calls.append(True); return 0

        sid = ecs.define_system(cb, require=[pos])
        ecs.disable_system(sid)
        ecs.enable_system(sid)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.run_systems(0)
        assert calls == [True]


# ===========================================================================
# 8. on_add / on_remove events — immediate mode (default)
# ===========================================================================

class TestImmediateEvents:
    def test_on_add_fires_on_add_call(self, world):
        ecs, pos, vel, *_ = world

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos, vel], on_add=on_add)
        eid = ecs.create()
        ecs.add(eid, pos)
        assert added == []          # vel not yet added

        ecs.add(eid, vel)
        assert added == [eid]       # now both present → on_add

    def test_on_remove_fires_on_remove_call(self, world):
        ecs, pos, vel, *_ = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos, vel], on_remove=on_remove)
        eid = ecs.create()
        ecs.add(eid, pos); ecs.add(eid, vel)
        ecs.remove(eid, vel)
        assert removed == [eid]

    def test_on_remove_fires_on_destroy(self, world):
        ecs, pos, *_ = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.destroy(eid)
        assert removed == [eid]

    def test_adding_excluded_comp_fires_on_remove(self, world):
        """Adding a component that is excluded by a system should fire on_remove."""
        ecs, pos, vel, hp, tag = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        # System requires pos, excludes tag
        ecs.define_system(cb, require=[pos], exclude=[tag], on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos)
        # Adding tag causes entity to leave the system's set → on_remove
        ecs.add(eid, tag)
        assert removed == [eid]

    def test_constructor_exception_does_not_fire_on_add(self, ecs):
        """If constructor raises, the mask is rolled back and on_add must NOT fire."""
        def bad_ctor(e, v, a):
            raise RuntimeError("fail")

        pos = ecs.define_component(POS_DTYPE, constructor=bad_ctor)

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add)
        eid = ecs.create()
        with pytest.raises(RuntimeError):
            ecs.add(eid, pos)
        assert added == []


# ===========================================================================
# 9. on_add / on_remove events — deferred mode
# ===========================================================================

class TestDeferredEvents:
    def test_deferred_on_add_fires_on_run_system(self, world):
        ecs, pos, vel, *_ = world
        ecs.immediate_events = False

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos, vel], on_add=on_add)
        eid = ecs.create(); ecs.add(eid, pos); ecs.add(eid, vel)

        # Not fired yet
        assert added == []
        ecs.run_systems(0)
        assert added == [eid]

    def test_deferred_on_remove_fires_on_run_system(self, world):
        ecs, pos, vel, *_ = world
        ecs.immediate_events = False

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos, vel], on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos); ecs.add(eid, vel)
        ecs.run_systems(0)   # registers in prev_sel

        ecs.remove(eid, vel)
        assert removed == []
        ecs.run_systems(0)
        assert removed == [eid]

    def test_deferred_no_duplicate_on_add(self, world):
        """on_add should fire only once per entity across consecutive runs."""
        ecs, pos, *_ = world
        ecs.immediate_events = False

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.run_systems(0)
        ecs.run_systems(0)
        assert added.count(eid) == 1


# ===========================================================================
# 10. Batch view / set_view
# ===========================================================================

class TestBatchView:
    def test_view_returns_correct_values(self, world):
        ecs, pos, *_ = world

        e1 = ecs.create(); v = ecs.add(e1, pos); v["x"][0] = 1.0
        e2 = ecs.create(); v = ecs.add(e2, pos); v["x"][0] = 2.0
        e3 = ecs.create(); v = ecs.add(e3, pos); v["x"][0] = 3.0

        eids = np.array([e1, e2, e3], dtype=np.int64)
        xs = ecs.gather(pos, eids)["x"]
        np.testing.assert_array_almost_equal(xs, [1.0, 2.0, 3.0])

    def test_set_view_writes_back(self, world):
        ecs, pos, *_ = world

        eids = np.array([ecs.create() for _ in range(3)], dtype=np.int64)
        for e in eids:
            ecs.add(e, pos)

        new_data = np.zeros(3, dtype=POS_DTYPE)
        new_data["x"] = [10.0, 20.0, 30.0]
        ecs.scatter(pos, eids, new_data)

        result = ecs.gather(pos, eids)["x"]
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])


# ===========================================================================
# 11. Capacity growth consistency
# ===========================================================================

class TestCapacityGrowth:
    def test_components_survive_capacity_growth(self, ecs):
        pos = ecs.define_component(POS_DTYPE)

        # Fill initial capacity and overflow it
        entities = []
        for i in range(20):
            eid = ecs.create()
            v = ecs.add(eid, pos)
            v["x"][0] = float(i)
            entities.append(eid)

        for i, eid in enumerate(entities):
            assert ecs.get(eid, pos)["x"] == pytest.approx(float(i))

    def test_masks_survive_capacity_growth(self, ecs):
        pos = ecs.define_component(POS_DTYPE)
        vel = ecs.define_component(VEL_DTYPE)

        entities = [ecs.create() for _ in range(20)]
        for eid in entities:
            ecs.add(eid, pos)
            ecs.add(eid, vel)

        for eid in entities:
            assert ecs.has(eid, pos)
            assert ecs.has(eid, vel)