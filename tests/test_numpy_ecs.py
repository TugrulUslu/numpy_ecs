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
# 9. on_add / on_remove events
# ===========================================================================

class TestDeferredEvents:
    def test_deferred_on_add_fires_on_run_system(self, world):
        ecs, pos, vel, *_ = world

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

# ===========================================================================
# 12. Entity ID recycling and _prev_sel contamination
# ===========================================================================

class TestIdRecycling:
    def test_recycled_id_gets_on_add(self, world):
        """
        Destroy entity 0, create a new entity that recycles ID 0, add the
        required component, then run — the new entity MUST receive on_add.
        If _prev_sel[0] is not cleared on destroy, this silently fails.
        """
        ecs, pos, vel, hp, tag = world

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add)

        # Create, match, run to register in prev_sel
        e0 = ecs.create()
        ecs.add(e0, pos)
        ecs.run_systems(0)
        assert added == [e0]
        added.clear()

        # Destroy and recycle
        ecs.destroy(e0)
        e_new = ecs.create()
        assert e_new == e0, "expected ID to be recycled"
        ecs.add(e_new, pos)

        # New entity must trigger on_add, not be silently swallowed
        ecs.run_systems(0)
        assert added == [e_new]

    def test_recycled_id_no_spurious_on_remove(self, world):
        """
        After destroying an entity and recycling its ID without adding the
        required component, on_remove must NOT fire for the new bare entity.
        """
        ecs, pos, vel, hp, tag = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_remove=on_remove)

        e0 = ecs.create()
        ecs.add(e0, pos)
        ecs.run_systems(0)

        ecs.destroy(e0)
        e_new = ecs.create()
        assert e_new == e0

        # e_new has no components — must not trigger on_remove
        ecs.run_systems(0)
        assert removed == [e0]   # only the original destroy is detected
        removed.clear()

        ecs.run_systems(0)
        assert removed == []     # no further spurious events


# ===========================================================================
# 13. on_add / on_remove fire exactly once
# ===========================================================================

class TestEventDeduplication:
    def test_on_add_fires_once_across_runs(self, world):
        ecs, pos, vel, hp, tag = world

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add)
        eid = ecs.create(); ecs.add(eid, pos)

        for _ in range(5):
            ecs.run_systems(0)

        assert added.count(eid) == 1

    def test_on_remove_fires_once_after_remove(self, world):
        ecs, pos, vel, hp, tag = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.run_systems(0)

        ecs.remove(eid, pos)
        for _ in range(5):
            ecs.run_systems(0)

        assert removed.count(eid) == 1

    def test_add_remove_add_fires_two_on_add_one_on_remove(self, world):
        """Component added, removed, then re-added across frames."""
        ecs, pos, vel, hp, tag = world

        added = []; removed = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add, on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos)

        ecs.run_systems(0)          # on_add fires
        ecs.remove(eid, pos)
        ecs.run_systems(0)          # on_remove fires
        ecs.add(eid, pos)
        ecs.run_systems(0)          # on_add fires again

        assert added.count(eid) == 2
        assert removed.count(eid) == 1


# ===========================================================================
# 14. Excluded component interactions
# ===========================================================================

class TestExcludeMask:
    def test_adding_excluded_comp_fires_on_remove_next_run(self, world):
        """Adding a component that is excluded by a system removes the entity
        from that system's set — on_remove must fire on the next run."""
        ecs, pos, vel, hp, tag = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], exclude=[tag], on_remove=on_remove)
        eid = ecs.create(); ecs.add(eid, pos)
        ecs.run_systems(0)          # entity enters system

        ecs.add(eid, tag)           # now excluded
        ecs.run_systems(0)          # on_remove should fire
        assert removed == [eid]

    def test_removing_excluded_comp_fires_on_add_next_run(self, world):
        """Removing the excluded component lets the entity re-enter."""
        ecs, pos, vel, hp, tag = world

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], exclude=[tag], on_add=on_add)
        eid = ecs.create(); ecs.add(eid, pos); ecs.add(eid, tag)
        ecs.run_systems(0)          # excluded, no on_add yet
        assert added == []

        ecs.remove(eid, tag)        # no longer excluded
        ecs.run_systems(0)          # on_add fires
        assert added == [eid]

    def test_excluded_entity_not_passed_to_callback(self, world):
        ecs, pos, vel, hp, tag = world

        seen = []
        def cb(e, eids, ud): seen.extend(eids.tolist()); return 0

        ecs.define_system(cb, require=[pos], exclude=[tag])
        e1 = ecs.create(); ecs.add(e1, pos)
        e2 = ecs.create(); ecs.add(e2, pos); ecs.add(e2, tag)

        ecs.run_systems(0)
        assert e1 in seen
        assert e2 not in seen


# ===========================================================================
# 15. Destroy during iteration / mid-frame mutations
# ===========================================================================

class TestMidFrameMutations:
    def test_destroy_inside_system_not_seen_same_frame(self, world):
        """
        An entity destroyed inside a system callback should not corrupt the
        current eids array (it was already selected before the callback ran).
        The removal should be detected by on_remove on the *next* run.
        """
        ecs, pos, vel, hp, tag = world

        removed = []
        def on_remove(e, eids, ud): removed.extend(eids.tolist())

        def cb(e, eids, ud):
            # Destroy the first entity we see
            if eids.size:
                e.destroy(eids[0])
            return 0

        ecs.define_system(cb, require=[pos], on_remove=on_remove)
        e0 = ecs.create(); ecs.add(e0, pos)
        ecs.run_systems(0)          # e0 selected and then destroyed inside cb

        # on_remove fires on the next run
        ecs.run_systems(0)
        assert e0 in removed

    def test_add_component_inside_system(self, world):
        """Adding a component inside a system callback should not raise."""
        ecs, pos, vel, hp, tag = world

        def cb(e, eids, ud):
            for eid in eids:
                if not e.has(eid, vel):
                    e.add(eid, vel)
            return 0

        ecs.define_system(cb, require=[pos])
        eid = ecs.create(); ecs.add(eid, pos)

        ecs.run_systems(0)
        assert ecs.has(eid, vel)


# ===========================================================================
# 16. Constructor rollback correctness
# ===========================================================================

class TestConstructorRollback:
    def test_no_on_add_after_failed_constructor(self, ecs):
        added = []
        def bad_ctor(e, view, args): raise RuntimeError("fail")
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        pos = ecs.define_component(POS_DTYPE, constructor=bad_ctor)
        ecs.define_system(cb, require=[pos], on_add=on_add)

        eid = ecs.create()
        with pytest.raises(RuntimeError):
            ecs.add(eid, pos)

        ecs.run_systems(0)
        assert added == []

    def test_entity_reusable_after_failed_constructor(self, ecs):
        call_count = [0]

        def flaky_ctor(e, view, args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            view["x"][0] = 99.0

        pos = ecs.define_component(POS_DTYPE, constructor=flaky_ctor)
        eid = ecs.create()

        with pytest.raises(RuntimeError):
            ecs.add(eid, pos)

        # Second attempt must succeed
        ecs.add(eid, pos)
        assert ecs.has(eid, pos)
        assert ecs.get(eid, pos)["x"] == pytest.approx(99.0)


# ===========================================================================
# 17. Multiple systems with overlapping requirements
# ===========================================================================

class TestMultipleSystems:
    def test_each_system_tracks_prev_sel_independently(self, world):
        """Two systems with different requirements must have independent
        on_add/on_remove tracking."""
        ecs, pos, vel, hp, tag = world

        added_a = []; added_b = []
        def on_add_a(e, eids, ud): added_a.extend(eids.tolist())
        def on_add_b(e, eids, ud): added_b.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos],      on_add=on_add_a)
        ecs.define_system(cb, require=[pos, vel], on_add=on_add_b)

        eid = ecs.create()
        ecs.add(eid, pos)
        ecs.run_systems(0)
        # Only system A should fire — system B needs vel too
        assert eid in added_a
        assert eid not in added_b

        ecs.add(eid, vel)
        ecs.run_systems(0)
        # Now system B fires; system A must NOT fire again
        assert added_a.count(eid) == 1
        assert eid in added_b

    def test_run_systems_early_exit_skips_remaining(self, world):
        ecs, pos, vel, hp, tag = world

        order = []
        def cb_a(e, eids, ud): order.append("A"); return 1
        def cb_b(e, eids, ud): order.append("B"); return 0

        ecs.define_system(cb_a, require=[pos])
        ecs.define_system(cb_b, require=[pos])
        eid = ecs.create(); ecs.add(eid, pos)

        ret = ecs.run_systems(0)
        assert ret == 1
        assert order == ["A"]


# ===========================================================================
# 18. Capacity growth with active systems
# ===========================================================================

class TestCapacityGrowthWithSystems:
    def test_on_add_fires_for_entities_created_after_growth(self, ecs):
        """prev_sel must resize correctly when new entities push a capacity
        growth, so on_add fires for the newly created entities."""
        pos = ecs.define_component(POS_DTYPE)

        added = []
        def on_add(e, eids, ud): added.extend(eids.tolist())
        def cb(e, eids, ud): return 0

        ecs.define_system(cb, require=[pos], on_add=on_add)

        # Create enough entities to force at least one capacity doubling
        # (initial_capacity not set here so default 1024 is used — use small ecs)
        small = ECS(initial_capacity=2)
        p = small.define_component(POS_DTYPE)
        fired = []
        small.define_system(
            lambda e, eids, ud: 0,
            require=[p],
            on_add=lambda e, eids, ud: fired.extend(eids.tolist()),
        )

        entities = []
        for _ in range(10):
            eid = small.create()
            small.add(eid, p)
            entities.append(eid)

        small.run_systems(0)
        assert sorted(fired) == sorted(entities)

    def test_prev_sel_shape_matches_cap_after_growth(self, ecs):
        pos = ecs.define_component(POS_DTYPE)
        sid = ecs.define_system(
            lambda e, eids, ud: 0,
            require=[pos],
            on_add=lambda e, eids, ud: None,
        )

        # Trigger growth
        for _ in range(10):
            eid = ecs.create()
            ecs.add(eid, pos)

        ecs.run_systems(0)
        sys = ecs._systems[sid]
        assert sys._prev_sel is not None
        assert sys._prev_sel.shape[0] == ecs._cap


# ===========================================================================
# 19. gather / scatter edge cases
# ===========================================================================

class TestGatherScatter:
    def test_gather_empty_eids(self, world):
        ecs, pos, *_ = world
        result = ecs.gather(pos, np.array([], dtype=np.int64))
        assert result.shape[0] == 0

    def test_scatter_partial_update(self, world):
        ecs, pos, *_ = world

        eids = np.array([ecs.create() for _ in range(4)], dtype=np.int64)
        for eid in eids:
            ecs.add(eid, pos)

        # Only update a subset
        subset = eids[[0, 2]]
        data = np.zeros(2, dtype=POS_DTYPE)
        data["x"] = [11.0, 33.0]
        ecs.scatter(pos, subset, data)

        assert ecs.get(eids[0], pos)["x"] == pytest.approx(11.0)
        assert ecs.get(eids[1], pos)["x"] == pytest.approx(0.0)   # untouched
        assert ecs.get(eids[2], pos)["x"] == pytest.approx(33.0)
        assert ecs.get(eids[3], pos)["x"] == pytest.approx(0.0)   # untouched

    def test_gather_slice_returns_view(self, world):
        ecs, pos, *_ = world

        for _ in range(4):
            eid = ecs.create()
            ecs.add(eid, pos)

        view = ecs.gather(pos, slice(0, 4))
        # Modifying the view should modify the underlying array
        view["x"][0] = 55.0
        assert ecs.get(0, pos)["x"] == pytest.approx(55.0)

    def test_gather_array_returns_copy(self, world):
        ecs, pos, *_ = world

        eid = ecs.create(); ecs.add(eid, pos)
        copy = ecs.gather(pos, np.array([eid], dtype=np.int64))
        copy["x"][0] = 77.0
        # Underlying array must be unchanged
        assert ecs.get(eid, pos)["x"] == pytest.approx(0.0)


# ===========================================================================
# 20. Destructor ordering
# ===========================================================================

class TestDestructorOrdering:
    def test_destructor_sees_data_before_zeroing(self, ecs):
        """Destructor must be called with the component value before the slot
        is cleared."""
        seen = []

        def dtor(e, val):
            seen.append(float(val["x"]))

        pos = ecs.define_component(POS_DTYPE, destructor=dtor)
        eid = ecs.create()
        v = ecs.add(eid, pos); v["x"][0] = 42.0

        ecs.remove(eid, pos)
        assert seen == [pytest.approx(42.0)]
        # Slot must be zeroed after
        assert ecs.get(eid, pos)["x"] == pytest.approx(0.0)

    def test_all_destructors_called_on_destroy(self, ecs):
        calls = []
        pos = ecs.define_component(POS_DTYPE, destructor=lambda e, v: calls.append("pos"))
        vel = ecs.define_component(VEL_DTYPE, destructor=lambda e, v: calls.append("vel"))

        eid = ecs.create()
        ecs.add(eid, pos); ecs.add(eid, vel)
        ecs.destroy(eid)

        assert "pos" in calls
        assert "vel" in calls
        assert len(calls) == 2