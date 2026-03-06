from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, NamedTuple
import numpy as np
from numpy.typing import NDArray


EntityId = int
CompId = int
SystemId = int

EidArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


# ------------------------------------------------------------
# Component and System definitions
# ------------------------------------------------------------

@dataclass
class ComponentDef:
    """Definition/metadata for a component type.

    Attributes
    ----------
    dtype : np.dtype
        NumPy dtype for the component storage (can be structured).
    constructor : Optional[Callable[[ECS, Any, Any], None]]
        Optional callback invoked when the component is added to an entity.
        Signature: (ecs, comp_view, args) where 'comp_view' is a **writable 1-element view**
        of the component array for that entity. For structured dtypes, write via
        'comp_view['field'][0] = value'.
    destructor : Optional[Callable[[ECS, Any], None]]
        Optional callback invoked when the component is removed or the entity is destroyed.
        Signature: (ecs, comp_scalar_or_view).
    """
    dtype: np.dtype
    constructor: Optional[Callable[["ECS", Any, Any], None]] = None
    destructor: Optional[Callable[["ECS", Any], None]] = None


@dataclass
class SystemDef:
    """Definition/metadata for a system."""
    require_mask: np.uint64
    exclude_mask: np.uint64
    category_mask: np.uint64
    callback: Callable[["ECS", EidArray, Any], int]
    on_add: Optional[Callable[["ECS", EidArray, Any], None]] = None
    on_remove: Optional[Callable[["ECS", EidArray, Any], None]] = None
    udata: Any = None
    active: bool = True
    _prev_sel: Optional[BoolArray] = None


class Selection(NamedTuple):
    """Result of a selection query for a system."""
    eids: EidArray
    sel: BoolArray


# ------------------------------------------------------------
# ECS implementation
# ------------------------------------------------------------

class ECS:
    """NumPy-first ECS with vectorized selection and typed component storage.

    Notes
    -----
    - Entities are integer IDs; components are SoA NumPy arrays indexed by entity ID.
    - Each entity has a uint64 bitmask 'entity_masks[eid]' of owned components (max 64).
    - Systems define 'require' and 'exclude' sets; matching is done via bit tests.
    - Systems can be grouped by a 'category_mask' and run conditionally.
    - 'on_add'/'on_remove' events:
        * If 'immediate_events=True' (default), they are emitted **immediately** by
          'add(...)', 'remove(...)', and 'destroy(...)', and '_prev_sel' is kept in sync.
        * If 'immediate_events=False', events are emitted **only** during 'run_system(...)'
          by diffing current selection against '_prev_sel'.
    """

    # ---------------------------
    # Construction and capacity
    # ---------------------------

    def __init__(self, initial_capacity: int = 1024):
        """Create an ECS with an initial entity capacity.

        Parameters
        ----------
        initial_capacity : int, default 1024
            Initial arrays length. Storage grows by powers of two as needed.

        Raises
        ------
        ValueError
            If 'initial_capacity <= 0'.
        """
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be > 0")

        self._cap: int = int(initial_capacity)
        self._next_entity: int = 0
        self._free_list: list[int] = []

        # Public toggle for immediate vs. deferred system events
        self.immediate_events: bool = True

        # Entity bookkeeping
        self.entity_active: BoolArray = np.zeros(self._cap, dtype=np.bool_)
        self.entity_ready: BoolArray = np.zeros(self._cap, dtype=np.bool_)
        self.entity_masks: NDArray[np.uint64] = np.zeros(self._cap, dtype=np.uint64)

        # Component registry (definitions + SoA arrays)
        self._components: list[ComponentDef] = []
        self._comp_arrays: list[NDArray[Any]] = []

        # Systems
        self._systems: list[SystemDef] = []

    def _ensure_capacity(self, min_cap: int) -> None:
        """Ensure internal arrays can index up to (min_cap - 1)."""
        if min_cap <= self._cap:
            return

        new_cap = self._cap
        while new_cap < min_cap:
            new_cap *= 2

        # Resize entity data
        self.entity_active = self._resize_bool(self.entity_active, new_cap)
        self.entity_ready = self._resize_bool(self.entity_ready, new_cap)
        self.entity_masks = self._resize_u64(self.entity_masks, new_cap)

        # Resize component arrays
        for i, arr in enumerate(self._comp_arrays):
            self._comp_arrays[i] = self._resize_array(arr, new_cap)

        # Resize system prev_sel
        for sys in self._systems:
            if sys._prev_sel is not None:
                sys._prev_sel = self._resize_bool(sys._prev_sel, new_cap)

        self._cap = new_cap

    @staticmethod
    def _resize_bool(a: BoolArray, new_cap: int) -> BoolArray:
        """Internal: resize a boolean array, zero-filling the tail."""
        out = np.zeros(new_cap, dtype=np.bool_)
        out[: a.shape[0]] = a
        return out

    @staticmethod
    def _resize_u64(a: NDArray[np.uint64], new_cap: int) -> NDArray[np.uint64]:
        """Internal: resize a uint64 array, zero-filling the tail."""
        out = np.zeros(new_cap, dtype=np.uint64)
        out[: a.shape[0]] = a
        return out

    @staticmethod
    def _resize_array(a: NDArray[Any], new_cap: int) -> NDArray[Any]:
        """Internal: generic array resize, zero-filling the tail."""
        out = np.zeros(new_cap, dtype=a.dtype)
        out[: a.shape[0]] = a
        return out

    # ---------------------------
    # Component API
    # ---------------------------

    def define_component(
        self,
        dtype: Any,
        constructor: Optional[Callable[["ECS", Any, Any], None]] = None,
        destructor: Optional[Callable[["ECS", Any], None]] = None,
    ) -> CompId:
        """Register a component type.

        Parameters
        ----------
        dtype : Any
            Any NumPy dtype spec (including structured dtypes).
        constructor : Optional[callable]
            Called on 'add'; gets writable 1-element view and 'args'.
        destructor : Optional[callable]
            Called on 'remove' and 'destroy'.

        Returns
        -------
        CompId
            The integer ID assigned to this component type.

        Raises
        ------
        ValueError
            If more than 64 components are defined.
        """
        dt = np.dtype(dtype)
        comp_id = len(self._components)

        if comp_id >= 64:
            raise ValueError("Max 64 components supported (uint64 mask).")

        self._components.append(ComponentDef(dt, constructor, destructor))
        self._comp_arrays.append(np.zeros(self._cap, dtype=dt))
        return comp_id

    def _comp_bit(self, comp_id: CompId) -> np.uint64:
        """Internal: compute the bit for a component ID."""
        return np.uint64(1) << np.uint64(comp_id)

    # ---------------------------
    # Entity API
    # ---------------------------

    def create(self) -> EntityId:
        """Create a new entity and return its ID."""
        if self._free_list:
            eid = self._free_list.pop()
        else:
            eid = self._next_entity
            self._next_entity += 1
            self._ensure_capacity(self._next_entity)

        self.entity_active[eid] = True
        self.entity_ready[eid] = True
        self.entity_masks[eid] = np.uint64(0)
        return eid

    def destroy(self, eid: EntityId) -> None:
        """Destroy an entity.

        Behavior
        --------
        If 'immediate_events=True':
          - Any system where the entity is currently matched receives 'on_remove([eid])'.
          - '_prev_sel[eid]' is cleared for all systems (kept in sync).
        If 'immediate_events=False':
          - No immediate callbacks; '_prev_sel' is left as-is so the next 'run_system'
            emits the removal diffs.
        Component destructors then run; the entity ID is recycled.
        """
        if not self.entity_active[eid]:
            return

        mask = self.entity_masks[eid]

        # Immediate removals for systems (if enabled)
        if self.immediate_events:
            for sys in self._systems:
                matched = self._matches_mask(sys, mask)
                if matched and sys.on_remove is not None:
                    sys.on_remove(self, np.array([eid], dtype=np.int64), sys.udata)
                if sys._prev_sel is not None:
                    sys._prev_sel[eid] = False

        # Call component destructors
        if mask != 0:
            for comp_id, comp_def in enumerate(self._components):
                if (mask >> np.uint64(comp_id)) & np.uint64(1):
                    if comp_def.destructor is not None:
                        comp_def.destructor(self, self._comp_arrays[comp_id][eid])

        # Clear and recycle
        self.entity_active[eid] = False
        self.entity_ready[eid] = False
        self.entity_masks[eid] = np.uint64(0)
        self._free_list.append(eid)

    # ---------------------------
    # Component instance API
    # ---------------------------

    def has(self, eid: EntityId, comp_id: CompId) -> bool:
        """Return True if the entity currently owns the component."""
        return bool(self.entity_masks[eid] & self._comp_bit(comp_id))

    def add(self, eid: EntityId, comp_id: CompId, args: Any = None) -> Any:
        """Add a component to an entity and return a **writable 1-element view**.

        Semantics
        ---------
        - The component mask bit is *tentatively committed before the constructor* so that
        code inside the constructor can detect ownership via 'has(eid, comp_id)'.
        - The constructor (if any) receives a **writable 1-element view** ('arr[eid:eid+1]')
        so structured dtypes can be initialized in place, e.g. 'view['x'][0] = 1.0'.
        - If the constructor raises, the mask bit and slot are **rolled back** and the
        exception is re-raised (no partial state).
        - 'on_add'/'on_remove' are fired **after** the constructor if 'immediate_events=True',
        and '_prev_sel[eid]' is kept in sync.

        Parameters
        ----------
        eid : EntityId
            Target entity.
        comp_id : CompId
            Component type identifier.
        args : Any
            Optional args passed to the constructor.

        Returns
        -------
        Any
            A writable 1-element view into the component array for 'eid'.
        """
        bit = self._comp_bit(comp_id)

        if not self.entity_ready[eid]:
            raise RuntimeError("Entity not ready")

        # Already owned -> just return a writable 1-element view
        if self.entity_masks[eid] & bit:
            return self._comp_arrays[comp_id][eid:eid + 1]

        # Prepare values
        mask_before = self.entity_masks[eid]
        mask_after = mask_before | bit

        # Zero-initialize the slot (scalar write), then get the writable slice
        self._comp_arrays[comp_id][eid] = np.zeros((), dtype=self._components[comp_id].dtype)
        comp_def = self._components[comp_id]
        comp_view_writable = self._comp_arrays[comp_id][eid:eid + 1]

        # Tentatively commit the bit to make has(eid, comp_id) true during constructor
        self.entity_masks[eid] = mask_after

        try:
            # Run constructor before on_add so on_add sees initialized data
            if comp_def.constructor is not None:
                comp_def.constructor(self, comp_view_writable, args)
        except Exception:
            # Roll back on failure: clear bit and clear slot
            # events haven't fired yet so no on_remove needed.
            self._comp_arrays[comp_id][eid] = np.zeros((), dtype=comp_def.dtype)
            self.entity_masks[eid] = mask_before
            raise

        # Immediate event notifications and prev_sel sync (optional)
        if self.immediate_events:
            for sys in self._systems:
                before = self._matches_mask(sys, mask_before)
                after = self._matches_mask(sys, mask_after)

                if not before and after and sys.on_add is not None:
                    sys.on_add(self, np.array([eid], dtype=np.int64), sys.udata)
                elif before and not after and sys.on_remove is not None:
                    # This can happen if adding this component causes exclusion in a system
                    sys.on_remove(self, np.array([eid], dtype=np.int64), sys.udata)

                if sys._prev_sel is not None:
                    sys._prev_sel[eid] = after

        return comp_view_writable

    def get(self, eid: EntityId, comp_id: CompId, *, writable: bool = False) -> Any:
        """Get the component storage for an entity.

        Parameters
        ----------
        writable : bool, default False
            If True, returns a **writable 1-element view** ('arr[eid:eid+1]').
            If False, returns the element ('arr[eid]', typically a scalar).

        Returns
        -------
        Any
            Either a NumPy scalar (read-mostly semantics for structured types) or
            a 1-element array view (writable).
        """
        return (
            self._comp_arrays[comp_id][eid:eid + 1]
            if writable
            else self._comp_arrays[comp_id][eid]
        )

    def remove(self, eid: EntityId, comp_id: CompId) -> None:
        """Remove a component from an entity.

        Behavior
        --------
        - Calls the component's destructor (if any).
        - If 'immediate_events=True', emits **immediate** system events and keeps
          '_prev_sel[eid]' in sync:
            * If membership changes True -> False: 'on_remove([eid])'
            * If membership changes False -> True (e.g., removing an *excluded* comp): 'on_add([eid])'
        - If 'immediate_events=False', no events are fired here; 'run_system' diffs will
          detect and emit events next frame.
        """
        bit = self._comp_bit(comp_id)
        if not (self.entity_masks[eid] & bit):
            return

        # Membership before/after
        mask_before = self.entity_masks[eid]
        mask_after = mask_before & ~bit

        if self.immediate_events:
            for sys in self._systems:
                before = self._matches_mask(sys, mask_before)
                after = self._matches_mask(sys, mask_after)

                if before and not after and sys.on_remove is not None:
                    sys.on_remove(self, np.array([eid], dtype=np.int64), sys.udata)
                elif not before and after and sys.on_add is not None:
                    sys.on_add(self, np.array([eid], dtype=np.int64), sys.udata)

                if sys._prev_sel is not None:
                    sys._prev_sel[eid] = after

        # Destroy data then clear mask bit
        comp_def = self._components[comp_id]
        if comp_def.destructor is not None:
            comp_def.destructor(self, self._comp_arrays[comp_id][eid])
        self._comp_arrays[comp_id][eid] = np.zeros((), dtype=comp_def.dtype)
        self.entity_masks[eid] = mask_after

    # ---------------------------
    # System API
    # ---------------------------

    def define_system(
        self,
        callback: Callable[["ECS", EidArray, Any], int],
        *,
        require: Sequence[CompId] = (),
        exclude: Sequence[CompId] = (),
        category_mask: int = 0,
        on_add: Optional[Callable[["ECS", EidArray, Any], None]] = None,
        on_remove: Optional[Callable[["ECS", EidArray, Any], None]] = None,
        udata: Any = None,
        active: bool = True,
    ) -> SystemId:
        """Define a system.

        Parameters
        ----------
        callback : (ecs, eids, udata) -> int
            The per-run function. Return nonzero to stop 'run_systems'.
        require, exclude : Sequence[CompId]
            Sets of components to require / forbid.
        category_mask : int, default 0
            Category bits for selective running (0 = matches all categories).
        on_add, on_remove : Optional callbacks
            Entity membership change notifications.
        udata : Any
            User data passed to callbacks.
        active : bool
            If False, the system is skipped.

        Returns
        -------
        SystemId
        """
        req_mask = np.uint64(0)
        exc_mask = np.uint64(0)
        for c in require:
            req_mask |= self._comp_bit(c)
        for c in exclude:
            exc_mask |= self._comp_bit(c)

        sys = SystemDef(
            require_mask=req_mask,
            exclude_mask=exc_mask,
            category_mask=np.uint64(category_mask),
            callback=callback,
            on_add=on_add,
            on_remove=on_remove,
            udata=udata,
            active=active,
            _prev_sel=None,
        )
        self._systems.append(sys)
        return len(self._systems) - 1

    def enable_system(self, sys_id: SystemId) -> None:
        """Enable a system (it will run if category mask matches)."""
        self._systems[sys_id].active = True

    def disable_system(self, sys_id: SystemId) -> None:
        """Disable a system (it will not run)."""
        self._systems[sys_id].active = False

    def set_system_mask(self, sys_id: SystemId, category_mask: int) -> None:
        """Set the category mask for a system."""
        self._systems[sys_id].category_mask = np.uint64(category_mask)

    def get_system_mask(self, sys_id: SystemId) -> int:
        """Get the category mask for a system."""
        return int(self._systems[sys_id].category_mask)

    # ---------------------------
    # Selection / Running
    # ---------------------------

    def _matches_mask(self, sys: SystemDef, mask: np.uint64) -> bool:
        """Internal: test whether a single mask matches a system."""
        if sys.exclude_mask != 0 and (mask & sys.exclude_mask) != 0:
            return False
        return (mask & sys.require_mask) == sys.require_mask

    def _select_entities(self, sys: SystemDef) -> Selection:
        """Vectorized selection (active & ready & require & ~exclude)."""
        masks = self.entity_masks
        sel: BoolArray = (
            self.entity_active
            & self.entity_ready
            & ((masks & sys.require_mask) == sys.require_mask)
            & ((masks & sys.exclude_mask) == 0)
        )
        eids: EidArray = np.flatnonzero(sel).astype(np.int64, copy=False)
        return Selection(eids=eids, sel=sel)

    def run_system(self, sys_id: SystemId, run_mask: int) -> int:
        """Run a single system if its category matches 'run_mask'.

        Mask semantics
        --------------
        - If a system's category_mask == 0, it matches all categories and runs (if active).
        - Else it runs only if '(system.category_mask & run_mask) != 0'.
        """
        sys = self._systems[sys_id]
        if not sys.active:
            return 0

        rm = np.uint64(run_mask)
        if sys.category_mask != 0 and (sys.category_mask & rm) == 0:
            return 0

        sel = self._select_entities(sys)
        eids, mask = sel.eids, sel.sel

        # Diff tracking for on_add/on_remove (works for both immediate and deferred modes).
        if sys.on_add is not None or sys.on_remove is not None:
            if sys._prev_sel is None:
                sys._prev_sel = np.zeros(self._cap, dtype=np.bool_)

            prev = sys._prev_sel
            if prev.shape[0] != self._cap:
                prev = self._resize_bool(prev, self._cap)
                sys._prev_sel = prev

            # Additions/removals since previous run
            if sys.on_add is not None:
                added = np.flatnonzero(mask & ~prev).astype(np.int64, copy=False)
                if added.size:
                    sys.on_add(self, added, sys.udata)

            if sys.on_remove is not None:
                removed = np.flatnonzero(prev & ~mask).astype(np.int64, copy=False)
                if removed.size:
                    sys.on_remove(self, removed, sys.udata)

            prev[:] = mask

        return int(sys.callback(self, eids, sys.udata))

    def run_systems(self, run_mask: int) -> int:
        """Run all systems in definition order; stop on first nonzero return code."""
        for sys_id in range(len(self._systems)):
            code = self.run_system(sys_id, run_mask)
            if code != 0:
                return code
        return 0

    # ---------------------------
    # Batch component view
    # ---------------------------

    def gather(self, comp_id: CompId, eids: EidArray | slice) -> NDArray[Any]:
        """Return data for 'eids'. Copy if advanced indexing; view if slice."""
        arr = self._comp_arrays[comp_id]
        return arr[eids] if isinstance(eids, slice) else arr[np.asarray(eids)]

    def scatter(self, comp_id: CompId, eids: EidArray | slice, data: NDArray[Any]) -> None:
        """Write data back for 'eids' (advanced-index assignment)."""
        self._comp_arrays[comp_id][eids] = data
