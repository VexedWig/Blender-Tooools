# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
"""
Cable Generator  -  Geometry Nodes builder for Blender 5.1+
==========================================================

Run this from Blender's Scripting tab (Text Editor > Run Script).

It builds a "Cable Generator" Geometry Nodes modifier that lays a bundle of loose
cables and pipes between two empties against a floor and the adjacent wall,
plus a small demo scene (floor, wall, generator object, Start/End empties).

Corner convention (generator object's local space):
    floor = local Z = Floor Level        wall face = local Y = Wall Plane
    the room is on the +Y side, the wall/floor corner runs along X.
Put the generator object's origin on the wall/floor corner and rotate it so its
+Y axis points out of the wall, then move the Start / End empties around.

Re-running the script refreshes this generator's node groups and reuses its demo run.
"""

import math
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

import bpy

MAIN_NAME = "Cable Generator"
SUB_PREFIX = "CG · "
OWNER_ROLE = "rcg_cable_role"
_created_ids = ContextVar("rcg_created_ids", default=None)


@contextmanager
def _creation_scope():
    """Roll back only IDs allocated by this build, including nested operations."""
    outer = _created_ids.get()
    created = []
    token = _created_ids.set(created)
    try:
        yield
    except Exception:
        if created:
            bpy.data.batch_remove(created)
        raise
    else:
        if outer is not None:
            outer.extend(created)
    finally:
        _created_ids.reset(token)


def _track(block):
    created = _created_ids.get()
    if created is not None:
        created.append(block)
    return block


def _owned(block, role):
    return block is not None and block.library is None and block.get(OWNER_ROLE) == role


def current_group():
    recorded = bpy.context.scene.get("rcg_cable_group")
    if isinstance(recorded, bpy.types.GeometryNodeTree) and _owned(recorded, MAIN_NAME):
        return recorded
    matches = [tree for tree in bpy.data.node_groups
               if isinstance(tree, bpy.types.GeometryNodeTree) and _owned(tree, MAIN_NAME)]
    return matches[0] if len(matches) == 1 else None

SOCK = {
    "GEO": "NodeSocketGeometry", "FLOAT": "NodeSocketFloat", "INT": "NodeSocketInt",
    "BOOL": "NodeSocketBool", "VEC": "NodeSocketVector", "OBJ": "NodeSocketObject",
    "MAT": "NodeSocketMaterial", "MENU": "NodeSocketMenu", "ROT": "NodeSocketRotation",
}

FRAME_COLORS = {
    "input": (0.20, 0.26, 0.34), "strand": (0.22, 0.30, 0.25), "route": (0.32, 0.27, 0.20),
    "mesh": (0.27, 0.22, 0.32), "detail": (0.33, 0.22, 0.22), "out": (0.22, 0.22, 0.22),
}


# --------------------------------------------------------------------------------------
#  Small node-building layer
# --------------------------------------------------------------------------------------
def _available(sockets):
    return [s for s in sockets if not s.is_unavailable]


def _sock(sockets, key):
    if isinstance(key, int):
        return _available(sockets)[key]
    for s in sockets:
        if s.identifier == key and not s.is_unavailable:
            return s
    for s in _available(sockets):
        if s.name == key:
            return s
    for s in sockets:
        if s.identifier == key:
            return s
    raise KeyError(f"socket {key!r} not found in {[(s.identifier, s.name) for s in sockets]}")


class Builder:
    """Creates a node group, tracks frames, and lays the nodes out at the end."""

    def __init__(self, name, is_modifier=False, previous=None):
        self._existing = {}
        if previous is None:
            ng = _track(bpy.data.node_groups.new(name, "GeometryNodeTree"))
        else:
            # Keep socket identifiers, modifier overrides, drivers and public group references.
            ng = previous
            ng.nodes.clear()
            self._existing = {self._key(it): it for it in ng.interface.items_tree}
        ng[OWNER_ROLE] = name
        self._used = set()
        self.ng = ng
        self.ng.is_modifier = is_modifier
        self.itf = self.ng.interface
        self.frames = []
        self.cur = None
        self.loose = []
        self._gin = {}
        self._gout = None

    # ---- interface -------------------------------------------------------------------
    @staticmethod
    def _key(it):
        if it.item_type == "PANEL":
            return ("PANEL", it.name)
        return ("SOCKET", it.in_out, it.name, it.socket_type)

    def _reuse(self, key):
        it = self._existing.get(key)
        if it is not None:
            self._used.add(key)
        return it

    def finish_interface(self):
        """Drop interface items left over from an older build that were not reused."""
        for key, it in list(self._existing.items()):
            if key not in self._used and it.item_type == "SOCKET":
                self.itf.remove(it)
        for key, it in list(self._existing.items()):
            if key not in self._used and it.item_type == "PANEL":
                self.itf.remove(it)

    def panel(self, name, closed=True, desc=""):
        p = self._reuse(("PANEL", name))
        if p is None:
            p = self.itf.new_panel(name, default_closed=closed)
        p.description = desc
        return p

    def inp(self, name, kind, default=None, mn=None, mx=None, sub=None, desc="", panel=None,
            toggle=False, single=True):
        s = self._reuse(("SOCKET", "INPUT", name, SOCK[kind]))
        if s is None:
            s = self.itf.new_socket(name, in_out="INPUT", socket_type=SOCK[kind], parent=panel)
        if sub:
            s.subtype = sub
        if mn is not None:
            s.min_value = mn
        if mx is not None:
            s.max_value = mx
        if default is not None and kind not in ("MENU",):
            s.default_value = default
        s.description = desc
        if toggle:
            s.is_panel_toggle = True
        if single and kind in ("FLOAT", "INT", "BOOL", "VEC", "ROT"):
            try:
                s.structure_type = "SINGLE"
            except Exception:
                try:
                    s.force_non_field = True
                except Exception:
                    pass
        return s

    def outp(self, name, kind, desc=""):
        s = self._reuse(("SOCKET", "OUTPUT", name, SOCK[kind]))
        if s is None:
            s = self.itf.new_socket(name, in_out="OUTPUT", socket_type=SOCK[kind])
        s.description = desc
        return s

    # ---- frames / bookkeeping ---------------------------------------------------------
    @contextmanager
    def frame(self, label, color="route", slot=None, note=None):
        f = self.ng.nodes.new("NodeFrame")
        f.label = label
        f.label_size = 18
        f.shrink = True
        f.use_custom_color = True
        f.color = FRAME_COLORS.get(color, color)
        rec = {"frame": f, "nodes": [], "slot": slot if slot is not None else len(self.frames)}
        self.frames.append(rec)
        prev, self.cur = self.cur, rec
        try:
            yield rec
        finally:
            self.cur = prev

    def _reg(self, n):
        (self.cur["nodes"] if self.cur is not None else self.loose).append(n)

    def gin(self, name):
        """Group input socket, using one Group Input node per frame (no long wires)."""
        key = id(self.cur) if self.cur is not None else "loose"
        n = self._gin.get(key)
        if n is None:
            n = self.ng.nodes.new("NodeGroupInput")
            self._reg(n)
            self._gin[key] = n
        return n.outputs[name]

    def gout(self):
        if self._gout is None:
            self._gout = self.ng.nodes.new("NodeGroupOutput")
            self._gout.is_active_output = True
            self._reg(self._gout)
        return self._gout

    def set_out(self, name, sock):
        self.ng.links.new(sock, self.gout().inputs[name])

    # ---- generic node -------------------------------------------------------------------
    def n(self, idname, label=None, ins=None, **props):
        node = self.ng.nodes.new(idname)
        for k, v in props.items():
            setattr(node, k, v)
        if label:
            node.label = label
        self._reg(node)
        for k, v in (ins or {}).items():
            self.c(node, k, v)
        return node

    def c(self, node, key, val):
        if val is None:
            return
        s = _sock(node.inputs, key)
        if isinstance(val, bpy.types.NodeSocket):
            self.ng.links.new(val, s)
        else:
            if s.type == "VECTOR" and isinstance(val, (int, float)):
                val = (val, val, val)
            s.default_value = val

    def group(self, sub, label=None, **ins):
        node = self.n("GeometryNodeGroup", label=label)
        node.node_tree = sub
        node.width = 190
        for k, v in ins.items():
            self.c(node, k.replace("_", " "), v)
        return node

    # ---- expression helpers (return output sockets) ----------------------------------------
    def math(self, op, a, b=None, c=None, label=None, clamp=False):
        node = self.n("ShaderNodeMath", label=label, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.c(node, i, v)
        return node.outputs[0]

    def imath(self, op, a, b=None, label=None):
        node = self.n("FunctionNodeIntegerMath", label=label, operation=op)
        self.c(node, 0, a)
        if b is not None:
            self.c(node, 1, b)
        return node.outputs[0]

    def vmath(self, op, a, b=None, c=None, scale=None, label=None):
        node = self.n("ShaderNodeVectorMath", label=label, operation=op)
        self.c(node, "Vector", a)
        self.c(node, "Vector_001", b)
        self.c(node, "Vector_002", c)
        self.c(node, "Scale", scale)
        if op in ("DOT_PRODUCT", "DISTANCE", "LENGTH"):
            return node.outputs["Value"]
        return node.outputs["Vector"]

    def mix(self, a, b, f, kind="FLOAT", label=None, clamp=True):
        node = self.n("ShaderNodeMix", label=label, data_type=kind, clamp_factor=clamp)
        suf = {"FLOAT": "Float", "VECTOR": "Vector", "ROTATION": "Rotation"}[kind]
        self.c(node, "Factor_Float", f)
        self.c(node, "A_" + suf, a)
        self.c(node, "B_" + suf, b)
        return _sock(node.outputs, "Result_" + suf)

    def maprange(self, v, fmin, fmax, tmin, tmax, interp="LINEAR", clamp=True, label=None):
        node = self.n("ShaderNodeMapRange", label=label, data_type="FLOAT",
                      interpolation_type=interp, clamp=clamp)
        for k, val in zip(("Value", "From Min", "From Max", "To Min", "To Max"),
                          (v, fmin, fmax, tmin, tmax)):
            self.c(node, k, val)
        return node.outputs["Result"]

    def combine(self, x=0.0, y=0.0, z=0.0, label=None):
        node = self.n("ShaderNodeCombineXYZ", label=label, ins={"X": x, "Y": y, "Z": z})
        return node.outputs[0]

    def separate(self, v, label=None):
        node = self.n("ShaderNodeSeparateXYZ", label=label, ins={"Vector": v})
        return node.outputs["X"], node.outputs["Y"], node.outputs["Z"]

    def rand(self, kind, mn=None, mx=None, prob=None, id=None, seed=None, label=None):
        node = self.n("FunctionNodeRandomValue", label=label, data_type=kind)
        if kind == "FLOAT":
            self.c(node, "Min_001", mn); self.c(node, "Max_001", mx); out = "Value_001"
        elif kind == "FLOAT_VECTOR":
            self.c(node, "Min", mn); self.c(node, "Max", mx); out = "Value"
        elif kind == "INT":
            self.c(node, "Min_002", mn); self.c(node, "Max_002", mx); out = "Value_002"
        else:
            self.c(node, "Probability", prob); out = "Value_003"
        self.c(node, "ID", id)
        self.c(node, "Seed", seed)
        return _sock(node.outputs, out)

    def switch(self, cond, false, true, kind="FLOAT", label=None):
        node = self.n("GeometryNodeSwitch", label=label, input_type=kind)
        self.c(node, "Switch", cond)
        self.c(node, "False", false)
        self.c(node, "True", true)
        return node.outputs["Output"]

    def compare(self, op, a, b, kind="FLOAT", label=None):
        node = self.n("FunctionNodeCompare", label=label, data_type=kind, operation=op)
        ka, kb = {"FLOAT": ("A", "B"), "INT": ("A_INT", "B_INT")}[kind]
        self.c(node, ka, a)
        self.c(node, kb, b)
        return node.outputs["Result"]

    def bmath(self, op, a, b=None, label=None):
        node = self.n("FunctionNodeBooleanMath", label=label, operation=op)
        self.c(node, 0, a)
        if b is not None:
            self.c(node, 1, b)
        return node.outputs[0]

    def attr(self, name, kind="FLOAT", label=None):
        node = self.n("GeometryNodeInputNamedAttribute", label=label or name, data_type=kind,
                      ins={"Name": name})
        node.width = 150
        return node.outputs["Attribute"]

    def store(self, geo, name, value, kind="FLOAT", domain="POINT", sel=None, label=None):
        node = self.n("GeometryNodeStoreNamedAttribute", label=label, data_type=kind, domain=domain,
                      ins={"Geometry": geo, "Name": name, "Value": value, "Selection": sel})
        return node.outputs["Geometry"]

    def setpos(self, geo, pos=None, offset=None, sel=None, label=None):
        node = self.n("GeometryNodeSetPosition", label=label,
                      ins={"Geometry": geo, "Position": pos, "Offset": offset, "Selection": sel})
        return node.outputs["Geometry"]

    def join(self, *geos, label=None):
        node = self.n("GeometryNodeJoinGeometry", label=label)
        for g in reversed(geos):  # Join stacks the latest link on top
            self.ng.links.new(g, node.inputs[0])
        return node.outputs[0]

    def transform(self, geo, t=None, s=None, label=None):
        node = self.n("GeometryNodeTransform", label=label,
                      ins={"Geometry": geo, "Translation": t, "Scale": s})
        return node.outputs[0]

    def index(self):
        return self.n("GeometryNodeInputIndex").outputs[0]

    def position(self):
        return self.n("GeometryNodeInputPosition").outputs[0]


# --------------------------------------------------------------------------------------
#  Layout: columns by dependency depth inside each frame, frames left -> right
# --------------------------------------------------------------------------------------
_PROP_ROWS = {
    "ShaderNodeMath": 2, "ShaderNodeVectorMath": 1, "ShaderNodeMapRange": 3, "ShaderNodeMix": 3,
    "FunctionNodeCompare": 2, "FunctionNodeRandomValue": 1, "GeometryNodeSwitch": 1,
    "GeometryNodeStoreNamedAttribute": 2, "GeometryNodeInputNamedAttribute": 1,
    "GeometryNodeAccumulateField": 2, "FunctionNodeIntegerMath": 1, "FunctionNodeBooleanMath": 1,
    "GeometryNodeCurveSplineType": 1, "GeometryNodeSeparateGeometry": 1,
    "GeometryNodeDeleteGeometry": 2, "GeometryNodeCurveToPoints": 1, "GeometryNodeObjectInfo": 1,
    "GeometryNodeFieldOnDomain": 2, "GeometryNodeFieldAtIndex": 2, "GeometryNodeMeshCylinder": 1,
    "GeometryNodeMeshCone": 1, "GeometryNodeDuplicateElements": 1, "GeometryNodeSetShadeSmooth": 1,
    "FunctionNodeAlignRotationToVector": 2, "FunctionNodeAxesToRotation": 2,
    "ShaderNodeVectorRotate": 2, "GeometryNodeMenuSwitch": 1, "GeometryNodeMeshCircle": 1,
    "GeometryNodeGroup": 1, "GeometryNodeAttributeDomainSize": 1, "GeometryNodeWarning": 1,
    "GeometryNodeCurvePrimitiveQuadrilateral": 1, "GeometryNodeRemoveAttribute": 0,
}


def _node_height(n):
    if n.bl_idname == "NodeGroupInput":
        return 40 + 22 * len([s for s in n.outputs if not s.hide and s.is_linked])
    rows = 0
    for s in n.outputs:
        if not s.is_unavailable and not s.hide:
            rows += 1
    for s in n.inputs:
        if s.is_unavailable or s.hide:
            continue
        rows += 1
        if s.type == "VECTOR" and not s.is_linked and not s.hide_value:
            rows += 3
    return int((44 + 22 * rows + 28 * _PROP_ROWS.get(n.bl_idname, 0)) * 1.08)


def _tidy_sockets(ng):
    """Hide unused outputs on group inputs and multi-output nodes (Ctrl+H style)."""
    for n in ng.nodes:
        if n.bl_idname in ("NodeGroupInput", "GeometryNodeObjectInfo", "ShaderNodeSeparateXYZ",
                           "GeometryNodeSplineParameter", "GeometryNodeCurveToPoints",
                           "GeometryNodeInputMeshEdgeAngle", "GeometryNodeOffsetPointInCurve",
                           "GeometryNodeCurveOfPoint", "GeometryNodeAccumulateField",
                           "GeometryNodeMeshCylinder", "GeometryNodeMeshCone", "GeometryNodeMeshCube",
                           "GeometryNodeSplineLength", "GeometryNodeInputNamedAttribute"):
            for s in n.outputs:
                if not s.is_linked:
                    s.hide = True
        if n.bl_idname == "NodeGroupInput":
            n.width = 180


def layout(B):
    ng = B.ng
    B.finish_interface()
    _tidy_sockets(ng)
    links = [(l.from_node, l.to_node) for l in ng.links]
    order = {n: i for i, n in enumerate(ng.nodes)}

    def place(nodes, x0, y0):
        S = set(nodes)
        preds = defaultdict(list)
        for a, b in links:
            if a in S and b in S and a is not b:
                preds[b].append(a)
        depth = {n: 0 for n in nodes}
        for _ in range(len(nodes) + 1):
            changed = False
            for n in nodes:
                d = max((depth[p] + 1 for p in preds[n]), default=0)
                if d != depth[n]:
                    depth[n] = d
                    changed = True
            if not changed:
                break
        cols = defaultdict(list)
        for n in nodes:
            cols[depth[n]].append(n)
        ypos = {}
        x = x0 + 40
        bottom = y0
        for d in sorted(cols):
            col = cols[d]
            col.sort(key=lambda n: (sum(ypos[p] for p in preds[n] if p in ypos) /
                                    max(1, len([p for p in preds[n] if p in ypos]))
                                    if any(p in ypos for p in preds[n]) else 1e9 - order[n]),
                     reverse=True)
            y = y0 - 70
            for n in col:
                if n.parent is None and B.cur is None:
                    pass
                ypos[n] = y
                n.location_absolute = (x, y)
                y -= _node_height(n) + 40
            bottom = min(bottom, y)
            x += max(n.width for n in col) + 70
        return x - x0, y0 - bottom

    # frames grouped into slots (same slot = stacked vertically)
    slots = defaultdict(list)
    for rec in B.frames:
        slots[rec["slot"]].append(rec)
        for n in rec["nodes"]:
            n.parent = rec["frame"]
    x = 0
    for slot in sorted(slots):
        y = 0
        widest = 0
        for rec in slots[slot]:
            w, h = place(rec["nodes"], x, y)
            widest = max(widest, w)
            y -= h + 160
        x += widest + 200
    # loose nodes (group output etc.)
    if B.loose:
        place(B.loose, x, 0)


# --------------------------------------------------------------------------------------
#  Sub-groups (small, reusable, each does one job)
# --------------------------------------------------------------------------------------
def build_anchor_frame():
    B = Builder(SUB_PREFIX + "Anchor Frame")
    B.inp("Object", "OBJ", desc="Anchor empty", single=False)
    B.inp("Corner", "VEC", desc="Corner origin (0, wall plane, floor level)", single=False)
    B.inp("Fallback Position", "VEC", desc="Used when no empty is assigned (corner space)", single=False)
    B.inp("Fallback Rotation", "ROT", desc="Used when no empty is assigned", single=False)
    for nm in ("Position", "Exit Dir", "Side Dir", "Row Dir"):
        B.outp(nm, "VEC")
    with B.frame("Anchor empty -> position + axes in corner space (default anchor if none set)", "input"):
        oi = B.n("GeometryNodeObjectInfo", transform_space="RELATIVE",
                 ins={"Object": B.gin("Object"), "As Instance": True})
        size = B.n("GeometryNodeAttributeDomainSize", component="INSTANCES", ins={"Geometry": oi.outputs["Geometry"]})
        unset = B.compare("EQUAL", size.outputs["Instance Count"], 0, "INT", label="no empty assigned?")
        warn = B.n("GeometryNodeWarning", warning_type="INFO", label="tell the user",
                   ins={"Show": unset, "Message": "Start / End empty not assigned: using a default anchor "
                                                  "(set them in the Anchors panel)"})
        use_fb = warn.outputs["Show"]
        local = B.vmath("SUBTRACT", oi.outputs["Location"], B.gin("Corner"), label="position")
        pos = B.switch(use_fb, local, B.gin("Fallback Position"), "VECTOR", label="position")
        rot = B.switch(use_fb, oi.outputs["Rotation"], B.gin("Fallback Rotation"), "ROTATION", label="rotation")
        axes = []
        for lbl, v in (("exit dir = local Z", (0, 0, 1)), ("side dir = local X", (1, 0, 0)),
                       ("row dir = local Y", (0, 1, 0))):
            rv = B.n("FunctionNodeRotateVector", label=lbl, ins={"Vector": v, "Rotation": rot})
            axes.append(rv.outputs[0])
        B.set_out("Position", pos)
        B.set_out("Exit Dir", axes[0])
        B.set_out("Side Dir", axes[1])
        B.set_out("Row Dir", axes[2])
    layout(B)
    return B.ng


def build_unfold_anchor():
    """Map an anchor into the 'unfolded' wall+floor plane: (x, s) with s = y - z."""
    B = Builder(SUB_PREFIX + "Unfold Anchor")
    B.inp("Point", "VEC", single=False)
    B.inp("Radius", "FLOAT", single=False)
    B.inp("Pins", "VEC", desc="x = wall pin, y = floor pin", single=False)
    B.inp("Loose Gap", "FLOAT", single=False)
    B.outp("Plane", "VEC", desc="(x, s, 0): x along the corner, s < 0 on the wall, s > 0 on the floor")
    B.outp("Height", "FLOAT", desc="Distance kept from the surface at this anchor")
    with B.frame("Unfold: wall + floor become one flat plane", "route"):
        x, y, z = B.separate(B.gin("Point"))
        on_floor = B.compare("GREATER_THAN", y, z, label="on floor?")
        px, py, _ = B.separate(B.gin("Pins"), label="pins")
        pin = B.mix(px, py, on_floor, label="pin here")
        free = B.math("SUBTRACT", 1.0, pin, label="1 - pin")
        h = B.math("MULTIPLY_ADD", free, B.gin("Loose Gap"), B.gin("Radius"), label="height = r + (1-pin) gap")
        lifted = B.vmath("MAXIMUM", B.gin("Point"), h, label="lift off surfaces")
        _, ly, lz = B.separate(lifted)
        s = B.math("SUBTRACT", ly, lz, label="s = y - z")
        B.set_out("Plane", B.combine(x, s, 0.0, label="(x, s)"))
        B.set_out("Height", h)
    layout(B)
    return B.ng


def build_drop_window():
    B = Builder(SUB_PREFIX + "Drop Window")
    B.inp("S Start", "FLOAT", single=False)
    B.inp("S End", "FLOAT", single=False)
    B.inp("Drop", "FLOAT", single=False)
    B.outp("T0", "FLOAT", desc="Route factor where sideways travel starts")
    B.outp("T1", "FLOAT", desc="Route factor where sideways travel ends")
    with B.frame("Drop: hang straight down the wall first, travel sideways later", "route"):
        sa, sb = B.gin("S Start"), B.gin("S End")
        ha = B.math("MAXIMUM", B.math("MULTIPLY", sa, -1.0), 0.0, label="wall height at start")
        hb = B.math("MAXIMUM", B.math("MULTIPLY", sb, -1.0), 0.0, label="wall height at end")
        den = B.math("ADD", B.math("ADD", B.math("ABSOLUTE", sa), B.math("ABSOLUTE", sb)), 1e-4,
                     label="route length along s")
        k = B.math("DIVIDE", B.gin("Drop"), den)
        t0 = B.math("MULTIPLY", ha, k, label="drop share at start")
        t1 = B.math("SUBTRACT", 1.0, B.math("MULTIPLY", hb, k), label="1 - drop share at end")
        mid = B.math("MULTIPLY", B.math("ADD", t0, t1), 0.5, label="centre")
        half = B.math("MAXIMUM", B.math("MULTIPLY", B.math("SUBTRACT", t1, t0), 0.5), 0.18,
                      label="half width (min 0.18, no boxy U)")
        B.set_out("T0", B.math("MAXIMUM", B.math("SUBTRACT", mid, half), 0.0, label="T0"))
        B.set_out("T1", B.math("MINIMUM", B.math("ADD", mid, half), 1.0, label="T1"))
    layout(B)
    return B.ng


def build_fold_corner():
    """Fold the unfolded (x, s, h) route back onto wall + floor with a rounded corner.

    y = h + (s + q) / 2,  z = h + (q - s) / 2,  q = sqrt(s^2 + c^2)
    -> a hyperbola (y - h)(z - h) = c^2 / 4 whose asymptotes are the wall and floor.
    """
    B = Builder(SUB_PREFIX + "Fold Corner")
    B.inp("Position", "VEC", desc="(x, s, h) in the unfolded plane", single=False)
    B.inp("Bend", "FLOAT", desc="Corner rounding (sqrt(2) x bend radius)", single=False)
    B.outp("Position", "VEC")
    with B.frame("Fold the plane onto wall + floor (smooth hyperbolic corner)", "route"):
        x, s, h = B.separate(B.gin("Position"), label="x, s, h")
        c = B.gin("Bend")
        cc = B.math("MULTIPLY", c, c, label="c^2")
        q = B.math("SQRT", B.math("MULTIPLY_ADD", s, s, cc, label="s^2 + c^2"), label="q")
        hq = B.math("MULTIPLY_ADD", q, 0.5, h, label="h + q/2")
        base = B.combine(x, hq, hq)
        p = B.vmath("MULTIPLY_ADD", (0.0, 0.5, -0.5), s, base, label="+/- s/2 on y / z")
        B.set_out("Position", p)
    layout(B)
    return B.ng


def build_align_ends():
    B = Builder(SUB_PREFIX + "Align Ends")
    B.inp("Position", "VEC", single=False)
    B.inp("A", "VEC", desc="Start anchor", single=False)
    B.inp("Dir A", "VEC", single=False)
    B.inp("B", "VEC", desc="End anchor", single=False)
    B.inp("Dir B", "VEC", single=False)
    B.inp("Exit", "FLOAT", desc="Straight length", single=False)
    B.inp("Blend", "FLOAT", desc="Length over which the straight exit eases into the route", single=False)
    B.outp("Position", "VEC")
    with B.frame("Leave each anchor straight along its arrow, then ease into the route", "route"):
        sp = B.n("GeometryNodeSplineParameter")
        sl = B.n("GeometryNodeSplineLength")
        d = sp.outputs["Length"]
        de = B.math("SUBTRACT", sl.outputs["Length"], d, label="distance to end")
        eb = B.math("ADD", B.gin("Exit"), B.gin("Blend"), label="exit + blend")
        ws = B.maprange(d, eb, B.gin("Exit"), 0.0, 1.0, "SMOOTHSTEP", label="weight start")
        we = B.maprange(de, eb, B.gin("Exit"), 0.0, 1.0, "SMOOTHSTEP", label="weight end")
        la = B.vmath("MULTIPLY_ADD", B.gin("Dir A"), d, B.gin("A"), label="straight line from A")
        lb = B.vmath("MULTIPLY_ADD", B.gin("Dir B"), de, B.gin("B"), label="straight line from B")
        p = B.mix(B.gin("Position"), la, ws, "VECTOR", label="ease start")
        p = B.mix(p, lb, we, "VECTOR", label="ease end")
        B.set_out("Position", p)
    layout(B)
    return B.ng


def build_adaptive():
    """Keep a point only where the accumulated bend angle or length crosses a step."""
    B = Builder(SUB_PREFIX + "Adaptive Simplify")
    B.inp("Curves", "GEO")
    B.inp("Angle", "FLOAT", sub="ANGLE", single=False)
    B.inp("Max Length", "FLOAT", sub="DISTANCE", single=False)
    B.outp("Curves", "GEO")
    with B.frame("Turn angle at each point", "mesh"):
        pos = B.position()
        prv = B.n("GeometryNodeOffsetPointInCurve", label="previous", ins={"Offset": -1})
        nxt = B.n("GeometryNodeOffsetPointInCurve", label="next", ins={"Offset": 1})
        pp = B.n("GeometryNodeFieldAtIndex", label="prev position", domain="POINT", data_type="FLOAT_VECTOR",
                 ins={"Value": pos, "Index": prv.outputs["Point Index"]}).outputs[0]
        pn = B.n("GeometryNodeFieldAtIndex", label="next position", domain="POINT", data_type="FLOAT_VECTOR",
                 ins={"Value": pos, "Index": nxt.outputs["Point Index"]}).outputs[0]
        d1 = B.vmath("NORMALIZE", B.vmath("SUBTRACT", pos, pp))
        d2 = B.vmath("NORMALIZE", B.vmath("SUBTRACT", pn, pos))
        cosv = B.math("MINIMUM", B.vmath("DOT_PRODUCT", d1, d2), 1.0)
        ang = B.math("ARCCOSINE", cosv, label="turn angle")
        valid = B.bmath("AND", prv.outputs["Is Valid Offset"], nxt.outputs["Is Valid Offset"])
        ang = B.math("MULTIPLY", ang, valid, label="0 at ends")
    with B.frame("Bucket by accumulated angle + length, keep first point of each bucket", "mesh"):
        cop = B.n("GeometryNodeCurveOfPoint")
        acc = B.n("GeometryNodeAccumulateField", data_type="FLOAT", domain="POINT",
                  ins={"Value": ang, "Group Index": cop.outputs["Curve Index"]})
        sp = B.n("GeometryNodeSplineParameter")
        ka = B.math("FLOOR", B.math("DIVIDE", acc.outputs["Leading"], B.gin("Angle")), label="angle step")
        kl = B.math("FLOOR", B.math("DIVIDE", sp.outputs["Length"], B.gin("Max Length")), label="length step")
        key = B.math("ADD", ka, kl, label="bucket")
        kp = B.n("GeometryNodeFieldAtIndex", label="previous bucket", domain="POINT", data_type="FLOAT",
                 ins={"Value": key, "Index": prv.outputs["Point Index"]}).outputs[0]
        changed = B.compare("NOT_EQUAL", key, kp, label="new bucket?")
        keep = B.bmath("OR", changed, B.bmath("NOT", valid), label="keep (or end point)")
        dg = B.n("GeometryNodeDeleteGeometry", domain="POINT", mode="ALL",
                 ins={"Geometry": B.gin("Curves"), "Selection": B.bmath("NOT", keep)})
        B.set_out("Curves", dg.outputs[0])
    layout(B)
    return B.ng


def build_strand_profile():
    B = Builder(SUB_PREFIX + "Strand Profile")
    B.inp("Strands", "INT", 1, 1, 4, single=False)
    B.inp("Sides", "INT", 8, 3, 64, single=False)
    B.outp("Profile", "GEO")
    with B.frame("1 round wire, or 2-4 twisted wires packed in a unit circle", "mesh"):
        n = B.gin("Strands")
        sinv = B.math("SINE", B.math("DIVIDE", math.pi, n), label="sin(pi / n)")
        rs = B.math("DIVIDE", sinv, B.math("ADD", sinv, 1.0), label="wire radius")
        multi = B.compare("GREATER_THAN", n, 1, "INT", label="twisted?")
        rs = B.switch(multi, 1.0, rs, label="wire radius")
        off = B.math("SUBTRACT", 1.0, rs, label="wire offset")
        ang = B.math("MULTIPLY", B.math("DIVIDE", B.index(), n), 2 * math.pi, label="angle")
        px = B.math("MULTIPLY", B.math("COSINE", ang), off)
        py = B.math("MULTIPLY", B.math("SINE", ang), off)
        pts = B.n("GeometryNodePoints", ins={"Count": n, "Position": B.combine(px, py, 0.0)})
        circ = B.n("GeometryNodeCurvePrimitiveCircle", ins={"Resolution": B.gin("Sides"), "Radius": rs})
        iop = B.n("GeometryNodeInstanceOnPoints", ins={"Points": pts.outputs[0], "Instance": circ.outputs[0]})
        real = B.n("GeometryNodeRealizeInstances", ins={"Geometry": iop.outputs[0]})
        sp = B.n("GeometryNodeSplineParameter")
        prof = B.store(real.outputs[0], "cg_v", sp.outputs["Factor"], label="cg_v (around)")
        B.set_out("Profile", prof)
    layout(B)
    return B.ng


def build_tube():
    B = Builder(SUB_PREFIX + "Tube")
    B.inp("Curves", "GEO")
    B.inp("Profile", "GEO")
    B.inp("Fill Caps", "BOOL", True, single=False)
    B.inp("Part", "INT", 0, single=False)
    B.outp("Mesh", "GEO")
    with B.frame("Sweep the profile, radius from cg_r", "mesh"):
        c2m = B.n("GeometryNodeCurveToMesh", ins={"Curve": B.gin("Curves"), "Profile Curve": B.gin("Profile"),
                                                  "Scale": B.attr("cg_r"), "Fill Caps": B.gin("Fill Caps")})
    with B.frame("UVs: U around (seam fixed), V along at matching texel density", "mesh"):
        v = B.attr("cg_v")
        favg = B.n("GeometryNodeFieldOnDomain", label="face average", domain="FACE", data_type="FLOAT",
                   ins={"Value": v}).outputs[0]
        seam = B.compare("LESS_THAN", v, B.math("SUBTRACT", favg, 0.25), label="seam corner?")
        u = B.math("ADD", v, seam, label="U")
        circ = B.math("MULTIPLY", B.attr("cg_r"), 2 * math.pi, label="circumference")
        vv = B.math("DIVIDE", B.attr("cg_d"), circ, label="V")
        g = B.store(c2m.outputs[0], "UVMap", B.combine(u, vv, 0.0), "FLOAT2", "CORNER", label="UVMap")
        g = B.store(g, "cg_part", B.gin("Part"), "INT", "FACE", label="cg_part")
        B.set_out("Mesh", g)
    layout(B)
    return B.ng


def build_box_profile():
    B = Builder(SUB_PREFIX + "Box Profile")
    B.inp("Width", "FLOAT", 1.0, single=False)
    B.inp("Height", "FLOAT", 1.0, single=False)
    B.outp("Profile", "GEO")
    with B.frame("Rectangle profile with cg_v for UVs", "mesh"):
        q = B.n("GeometryNodeCurvePrimitiveQuadrilateral", mode="RECTANGLE",
                ins={"Width": B.gin("Width"), "Height": B.gin("Height")})
        sp = B.n("GeometryNodeSplineParameter")
        B.set_out("Profile", B.store(q.outputs[0], "cg_v", sp.outputs["Factor"], label="cg_v (around)"))
    layout(B)
    return B.ng


def _prism(B, S, profile, rb, rt, z0, z1, part, label, offset=None):
    """Swept solid along local Z (built from curves, so no default material slot; gets UVs)."""
    line = B.n("GeometryNodeCurvePrimitiveLine", label=label, ins={"Start": (0.0, 0.0, z0), "End": (0.0, 0.0, z1)})
    sp = B.n("GeometryNodeSplineParameter")
    g = B.store(line.outputs[0], "cg_r", B.mix(rb, rt, sp.outputs["Factor"]), label="cg_r")
    g = B.store(g, "cg_d", sp.outputs["Length"], label="cg_d")
    mesh = B.group(S["tube"], label, Curves=g, Profile=profile, Fill_Caps=True, Part=part).outputs[0]
    return B.transform(mesh, offset) if offset is not None else mesh


def _circle(B, S, sides, label="circle profile"):
    return B.group(S["profile"], label, Strands=1, Sides=sides).outputs[0]


def build_connector_mesh(S):
    """Unit connector: radius 1 = connector radius, z 0..1 = exit length (anchor at z=0)."""
    B = Builder(SUB_PREFIX + "Connector Mesh")
    B.inp("Sides", "INT", 12, 3, 64, single=False)
    B.outp("Mesh", "GEO")
    with B.frame("Hex nut + body + strain-relief boot (unit size)", "detail"):
        hexp = _circle(B, S, 6, "hex profile")
        rnd = _circle(B, S, B.gin("Sides"), "round profile")
        nut = _prism(B, S, hexp, 1.22, 1.22, 0.0, 0.22, 2, "hex nut")
        body = _prism(B, S, rnd, 1.0, 1.0, 0.22, 0.68, 2, "body")
        boot = _prism(B, S, rnd, 0.86, 0.6, 0.68, 1.0, 2, "boot")
        B.set_out("Mesh", B.join(nut, body, boot))
    layout(B)
    return B.ng


def build_tie_mesh(S):
    B = Builder(SUB_PREFIX + "Tie Mesh")
    B.inp("Sides", "INT", 12, 3, 64, single=False)
    B.outp("Mesh", "GEO")
    with B.frame("Zip tie: band + locking head (unit size)", "detail"):
        band = _prism(B, S, _circle(B, S, B.gin("Sides")), 1.14, 1.14, -0.5, 0.5, 3, "band")
        boxp = B.group(S["box"], "head profile", Width=0.75, Height=0.55).outputs[0]
        head = _prism(B, S, boxp, 1.0, 1.0, -0.65, 0.65, 3, "head", offset=(0.0, -1.32, 0.0))
        B.set_out("Mesh", B.join(band, head))
    layout(B)
    return B.ng


def build_clip_mesh(S):
    B = Builder(SUB_PREFIX + "Clip Mesh")
    B.inp("Sides", "INT", 12, 3, 64, single=False)
    B.outp("Mesh", "GEO")
    with B.frame("Saddle clip: band + feet resting on the surface at y = -1 (unit size)", "detail"):
        band = _prism(B, S, _circle(B, S, B.gin("Sides")), 1.16, 1.16, -0.5, 0.5, 3, "band")
        boxp = B.group(S["box"], "feet profile", Width=4.4, Height=0.16).outputs[0]
        feet = _prism(B, S, boxp, 1.0, 1.0, -0.5, 0.5, 3, "feet", offset=(0.0, -0.92, 0.0))
        B.set_out("Mesh", B.join(band, feet))
    layout(B)
    return B.ng


def build_flange_mesh(S):
    B = Builder(SUB_PREFIX + "Flange Mesh")
    B.inp("Sides", "INT", 16, 3, 64, single=False)
    B.inp("Flange Scale", "FLOAT", 1.5, single=False)
    B.inp("Bolts", "INT", 6, 0, 32, single=False)
    B.outp("Mesh", "GEO")
    with B.frame("Flange disc + bolt ring (unit size, z 0..1 = thickness)", "detail"):
        fs = B.gin("Flange Scale")
        disc = _prism(B, S, _circle(B, S, B.gin("Sides")), fs, fs, 0.0, 1.0, 3, "disc")
        bolt = _prism(B, S, _circle(B, S, 6, "hex profile"), 0.1, 0.1, 0.0, 1.45, 3, "bolt")
        ring_r = B.math("MULTIPLY_ADD", fs, 0.5, 0.5, label="bolt circle")
        circle = B.n("GeometryNodeMeshCircle", fill_type="NONE",
                     ins={"Vertices": B.imath("MAXIMUM", B.gin("Bolts"), 3), "Radius": ring_r})
        iop = B.n("GeometryNodeInstanceOnPoints", ins={"Points": circle.outputs[0], "Instance": bolt})
        bolts = B.n("GeometryNodeRealizeInstances", ins={"Geometry": iop.outputs[0]}).outputs[0]
        has = B.compare("GREATER_EQUAL", B.gin("Bolts"), 3, "INT", label="bolts on?")
        bolts = B.switch(has, None, bolts, "GEOMETRY")
        B.set_out("Mesh", B.join(disc, bolts))
    layout(B)
    return B.ng


# --------------------------------------------------------------------------------------
#  Main group
# --------------------------------------------------------------------------------------
def build_main(S, materials, previous=None):
    B = Builder(MAIN_NAME, is_modifier=True, previous=previous)
    ng = B.ng
    ng.description = ("Loose cables and pipes between two empties against a floor and the adjacent "
                      "wall. Floor = local Z, wall face = local Y, room on +Y.")
    B.outp("Geometry", "GEO")

    # ---------------- interface (modifier panel) ----------------
    p = B.panel("Anchors", closed=False, desc="Where the run starts and ends")
    B.inp("Start", "OBJ", desc="Empty where the run starts (put it on the wall). Its arrow (local Z) is the "
                               "direction the cables leave in", panel=p)
    B.inp("End", "OBJ", desc="Empty where the run ends (put it on the floor). Its arrow (local Z) is the "
                             "direction the cables leave in", panel=p)
    B.inp("Floor Level", "FLOAT", 0.0, sub="DISTANCE", desc="Floor height in this object's local space", panel=p)
    B.inp("Wall Plane", "FLOAT", 0.0, sub="DISTANCE",
          desc="Wall face position along this object's local Y. The room is on the +Y side", panel=p)
    B.inp("Exit Length", "FLOAT", 0.08, 0.0, 2.0, "DISTANCE",
          "Straight section where strands leave each anchor. Connectors and end flanges fill it", p)

    p = B.panel("Bundle", closed=False, desc="How many strands, what kind, how thick")
    B.inp("Count", "INT", 14, 1, 400, desc="Number of strands", panel=p)
    B.inp("Pipe Ratio", "FLOAT", 0.2, 0.0, 1.0, "FACTOR",
          "Share of strands that become pipes (Pipe material, flanges, stiffer shape). 0 = all cables", p)
    B.inp("Cable Radius", "FLOAT", 0.012, 0.0005, 1.0, "DISTANCE", "Largest cable radius", p)
    B.inp("Pipe Radius", "FLOAT", 0.035, 0.001, 2.0, "DISTANCE", "Largest pipe radius", p)
    B.inp("Radius Random", "FLOAT", 0.4, 0.0, 0.95, "FACTOR",
          "Random thinning per strand, as a share of the radius", p)
    B.inp("Rows", "INT", 2, 1, 16, desc="Rows of strands stacked at each anchor", panel=p)
    B.inp("Start Gap", "FLOAT", 0.004, 0.0, 1.0, "DISTANCE", "Space between strands at the start anchor", p)
    B.inp("End Gap", "FLOAT", 0.02, 0.0, 1.0, "DISTANCE",
          "Space between strands at the end anchor (bigger fans the bundle out)", p)
    B.inp("Jitter", "FLOAT", 0.006, 0.0, 1.0, "DISTANCE", "Random offset of each strand at the anchors", p)
    B.inp("Strands per Cable", "INT", 1, 1, 4,
          desc="1 = round cable, 2-4 = twisted wires (twisted pair look)", panel=p)
    B.inp("Twist", "FLOAT", 6.0, 0.0, 100.0, desc="Twist rate of multi-wire cables, turns per meter", panel=p)
    B.inp("Seed", "INT", 0, 0, 100000, desc="Random seed", panel=p)

    p = B.panel("Shape", closed=False, desc="How loose and noodly the run is")
    B.inp("Slack", "FLOAT", 0.35, 0.0, 10.0, "DISTANCE",
          "Extra length: the run bows sideways on the floor and sags on the wall", p)
    B.inp("Slack Random", "FLOAT", 0.5, 0.0, 1.0, "FACTOR", "Per-strand slack variation", p)
    B.inp("Drop", "FLOAT", 0.85, 0.0, 1.0, "FACTOR",
          "1 = strands fall straight down the wall before heading to the end, 0 = diagonal shortcut", p)
    B.inp("Noodle Amount", "FLOAT", 0.14, 0.0, 5.0, "DISTANCE", "Side-to-side wiggle", p)
    B.inp("Noodle Frequency", "FLOAT", 1.6, 0.05, 30.0, desc="Wiggles per meter", panel=p)
    B.inp("Bundle Cohesion", "FLOAT", 0.55, 0.0, 1.0, "FACTOR",
          "1 = the whole bundle snakes together, 0 = every strand wanders on its own", p)
    B.inp("Bend Radius", "FLOAT", 0.05, 0.0, 5.0, "DISTANCE",
          "Bend radius in the wall/floor corner (never tighter than 2.5x the strand radius)", p)
    B.inp("Pipe Stiffness", "FLOAT", 0.85, 0.0, 1.0, "FACTOR",
          "How much less pipes wiggle, sag and arch than cables", p)

    p = B.panel("Pins", closed=False, desc="How closely strands sit on the floor and against the wall")
    B.inp("Wall Pin", "FLOAT", 0.85, 0.0, 1.0, "FACTOR", "1 = flush against the wall, 0 = hangs loose", p)
    B.inp("Floor Pin", "FLOAT", 1.0, 0.0, 1.0, "FACTOR", "1 = flat on the floor, 0 = arches up off it", p)
    B.inp("Loose Gap", "FLOAT", 0.25, 0.0, 5.0, "DISTANCE", "Distance from the surfaces when a pin is 0", p)
    B.inp("Pin Random", "FLOAT", 0.35, 0.0, 1.0, "FACTOR", "Per-strand variation of both pins", p)

    p = B.panel("Connectors", desc="Plugs / glands where cables meet the anchors")
    B.inp("Connectors", "BOOL", True, desc="Add connectors at both ends of every cable", panel=p, toggle=True)
    B.inp("Connector Scale", "FLOAT", 1.7, 1.0, 6.0, desc="Connector radius relative to its cable", panel=p)

    p = B.panel("Ties & Clips", desc="Zip ties on loose cables, saddle clips where strands touch a surface")
    B.inp("Ties & Clips", "BOOL", True, desc="Add ties and clips", panel=p, toggle=True)
    B.inp("Tie Spacing", "FLOAT", 0.4, 0.02, 20.0, "DISTANCE", "Distance between tie/clip candidates", p)
    B.inp("Tie Chance", "FLOAT", 0.5, 0.0, 1.0, "FACTOR", "Chance that a candidate gets a tie or clip", p)
    B.inp("Tie Width", "FLOAT", 0.01, 0.001, 0.5, "DISTANCE", "Width of ties and clips along the strand", p)

    p = B.panel("Pipe Flanges", desc="Wall plates at the ends and couplings along pipes")
    B.inp("Pipe Flanges", "BOOL", True, desc="Add flanges to pipes", panel=p, toggle=True)
    B.inp("Flange Spacing", "FLOAT", 1.0, 0.0, 50.0, "DISTANCE", "Couplings along each pipe (0 = only at the ends)", p)
    B.inp("Flange Scale", "FLOAT", 1.55, 1.05, 4.0, desc="Flange radius relative to its pipe", panel=p)
    B.inp("Flange Thickness", "FLOAT", 0.022, 0.001, 0.5, "DISTANCE", "Flange thickness", p)
    B.inp("Bolt Count", "INT", 6, 0, 32, desc="Bolts per flange (below 3 = none)", panel=p)

    p = B.panel("Topology", desc="Geometry density and output")
    B.inp("Segments per Meter", "FLOAT", 10.0, 1.0, 1000.0,
          desc="Density along strands. With Adaptive on, this only applies to straight runs", panel=p)
    B.inp("Adaptive", "BOOL", True,
          desc="Fewer segments on straight runs, more in bends (game friendly topology)", panel=p)
    B.inp("Adaptive Angle", "FLOAT", math.radians(8.0), math.radians(0.5), math.radians(60.0), "ANGLE",
          "Max bend between kept segments when Adaptive is on (smaller = smoother bends)", p)
    B.inp("Cable Sides", "INT", 8, 3, 64, desc="Profile resolution of cables", panel=p)
    B.inp("Pipe Sides", "INT", 16, 3, 64, desc="Profile resolution of pipes", panel=p)
    B.inp("Detail Sides", "INT", 12, 3, 64, desc="Resolution of connectors, ties and flanges", panel=p)
    B.inp("Fill Caps", "BOOL", True, desc="Close the strand ends", panel=p)
    B.inp("Shade Smooth", "BOOL", True, desc="Smooth shading with sharp edges on hard angles", panel=p)
    B.inp("Output", "MENU", desc="Mesh = final geometry. Curves = center lines with radius, for further editing",
          panel=p)

    p = B.panel("Materials", desc="Separate material slots (left untextured)")
    for nm, desc in (("Cable Material", "Slot 1: cables"), ("Pipe Material", "Slot 2: pipes"),
                     ("Connector Material", "Slot 3: connectors / plugs"),
                     ("Hardware Material", "Slot 4: ties, clips, flanges and bolts")):
        sock = B.inp(nm, "MAT", desc=desc, panel=p)
        sock.default_value = materials[nm]

    # ---------------- 1 · anchors ----------------
    with B.frame("1 · Anchors: Start / End empties in corner space", "input", slot=0):
        corner = B.combine(0.0, B.gin("Wall Plane"), B.gin("Floor Level"), label="corner origin")
        aS = B.group(S["anchor"], "Start anchor", Object=B.gin("Start"), Corner=corner,
                     Fallback_Position=(0.0, 0.0, 1.5), Fallback_Rotation=(-math.pi / 2, 0.0, 0.0))
        aE = B.group(S["anchor"], "End anchor", Object=B.gin("End"), Corner=corner,
                     Fallback_Position=(1.2, 1.5, 0.0), Fallback_Rotation=(0.0, 0.0, 0.0))
        A, dA, uA, vA = (aS.outputs[k] for k in ("Position", "Exit Dir", "Side Dir", "Row Dir"))
        Bp, dB, uB, vB = (aE.outputs[k] for k in ("Position", "Exit Dir", "Side Dir", "Row Dir"))

    # ---------------- 2 · strands ----------------
    with B.frame("2 · Strands: one point per strand, random type / radius / pins", "strand", slot=1):
        pts = B.n("GeometryNodePoints", label="one point per strand", ins={"Count": B.gin("Count")})
        i = B.index()
        seed = B.gin("Seed")
        r1 = B.rand("FLOAT_VECTOR", 0.0, 1.0, id=i, seed=seed, label="type, radius, slack")
        u_type, u_rad, u_slack = B.separate(r1)
        is_pipe = B.compare("LESS_THAN", u_type, B.gin("Pipe Ratio"), label="is pipe?")
        flex = B.switch(is_pipe, 1.0, B.math("SUBTRACT", 1.0, B.gin("Pipe Stiffness")), label="flex (pipes stiffer)")
        base_r = B.switch(is_pipe, B.gin("Cable Radius"), B.gin("Pipe Radius"), label="max radius")
        thin = B.math("SUBTRACT", 1.0, B.gin("Radius Random"))
        r = B.maprange(u_rad, 0.0, 1.0, base_r, B.math("MULTIPLY", base_r, thin), label="radius")
        sl_thin = B.math("SUBTRACT", 1.0, B.gin("Slack Random"))
        sl_rand = B.switch(is_pipe, B.maprange(u_slack, 0.0, 1.0, 1.0, sl_thin), 1.0,
                           label="pipes: no slack randomness")
        slack = B.math("MULTIPLY", sl_rand, B.math("MULTIPLY", B.gin("Slack"), flex), label="slack")
        coh = B.switch(is_pipe, B.gin("Bundle Cohesion"), 1.0, label="cohesion (pipes run parallel)")
        pr = B.math("SUBTRACT", 1.0, B.gin("Pin Random"))
        r2 = B.rand("FLOAT_VECTOR", B.combine(pr, pr, -1.0), (1.0, 1.0, 1.0), id=i,
                    seed=B.imath("ADD", seed, 1), label="pin scale x/y, side z")
        pins = B.vmath("MULTIPLY", B.combine(B.gin("Wall Pin"), B.gin("Floor Pin"), 0.0), r2, label="pins")
        _, _, side = B.separate(r2)

    with B.frame("2b · Strand layout at the anchors (rows, gaps, jitter)", "strand", slot=1):
        row = B.imath("FLOORED_MODULO", i, B.gin("Rows"), label="row")
        w = B.combine(B.math("MULTIPLY_ADD", r, 2.0, B.gin("Start Gap")),
                      B.math("MULTIPLY_ADD", r, 2.0, B.gin("End Gap")), 0.0, label="slot width start / end")
        acc = B.n("GeometryNodeAccumulateField", data_type="FLOAT_VECTOR", domain="POINT",
                  ins={"Value": w, "Group Index": row})
        centre = B.vmath("MULTIPLY_ADD", B.vmath("ADD", w, acc.outputs["Total"]), -0.5, acc.outputs["Leading"],
                         label="centred side offset")
        odd = B.imath("FLOORED_MODULO", row, 2, label="odd row?")
        centre = B.vmath("MULTIPLY_ADD", w, B.math("MULTIPLY", odd, 0.5), centre,
                         label="odd rows sit in the grooves")
        maxr = B.math("MAXIMUM", B.gin("Cable Radius"), B.gin("Pipe Radius"))
        pitch = B.combine(B.math("MULTIPLY_ADD", maxr, 2.0, B.gin("Start Gap")),
                          B.math("MULTIPLY_ADD", maxr, 2.0, B.gin("End Gap")), 0.0, label="row pitch")
        rowc = B.math("SUBTRACT", row, B.math("MULTIPLY_ADD", B.gin("Rows"), 0.5, -0.5), label="centred row")
        rowo = B.vmath("SCALE", pitch, scale=rowc)
        jit = B.gin("Jitter")
        nj = B.math("MULTIPLY", jit, -1.0)
        jl = B.rand("FLOAT_VECTOR", nj, jit, id=i, seed=B.imath("ADD", seed, 2), label="jitter side")
        jr = B.rand("FLOAT_VECTOR", nj, jit, id=i, seed=B.imath("ADD", seed, 3), label="jitter row")
        lat_s, lat_e, _ = B.separate(B.vmath("ADD", centre, jl), label="side offset start / end")
        row_s, row_e, _ = B.separate(B.vmath("ADD", rowo, jr), label="row offset start / end")
        Ai = B.vmath("MULTIPLY_ADD", vA, row_s, B.vmath("MULTIPLY_ADD", uA, lat_s, A), label="strand start")
        Bi = B.vmath("MULTIPLY_ADD", vB, row_e, B.vmath("MULTIPLY_ADD", uB, lat_e, Bp), label="strand end")

    with B.frame("2c · Store per-strand attributes (read later by name, no long wires)", "strand", slot=1):
        g = pts.outputs[0]
        for name, val, kind in (("cg_id", i, "INT"), ("cg_layer", row, "FLOAT"),
                                ("cg_A", Ai, "FLOAT_VECTOR"), ("cg_B", Bi, "FLOAT_VECTOR"),
                                ("cg_dA", dA, "FLOAT_VECTOR"), ("cg_dB", dB, "FLOAT_VECTOR"),
                                ("cg_r", r, "FLOAT"), ("cg_pipe", is_pipe, "BOOLEAN"),
                                ("cg_pins", pins, "FLOAT_VECTOR"), ("cg_slack", slack, "FLOAT"),
                                ("cg_flex", flex, "FLOAT"), ("cg_side", side, "FLOAT"), ("cg_coh", coh, "FLOAT")):
            g = B.store(g, name, val, kind, "POINT", label=name)
        strands = g

    # ---------------- 3 · keypoints ----------------
    with B.frame("3 · Keypoints: K points per strand, more for longer / noodlier runs", "route", slot=2):
        ax, ay, az = B.separate(A, label="A")
        bx, by, bz = B.separate(Bp, label="B")
        ds = B.math("SUBTRACT", B.math("SUBTRACT", by, bz), B.math("SUBTRACT", ay, az), label="ds (unfolded)")
        run = B.math("ADD", B.math("ADD", B.math("ABSOLUTE", B.math("SUBTRACT", bx, ax)), B.math("ABSOLUTE", ds)),
                     B.gin("Slack"), label="rough run length")
        kf = B.math("CEIL", B.math("MULTIPLY", B.math("MULTIPLY", run, B.gin("Noodle Frequency")), 2.0))
        K = B.n("ShaderNodeClamp", label="keypoints", ins={"Value": kf, "Min": 7.0, "Max": 400.0}).outputs[0]
        K = B.math("ADD", K, 1.0, label="K")
        fine = B.n("ShaderNodeClamp", label="fine samples",
                   ins={"Value": B.math("MULTIPLY", run, 90.0), "Min": 64.0, "Max": 3000.0}).outputs[0]
        dup = B.n("GeometryNodeDuplicateElements", domain="POINT", ins={"Geometry": strands, "Amount": K})
        crv = B.n("GeometryNodePointsToCurves",
                  ins={"Points": dup.outputs["Geometry"], "Curve Group ID": B.attr("cg_id", "INT"),
                       "Weight": dup.outputs["Duplicate Index"]})

    # ---------------- 4 · route in the unfolded plane ----------------
    with B.frame("4 · Route in the unfolded wall+floor plane (x along corner, s down-the-wall / out-on-floor)",
                 "route", slot=3):
        sp = B.n("GeometryNodeSplineParameter")
        t = B.math("DIVIDE", sp.outputs["Index"], B.math("SUBTRACT", K, 1.0), label="t (0..1)")
        r_ = B.attr("cg_r")
        pins_ = B.attr("cg_pins", "FLOAT_VECTOR")
        ua = B.group(S["unfold"], "Unfold start", Point=B.attr("cg_A", "FLOAT_VECTOR"), Radius=r_, Pins=pins_,
                     Loose_Gap=B.gin("Loose Gap"))
        ub = B.group(S["unfold"], "Unfold end", Point=B.attr("cg_B", "FLOAT_VECTOR"), Radius=r_, Pins=pins_,
                     Loose_Gap=B.gin("Loose Gap"))
        pa, pb = ua.outputs["Plane"], ub.outputs["Plane"]
        xa, sa, _ = B.separate(pa, label="start (x, s)")
        xb, sb, _ = B.separate(pb, label="end (x, s)")
        dw = B.group(S["drop"], "Drop window", S_Start=sa, S_End=sb, Drop=B.gin("Drop"))
        x_t = B.maprange(t, dw.outputs["T0"], dw.outputs["T1"], xa, xb, "SMOOTHSTEP", label="x(t)")
        s_t = B.mix(sa, sb, t, label="s(t)")
        bump = B.math("SINE", B.math("MULTIPLY", t, math.pi), label="bump sin(pi t)")
        perp = B.n("ShaderNodeVectorRotate", rotation_type="Z_AXIS", label="perpendicular",
                   ins={"Vector": B.vmath("SUBTRACT", pb, pa), "Angle": math.pi / 2}).outputs[0]
        n_hat = B.vmath("NORMALIZE", perp)
        _, ny, _ = B.separate(n_hat)
        shared_side = B.rand("FLOAT", -1.0, 1.0, id=0, seed=B.imath("ADD", B.gin("Seed"), 7), label="shared side")
        side = B.mix(B.attr("cg_side"), shared_side, B.attr("cg_coh"), label="side")
        sgn = B.math("SIGN", B.math("MULTIPLY_ADD", side, 0.15, ny), label="bow downhill / random")
        n_dir = B.vmath("SCALE", n_hat, scale=sgn, label="sideways dir")

    with B.frame("4b · Bow (slack), noodle wiggle, pins -> height off the surfaces", "route", slot=3):
        seed = B.gin("Seed")
        rk = B.rand("FLOAT_VECTOR", -1.0, 1.0, id=B.index(), seed=B.imath("ADD", seed, 5), label="per keypoint")
        mk, ak, _ = B.separate(rk, label="wiggle, arch")
        msh = B.rand("FLOAT", -1.0, 1.0, id=sp.outputs["Index"], seed=B.imath("ADD", seed, 6),
                     label="shared wiggle")
        wig = B.mix(mk, msh, B.attr("cg_coh"), label="wiggle")
        floorness = B.maprange(s_t, -0.15, 0.15, 0.0, 1.0, "SMOOTHSTEP", label="on floor (0..1)")
        damp = B.mix(0.35, 1.0, floorness, label="less wiggle on wall")
        flex_ = B.attr("cg_flex")
        amp = B.math("MULTIPLY", B.math("MULTIPLY", B.gin("Noodle Amount"), flex_),
                     B.math("MULTIPLY", bump, damp), label="wiggle amount")
        lateral = B.math("MULTIPLY_ADD", wig, amp, B.math("MULTIPLY", B.attr("cg_slack"), bump),
                         label="sideways = bow + wiggle")
        plane = B.vmath("MULTIPLY_ADD", n_dir, lateral, B.combine(x_t, s_t, 0.0), label="(x, s)")
        px, py, _ = B.separate(pins_, label="pins")
        pin_t = B.mix(px, py, floorness, label="pin here")
        loose = B.math("MULTIPLY", B.math("SUBTRACT", 1.0, pin_t), B.gin("Loose Gap"), label="loose gap")
        hump = B.math("MAXIMUM", B.maprange(ak, -1.0, 1.0, -1.0, 1.6, label="hump size"), 0.0,
                      label="humps touch down between")
        arch = B.mix(1.0, hump, B.math("MULTIPLY", floorness, flex_), label="floor humps (cables only)")
        h = B.math("MULTIPLY_ADD", loose, arch, r_, label="height = r + gap")
        h = B.math("MULTIPLY_ADD", B.attr("cg_layer"), B.math("MULTIPLY", r_, 1.7), h,
                   label="+ layer: rows stack on top of each other")
        upos = B.vmath("MULTIPLY_ADD", (0.0, 0.0, 1.0), h, plane, label="(x, s, h)")
        g = B.setpos(crv.outputs[0], upos)
        g = B.n("GeometryNodeCurveSplineType", spline_type="CATMULL_ROM", ins={"Curve": g}).outputs[0]
        g = B.n("GeometryNodeResampleCurve", label="smooth + dense", ins={"Curve": g, "Count": fine}).outputs[0]

    # ---------------- 5 · fold onto the surfaces + straight exits ----------------
    with B.frame("5 · Fold onto wall + floor, then leave each anchor straight", "route", slot=4):
        r_ = B.attr("cg_r")
        c = B.math("MULTIPLY", B.math("MAXIMUM", B.math("MULTIPLY", r_, 2.5), B.gin("Bend Radius")), math.sqrt(2.0),
                   label="corner rounding")
        fold = B.group(S["fold"], "Fold corner", Position=B.position(), Bend=c)
        g = B.setpos(g, fold.outputs[0])
        blend = B.math("MULTIPLY_ADD", r_, 6.0, 0.06, label="ease length")
        al = B.group(S["align"], "Align ends", Position=B.position(), A=B.attr("cg_A", "FLOAT_VECTOR"),
                     Dir_A=B.attr("cg_dA", "FLOAT_VECTOR"), B=B.attr("cg_B", "FLOAT_VECTOR"),
                     Dir_B=B.attr("cg_dB", "FLOAT_VECTOR"), Exit=B.gin("Exit Length"), Blend=blend)
        g = B.setpos(g, al.outputs[0])

    # ---------------- 6 · density / topology ----------------
    with B.frame("6 · Density: even segments, or adaptive (dense in bends only)", "mesh", slot=5):
        seg = B.math("DIVIDE", 1.0, B.gin("Segments per Meter"), label="segment length")
        tw_seg = B.math("DIVIDE", 0.1, B.math("MAXIMUM", B.gin("Twist"), 0.01), label="10 segments per twist")
        multi = B.compare("GREATER_THAN", B.gin("Strands per Cable"), 1, "INT", label="multi-wire?")
        seg = B.switch(multi, seg, B.math("MINIMUM", seg, tw_seg), label="segment length")
        fine_len = B.math("MINIMUM", seg, 0.012)
        rlen = B.switch(B.gin("Adaptive"), seg, fine_len, label="resample length")
        rs = B.n("GeometryNodeResampleCurve", keep_last_segment=True,
                 ins={"Curve": g, "Mode": "Length", "Length": rlen}).outputs[0]
        simp = B.group(S["adaptive"], "Adaptive simplify", Curves=rs, Angle=B.gin("Adaptive Angle"), Max_Length=seg)
        g = B.switch(B.gin("Adaptive"), rs, simp.outputs[0], "GEOMETRY", label="adaptive?")
        sp = B.n("GeometryNodeSplineParameter")
        sl = B.n("GeometryNodeSplineLength")
        g = B.store(g, "cg_d", sp.outputs["Length"], label="cg_d (distance from start)")
        g = B.store(g, "cg_de", B.math("SUBTRACT", sl.outputs["Length"], sp.outputs["Length"]),
                    label="cg_de (distance to end)")
        centerlines = g

    # ---------------- 7 · tubes ----------------
    with B.frame("7 · Tubes: cables (slot 1) and pipes (slot 2)", "mesh", slot=6):
        sep = B.n("GeometryNodeSeparateGeometry", domain="CURVE",
                  ins={"Geometry": centerlines, "Selection": B.attr("cg_pipe", "BOOLEAN")})
        pipes, cables = sep.outputs["Selection"], sep.outputs["Inverted"]
        twisted = B.compare("GREATER_THAN", B.gin("Strands per Cable"), 1, "INT", label="multi-wire?")
        rate = B.math("MULTIPLY", B.math("MULTIPLY", B.gin("Twist"), 2 * math.pi), twisted, label="twist rate")
        tilt = B.math("MULTIPLY", B.attr("cg_d"), rate, label="twist")
        cables = B.n("GeometryNodeSetCurveTilt", ins={"Curve": cables, "Tilt": tilt}).outputs[0]
        cprof = B.group(S["profile"], "Cable profile", Strands=B.gin("Strands per Cable"), Sides=B.gin("Cable Sides"))
        pprof = B.group(S["profile"], "Pipe profile", Strands=1, Sides=B.gin("Pipe Sides"))
        cmesh = B.group(S["tube"], "Cable tubes", Curves=cables, Profile=cprof.outputs[0],
                        Fill_Caps=B.gin("Fill Caps"), Part=0).outputs[0]
        pmesh = B.group(S["tube"], "Pipe tubes", Curves=pipes, Profile=pprof.outputs[0],
                        Fill_Caps=B.gin("Fill Caps"), Part=1).outputs[0]

    # ---------------- 8 · connectors + end flanges ----------------
    with B.frame("8 · Ends: cable connectors / pipe wall flanges, aimed along each anchor arrow", "detail", slot=7):
        ends = B.n("GeometryNodeCurveEndpointSelection", ins={"Start Size": 1, "End Size": 1}).outputs[0]
        is_pipe = B.attr("cg_pipe", "BOOLEAN")
        at_start = B.compare("LESS_THAN", B.attr("cg_d"), B.attr("cg_de"), label="start end?")
        aim = B.switch(at_start, B.attr("cg_dB", "FLOAT_VECTOR"), B.attr("cg_dA", "FLOAT_VECTOR"), "VECTOR",
                       label="anchor arrow")
        rot = B.n("FunctionNodeAlignRotationToVector", axis="Z", ins={"Vector": aim}).outputs[0]
        r_ = B.attr("cg_r")
        cr = B.math("MULTIPLY", r_, B.gin("Connector Scale"))
        csel = B.bmath("AND", B.bmath("AND", ends, B.bmath("NOT", is_pipe)), B.gin("Connectors"),
                       label="cable ends")
        cmeshu = B.group(S["connector"], "Connector", Sides=B.gin("Detail Sides"))
        conn = B.n("GeometryNodeInstanceOnPoints",
                   ins={"Points": centerlines, "Selection": csel, "Instance": cmeshu.outputs[0],
                        "Rotation": rot, "Scale": B.combine(cr, cr, B.gin("Exit Length"))}).outputs[0]
        fmesh = B.group(S["flange"], "Flange", Sides=B.gin("Detail Sides"), Flange_Scale=B.gin("Flange Scale"),
                        Bolts=B.gin("Bolt Count")).outputs[0]
        fsel = B.bmath("AND", B.bmath("AND", ends, is_pipe), B.gin("Pipe Flanges"), label="pipe ends")
        fscale = B.combine(r_, r_, B.gin("Flange Thickness"))
        endfl = B.n("GeometryNodeInstanceOnPoints",
                    ins={"Points": centerlines, "Selection": fsel, "Instance": fmesh, "Rotation": rot,
                         "Scale": fscale}).outputs[0]

    # ---------------- 9 · ties & clips ----------------
    with B.frame("9 · Ties & clips: clip (with feet) where touching a surface, zip tie elsewhere", "detail", slot=7):
        tsp = B.math("MULTIPLY", B.gin("Tie Spacing"),
                     B.rand("FLOAT", 0.8, 1.2, id=B.index(), seed=B.imath("ADD", B.gin("Seed"), 10),
                            label="per strand (staggered)"), label="spacing")
        c2p = B.n("GeometryNodeCurveToPoints", mode="LENGTH", ins={"Curve": centerlines, "Length": tsp})
        margin = B.math("ADD", B.gin("Exit Length"), 0.06, label="keep clear of ends")
        away = B.bmath("AND", B.compare("GREATER_THAN", B.attr("cg_d"), margin),
                       B.compare("GREATER_THAN", B.attr("cg_de"), margin), label="not near ends")
        lucky = B.rand("BOOLEAN", prob=B.gin("Tie Chance"), id=B.index(),
                       seed=B.imath("ADD", B.gin("Seed"), 9), label="tie chance")
        px, py, pz = B.separate(B.position(), label="position")
        r_ = B.attr("cg_r")
        touch = B.compare("LESS_THAN", B.math("MINIMUM", py, pz), B.math("MULTIPLY_ADD", r_, 1.35, 0.002),
                          label="touching a surface?")
        clear = B.compare("GREATER_THAN", B.math("MAXIMUM", py, pz), B.math("MULTIPLY", r_, 4.5),
                          label="clear of the corner?")
        near = B.bmath("AND", touch, clear, label="clip here")
        is_pipe = B.attr("cg_pipe", "BOOLEAN")
        pipe_ok = B.bmath("OR", B.bmath("NOT", is_pipe), near, label="pipes: clamps only")
        tsel = B.bmath("AND", B.bmath("AND", B.bmath("AND", away, lucky), pipe_ok), B.gin("Ties & Clips"),
                       label="place?")
        normal = B.switch(B.compare("LESS_THAN", py, pz), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "VECTOR",
                          label="nearest surface normal")
        trot = B.n("FunctionNodeAxesToRotation", primary_axis="Z", secondary_axis="Y",
                   ins={"Primary Axis": c2p.outputs["Tangent"], "Secondary Axis": normal}).outputs[0]
        wide = B.switch(is_pipe, 1.0, 2.5, label="wider on pipes")
        tscale = B.combine(r_, r_, B.math("MULTIPLY", B.gin("Tie Width"), wide))
        kinds = B.n("GeometryNodeGeometryToInstance")
        for sub in (S["clip"], S["tie"]):  # instance 0 = tie, 1 = clip (links stack on top)
            gnode = B.group(sub, sub.name.replace(SUB_PREFIX, ""), Sides=B.gin("Detail Sides"))
            B.ng.links.new(gnode.outputs[0], kinds.inputs[0])
        ties = B.n("GeometryNodeInstanceOnPoints",
                   ins={"Points": c2p.outputs["Points"], "Selection": tsel, "Instance": kinds.outputs[0],
                        "Pick Instance": True, "Instance Index": near, "Rotation": trot,
                        "Scale": tscale}).outputs[0]

    # ---------------- 10 · couplings ----------------
    with B.frame("10 · Pipe couplings along the run", "detail", slot=7):
        on = B.bmath("AND", B.gin("Pipe Flanges"), B.compare("GREATER_THAN", B.gin("Flange Spacing"), 0.0), label="couplings?")
        spacing = B.math("MULTIPLY", B.math("MAXIMUM", B.gin("Flange Spacing"), 0.05),
                         B.rand("FLOAT", 0.75, 1.25, id=B.index(), seed=B.imath("ADD", B.gin("Seed"), 11),
                                label="per pipe (staggered)"), label="spacing")
        c2p2 = B.n("GeometryNodeCurveToPoints", mode="LENGTH", ins={"Curve": pipes, "Length": spacing})
        margin2 = B.math("ADD", B.gin("Exit Length"), 0.15)
        away2 = B.bmath("AND", B.compare("GREATER_THAN", B.attr("cg_d"), margin2),
                        B.compare("GREATER_THAN", B.attr("cg_de"), margin2))
        r_ = B.attr("cg_r")
        coup = B.n("GeometryNodeInstanceOnPoints",
                   ins={"Points": c2p2.outputs["Points"], "Selection": B.bmath("AND", away2, on),
                        "Instance": fmesh, "Rotation": c2p2.outputs["Rotation"],
                        "Scale": B.combine(r_, r_, B.gin("Flange Thickness"))}).outputs[0]

    # ---------------- 11 · finish ----------------
    with B.frame("11 · Finish: realize, shading, material slots, clean-up, back to object space", "out", slot=8):
        allg = B.join(cmesh, pmesh, conn, endfl, ties, coup)
        g = B.n("GeometryNodeRealizeInstances", ins={"Geometry": allg}).outputs[0]
        g = B.n("GeometryNodeSetShadeSmooth", domain="FACE",
                ins={"Geometry": g, "Shade Smooth": B.gin("Shade Smooth")}).outputs[0]
        ea = B.n("GeometryNodeInputMeshEdgeAngle").outputs["Unsigned Angle"]
        g = B.n("GeometryNodeSetShadeSmooth", domain="EDGE", label="sharp hard edges",
                ins={"Geometry": g, "Selection": B.compare("GREATER_THAN", ea, math.radians(60.0)),
                     "Shade Smooth": False}).outputs[0]
        part = B.attr("cg_part", "INT")
        # Joined geometry starts with one empty slot (index 0) that every face uses.
        # Append slots 2-4 by part, then fill the empty slot with the cable material,
        # so the slot order is always Cable, Pipe, Connector, Hardware.
        for k, mname in enumerate(("Pipe Material", "Connector Material", "Hardware Material"), start=1):
            g = B.n("GeometryNodeSetMaterial", label=f"slot {k + 1}: {mname.split()[0].lower()}",
                    ins={"Geometry": g, "Selection": B.compare("EQUAL", part, k, "INT"),
                         "Material": B.gin(mname)}).outputs[0]
        g = B.n("GeometryNodeReplaceMaterial", label="slot 1: cable (fills the empty slot)",
                ins={"Geometry": g, "New": B.gin("Cable Material")}).outputs[0]
        mesh_out = g
        crv_out = B.n("GeometryNodeSetCurveRadius", ins={"Curve": centerlines, "Radius": B.attr("cg_r")}).outputs[0]
        menu = B.n("GeometryNodeMenuSwitch", data_type="GEOMETRY", label="Output")
        menu.enum_items[0].name = "Mesh"
        menu.enum_items[1].name = "Curves"
        B.c(menu, 1, mesh_out)
        B.c(menu, 2, crv_out)
        B.ng.links.new(B.gin("Output"), menu.inputs[0])
        g = B.n("GeometryNodeRemoveAttribute", label="drop cg_* temp attributes",
                ins={"Geometry": menu.outputs[0], "Pattern Mode": "Wildcard", "Name": "cg_*"}).outputs[0]
        corner = B.combine(0.0, B.gin("Wall Plane"), B.gin("Floor Level"), label="corner origin")
        g = B.transform(g, corner, label="back to object space")
        B.set_out("Geometry", g)

    # menu default after the menu socket has items
    for item in ng.interface.items_tree:
        if getattr(item, "name", "") == "Output" and getattr(item, "in_out", "") == "INPUT":
            item.default_value = "Mesh"
    layout(B)
    _add_readme_frame(ng)
    return ng


README = """CABLE GENERATOR  (Blender 5.1)

SETUP
- Origin of this object = the floor/wall corner. +Y points out of the wall into the room, +Z up.
- Start / End empties: their arrow (local Z) is the direction the strands leave in.
  Start usually sits on the wall pointing out, End on the floor pointing up.
- Floor Level / Wall Plane nudge the surfaces without moving the object.

HOW IT WORKS
1-2  One point per strand: random type (cable / pipe), radius, pins, layout at the anchors.
3-4  Each strand gets K keypoints laid out in an 'unfolded' plane where the wall and floor are
     flattened into one sheet (s < 0 = on the wall, s > 0 = on the floor). Slack bows the run,
     noodle noise wiggles it, pins set the height off the surface.
5    Catmull-Rom smoothing, then the plane is folded back onto wall + floor with a rounded corner,
     and each end is eased into a straight exit along the empty's arrow.
6    Even or adaptive (bend-only) segment density.
7    Tubes with UVs: cables -> slot 1, pipes -> slot 2.
8-10 Connectors (slot 3), wall flanges, couplings, ties and clips (slot 4).
11   Shading, materials, cleanup.

Per-strand values travel as cg_* named attributes and are removed at the end.
"""


def _add_readme_frame(ng):
    name = "Cable Generator README"
    txt = next((t for t in bpy.data.texts if _owned(t, name) and t.as_string() == README), None)
    if txt is None:
        txt = _track(bpy.data.texts.new(name))
        txt[OWNER_ROLE] = name
        txt.write(README)
    f = ng.nodes.new("NodeFrame")
    f.label = "README"
    f.text = txt
    f.label_size = 18
    f.shrink = False
    f.width, f.height = 760, 560
    f.location = (-900, 0)
    f.use_custom_color = True
    f.color = (0.16, 0.16, 0.16)


# --------------------------------------------------------------------------------------
#  Build everything + demo scene
# --------------------------------------------------------------------------------------
def _build_graph(previous):
    materials = {
        "Cable Material": _material("CG_Cable", (0.04, 0.04, 0.045, 1.0)),
        "Pipe Material": _material("CG_Pipe", (0.62, 0.63, 0.6, 1.0)),
        "Connector Material": _material("CG_Connector", (0.16, 0.22, 0.3, 1.0)),
        "Hardware Material": _material("CG_Hardware", (0.72, 0.6, 0.32, 1.0)),
    }
    S = {
        "anchor": build_anchor_frame(), "unfold": build_unfold_anchor(), "drop": build_drop_window(),
        "fold": build_fold_corner(), "align": build_align_ends(), "adaptive": build_adaptive(),
        "profile": build_strand_profile(), "tube": build_tube(), "box": build_box_profile(),
    }
    S["connector"] = build_connector_mesh(S)
    S["tie"] = build_tie_mesh(S)
    S["clip"] = build_clip_mesh(S)
    S["flange"] = build_flange_mesh(S)
    template = _track(previous.copy()) if previous is not None else None
    main = build_main(S, materials, template)
    try:
        main.asset_mark()
        main.asset_data.description = main.description
        for tag in ("cables", "pipes", "procedural", "kitbash"):
            main.asset_data.tags.new(tag, skip_if_exists=True)
    except Exception:
        pass
    return main, S, materials


def _material(name, color):
    m = next((m for m in bpy.data.materials if _owned(m, name)), None)
    if m is None:
        m = _track(bpy.data.materials.new(name))
        m[OWNER_ROLE] = name
        m.diffuse_color = color
    return m


def _mod_set(mod, name, value):
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == name:
            mod[item.identifier] = value
            return
    raise KeyError(name)


def _dependencies(tree):
    groups = set()
    def visit(parent):
        for node in parent.nodes:
            group = getattr(node, "node_tree", None)
            if group and group.library is None and group.get(OWNER_ROLE, "").startswith(SUB_PREFIX) and group not in groups:
                groups.add(group)
                visit(group)
    visit(tree)
    return groups


@contextmanager
def _refresh_graph():
    previous = current_group()
    with _creation_scope():
        main, helpers, materials = _build_graph(previous)
        yield main
        if previous is not None:
            # Rebuild the published main tree only after the entire candidate and run
            # succeeded. Existing modifiers and RNA references keep the same interface.
            old_helpers = _dependencies(previous)
            backup = previous.copy()
            try:
                build_main(helpers, materials, previous)
            except Exception:
                name = previous.name
                previous.user_remap(backup)
                bpy.data.node_groups.remove(previous)
                backup.name = name
                raise
            bpy.data.node_groups.remove(backup)
            main.user_remap(previous)
            bpy.data.node_groups.remove(main)
            main = previous
            # Retain helpers an artist has linked elsewhere. Only discard unused dependencies.
            while unused := {g for g in old_helpers if not g.users}:
                old_helpers -= unused
                bpy.data.batch_remove(unused)
            for group in helpers.values():
                if group[OWNER_ROLE] not in bpy.data.node_groups:
                    group.name = group[OWNER_ROLE]
        bpy.context.scene["rcg_cable_group"] = main


def build_all():
    with _refresh_graph():
        pass
    return current_group()


def create_run(matrix, demo_surfaces=False, refresh=False):
    main = current_group()
    if main is None or refresh:
        with _refresh_graph() as main:
            return add_cable_run(main, matrix, demo_surfaces)
    return add_cable_run(main, matrix, demo_surfaces)


def setup_demo(ng):
    from mathutils import Matrix
    scene = bpy.context.scene
    demo = scene.get("rcg_cable_demo")
    if isinstance(demo, bpy.types.Object) and demo.name in scene.objects and _owned(demo, "demo"):
        return demo
    demo = add_cable_run(ng, Matrix.Identity(4), demo_surfaces=True)
    demo[OWNER_ROLE] = "demo"
    scene["rcg_cable_demo"] = demo
    return demo


def add_cable_run(ng, matrix, demo_surfaces=False):
    """New generator object + Start / End empties at `matrix` (used by the add-on).
    The object's origin is the wall/floor corner, +Y out of the wall, +Z up."""
    with _creation_scope():
        return _add_run(ng, matrix, demo_surfaces)


def _add_run(ng, matrix, demo_surfaces):
    scene = bpy.context.scene
    coll = next((child for child in scene.collection.children
                 if child.users - int(child.use_fake_user) == 1
                 and child.get("rcg_cable_runs") and child.library is None), None)
    if coll is None:
        coll = _track(bpy.data.collections.new("Cable Generator"))
        scene.collection.children.link(coll)
    coll["rcg_cable_runs"] = True
    mesh = _track(bpy.data.meshes.new("CableRun"))
    gen = _track(bpy.data.objects.new("CableRun", mesh))
    coll.objects.link(gen)
    gen.matrix_world = matrix
    mod = gen.modifiers.new("Cable Generator", "NODES")
    mod.node_group = ng
    ends = []
    for suffix, loc, rot in (("Start", (-0.7, 0.0, 1.75), (-math.pi / 2, 0.0, 0.0)),
                             ("End", (1.3, 1.7, 0.0), (0.0, 0.0, 0.0))):
        e = _track(bpy.data.objects.new(f"{gen.name}_{suffix}", None))
        coll.objects.link(e)
        e.empty_display_type = "SINGLE_ARROW"
        e.empty_display_size = 0.3
        e.parent = gen
        e.location, e.rotation_euler = loc, rot
        ends.append(e)
    _mod_set(mod, "Start", ends[0])
    _mod_set(mod, "End", ends[1])
    for socket in ng.interface.items_tree:
        if getattr(socket, "socket_type", None) == "NodeSocketMaterial" and socket.in_out == "INPUT":
            mod[socket.identifier] = socket.default_value
    if demo_surfaces:
        for name, verts in (("CG_Demo_Floor", [(-3, 0, 0), (3, 0, 0), (3, 4, 0), (-3, 4, 0)]),
                            ("CG_Demo_Wall", [(-3, 0, 0), (-3, 0, 3), (3, 0, 3), (3, 0, 0)])):
            me = _track(bpy.data.meshes.new(name))
            me.from_pydata(verts, [], [(0, 1, 2, 3)])
            o = _track(bpy.data.objects.new(name, me))
            coll.objects.link(o)
            o.parent = gen
    return gen


def main():
    ng = build_all()
    gen = setup_demo(ng)
    return ng, gen


if __name__ == "__main__":
    main()
