# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
bl_info = {
    "name": "Hanging Cable Generator",
    "author": "Aidan Hart",
    "version": (1, 0, 2),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Generators",
    "description": "Click anchor points and loose cables hang between them, with exact control over sides "
                   "and loop cuts",
    "category": "Add Mesh",
}

"""
Hanging Cable Generator
-----------------------
Draw Cables, then click in the viewport. The first click sets point A, the second point B and
the cables hang between them. Keep clicking: every new point adds a span from the last one.

  Click               add a point on whatever surface is under the mouse
  Shift+Click         start a new run at this point (no span from the previous point)
  Ctrl (held)         snap to the nearest vertex of the surface under the mouse
  hover an anchor     snaps to it; clicking connects to it (branches, loops)
  Alt+Wheel           sag, Shift+Alt+Wheel weight bias, live
  Backspace / Ctrl+Z  remove the last point
  Enter / Right-click / Esc   done
  Middle mouse, wheel and numpad still navigate while drawing.

The points live in a wire guide object. Edit Points tabs into it: move, extrude or delete
anchors and the cables follow live. Every setting stays editable afterwards, and selected
spans (guide edges) can get their own sag multiplier and bias.

Loop cuts: Sides sets the cross-section (3 and up). Segments is the exact number of segments
per span, or a density per metre clamped to min/max. Curve Focus decides where those loops
go: 0 spaces them evenly, 1 packs them where the cable bends and leaves straight runs bare,
so a handful of segments still gives a clean sag.
"""

import math
import random
import time
import traceback
from functools import lru_cache

import bmesh
import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

DEFAULT_TAB = "Generators"
A_NRM = "hc_normal"     # per anchor (point): normal of the surface it was placed on, zero if unknown
A_SAG = "hc_sag"        # per span (edge): sag multiplier offset, 0 = x1
A_BIAS = "hc_bias"      # per span (edge): added to the weight bias
UP = np.array((0.0, 0.0, 1.0))
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


# ======================================================================================
#  Solver (pure numpy, world space)
# ======================================================================================
class Cfg:
    """Plain snapshot of the settings the solver reads."""
    DEFAULTS = dict(count=3, thickness=0.04, thickness_random=0.25, sag=0.15, bias=0.0, sag_random=0.3,
                    swing=0.0, swing_random=math.radians(6.0), layout="ROW", spread=0.07, gather=0.6,
                    sides=6, seg_mode="FIXED", segments=10, density=2.0, seg_min=4, seg_max=48, focus=0.6,
                    aspect=1.0, profile_rot=0.0, caps=False, smooth=True, uv_tiling=1.0, exit_length=0.0,
                    floor=False, floor_z=0.0, seed=0)

    def __init__(self, **kw):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, kw.pop(k, v))
        if kw:
            raise TypeError(f"unknown settings {sorted(kw)}")

    @classmethod
    def from_props(cls, p):
        return cls(**{k: getattr(p, k) for k in cls.DEFAULTS})


def _low_point(k):
    """Where the sag peaks along the span (0..1) for a load exponent k >= 0."""
    if k < 1e-6:
        return 0.5
    return 0.5 + math.log(2.0 * math.sinh(0.5 * k) / k) / k


@lru_cache(maxsize=1024)
def bias_to_k(b):
    """Weight bias -1..1 -> load exponent. The sag peak moves up to 35 % of the span off centre."""
    b = max(-1.0, min(1.0, b))
    target = 0.35 * abs(b)
    if target < 1e-6:
        return 0.0
    lo, hi = 0.0, 120.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _low_point(mid) - 0.5 < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k if b > 0 else -k


def sag_profile(u, k):
    """Sag below the chord, peak 1. k = 0 is the even-load parabola; otherwise the cable carries a
    load that grows exponentially toward one end, so that end droops more (smooth, finite slopes)."""
    u = np.asarray(u, dtype=float)
    if abs(k) < 1e-3:
        return 4.0 * u * (1.0 - u)
    a, b = math.exp(-0.5 * k), math.exp(0.5 * k)
    d = a + u * (b - a) - np.exp(k * (u - 0.5))
    us = _low_point(abs(k))
    if k < 0:
        us = 1.0 - us
    dmax = a + us * (b - a) - math.exp(k * (us - 0.5))
    return d / dmax


def anchor_rows(P, edges):
    """Per anchor: the horizontal direction cables are laid out along (across the run)."""
    V = len(P)
    acc = np.zeros((V, 3))
    ref = np.zeros((V, 3))
    for a, b in edges:
        d = P[b] - P[a]
        d = np.array((d[0], d[1], 0.0))
        n = math.hypot(d[0], d[1])
        if n < 1e-6:
            continue
        d /= n
        for v in (a, b):
            if not ref[v].any():
                ref[v] = d
            acc[v] += d if np.dot(d, ref[v]) >= 0.0 else -d
    rows = np.cross(UP, acc) if V else np.zeros((0, 3))
    n = np.linalg.norm(rows, axis=1) if V else np.zeros(0)
    rows = rows / np.maximum(n, 1e-12)[:, None]
    rows[n < 1e-6] = (1.0, 0.0, 0.0)
    return rows


def slot_offsets(n, layout, spread):
    """(n, 2) offset of each cable at an anchor: across the run, and up."""
    if n <= 1:
        return np.zeros((max(n, 0), 2))
    i = np.arange(n, dtype=float)
    if layout == "ROW":
        return np.stack([(i - (n - 1) / 2.0) * spread, np.zeros(n)], 1)
    if layout == "STACK":
        return np.stack([np.zeros(n), ((n - 1) / 2.0 - i) * spread], 1)
    r = spread * 0.55 * np.sqrt(i + 0.5)        # sunflower packing, neighbours about `spread` apart
    t = i * GOLDEN
    pts = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return pts - pts.mean(0)


def rotate(v, axis, ang):
    if abs(ang) < 1e-9:
        return v
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _bezier(q, t):
    t = t[:, None]
    s = 1.0 - t
    return s * s * s * q[0] + 3.0 * s * s * t * q[1] + 3.0 * s * t * t * q[2] + t * t * t * q[3]


class StrandCurve:
    """One cable between two anchors in world space, evaluated at any u in 0..1."""

    def __init__(self, A, B, S, k, dirv, oA, oB, gather, nA, nB, exit_len, floor_z):
        self.A, self.chord = A, B - A
        self.S, self.k, self.dirv = S, k, dirv
        self.oA, self.oB, self.g = oA, oB, gather
        self.floor = floor_z
        L = float(np.linalg.norm(self.chord))
        use_a = nA is not None and exit_len > 0.0
        use_b = nB is not None and exit_len > 0.0
        dA = min(0.4, 3.0 * exit_len / L) if use_a else 0.0
        dB = min(0.4, 3.0 * exit_len / L) if use_b else 0.0
        if dA + dB > 0.8:
            f = 0.8 / (dA + dB)
            dA, dB = dA * f, dB * f
        self.dA, self.dB = dA, dB
        # exits: a cubic that leaves the anchor along the surface normal and joins the hang with
        # matching tangent (C1), so the cable comes straight out of its socket before it droops
        if dA > 0.0:
            q0 = self.base(np.array([0.0]))[0]
            q3 = self.base(np.array([dA]))[0]
            self.qA = (q0, q0 + nA * exit_len * 1.5, q3 - self.deriv(dA) * dA / 3.0, q3)
        if dB > 0.0:
            q0 = self.base(np.array([1.0]))[0]
            q3 = self.base(np.array([1.0 - dB]))[0]
            self.qB = (q0, q0 + nB * exit_len * 1.5, q3 + self.deriv(1.0 - dB) * dB / 3.0, q3)

    def base(self, u):
        wide = 1.0 - self.g * (1.0 - np.sin(np.pi * u))
        off = (np.outer(1.0 - u, self.oA) + np.outer(u, self.oB)) * wide[:, None]
        return self.A + np.outer(u, self.chord) + np.outer(self.S * sag_profile(u, self.k), self.dirv) + off

    def deriv(self, u, h=1e-4):
        lo, hi = max(0.0, u - h), min(1.0, u + h)
        p = self.base(np.array([lo, hi]))
        return (p[1] - p[0]) / (hi - lo)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        out = self.base(u)
        if self.dA > 0.0:
            m = u < self.dA
            if m.any():
                out[m] = _bezier(self.qA, u[m] / self.dA)
        if self.dB > 0.0:
            m = u > 1.0 - self.dB
            if m.any():
                out[m] = _bezier(self.qB, (1.0 - u[m]) / self.dB)
        if self.floor is not None:
            inner = (u > 0.0) & (u < 1.0)
            out[inner, 2] = np.maximum(out[inner, 2], self.floor)
        return out


def segment_count(curve, cfg):
    if cfg.seg_mode == "FIXED":
        return max(1, int(cfg.segments))
    p = curve(np.linspace(0.0, 1.0, 65))
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    lo = max(1, int(cfg.seg_min))
    hi = max(lo, int(cfg.seg_max))
    return int(min(max(math.ceil(length * cfg.density), lo), hi))


def resample(curve, nseg, focus):
    """Pick nseg + 1 ring positions. The spacing blends even arc length (focus 0) with sqrt(curvature)
    spacing (focus 1), which is the spacing that keeps a polyline closest to the true curve for a
    fixed number of segments."""
    if nseg <= 1:
        u = np.array([0.0, 1.0])
        return u, curve(u)
    M = max(129, 6 * nseg + 1)
    ud = np.linspace(0.0, 1.0, M)
    pd = curve(ud)
    seg = np.diff(pd, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    total = float(ds.sum())
    if total < 1e-9:
        u = np.linspace(0.0, 1.0, nseg + 1)
        return u, curve(u)
    dirs = seg / np.maximum(ds, 1e-12)[:, None]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", dirs[:-1], dirs[1:]), -1.0, 1.0))
    turn = np.zeros(M - 1)
    turn[:-1] += 0.5 * ang
    turn[1:] += 0.5 * ang
    wk = np.sqrt(turn * ds)
    even = ds / total
    meas = (1.0 - focus) * even + (focus * wk / wk.sum() if wk.sum() > 1e-9 else focus * even) + 1e-6 * even
    cum = np.concatenate([[0.0], np.cumsum(meas)])
    cum /= cum[-1]
    u = np.interp(np.linspace(0.0, 1.0, nseg + 1), cum, ud)
    u[0], u[-1] = 0.0, 1.0
    return u, curve(u)


def tube(C, radius, sides, aspect, rot, v0, uv_tiling):
    """Rings around the centre points C. Rings sit on the bisector of their two segments, widened
    across the bend (mitre) so low segment counts keep their thickness; frames are parallel
    transported so the profile never twists."""
    K = len(C)
    d = np.diff(C, axis=0)
    dl = np.linalg.norm(d, axis=1)
    dh = d / np.maximum(dl, 1e-12)[:, None]
    T = np.empty_like(C)
    T[0], T[-1] = dh[0], dh[-1]
    axis = np.zeros_like(C)
    scale = np.ones(K)
    if K > 2:
        s = dh[:-1] + dh[1:]
        sn = np.linalg.norm(s, axis=1)
        T[1:-1] = np.where((sn > 1e-6)[:, None], s / np.maximum(sn, 1e-12)[:, None], dh[1:])
        m = dh[1:] - dh[:-1]
        mn = np.linalg.norm(m, axis=1)
        axis[1:-1] = np.where((mn > 1e-6)[:, None], m / np.maximum(mn, 1e-12)[:, None], 0.0)
        scale[1:-1] = 1.0 / np.clip(np.einsum("ij,ij->i", T[1:-1], dh[1:]), 0.5, 1.0)
    N = np.empty_like(C)
    n0 = UP - np.dot(UP, T[0]) * T[0]
    if np.linalg.norm(n0) < 1e-4:
        x = np.array((1.0, 0.0, 0.0))
        n0 = x - np.dot(x, T[0]) * T[0]
    N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, K):
        n = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        nn = np.linalg.norm(n)
        N[i] = n / nn if nn > 1e-9 else N[i - 1]
    Bn = np.cross(T, N)
    ang = 2.0 * np.pi * np.arange(sides) / sides + np.pi / sides
    ph = np.cos(ang) * radius
    pw = np.sin(ang) * radius * aspect
    cosine, sine = math.cos(rot), math.sin(rot)
    ph, pw = cosine * ph - sine * pw, sine * ph + cosine * pw
    off = ph[None, :, None] * N[:, None, :] + pw[None, :, None] * Bn[:, None, :]
    if K > 2:
        proj = np.einsum("ksj,kj->ks", off, axis)
        off = off + proj[..., None] * (scale - 1.0)[:, None, None] * axis[:, None, :]
    verts = C[:, None, :] + off
    v = v0 + np.concatenate([[0.0], np.cumsum(dl)]) * uv_tiling
    return verts.reshape(-1, 3), v


def tube_faces(K, sides, vbase, V):
    idx = np.arange(K * sides).reshape(K, sides) + vbase
    j = np.arange(sides)
    jn = (j + 1) % sides
    quads = np.stack([idx[:-1, j], idx[:-1, jn], idx[1:, jn], idx[1:, j]], axis=-1).reshape(-1)
    shp = (K - 1, sides)
    u0 = np.broadcast_to(j / sides, shp)
    u1 = np.broadcast_to((j + 1) / sides, shp)
    va = np.broadcast_to(V[:-1, None], shp)
    vb = np.broadcast_to(V[1:, None], shp)
    uv = np.stack([np.stack([u0, va], -1), np.stack([u1, va], -1), np.stack([u1, vb], -1),
                   np.stack([u0, vb], -1)], axis=2)
    return quads, uv.reshape(-1, 2), idx


def build_cables(P, NRM, edges, span_sag, span_bias, cfg):
    """P, NRM: (V, 3) world anchors and surface normals (zero = unknown). edges: (E, 2) spans.
    span_sag / span_bias: (E,) per-span offsets. Returns arrays ready to write into a mesh."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    NRM = np.asarray(NRM, dtype=float).reshape(-1, 3)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    span_sag = np.asarray(span_sag, dtype=float).reshape(-1)
    span_bias = np.asarray(span_bias, dtype=float).reshape(-1)
    n = max(1, int(cfg.count))
    sides = max(3, int(cfg.sides))
    rows = anchor_rows(P, edges)
    slots = slot_offsets(n, cfg.layout, cfg.spread)
    R = 0.5 * cfg.thickness
    radii, uvoff = [], []
    for i in range(n):
        rs = random.Random(f"s{cfg.seed}:{i}")
        radii.append(max(R * 0.05, R * (1.0 + cfg.thickness_random * (2.0 * rs.random() - 1.0))))
        uvoff.append(rs.random())
    nlen = np.linalg.norm(NRM, axis=1) if len(NRM) else np.zeros(0)

    co, loops, sizes, uvs, smooth, rims = [], [], [], [], [], []
    vbase = tubes = spans = 0
    for ei, (a, b) in enumerate(edges):
        if a == b or a >= len(P) or b >= len(P):
            continue
        A, B = P[a], P[b]
        chord = B - A
        L = float(np.linalg.norm(chord))
        if L < 1e-4:
            continue
        spans += 1
        c = chord / L
        # hang direction: straight down, turning toward the chord's perpendicular as the span
        # approaches vertical so a plumb span bows outward instead of folding onto itself
        gp = -UP + c[2] * c
        gl = float(np.linalg.norm(gp))
        if gl > 1e-4:
            p_hat = gp / gl
        else:
            F = NRM[a] + NRM[b]
            F = np.array((F[0], F[1], 0.0))
            if np.linalg.norm(F) < 1e-4:
                F = np.array((1.0, 0.0, 0.0))
            F = F - np.dot(F, c) * c
            p_hat = F / max(float(np.linalg.norm(F)), 1e-9)
        w = 1.0 - _smooth01(gl / 0.5)
        base_dir = -UP * (1.0 - w) + p_hat * w
        base_dir /= np.linalg.norm(base_dir)
        h = np.array((chord[0], chord[1], 0.0))
        hl = float(np.linalg.norm(h))
        r_span = np.cross(UP, h / hl) if hl > 1e-6 else rows[a]
        sa = 1.0 if np.dot(rows[a], r_span) >= 0.0 else -1.0
        sb = 1.0 if np.dot(rows[b], r_span) >= 0.0 else -1.0
        k = bias_to_k(round(max(-1.0, min(1.0, cfg.bias + float(span_bias[ei]))), 4))
        S_span = cfg.sag * max(0.0, 1.0 + float(span_sag[ei])) * L
        nA = NRM[a] / nlen[a] if nlen[a] > 0.5 else None
        nB = NRM[b] / nlen[b] if nlen[b] > 0.5 else None
        for i in range(n):
            rr = random.Random(f"e{cfg.seed}:{ei}:{i}")
            S = S_span * max(0.0, 1.0 + cfg.sag_random * (2.0 * rr.random() - 1.0))
            sw = cfg.swing + cfg.swing_random * (2.0 * rr.random() - 1.0)
            dirv = rotate(base_dir, c, sw)
            oA = sa * slots[i, 0] * rows[a] + slots[i, 1] * UP
            oB = sb * slots[i, 0] * rows[b] + slots[i, 1] * UP
            floor = (cfg.floor_z + radii[i]) if cfg.floor else None
            curve = StrandCurve(A, B, S, k, dirv, oA, oB, cfg.gather, nA, nB, cfg.exit_length, floor)
            _, C = resample(curve, segment_count(curve, cfg), cfg.focus)
            verts, V = tube(C, radii[i], sides, cfg.aspect, cfg.profile_rot, uvoff[i], cfg.uv_tiling)
            K = len(C)
            q, uv, idx = tube_faces(K, sides, vbase, V)
            co.append(verts)
            loops.append(q)
            uvs.append(uv)
            nq = (K - 1) * sides
            sizes.append(np.full(nq, 4, np.int32))
            smooth.append(np.full(nq, bool(cfg.smooth)))
            if cfg.caps:
                circ = np.stack([0.5 + 0.5 * np.cos(2 * np.pi * np.arange(sides) / sides),
                                 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(sides) / sides)], 1)
                loops += [idx[0, ::-1], idx[-1]]
                uvs += [circ[::-1], circ]
                sizes.append(np.array([sides, sides], np.int32))
                smooth.append(np.zeros(2, bool))
                for ring in (idx[0], idx[-1]):
                    rims.append(np.stack([ring, np.roll(ring, -1)], 1))
            vbase += K * sides
            tubes += 1

    if co:
        out = dict(co=np.concatenate(co), loops=np.concatenate(loops).astype(np.int32),
                   sizes=np.concatenate(sizes), uv=np.concatenate(uvs), smooth=np.concatenate(smooth),
                   rims=np.concatenate(rims) if rims else np.zeros((0, 2), int))
    else:
        out = dict(co=np.zeros((0, 3)), loops=np.zeros(0, np.int32), sizes=np.zeros(0, np.int32),
                   uv=np.zeros((0, 2)), smooth=np.zeros(0, bool), rims=np.zeros((0, 2), int))
    out["info"] = dict(spans=spans, tubes=tubes, verts=len(out["co"]),
                       tris=int(np.sum(out["sizes"] - 2)) if len(out["sizes"]) else 0)
    return out


# ======================================================================================
#  Blender side
# ======================================================================================
_pending = {}
_last_info = {}
_last_transform = {}
_sessions = {}
DELAY = 0.04


def _attr(me, name, domain, dtype, n, width):
    out = np.zeros(n * width, np.float32)
    a = me.attributes.get(name)
    if n and a is not None and a.domain == domain and a.data_type == dtype and len(a.data) == n:
        a.data.foreach_get("vector" if width == 3 else "value", out)
    return out.reshape(n, width) if width > 1 else out


def read_guide(obj):
    """Anchors, normals (local), spans and per-span offsets, from Object or Edit Mode."""
    me = obj.data
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        bm.verts.index_update()
        nl = bm.verts.layers.float_vector.get(A_NRM)
        sl = bm.edges.layers.float.get(A_SAG)
        bl = bm.edges.layers.float.get(A_BIAS)
        co = np.array([v.co[:] for v in bm.verts], float).reshape(-1, 3)
        nrm = (np.array([v[nl][:] for v in bm.verts], float).reshape(-1, 3) if nl is not None
               else np.zeros((len(co), 3)))
        edges = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges], int).reshape(-1, 2)
        sag = np.array([e[sl] for e in bm.edges], float) if sl is not None else np.zeros(len(edges))
        bias = np.array([e[bl] for e in bm.edges], float) if bl is not None else np.zeros(len(edges))
        return co, nrm, edges, sag, bias
    nv, ne = len(me.vertices), len(me.edges)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    ev = np.empty(ne * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    return (co.reshape(-1, 3).astype(float), _attr(me, A_NRM, "POINT", "FLOAT_VECTOR", nv, 3).astype(float),
            ev.reshape(-1, 2), _attr(me, A_SAG, "EDGE", "FLOAT", ne, 1).astype(float),
            _attr(me, A_BIAS, "EDGE", "FLOAT", ne, 1).astype(float))


def write_mesh(me, data):
    me.clear_geometry()
    co = data["co"]
    if len(co):
        me.vertices.add(len(co))
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        loops, sizes = data["loops"], data["sizes"]
        me.loops.add(len(loops))
        me.loops.foreach_set("vertex_index", loops)
        me.polygons.add(len(sizes))
        starts = np.zeros(len(sizes), np.int32)
        starts[1:] = np.cumsum(sizes)[:-1]
        me.polygons.foreach_set("loop_start", starts)
        me.polygons.foreach_set("loop_total", sizes)
        uvl = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", data["uv"].astype(np.float32).ravel())
        me.polygons.foreach_set("use_smooth", data["smooth"])
    me.update(calc_edges=True)
    rims = data["rims"]
    if len(rims):
        ev = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", ev)
        ev = np.sort(ev.reshape(-1, 2), axis=1)
        keys = {(int(a), int(b)) for a, b in np.sort(rims, axis=1)}
        flags = np.array([(int(a), int(b)) in keys for a, b in ev], bool)
        attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
        attr.data.foreach_set("value", flags)


def _material():
    mat = bpy.data.materials.get("HC Cable")
    if mat is None:
        mat = bpy.data.materials.new("HC Cable")
        rgb = (0.03, 0.03, 0.032)
        mat.diffuse_color = (*rgb, 1.0)
        if mat.node_tree is not None:
            b = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if b is not None:
                b.inputs["Base Color"].default_value = (*rgb, 1.0)
                b.inputs["Roughness"].default_value = 0.55
    return mat


def ensure_output(guide):
    cfg = guide.hanging_cable
    out = cfg.output
    if out is not None and out.parent is not None and out.parent is not guide:
        out = None                       # a duplicated guide: give it its own cables
    if out is None or bpy.data.objects.get(out.name) is not out:
        me = bpy.data.meshes.new(guide.name + "_Mesh")
        out = bpy.data.objects.new(guide.name + "_Mesh", me)
        colls = guide.users_collection or [bpy.context.scene.collection]
        colls[0].objects.link(out)
        cfg.output = out
    if out.parent is not guide:
        out.parent = guide
    if out.matrix_parent_inverse != Matrix.Identity(4):
        out.matrix_parent_inverse = Matrix.Identity(4)
    if out.matrix_basis != Matrix.Identity(4):
        out.matrix_basis = Matrix.Identity(4)
    return out


def _transform_key(ob):
    key = tuple(round(x, 6) for row in ob.matrix_world.to_3x3() for x in row)
    # Floor height is in world space, so a vertical move also changes the sag.
    return key + (round(ob.matrix_world.translation.z, 6),) if ob.hanging_cable.floor else key


def rebuild(guide):
    if guide is None or guide.type != "MESH" or not guide.hanging_cable.is_guide:
        return None
    M = np.array(guide.matrix_world, float)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    out = ensure_output(guide)
    if out.mode == "EDIT":
        return None
    t0 = time.perf_counter()
    co, nrm, edges, sag, bias = read_guide(guide)
    sess = _sessions.get(guide.name)
    if sess is not None and sess.preview is not None:
        i0, p, nn = sess.preview
        if 0 <= i0 < len(co):           # live span from the last point to the mouse
            co = np.vstack([co, p])
            nrm = np.vstack([nrm, nn])
            edges = np.vstack([edges, [[i0, len(co) - 1]]])
            sag, bias = np.append(sag, 0.0), np.append(bias, 0.0)
    Pw = co @ M[:3, :3].T + M[:3, 3]
    Nw = nrm @ Minv[:3, :3]
    ln = np.linalg.norm(Nw, axis=1)
    Nw = np.where((ln > 1e-9)[:, None], Nw / np.maximum(ln, 1e-12)[:, None], 0.0)
    data = build_cables(Pw, Nw, edges, sag, bias, Cfg.from_props(guide.hanging_cable))
    data["co"] = data["co"] @ Minv[:3, :3].T + Minv[:3, 3]
    me = out.data
    if me.users > 1:
        me = out.data = me.copy()
    write_mesh(me, data)
    if not me.materials:
        me.materials.append(_material())
    info = data["info"]
    info["seconds"] = time.perf_counter() - t0
    _last_info[guide.name] = info
    _last_transform[guide.name] = _transform_key(guide)
    return info


def _tick():
    now = time.monotonic()
    for name, t in list(_pending.items()):
        if now - t >= DELAY:
            _pending.pop(name, None)
            ob = bpy.data.objects.get(name)
            if ob is not None:
                try:
                    rebuild(ob)
                except Exception as ex:          # keep the timer alive, report in the console
                    print("Hanging Cables rebuild failed:", ex)
    return DELAY / 2 if _pending else None


def request_rebuild(guide):
    _pending.setdefault(guide.name, time.monotonic())   # throttle: cables follow while you drag
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=DELAY)


def _guides_from_id(idd):
    if isinstance(idd, bpy.types.Object):
        ob = idd.original
        return [ob] if ob.type == "MESH" and ob.hanging_cable.is_guide else []
    if isinstance(idd, bpy.types.Mesh):
        me = idd.original
        return [o for o in bpy.data.objects if o.data is me and o.type == "MESH" and o.hanging_cable.is_guide]
    return []


@persistent
def _on_depsgraph(scene, depsgraph):
    for upd in depsgraph.updates:
        for g in _guides_from_id(upd.id):
            if not g.hanging_cable.live:
                continue
            if upd.is_updated_geometry:
                request_rebuild(g)
            elif upd.is_updated_transform and _last_transform.get(g.name) != _transform_key(g):
                request_rebuild(g)       # rotation or scale changes which way is down


def guide_of(obj):
    if obj is None:
        return None
    if obj.type == "MESH" and obj.hanging_cable.is_guide:
        return obj
    par = obj.parent
    if par is not None and par.hanging_cable.is_guide and par.hanging_cable.output is obj:
        return par
    return None


def init_guide(obj):
    obj.hanging_cable.is_guide = True
    me = obj.data
    for name, dtype, domain in ((A_NRM, "FLOAT_VECTOR", "POINT"), (A_SAG, "FLOAT", "EDGE"),
                                (A_BIAS, "FLOAT", "EDGE")):
        if me.attributes.get(name) is None:
            me.attributes.new(name, dtype, domain)
    obj.display_type = "WIRE"
    obj.hide_render = True


SKIP = {"rna_type", "name", "is_guide", "output", "end_vertex"}


def copy_settings(src, dst):
    for p in src.bl_rna.properties:
        if p.identifier not in SKIP and not p.is_readonly:
            setattr(dst, p.identifier, getattr(src, p.identifier))


def new_guide(context, src=None):
    me = bpy.data.meshes.new("Cables")
    ob = bpy.data.objects.new("Cables", me)
    context.collection.objects.link(ob)
    if src is not None:
        copy_settings(src, ob.hanging_cable)
    init_guide(ob)
    for o in context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob
    return ob


def _local_normal(ob, n):
    if n is None or n.length < 1e-9:
        return Vector((0.0, 0.0, 0.0))
    v = ob.matrix_world.to_3x3().transposed() @ n
    return v.normalized() if v.length > 1e-9 else Vector((0.0, 0.0, 0.0))


def _world_normal(ob, n):
    if n.length < 1e-9:
        return Vector((0.0, 0.0, 0.0))
    v = ob.matrix_world.to_3x3().inverted_safe().transposed() @ n
    return v.normalized() if v.length > 1e-9 else Vector((0.0, 0.0, 0.0))


# ======================================================================================
#  Click-to-place drawing
# ======================================================================================
NAV = {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "WHEELINMOUSE", "WHEELOUTMOUSE", "TRACKPADPAN",
       "TRACKPADZOOM", "MOUSEROTATE", "MOUSESMARTZOOM", "HOME", "BUTTON4MOUSE", "BUTTON5MOUSE",
       "BUTTON6MOUSE", "BUTTON7MOUSE", "WINDOW_DEACTIVATE"}
OVERLAYS = {"UI", "TOOLS", "HEADER", "TOOL_HEADER", "ASSET_SHELF", "ASSET_SHELF_HEADER", "HUD",
            "NAVIGATION_BAR", "FOOTER"}
BLOCK = {"TAB", "DEL", "F2", "F3", "F4", "ACCENT_GRAVE", "PERIOD", "COMMA"}


class DrawSession:
    """State of one Draw Cables run. Kept apart from the operator so it can be driven directly."""
    SNAP_PX = 14.0

    def __init__(self, context, guide, created):
        self.name = guide.name
        self.created = created
        self.area = context.area
        self.region = next(r for r in self.area.regions if r.type == "WINDOW")
        self.rv3d = self.area.spaces.active.region_3d
        cfg = guide.hanging_cable
        nv = len(guide.data.vertices)
        self.last = None if created or not (0 <= cfg.end_vertex < nv) else cfg.end_vertex
        self.history = []
        self.hover = None              # (world position, world normal, existing anchor index or None)
        self.preview = None            # (from index, local position, local normal)
        self.snapped = ""
        self.t_preview = 0.0
        try:
            self.ui_scale = max(0.5, context.preferences.system.ui_scale or 1.0)
        except AttributeError:
            self.ui_scale = 1.0
        _sessions[self.name] = self

    @property
    def guide(self):
        return bpy.data.objects.get(self.name)

    # ---------------------------------------------------------------- events
    def handle(self, context, event):
        g = self.guide
        if g is None or not g.hanging_cable.is_guide or g.mode != "OBJECT":
            return self.finish(context)
        t, v = event.type, event.value
        if t in ("WHEELUPMOUSE", "WHEELDOWNMOUSE") and event.alt:
            step = 1.0 if t == "WHEELUPMOUSE" else -1.0
            cfg = g.hanging_cable
            if event.shift:
                cfg.bias = max(-1.0, min(1.0, cfg.bias + 0.05 * step))
            else:
                cfg.sag = max(0.0, cfg.sag + 0.01 * step)
            self.update_header()
            return {"RUNNING_MODAL"}
        if t in NAV or t.startswith("NDOF") or t.startswith("TIMER") or t.startswith("NUMPAD_"):
            return {"PASS_THROUGH"}
        if t in ("MOUSEMOVE", "INBETWEEN_MOUSEMOVE"):
            if not self.in_view(event):
                return {"PASS_THROUGH"}
            self.update_hover(context, event)
            return {"RUNNING_MODAL"}
        if t == "LEFTMOUSE":
            if event.alt or not self.in_view(event):
                return {"PASS_THROUGH"}
            if v == "PRESS":
                self.update_hover(context, event)
                self.commit(new_run=event.shift)
            return {"RUNNING_MODAL"}
        if v == "PRESS" and (t == "BACK_SPACE" or (t == "Z" and (event.ctrl or event.oskey))):
            self.undo()
            return {"RUNNING_MODAL"}
        if v == "PRESS" and t in ("RIGHTMOUSE", "RET", "NUMPAD_ENTER", "SPACE", "ESC"):
            if t == "RIGHTMOUSE" and not self.in_view(event):
                return {"PASS_THROUGH"}
            return self.finish(context)
        if len(t) == 1 or t in BLOCK:
            return {"RUNNING_MODAL"}      # keep stray shortcuts off the guide while drawing
        return {"PASS_THROUGH"}

    def in_view(self, event):
        x, y = event.mouse_x, event.mouse_y
        r = self.region
        if not (r.x <= x < r.x + r.width and r.y <= y < r.y + r.height):
            return False
        for o in self.area.regions:
            if o.type in OVERLAYS and o.width > 1 and o.height > 1:
                if o.x <= x < o.x + o.width and o.y <= y < o.y + o.height:
                    return False
        return True

    # ---------------------------------------------------------------- picking
    def pick(self, context, event):
        g = self.guide
        region, rv3d = self.region, self.rv3d
        coord = (event.mouse_x - region.x, event.mouse_y - region.y)
        M = g.matrix_world
        me = g.data
        attr = me.attributes.get(A_NRM)
        best, best_d = None, self.SNAP_PX * self.ui_scale
        for vtx in me.vertices:
            w = M @ vtx.co
            p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, w)
            if p2 is None:
                continue
            d = math.hypot(p2.x - coord[0], p2.y - coord[1])
            if d < best_d:
                best, best_d = (w, vtx.index), d
        if best is not None:
            n = Vector(attr.data[best[1]].vector) if attr is not None and attr.domain == "POINT" else Vector()
            self.snapped = "anchor"
            return best[0], _world_normal(g, n), best[1]
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        o = origin
        for _ in range(24):
            ok, loc, nrm, idx, ob, mat = scene.ray_cast(depsgraph, o, direction)
            if not ok:
                break
            if guide_of(ob.original) is not None:          # look through cables and guides
                o = loc + direction * 1e-3
                continue
            self.snapped = ""
            if event.ctrl:
                vloc = self._face_vertex(ob, depsgraph, idx, mat, loc)
                if vloc is not None:
                    loc = vloc
                    self.snapped = "vertex"
            nrm = nrm.normalized()
            if nrm.dot(direction) > 0.0:                 # back face: point the normal at the viewer
                nrm = -nrm
            return loc + nrm * g.hanging_cable.surface_offset, nrm, None
        self.snapped = "air"
        if self.last is not None and self.last < len(me.vertices):
            depth = M @ me.vertices[self.last].co
        else:
            depth = scene.cursor.location
        loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, depth)
        return loc, Vector((0.0, 0.0, 0.0)), None

    @staticmethod
    def _face_vertex(ob, depsgraph, idx, mat, loc):
        try:
            me = ob.original.evaluated_get(depsgraph).data
            poly = me.polygons[idx]
            pts = [mat @ me.vertices[i].co for i in poly.vertices]
        except (AttributeError, IndexError, RuntimeError, TypeError):
            return None
        return min(pts, key=lambda p: (p - loc).length) if pts else None

    def update_hover(self, context, event):
        g = self.guide
        self.hover = self.pick(context, event)
        self.set_preview()
        self.refresh()
        self.update_header()
        return g

    def set_preview(self):
        g = self.guide
        hit = self.hover
        if hit is None or self.last is None or (hit[2] is not None and hit[2] == self.last):
            self.preview = None
            return
        p = g.matrix_world.inverted_safe() @ hit[0]
        self.preview = (self.last, np.array(p[:], float), np.array(_local_normal(g, hit[1])[:], float))

    def refresh(self):
        """Rebuild now if that keeps up with the mouse, otherwise let the timer catch up."""
        now = time.perf_counter()
        info = _last_info.get(self.name)
        cost = info["seconds"] if info else 0.0
        if now - self.t_preview >= min(0.25, 2.0 * cost):
            self.t_preview = now
            rebuild(self.guide)
            _pending.pop(self.name, None)
            self.redraw()
        else:
            request_rebuild(self.guide)

    def redraw(self):
        try:
            self.area.tag_redraw()
        except (AttributeError, ReferenceError):
            pass

    # ---------------------------------------------------------------- edits
    def commit(self, new_run=False):
        hit = self.hover
        g = self.guide
        if hit is None or g is None:
            return
        pos, nrm, snap = hit
        me = g.data
        if self.created and len(me.vertices) == 0:
            g.matrix_world = Matrix.Translation(pos)       # object origin at the first anchor
        inv = g.matrix_world.inverted_safe()
        bm = bmesh.new()
        bm.from_mesh(me)
        nl = bm.verts.layers.float_vector.get(A_NRM) or bm.verts.layers.float_vector.new(A_NRM)
        bm.verts.ensure_lookup_table()
        if snap is None or snap >= len(bm.verts):
            vert = bm.verts.new(inv @ pos)
            vert[nl] = _local_normal(g, nrm)
            made_vert = True
        else:
            vert = bm.verts[snap]
            made_vert = False
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        made_edge = None
        if self.last is not None and not new_run and self.last != vert.index and self.last < len(bm.verts):
            a = bm.verts[self.last]
            if bm.edges.get((a, vert)) is None:
                bm.edges.new((a, vert))
                made_edge = (self.last, vert.index)
        self.history.append((made_vert, vert.index, made_edge, self.last))
        self.last = vert.index
        bm.to_mesh(me)
        bm.free()
        me.update()
        g.hanging_cable.end_vertex = self.last
        self.set_preview()
        rebuild(g)
        self.t_preview = time.perf_counter()
        self.redraw()

    def undo(self):
        g = self.guide
        if not self.history or g is None:
            return
        made_vert, vi, made_edge, prev = self.history.pop()
        me = g.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        if made_edge is not None and max(made_edge) < len(bm.verts):
            e = bm.edges.get((bm.verts[made_edge[0]], bm.verts[made_edge[1]]))
            if e is not None:
                bm.edges.remove(e)
        if made_vert and vi < len(bm.verts):
            bm.verts.remove(bm.verts[vi])
        bm.to_mesh(me)
        bm.free()
        me.update()
        self.last = prev
        g.hanging_cable.end_vertex = -1 if prev is None else prev
        self.set_preview()
        rebuild(g)
        self.update_header()
        self.redraw()

    def update_header(self):
        g = self.guide
        if g is None:
            return
        cfg = g.hanging_cable
        snap = {"anchor": "   [on anchor]", "vertex": "   [vertex]", "air": "   [in air]"}.get(self.snapped, "")
        first = "Click: point A" if self.last is None else "Click: next point"
        txt = (f"Hanging Cables   {first}   Shift+Click: new run   Ctrl: snap to vertex   "
               f"Alt+Wheel: sag {cfg.sag * 100:.0f}%   Shift+Alt+Wheel: bias {cfg.bias:+.2f}   "
               f"Backspace: undo   Enter / Right-click / Esc: done{snap}")
        try:
            self.area.header_text_set(txt)
        except (AttributeError, ReferenceError):
            pass

    def finish(self, context):
        _sessions.pop(self.name, None)
        self.preview = None
        try:
            self.area.header_text_set(None)
        except (AttributeError, ReferenceError):
            pass
        try:
            context.window.cursor_modal_restore()
        except (AttributeError, ReferenceError):
            pass
        g = self.guide
        if g is None:
            return {"CANCELLED"}
        if self.created and len(g.data.edges) == 0:          # nothing drawn: leave no trace
            for ob in (g.hanging_cable.output, g):
                if ob is not None:
                    me = ob.data
                    bpy.data.objects.remove(ob)
                    if me is not None and me.users == 0:
                        bpy.data.meshes.remove(me)
            return {"CANCELLED"}
        rebuild(g)
        g.select_set(True)
        context.view_layer.objects.active = g
        return {"FINISHED"}


class HC_OT_draw(bpy.types.Operator):
    bl_idname = "hanging_cables.draw"
    bl_label = "Draw Cables"
    bl_description = ("Click points in the viewport and cables hang between them. Keep clicking to chain "
                      "spans; Enter, right-click or Esc when done")
    bl_options = {"REGISTER", "UNDO"}

    extend: BoolProperty(name="Continue", default=False, options={"SKIP_SAVE"},
                         description="Add points to the selected cables instead of starting new ones")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    @classmethod
    def description(cls, context, properties):
        if properties.extend:
            return "Keep drawing on the selected cables, starting from their last point"
        return cls.bl_description

    def invoke(self, context, event):
        if context.mode != "OBJECT" and context.active_object is not None:
            bpy.ops.object.mode_set(mode="OBJECT")
        active = guide_of(context.active_object)
        if self.extend and active is not None:
            guide, created = active, False
            guide.hide_set(False)
        else:
            src = active.hanging_cable if active is not None else context.scene.hanging_cable_defaults
            guide, created = new_guide(context, src), True
        self.session = DrawSession(context, guide, created)
        context.window_manager.modal_handler_add(self)
        try:
            context.window.cursor_modal_set("CROSSHAIR")
        except (AttributeError, TypeError):
            pass
        self.session.update_header()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        try:
            return self.session.handle(context, event)
        except Exception:
            traceback.print_exc()
            return self.session.finish(context)


# ======================================================================================
#  Settings
# ======================================================================================
def _changed(self, context):
    ob = self.id_data
    if isinstance(ob, bpy.types.Object) and self.is_guide:
        request_rebuild(ob)


LAYOUT_ITEMS = [("ROW", "Row", "Side by side, across the run"),
                ("BUNDLE", "Bundle", "Packed round bundle"),
                ("STACK", "Stack", "One above the other")]
SEG_ITEMS = [("FIXED", "Fixed", "Exact number of segments per span"),
             ("DENSITY", "Per Metre", "Segments from the cable length, clamped to Min / Max")]


class HC_Settings(bpy.types.PropertyGroup):
    is_guide: BoolProperty(default=False)
    output: PointerProperty(type=bpy.types.Object)
    end_vertex: IntProperty(default=-1)
    live: BoolProperty(name="Live Update", default=True, description="Rebuild while the points change")

    count: IntProperty(name="Cables", default=3, min=1, soft_max=12, max=64, update=_changed,
                       description="Cables per span")
    thickness: FloatProperty(name="Thickness", default=0.04, min=0.001, soft_max=0.3, unit="LENGTH",
                             precision=3, update=_changed, description="Cable diameter")
    thickness_random: FloatProperty(name="Thickness Random", default=0.25, min=0.0, max=1.0, subtype="FACTOR",
                                    update=_changed, description="Per-cable thickness variation")
    sag: FloatProperty(name="Sag", default=0.15, min=0.0, soft_max=0.6, max=3.0, subtype="FACTOR",
                       update=_changed,
                       description="How far the cables hang below the line between anchors, as a fraction "
                                   "of the span length")
    bias: FloatProperty(name="Weight Bias", default=0.0, min=-1.0, max=1.0, subtype="FACTOR", update=_changed,
                        description="Move the weight toward one end so that side droops more "
                                    "(- toward where the span starts, + toward where it ends)")
    sag_random: FloatProperty(name="Sag Random", default=0.3, min=0.0, max=1.0, subtype="FACTOR",
                              update=_changed, description="Per-cable, per-span sag variation")
    swing: FloatProperty(name="Swing", default=0.0, min=-math.pi / 2, max=math.pi / 2, subtype="ANGLE",
                         update=_changed, description="Tilt the hang sideways around the span, like wind")
    swing_random: FloatProperty(name="Swing Random", default=math.radians(6.0), min=0.0, max=math.pi / 2,
                                subtype="ANGLE", update=_changed, description="Per-cable sideways variation")
    layout: EnumProperty(name="Layout", items=LAYOUT_ITEMS, default="ROW", update=_changed)
    spread: FloatProperty(name="Spread", default=0.07, min=0.0, soft_max=1.0, unit="LENGTH", update=_changed,
                          description="Spacing between cables")
    gather: FloatProperty(name="Gather", default=0.6, min=0.0, max=1.0, subtype="FACTOR", update=_changed,
                          description="Pull the cables together at the anchors (1 = all meet at the point)")

    sides: IntProperty(name="Sides", default=6, min=3, max=32, update=_changed,
                       description="Sides of the cross-section")
    seg_mode: EnumProperty(name="Segments", items=SEG_ITEMS, default="FIXED", update=_changed)
    segments: IntProperty(name="Segments", default=10, min=1, soft_max=64, max=256, update=_changed,
                          description="Segments along each span: exactly this many minus one loop cuts")
    density: FloatProperty(name="Per Metre", default=2.0, min=0.01, soft_max=20.0, update=_changed,
                           description="Segments per metre of cable")
    seg_min: IntProperty(name="Min", default=4, min=1, max=256, update=_changed)
    seg_max: IntProperty(name="Max", default=48, min=1, max=1024, update=_changed)
    focus: FloatProperty(name="Curve Focus", default=0.6, min=0.0, max=1.0, subtype="FACTOR", update=_changed,
                         description="0 spaces loops evenly; 1 packs them where the cable bends and leaves "
                                     "straight runs bare")

    aspect: FloatProperty(name="Aspect", default=1.0, min=0.1, soft_max=4.0, max=10.0, update=_changed,
                          description="Width / height of the cross-section. 1 is round, higher for flat cables")
    profile_rot: FloatProperty(name="Rotate", default=0.0, subtype="ANGLE", update=_changed,
                               description="Rotate the cross-section")
    caps: BoolProperty(name="Cap Ends", default=False, update=_changed, description="Close the cable ends")
    smooth: BoolProperty(name="Smooth", default=True, update=_changed, description="Smooth shading")
    uv_tiling: FloatProperty(name="UV Tiling", default=1.0, min=0.0, soft_max=10.0, update=_changed,
                             description="Texture repeats per metre along the cable (U wraps around once)")
    exit_length: FloatProperty(name="Exit Length", default=0.0, min=0.0, soft_max=1.0, unit="LENGTH",
                               update=_changed,
                               description="Cables leave each anchor straight out of the surface it was "
                                           "placed on before they droop. 0 = off")
    floor: BoolProperty(name="Rest on Floor", default=False, update=_changed,
                        description="Cables lie on the floor height instead of passing through it")
    floor_z: FloatProperty(name="Floor Height", default=0.0, unit="LENGTH", update=_changed)
    seed: IntProperty(name="Seed", default=0, min=0, update=_changed)
    surface_offset: FloatProperty(name="Surface Offset", default=0.01, soft_min=0.0, soft_max=0.2,
                                  unit="LENGTH",
                                  description="Draw Cables lifts new points off the surface by this much")


# ======================================================================================
#  Operators
# ======================================================================================
class HC_OT_span_set(bpy.types.Operator):
    bl_idname = "hanging_cables.span_set"
    bl_label = "Adjust Spans"
    bl_description = "Sag multiplier and weight bias for the selected spans (guide edges)"
    bl_options = {"REGISTER", "UNDO"}

    sag: FloatProperty(name="Sag x", default=1.0, min=0.0, soft_max=4.0, max=20.0,
                       description="Multiplies the sag of these spans")
    bias: FloatProperty(name="Bias", default=0.0, min=-2.0, max=2.0,
                        description="Added to the Weight Bias for these spans")
    reset: BoolProperty(default=False, options={"SKIP_SAVE", "HIDDEN"})

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return ob is not None and ob.type == "MESH" and ob.hanging_cable.is_guide

    def _layers(self, bm):
        if bm.edges.layers.float.get(A_SAG) is None:
            bm.edges.layers.float.new(A_SAG)
        if bm.edges.layers.float.get(A_BIAS) is None:
            bm.edges.layers.float.new(A_BIAS)
        return bm.edges.layers.float.get(A_SAG), bm.edges.layers.float.get(A_BIAS)

    def invoke(self, context, event):
        bm = bmesh.from_edit_mesh(context.edit_object.data)
        sel = [e for e in bm.edges if e.select]
        if not sel:
            self.report({"WARNING"}, "Select spans (edges between anchors) first")
            return {"CANCELLED"}
        if self.reset:
            return self.execute(context)
        sl, bl = self._layers(bm)
        act = bm.select_history.active
        e = act if isinstance(act, bmesh.types.BMEdge) and act.select else sel[0]
        self.sag, self.bias = 1.0 + e[sl], e[bl]
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        sl, bl = self._layers(bm)
        n = 0
        for e in bm.edges:
            if e.select:
                e[sl] = 0.0 if self.reset else self.sag - 1.0
                e[bl] = 0.0 if self.reset else self.bias
                n += 1
        bmesh.update_edit_mesh(ob.data)
        request_rebuild(ob)
        self.report({"INFO"}, f"{n} span(s) {'reset' if self.reset else 'set'}")
        return {"FINISHED"}


class HC_OT_edit_points(bpy.types.Operator):
    bl_idname = "hanging_cables.edit_points"
    bl_label = "Edit Points"
    bl_description = "Select the guide and switch to Edit Mode to move, extrude or delete anchors"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        guide = guide_of(context.active_object)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        guide.hide_set(False)
        for o in context.selected_objects:
            o.select_set(False)
        guide.select_set(True)
        context.view_layer.objects.active = guide
        bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


class HC_OT_make_guide(bpy.types.Operator):
    bl_idname = "hanging_cables.make_guide"
    bl_label = "Cables Along Edges"
    bl_description = "Turn the active mesh into cable points: every vertex is an anchor, every edge a span"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH" and guide_of(ob) is None

    def execute(self, context):
        ob = context.active_object
        if ob.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        copy_settings(context.scene.hanging_cable_defaults, ob.hanging_cable)
        init_guide(ob)
        rebuild(ob)
        return {"FINISHED"}


class HC_OT_rebuild(bpy.types.Operator):
    bl_idname = "hanging_cables.rebuild"
    bl_label = "Rebuild"
    bl_description = "Rebuild the cables now"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        context.view_layer.update()
        info = rebuild(guide_of(context.active_object))
        if info is None:
            self.report({"WARNING"}, "Check the guide for zero scale and leave Edit Mode on the cable mesh")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Cables: {info['tris']:,} tris")
        return {"FINISHED"}


class HC_OT_bake(bpy.types.Operator):
    bl_idname = "hanging_cables.bake"
    bl_label = "Bake to Mesh"
    bl_description = "Copy the cables into a plain mesh object for export and hide the live version"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        guide = guide_of(context.active_object)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.update()
        if rebuild(guide) is None:
            self.report({"WARNING"}, "Could not rebuild these cables. Check the guide for zero scale")
            return {"CANCELLED"}
        out = guide.hanging_cable.output
        baked = bpy.data.objects.new(guide.name + "_Baked", out.data.copy())
        (guide.users_collection or [context.scene.collection])[0].objects.link(baked)
        baked.matrix_world = guide.matrix_world.copy()
        guide.hide_set(True)
        out.hide_set(True)
        out.hide_render = True
        for o in context.selected_objects:
            o.select_set(False)
        baked.select_set(True)
        context.view_layer.objects.active = baked
        return {"FINISHED"}


class HC_OT_save_defaults(bpy.types.Operator):
    bl_idname = "hanging_cables.save_defaults"
    bl_label = "Use as Default"
    bl_description = "New cables start with these settings"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        copy_settings(guide_of(context.active_object).hanging_cable, context.scene.hanging_cable_defaults)
        return {"FINISHED"}


# ======================================================================================
#  Panel
# ======================================================================================
class HC_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = DEFAULT_TAB
    bl_label = "Hanging Cables"

    def draw(self, context):
        layout = self.layout
        guide = guide_of(context.active_object)
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("hanging_cables.draw", text="Draw Cables", icon="GREASEPENCIL")
        if guide is not None:
            row.operator("hanging_cables.draw", text="", icon="ADD").extend = True
            cfg = guide.hanging_cable
        else:
            cfg = context.scene.hanging_cable_defaults
            layout.label(text="Settings for new cables:")

        row = layout.row(align=True)
        row.prop(cfg, "count")
        row.prop(cfg, "thickness")
        col = layout.column(align=True)
        col.prop(cfg, "sag", slider=True)
        col.prop(cfg, "bias", slider=True)
        row = layout.row(align=True)
        row.prop(cfg, "layout", expand=True)
        row = layout.row(align=True)
        row.prop(cfg, "spread")
        row.prop(cfg, "gather", slider=True)

        header, body = layout.panel("HC_PT_loops", default_closed=False)
        header.label(text="Loop Cuts")
        if body:
            col = body.column(align=True)
            col.prop(cfg, "sides")
            col.row(align=True).prop(cfg, "seg_mode", expand=True)
            if cfg.seg_mode == "FIXED":
                col.prop(cfg, "segments")
            else:
                col.prop(cfg, "density")
                row = col.row(align=True)
                row.prop(cfg, "seg_min")
                row.prop(cfg, "seg_max")
            col.prop(cfg, "focus", slider=True)

        header, body = layout.panel("HC_PT_vary", default_closed=True)
        header.label(text="Variation")
        if body:
            col = body.column(align=True)
            col.prop(cfg, "thickness_random", slider=True)
            col.prop(cfg, "sag_random", slider=True)
            col.prop(cfg, "swing")
            col.prop(cfg, "swing_random")
            col.prop(cfg, "seed")

        header, body = layout.panel("HC_PT_shape", default_closed=True)
        header.label(text="Shape & Anchors")
        if body:
            col = body.column(align=True)
            col.prop(cfg, "aspect")
            col.prop(cfg, "profile_rot")
            col.prop(cfg, "uv_tiling")
            col = body.column(align=True)
            col.prop(cfg, "exit_length")
            col.prop(cfg, "surface_offset")
            row = body.row(align=True)
            row.prop(cfg, "floor")
            sub = row.row(align=True)
            sub.active = cfg.floor
            sub.prop(cfg, "floor_z", text="")
            row = body.row(align=True)
            row.prop(cfg, "caps")
            row.prop(cfg, "smooth")

        if guide is None:
            ob = context.active_object
            if ob is not None and ob.type == "MESH":
                layout.operator("hanging_cables.make_guide", icon="MOD_WIREFRAME")
            return

        if context.mode == "EDIT_MESH" and context.edit_object is guide:
            header, body = layout.panel("HC_PT_spans", default_closed=False)
            header.label(text="Selected Spans")
            if body:
                row = body.row(align=True)
                row.operator("hanging_cables.span_set", text="Sag / Bias", icon="IPO_EASE_IN_OUT")
                row.operator("hanging_cables.span_set", text="Reset", icon="LOOP_BACK").reset = True
        row = layout.row(align=True)
        if not (context.mode == "EDIT_MESH" and context.edit_object is guide):
            row.operator("hanging_cables.edit_points", icon="EDITMODE_HLT")
        row.operator("hanging_cables.bake", icon="CHECKMARK")
        row = layout.row(align=True)
        row.operator("hanging_cables.rebuild", text="", icon="FILE_REFRESH")
        row.operator("hanging_cables.save_defaults", icon="PRESET")
        row.prop(cfg, "live", text="", icon="FILE_REFRESH" if cfg.live else "PAUSE")
        info = _last_info.get(guide.name)
        if info:
            layout.label(text=f"{info['tris']:,} tris · {info['spans']} spans · {info['tubes']} cables",
                         icon="MESH_DATA")


def _register_panel(tab):
    try:
        bpy.utils.unregister_class(HC_PT_main)
    except RuntimeError:
        pass
    HC_PT_main.bl_category = (tab or "").strip() or DEFAULT_TAB
    bpy.utils.register_class(HC_PT_main)


def _tab_changed(self, context):
    _register_panel(self.tab_name)


class HC_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    tab_name: StringProperty(name="Sidebar Tab", default=DEFAULT_TAB, update=_tab_changed,
                             description="Sidebar (N) tab the panel lives in. Type an existing tab to share it")

    def draw(self, context):
        self.layout.prop(self, "tab_name")


CLASSES = (HC_Preferences, HC_Settings, HC_OT_draw, HC_OT_span_set, HC_OT_edit_points, HC_OT_make_guide,
           HC_OT_rebuild, HC_OT_bake, HC_OT_save_defaults)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.hanging_cable = PointerProperty(type=HC_Settings)
    bpy.types.Scene.hanging_cable_defaults = PointerProperty(type=HC_Settings)
    addon = bpy.context.preferences.addons.get(__name__)
    _register_panel(addon.preferences.tab_name if addon and addon.preferences else DEFAULT_TAB)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _pending.clear()
    _last_transform.clear()
    _last_info.clear()
    _sessions.clear()
    try:
        bpy.utils.unregister_class(HC_PT_main)
    except RuntimeError:
        pass
    del bpy.types.Scene.hanging_cable_defaults
    del bpy.types.Object.hanging_cable
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
