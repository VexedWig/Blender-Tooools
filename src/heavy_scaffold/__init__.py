# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
bl_info = {
    "name": "Heavy Scaffold Generator",
    "author": "Aidan Hart",
    "version": (1, 2, 5),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Generators",
    "description": "Extrude a guide mesh and heavy steel scaffolding fills it live (Monolith, Flange, Lattice)",
    "category": "Add Mesh",
}

"""
Heavy Scaffold Generator
------------------------
Add Scaffold puts a wireframe guide box at the 3D cursor. Edit the guide like any mesh:
extrude faces up, sideways or out, move vertices. The scaffold rebuilds a moment after
you stop moving (Live Update), or with Rebuild.

How the guide is read (guide's local Z is up):
  * every span is split into sections; the count snaps to the nearest whole number of
    Section W / Section H, so sections pop in and out as you pull a span longer or shorter
  * Adaptive off (default): rows of beams stay level at fixed heights; under a sloped edge
    only the top (or bottom) chord follows the slope. Adaptive on: every row slants to share
    the slope evenly
  * section lines become members: vertical = columns, the rest = beams. Columns turn to
    follow the walls around them; beams stop flush on the column faces, cut to their angle
  * every bay is read from its four real corners, so braces land on the nodes of
    trapezoid and skewed cells too
  * vertical faces become braced walls; the brace per face is a face attribute you set
    from the panel in Edit Mode (Auto, Large X, Skinny X, Z, Z flipped, Open)
  * horizontal faces become floor frames (optionally braced with Skinny X)
  * loose edges become single members
  * columns standing on the lowest level get base plates; the lowest beams sit Plinth
    above them

Auto bracing: bottom row Large X, top row Skinny X, rows between Z (zig-zag).
Bolt levels: 1 base plates, 2 + connections, 3 + small parts. The Bolts toggle removes all.
Bake copies the current scaffold into a plain mesh object for export. Watertight Bake also
seats touching parts, unions their closed shells, and checks the result before export.
"""

import math
import time
from bisect import bisect_right

import bmesh
import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

ATTR = "hs_brace"
DEFAULT_TAB = "Generators"
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
PAINT, STEEL = 0, 1
AUTO, LARGE_X, SKINNY_X, Z_BRACE, Z_FLIP, OPEN = range(6)


# ======================================================================================
#  2D profiles
# ======================================================================================
def ccw(pts):
    pts = list(pts)
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if a > 0 else pts[::-1]


def rect(w, h, c=0.0, cx=0.0, cy=0.0):
    hw, hh = w / 2, h / 2
    c = min(c, hw * 0.95, hh * 0.95)
    if c <= 0:
        pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    else:
        pts = [(-hw + c, -hh), (hw - c, -hh), (hw, -hh + c), (hw, hh - c),
               (hw - c, hh), (-hw + c, hh), (-hw, hh - c), (-hw, -hh + c)]
    return [(x + cx, y + cy) for x, y in pts]


def _distinct_corners(points):
    """Collapse each zero-width chamfer to one vertex, including at the closing edge."""
    return [point for i, point in enumerate(points) if point != points[i - 1]]


def grooved_box(w, h, c, g, gd):
    """Box section with a recessed groove down the middle of each face."""
    hw, hh, q = w / 2, h / 2, g / 2
    points = [(-hw + c, -hh), (-q, -hh), (-q, -hh + gd), (q, -hh + gd), (q, -hh), (hw - c, -hh),
            (hw, -hh + c), (hw, -q), (hw - gd, -q), (hw - gd, q), (hw, q), (hw, hh - c),
            (hw - c, hh), (q, hh), (q, hh - gd), (-q, hh - gd), (-q, hh), (-hw + c, hh),
            (-hw, hh - c), (-hw, q), (-hw + gd, q), (-hw + gd, -q), (-hw, -q), (-hw, -hh + c)]
    return points if c else _distinct_corners(points)


def ngon(r, n, cx=0.0, cy=0.0, rot=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n))
            for i in range(n)]


def h_prof(bf, d, tf, tw, ch=0.0):
    hb, hd, ht = bf / 2, d / 2, tw / 2
    points = [(-hb + ch, -hd), (hb - ch, -hd), (hb, -hd + ch), (hb, -hd + tf), (ht, -hd + tf), (ht, hd - tf),
            (hb, hd - tf), (hb, hd - ch), (hb - ch, hd), (-hb + ch, hd), (-hb, hd - ch), (-hb, hd - tf),
            (-ht, hd - tf), (-ht, -hd + tf), (-hb, -hd + tf), (-hb, -hd + ch)]
    return points if ch else _distinct_corners(points)


def c_prof(bf, d, tf, tw, toes=1, cx=0.0):
    hb, hd = bf / 2, d / 2
    pts = [(-hb, -hd), (hb, -hd), (hb, -hd + tf), (-hb + tw, -hd + tf), (-hb + tw, hd - tf),
           (hb, hd - tf), (hb, hd), (-hb, hd)]
    return ccw([(x * toes + cx, y) for x, y in pts])


def offset_convex(poly, d):
    poly = ccw(poly)
    out, n = [], len(poly)
    for i in range(n):
        p0, p1, p2 = Vector(poly[i - 1]), Vector(poly[i]), Vector(poly[(i + 1) % n])
        e0, e1 = (p1 - p0).normalized(), (p2 - p1).normalized()
        n0, n1 = Vector((e0.y, -e0.x)), Vector((e1.y, -e1.x))
        m = (n0 + n1).normalized()
        out.append(tuple(p1 + m * (d / max(m.dot(n0), 0.2))))
    return out


def chamfer_poly(poly, c):
    poly = ccw(poly)
    out, n = [], len(poly)
    for i in range(n):
        p0, p1, p2 = Vector(poly[i - 1]), Vector(poly[i]), Vector(poly[(i + 1) % n])
        e0, e1 = p1 - p0, p2 - p1
        if e0.x * e1.y - e0.y * e1.x > 1e-9:
            k = min(c, e0.length * 0.45, e1.length * 0.45)
            out += [tuple(p1 - e0.normalized() * k), tuple(p1 + e1.normalized() * k)]
        else:
            out.append(tuple(p1))
    return out


def hull2d(points):
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def knuckle_poly(dirs, length, half_w):
    """Angular plate covering a bend: hull of strips running along each member direction."""
    pts = []
    for e in dirs:
        p = Vector((-e.y, e.x))
        for q in (e * length + p * half_w, e * length - p * half_w, p * half_w, -p * half_w):
            pts.append((q.x, q.y))
    return hull2d(pts)


def node_outline(hw, hc, hb, L, left, right, c):
    pts = [(-hw, -hc), (hw, -hc)]
    if right:
        pts += [(hw, -hb), (hw + L, -hb), (hw + L, hb), (hw, hb)]
    pts += [(hw, hc), (-hw, hc)]
    if left:
        pts += [(-hw, hb), (-hw - L, hb), (-hw - L, -hb), (-hw, -hb)]
    return chamfer_poly(pts, c)


def basis(n):
    n = Vector(n).normalized()
    u = (Z if abs(n.dot(Z)) < 0.9 else X).cross(n).normalized()
    return u, n.cross(u), n


# ======================================================================================
#  Mesh builder
# ======================================================================================
def _plane_hit(q, t, plane):
    """Slide q along t until it lies on plane = (point, normal). None leaves q alone."""
    if plane is None:
        return q
    P, N = plane
    dn = t.dot(N)
    if abs(dn) < 0.12:
        dn = 0.12 if dn >= 0 else -0.12
    return q + t * ((P - q).dot(N) / dn)


def _faces(fx):
    """Column faces in its own frame: +fx, -fx, +fy, -fy (face k sits on axis k // 2)."""
    fy = Z.cross(fx)
    return (fx, -fx, fy, -fy)


def _isect(p1, d1, p2, d2):
    den = d1.x * d2.y - d1.y * d2.x
    if abs(den) < 1e-9:
        return None
    w = p2 - p1
    return p1 + d1 * ((w.x * d2.y - w.y * d2.x) / den)


def _ang(a, b):
    return math.acos(max(-1.0, min(1.0, a.normalized().dot(b.normalized()))))


def _screw_bake_winding(mesh):
    """Orient solid templates outward without turning a nested cavity into a solid."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if any(not e.is_manifold or not e.is_contiguous for e in bm.edges) or any(not v.is_manifold for v in bm.verts):
            raise ValueError("Watertight Bake needs a closed custom screw with consistent face winding.")
        bm.normal_update()
        original = [(f, f.normal.copy()) for f in bm.faces if f.normal.length_squared > 0]
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.normal_update()
        directions = {1 if face.normal.dot(normal) > 0 else -1 for face, normal in original}
        if len(directions) != 1:
            raise ValueError("Custom screw shells must share one orientation. Watertight Bake supports solid parts without nested cavities.")
        return directions.pop()
    finally:
        bm.free()


def screw_template(obj, closed=False):
    """Read the evaluated mesh once. Local +Z points out of the mounting surface."""
    if obj is None or obj.type != "MESH" or guide_of(obj) is not None:
        return None
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if not mesh or not mesh.polygons:
            return None
        scale = evaluated.scale
        coords = [v.co * scale for v in mesh.vertices]
        xmin, xmax = min(v.x for v in coords), max(v.x for v in coords)
        ymin, ymax = min(v.y for v in coords), max(v.y for v in coords)
        radius = max(xmax - xmin, ymax - ymin) / 2
        if radius < 1e-9:
            return None
        center = Vector(((xmin + xmax) / 2, (ymin + ymax) / 2, 0))
        points = [(v - center) / radius for v in coords]
        winding = -1 if scale.x * scale.y * scale.z < 0 else 1
        if closed:
            winding *= _screw_bake_winding(mesh)
        faces = [(poly.vertices[:][::winding], poly.use_smooth) for poly in mesh.polygons]
        return points, faces
    finally:
        evaluated.to_mesh_clear()


class Mesh:
    bolt_level = 1          # 0 none, 1 base plates, 2 + connections, 3 + small parts
    head_style = "hex"      # what a plain "hex" head call becomes
    screw = None
    closed = False          # True: every part is a closed shell (watertight bake)

    def __init__(self):
        self.bm = bmesh.new()
        self.gussets = []
        self.pending_bolts = None

    def face(self, verts, mi):
        f = self.bm.faces.new(verts)
        f.material_index = mi
        return f

    def _rings(self, ring0, ring1, mi, cap0, cap1):
        r0 = [self.bm.verts.new(p) for p in ring0]
        r1 = [self.bm.verts.new(p) for p in ring1]
        n = len(r0)
        for i in range(n):
            j = (i + 1) % n
            self.face((r0[i], r0[j], r1[j], r1[i]), mi)
        if cap0 or self.closed:
            self.face(r0[::-1], mi)
        if cap1 or self.closed:
            self.face(r1, mi)

    def extrude(self, poly, origin, bx, ay, tvec, mi=PAINT, cap0=True, cap1=True, cut0=None, cut1=None):
        """Extrude poly along tvec. cut0 / cut1 = (point, normal) planes the end rings are slid
        onto, so a member can end flush against an angled face. An end that butts another part
        is left open (cap False) so no face is buried inside that part."""
        bx, ay, tvec, origin = Vector(bx), Vector(ay), Vector(tvec), Vector(origin)
        poly = ccw(poly)
        if bx.cross(ay).dot(tvec) < 0:
            poly = poly[::-1]
        t = tvec.normalized()
        ring0 = [_plane_hit(origin + bx * u + ay * v, t, cut0) for u, v in poly]
        ring1 = [_plane_hit(origin + tvec + bx * u + ay * v, t, cut1) for u, v in poly]
        self._rings(ring0, ring1, mi, cap0, cap1)

    def prism(self, poly, p0, p1, up, mi=PAINT, cap0=True, cap1=True, cut0=None, cut1=None):
        p0, p1, up = Vector(p0), Vector(p1), Vector(up).normalized()
        t = p1 - p0
        if t.length < 1e-6:
            return
        t.normalize()
        if abs(t.dot(up)) > 0.98:
            up = X if abs(t.dot(X)) < 0.9 else Y
        a = (up - t * up.dot(t)).normalized()
        b = a.cross(t)
        self.extrude(poly, p0, b, a, p1 - p0, mi, cap0, cap1, cut0, cut1)

    def plate(self, poly, origin, u, v, thick, mi=PAINT):
        u, v = Vector(u), Vector(v)
        n = u.cross(v).normalized()
        self.extrude(poly, Vector(origin) - n * (thick / 2), u, v, n * thick, mi)

    def stack(self, layers, origin, n, mi=STEEL, rot_u=None):
        """Rings stacked along n, each joined to the next by quads, top capped, no bottom:
        a smaller ring at the same height is an inset, a taller one an extrusion."""
        u, v, n = basis(n)
        if rot_u is not None:
            u = (Vector(rot_u) - n * Vector(rot_u).dot(n)).normalized()
            v = n.cross(u)
        rings = []
        for poly, h in layers:
            o = Vector(origin) + n * h
            rings.append([self.bm.verts.new(o + u * x + v * y) for x, y in ccw(poly)])
        for ra, rb in zip(rings, rings[1:]):
            m = len(ra)
            for i in range(m):
                j = (i + 1) % m
                self.face((ra[i], ra[j], rb[j], rb[i]), mi)
        self.face(rings[-1], mi)
        if self.closed:
            self.face(rings[0][::-1], mi)

    def bolt(self, p, n, r=0.04, style="hex", rot_u=None, lvl=2):
        """Bolt head or nut on a surface at p, pointing along n. Flat insets and extrusions,
        no bevels, no hidden faces."""
        if lvl > self.bolt_level:
            return
        if self.pending_bolts is not None:
            self.pending_bolts.append((p, n, r, style, rot_u, lvl))
            return
        if self.screw:
            u, v, n = basis(n)
            if rot_u is not None:
                u = (rot_u - n * rot_u.dot(n)).normalized()
                v = n.cross(u)
            points, faces = self.screw
            verts = [self.bm.verts.new(p + r * (u * x + v * y + n * z)) for x, y, z in points]
            for indices, smooth in faces:
                self.face([verts[i] for i in indices], STEEL).smooth = smooth
            return
        if style == "hex":
            style = self.head_style
        h = r * 0.95
        h6 = lambda rr: ngon(rr, 6)
        if style == "hex":
            hw = h * 0.22
            layers = [(h6(r * 1.4), 0), (h6(r * 1.4), hw), (h6(r), hw), (h6(r), hw + h * 0.62)]
        elif style == "hex_plain":
            layers = [(h6(r), 0), (h6(r), h * 0.62)]
        elif style == "square":
            sq = ngon(r * 1.1, 4, rot=math.pi / 4)
            layers = [(sq, 0), (sq, h * 0.55)]
        elif style == "rivet":
            o = ngon(r * 1.1, 8)
            layers = [(o, 0), (o, h * 0.38)]
        elif style == "stud":
            layers = [(h6(r * 1.45), 0), (h6(r * 1.45), h * 0.2), (h6(r), h * 0.2), (h6(r), h * 0.82),
                      (h6(r * 0.5), h * 0.82), (h6(r * 0.5), h * 1.5)]
        else:
            raise ValueError(style)
        self.stack(layers, p, n, rot_u=rot_u)

    def finish_bolts(self):
        """Omit fasteners covered by a brace cheek, including custom screw heads."""
        bolts, self.pending_bolts = self.pending_bolts, None
        if not bolts:
            return
        if not self.gussets:
            for bolt in bolts:
                self.bolt(*bolt)
            return
        tree = KDTree(len(self.gussets))
        reach = 0.0
        for i, (center, radius, _) in enumerate(self.gussets):
            tree.insert(center, i)
            reach = max(radius, reach)
        tree.balance()
        custom = None
        if self.screw:
            xyz = self.screw[0]
            custom = (max(math.hypot(p.x, p.y) for p in xyz), min(p.z for p in xyz), max(p.z for p in xyz))
        bounds = {"hex": (1.4, 0, .80), "hex_plain": (1, 0, .60), "square": (1.1, 0, .53),
                  "rivet": (1.1, 0, .37), "stud": (1.45, 0, 1.425)}
        for p, n, r, style, rot, level in bolts:
            selected = self.head_style if style == "hex" else style
            width, lo, hi = custom or bounds[selected]
            a, b = p + n * (lo * r), p + n * (hi * r)
            middle = (a + b) / 2
            extent = math.hypot(width * r, (hi - lo) * r / 2)
            for _, index, distance in tree.find_range(middle, reach + extent):
                _, radius, planes = self.gussets[index]
                if distance <= radius + extent and all(
                        min((a-q).dot(v), (b-q).dot(v)) <
                        width*r*math.sqrt(max(0.0, 1.0 - v.dot(n)**2)) - 1e-6 for q, v in planes):
                    break
            else:
                self.bolt(p, n, r, style, rot, level)

    def bolt_grid(self, center, n, u, v, cols, rows, pitch_u, pitch_v, r, style="hex", lvl=2, keep=None):
        if lvl > self.bolt_level:
            return
        n, u, v = Vector(n).normalized(), Vector(u).normalized(), Vector(v).normalized()
        for i in range(cols):
            for j in range(rows):
                p = Vector(center) + u * (i - (cols - 1) / 2) * pitch_u + v * (j - (rows - 1) / 2) * pitch_v
                if keep is None or keep(p):
                    self.bolt(p, n, r, style, rot_u=u, lvl=lvl)

    def ring_plate(self, c, n, r_out, r_in, thick, segs=12, mi=PAINT, rot=0.0, u=None):
        bu, bv, n = basis(n)
        if u is not None:
            bu = (Vector(u) - n * Vector(u).dot(n)).normalized()
            bv = n.cross(bu)
        c = Vector(c)
        lay = []
        for z in (-thick / 2, thick / 2):
            for r in (r_out, r_in):
                lay.append([self.bm.verts.new(c + n * z + bu * x + bv * y) for x, y in ngon(r, segs, rot=rot)])
        ob, ib, ot, it = lay
        for i in range(segs):
            j = (i + 1) % segs
            self.face((ob[i], ob[j], ot[j], ot[i]), mi)
            self.face((ib[j], ib[i], it[i], it[j]), mi)
            self.face((ot[i], ot[j], it[j], it[i]), mi)
            self.face((ob[j], ob[i], ib[i], ib[j]), mi)

    def tris(self):
        return sum(len(f.verts) - 2 for f in self.bm.faces)

    def write(self, me):
        self.bm.to_mesh(me)
        self.bm.free()
        me.update()


# ======================================================================================
#  Bays and shared parts
# ======================================================================================
class Bay:
    """One cell of a wall or floor, built from its four real corners C0 BL, C1 BR, C2 TR, C3 TL
    (any quad: rectangle, trapezoid under a slope, parallelogram on a skew).
    Frame: u along the bottom edge, v across the cell in its plane, n out of the plane.
    ins = clearance from each edge (bottom, right, top, left) that braces stay inside."""

    def __init__(self, C, n, ins, mounts=None):
        C = [Vector(c) for c in C]
        mounts = ins if mounts is None else mounts
        nq = newell(C)
        if nq.dot(Vector(n)) < 0:                     # mirror so the corners run ccw around n
            C = [C[1], C[0], C[3], C[2]]
            ins = (ins[0], ins[3], ins[2], ins[1])
            mounts = (mounts[0], mounts[3], mounts[2], mounts[1])
            nq = newell(C)
        self.n = nq
        u = C[1] - C[0]
        u = u - nq * u.dot(nq)
        self.u = u.normalized() if u.length > 1e-9 else basis(nq)[0]
        self.v = nq.cross(self.u)
        self.O = C[0]
        self.Q = [Vector(((c - self.O).dot(self.u), (c - self.O).dot(self.v))) for c in C]
        self.ins = tuple(ins)
        self.mounts = tuple(mounts)
        self.folds = (0.0,) * 4
        self.post_planes = (None,) * 4
        self.chord_planes = [[] for _ in range(4)]
        self._inner = self.inset()

    def p(self, a, b, d=0.0):
        return self.O + self.u * a + self.v * b + self.n * d

    def P(self, ab, d=0.0):
        return self.p(ab[0], ab[1], d)

    def w(self, d2):
        return (self.u * d2[0] + self.v * d2[1]).normalized()

    def _line(self, k, extra):
        a, b = self.Q[k], self.Q[(k + 1) % 4]
        e = b - a
        e = e.normalized() if e.length > 1e-9 else Vector((1.0, 0.0))
        nin = Vector((-e.y, e.x))
        return a + nin * (self.ins[k] + extra), e, nin

    def inset(self, es=0.0, ec=0.0):
        """Inner quad: every edge moved in by its clearance (+ es on the sides, + ec top/bottom)."""
        lines = [self._line(k, ec if k in (0, 2) else es)[:2] for k in range(4)]
        out = []
        for k in range(4):
            (p1, e1), (p2, e2) = lines[k - 1], lines[k]
            q = _isect(p1, e1, p2, e2)
            out.append(q if q is not None else p2.copy())
        return out

    def size(self, q=None):
        """Smallest width and height of the inner quad, (0, 0) if it folded over."""
        q = q or self._inner
        for k in range(4):
            a, b, c = q[k - 1], q[k], q[(k + 1) % 4]
            if (b - a).x * (c - b).y - (b - a).y * (c - b).x <= 1e-6:
                return 0.0, 0.0
        w = min((q[1] - q[0]).length, (q[2] - q[3]).length)
        h = min((q[3] - q[0]).length, (q[2] - q[1]).length)
        return w, h

    def corners(self):
        """Inner corners with the unit directions into the cell along the top/bottom edge (ea)
        and along the side edge (eb)."""
        q, Q = self._inner, self.Q
        out = []
        for k in range(4):
            nxt, prv = Q[(k + 1) % 4] - Q[k], Q[k - 1] - Q[k]
            ea, eb = (nxt, prv) if k in (0, 2) else (prv, nxt)
            out.append((q[k], ea.normalized(), eb.normalized()))
        return out

    def diag(self, i, j, inset, hw=0.0):
        """Diagonal between inner corners i and j, pulled in so a member of half width hw
        clears both edges at each end."""
        c = self.corners()
        s, e = c[i][0], c[j][0]
        d = (e - s).normalized()

        def need(k, dd):
            ang = min(_ang(dd, c[k][1]), _ang(dd, c[k][2]))
            # A shallow brace may overlap the frame, but its end must stay on the gusset.
            return min(1.5 * inset, hw / math.tan(ang)) if hw > 0 and ang > 0.05 else 0.0

        return s + d * max(inset, need(i, d)), e - d * max(inset, need(j, -d)), d

    @property
    def center(self):
        q = self._inner
        c = _isect(q[0], q[2] - q[0], q[1], q[3] - q[1])
        return c if c is not None else (q[0] + q[1] + q[2] + q[3]) / 4

    def edge_at(self, k, extra, a):
        """Height (b) of edge k's clearance line (+ extra) at a."""
        p, e, _ = self._line(k, extra)
        if abs(e.x) < 1e-6:
            return p.y
        return p.y + e.y * (a - p.x) / e.x

    def edge_plane(self, k, extra):
        p, e, nin = self._line(k, extra)
        return self.P(p), self.w(nin)


def _clip_outline(poly, distance):
    """Keep the part of a convex plate on the positive side of a plane."""
    clipped = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        da, db = distance(a), distance(b)
        if da >= 0:
            clipped.append(a)
        if da * db < 0:
            clipped.append(a.lerp(b, da / (da - db)))
    return clipped


def corner_gusset(m, bay, k, g, t, depth=0.0, cut=0.24):
    """Seat a brace cheek on its chord while keeping its footprint outside the post."""
    pt, ea, eb = bay.corners()[k]
    side = 3 if k in (0, 3) else 1
    chord = 0 if k < 2 else 2
    sine = max(abs(ea.cross(eb)), 0.1)
    reach = abs(depth) + t / 2
    x = (max(bay.mounts[side], reach * bay.folds[side]) - bay.ins[side]) / sine
    y = (max(bay.mounts[chord], reach * bay.folds[chord]) - bay.ins[chord]) / sine
    poly = [Vector((x, y)), Vector((g, y)), Vector((g, g)), Vector((x, g))]
    poly = _clip_outline(poly, lambda q: g * (1 + cut) - q.x - q.y)
    poly = [pt + ea * q.x + eb * q.y for q in poly]
    for plane in (bay.post_planes[k], *bay.chord_planes[chord]):
        if plane is not None:
            origin, normal = plane
            margin = abs(bay.n.dot(normal)) * t / 2
            poly = _clip_outline(poly, lambda q: (bay.P(q, depth) - origin).dot(normal) - margin)
    if len(poly) < 3:
        return
    poly = ccw(poly)
    m.plate(poly, bay.P((0, 0), depth), bay.u, bay.v, t)
    if not m.bolt_level:
        return
    corners = [bay.P(q, depth) for q in poly]
    center = sum(corners, Vector()) / len(corners)
    planes = [(center + bay.n*t/2, bay.n), (center - bay.n*t/2, -bay.n)]
    planes.extend((a, (b-a).cross(bay.n).normalized()) for a,b in zip(corners,corners[1:] + corners[:1]))
    m.gussets.append((center, math.hypot(max((q-center).length for q in corners), t/2), planes))


def gusset_pair(m, bay, k, size, thickness, lo, hi):
    """Put the inner face of each cheek on the actual near or far face of the brace."""
    for depth in (lo - thickness/2, hi + thickness/2):
        corner_gusset(m, bay, k, size, thickness, depth)


def through_bolt(m, bay, q, d_head, d_nut, r, style="hex", lvl=2):
    sgn = 1 if d_head > d_nut else -1
    m.bolt(bay.P(q, d_head), bay.n * sgn, r, style, lvl=lvl)
    m.bolt(bay.P(q, d_nut), -bay.n * sgn, r * 0.92, "hex_plain", lvl=lvl)


def x_pair(bay, inset, hw, hw_split):
    """Large X: diagonal A (0-2) runs whole, diagonal B (1-3) is split where it meets A's sides,
    so the two never pass through each other. Returns A, B, crossing, and A's two side planes
    (the one facing B's start first)."""
    A = bay.diag(0, 2, inset, hw)
    B = bay.diag(1, 3, inset, hw)
    c = _isect(A[0], A[2], B[0], B[2])
    if c is None:
        c = bay.center
    nA = Vector((-A[2].y, A[2].x))
    sg = 1 if (B[0] - c).dot(nA) > 0 else -1
    pl1 = (bay.P(c + nA * sg * hw_split), bay.w(nA * sg))
    pl2 = (bay.P(c - nA * sg * hw_split), bay.w(nA * sg))
    return A, B, c, pl1, pl2


def miters(bay, pts):
    """Cut planes at the bends of a polyline in the bay plane, so the pieces meet flush."""
    cuts = [None] * len(pts)
    for k in range(1, len(pts) - 1):
        a, b = bay.w(pts[k] - pts[k - 1]), bay.w(pts[k + 1] - pts[k])
        nrm = a + b
        cuts[k] = (bay.P(pts[k]), (nrm if nrm.length > 1e-6 else a).normalized())
    return cuts


def z_layout(bay, zh, flip, s, gap=0.17, leg=0.85):
    q = bay.inset(0.05 * s, gap * s + zh / 2)
    w, h = bay.size(q)
    leg = min(leg * s, w * 0.32)
    if h < zh * 2 or w < leg * 2.5:
        return None
    start, toward, end, back = (1, 0, 3, 2) if flip else (0, 1, 2, 3)
    bottom = (q[toward] - q[start]).normalized()
    top = (q[back] - q[end]).normalized()

    def mounted(point, direction, edge):
        anchor, line, _ = bay._line(edge, bay.mounts[edge] - bay.ins[edge])
        hit = _isect(point, direction, anchor, line)
        return point if hit is None else hit

    return [mounted(q[start], bottom, 1 if flip else 3), q[start] + bottom * leg,
            q[end] + top * leg, mounted(q[end], top, 3 if flip else 1)]


def z_standoffs(m, bay, pts, zh, block, face, s, up_axis="n", gap=0.17):
    """Spacer blocks between the Z legs and the beams, cut flush to both faces."""
    up = bay.n if up_axis == "n" else bay.u
    for k0, k1, edge in ((0, 1, 0), (3, 2, 2)):
        for f in (0.25, 0.72):
            q = pts[k0].lerp(pts[k1], f)
            seat = bay.mounts[edge] - bay.ins[edge]
            b0, b1 = bay.edge_at(edge, seat, q.x), bay.edge_at(edge, gap * s, q.x)
            if abs(b1 - b0) < 0.02 * s:
                continue
            m.prism(block, bay.p(q.x, b0), bay.p(q.x, b1), up, cap0=False, cap1=False,
                    cut0=bay.edge_plane(edge, seat), cut1=bay.edge_plane(edge, gap * s))
            mid = (b0 + b1) / 2
            for sgn in (1, -1):
                m.bolt(bay.p(q.x, mid, sgn * face), bay.n * sgn, 0.036 * s, lvl=3)


def spaced(points, gap):
    """Drop points closer than gap to one already kept (bolts on tight bends)."""
    out = []
    for q in points:
        if all((q - o).length >= gap for o in out):
            out.append(q)
    return out


def z_knuckles(pts):
    """Bend points with the unit directions toward the leg and the diagonal."""
    out = []
    for k in (1, 2):
        q = pts[k]
        leg = pts[0] if k == 1 else pts[3]
        dia = pts[2] if k == 1 else pts[1]
        out.append((q, (leg - q).normalized(), (dia - q).normalized()))
    return out


def base_plate(m, p, s, fx, starts, max_h=None):
    """Base plate turned with the column; fins start on the column's real faces and stop
    under anything mounted low on the column (max_h)."""
    fy = Z.cross(fx)
    size, t, fin_h = 1.0 * s, 0.09 * s, 0.62 * s
    if max_h is not None:
        fin_h = max(t + 0.04 * s, min(fin_h, max_h))
    m.plate(rect(size, size, c=size * 0.16), p + Z * (t / 2), fx, fy, t)
    reach = size / 2 - 0.05 * s
    for d, c0 in ((fx, starts[0]), (-fx, starts[0]), (fy, starts[1]), (-fy, starts[1])):
        # Bury the inside edge slightly so fins stay seated on a leaning column.
        inside = c0 - 0.025 * s
        poly = [(inside, t), (reach, t), (reach, t + 0.08 * s), (inside, fin_h)]
        m.plate(poly, p, d, Z, 0.045 * s)
    k = size / 2 - 0.14 * s
    for su in (-1, 1):
        for sn in (-1, 1):
            m.bolt(p + fx * su * k + fy * sn * k + Z * t, Z, 0.046 * s, "stud", rot_u=fx, lvl=1)
    return t


def cap_plate(m, p, w, t, fx, down=False, corner=None):
    m.plate(rect(w, w, c=w * 0.22 if corner is None else corner),
            p + Z * (-t / 2 if down else t / 2), fx, Z.cross(fx), t)


def laced(m, p0, p1, up, W, D, chw, tf, tw, pitch, lw, lt, batten, rivet,
          cut0=None, cut1=None, cap0=True, cap1=True, rivets=(True, True)):
    """Two channels (toes in) joined by zig-zag lacing on both faces, battens at the ends.
    cut0 / cut1 end the member flush on a plane; lacing strips are mitred where they meet."""
    p0, p1, up = Vector(p0), Vector(p1), Vector(up).normalized()
    t = p1 - p0
    L = t.length
    if L < 1e-4:
        return
    t.normalize()
    if abs(t.dot(up)) > 0.98:
        up = X if abs(t.dot(X)) < 0.9 else Y
    a = (up - t * up.dot(t)).normalized()
    b = a.cross(t)

    def P(s, x, y):
        return p0 + t * s + b * x + a * y

    for toes, cx in ((1, -W / 2 + chw / 2), (-1, W / 2 - chw / 2)):
        m.prism(c_prof(chw, D, tf, tw, toes, cx), p0, p1, a, cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1)
    bat = min(batten, L * 0.3)
    xi = W / 2 - chw * 0.4

    def reach(plane, s_ref):
        """How far along t an angled end cut runs past s_ref (worst corner of the section)."""
        if plane is None:
            return 0.0
        Pp, N = plane
        dn = t.dot(N)
        if abs(dn) < 0.12:
            dn = 0.12 if dn >= 0 else -0.12
        out = []
        for x in (-W / 2, W / 2):
            for y in (-D / 2 - lt, D / 2 + lt):
                q = P(s_ref, x, y)
                out.append((Pp - q).dot(N) / dn)
        return out

    r0, r1 = reach(cut0, 0.0), reach(cut1, L)
    s_a = (max(r0) if r0 else 0.0) + bat              # lacing starts here...
    s_b = L + (min(r1) if r1 else 0.0) - bat          # ...and ends here
    for sgn in (1, -1):
        y = sgn * (D / 2 + lt / 2)
        for s0, s1, first in ((s_a - bat, s_a, True), (s_b, s_b + bat, False)):
            m.prism(rect(W - 0.01 * W, lt, cy=y), p0 + t * (0.0 if first else s0), p0 + t * (s1 if first else L), a,
                    cap0=cap0 if first else True, cap1=True if first else cap1,
                    cut0=cut0 if first else None, cut1=None if first else cut1)
            if not rivets[0 if first else 1]:
                continue                     # this end sits under a plate or ring
            for x in (-xi * 0.8, xi * 0.8):
                for f in (0.3, 0.7):
                    m.bolt(P(s0 + bat * f, x, sgn * (D / 2 + lt)), a * sgn, rivet, "rivet", lvl=3)
        span = s_b - s_a
        if span <= 0:
            continue
        k = max(1, round(span / (pitch / 2)))
        step = span / k
        nodes = [P(s_a + step * i, (-xi if i % 2 == 0 else xi), y) for i in range(k + 1)]
        for i in range(k):
            q0, q1 = nodes[i], nodes[i + 1]
            d0 = (q1 - q0).normalized()
            if i == 0:
                c0 = (P(s_a, 0, y), t)
            else:
                nb = d0 + (q0 - nodes[i - 1]).normalized()
                c0 = (q0, nb.normalized())
            if i == k - 1:
                c1 = (P(s_b, 0, y), t)
            else:
                nb = d0 + (nodes[i + 2] - q1).normalized()
                c1 = (q1, nb.normalized())
            m.prism(rect(lw, lt), q0, q1, a, cap0=False, cap1=False, cut0=c0, cut1=c1)


# ======================================================================================
#  Styles
# ======================================================================================
class Style:
    key = name = ""
    has_frame_bevels = True
    chord_skin = 0.0         # wall-facing flange or channel web thickness on open sections
    plate_out = 0.0          # column face -> where beams stop (node plate thickness etc.)
    side_plate = 0.0         # plates riding on the beams' sides at a node (Lattice cheeks)
    face_clear = 0.0         # extra room braces leave past that (bolt heads on the node plates)

    def setup(self, s1, s2, s3, bevel_horizontal=True, bevel_vertical=True):
        self.s1, self.s2, self.s3 = s1, s2, s3
        self.bevel_horizontal = bevel_horizontal
        self.bevel_vertical = bevel_vertical
        self.dims()

    def dims(self):
        pass

    def half(self, axis):
        """Column half size along its own frame axis (0 = fx, 1 = fy)."""
        return self.M / 2

    def end_out(self, k, used):
        return self.plate_out

    def fins(self):
        return self.M / 2, self.M / 2

    def beam_side(self, t):
        """Unit normal of a beam's side face for direction t, and its half width."""
        a = Z - t * Z.dot(t)
        a = a.normalized() if a.length > 1e-6 else X
        return a.cross(t).normalized(), self.MBD / 2

    def beam_extent(self, t, w):
        """How far a beam along t reaches in direction w (side half width and depth together)."""
        sn, hw = self.beam_side(t)
        up = sn.cross(t)
        return abs(w.dot(sn)) * hw + abs(w.dot(up)) * self.MB / 2

    def base(self, m, p, fx, max_h=None):
        return base_plate(m, p, self.s1, fx, self.fins(), max_h)

    def cap(self, m, p, fx, down=False):
        # Leave a rim over the corners of a square post's open end.
        corner = 0.04 * self.s1 if self.has_frame_bevels and not self.bevel_vertical else None
        cap_plate(m, p, self.M + 0.08 * self.s1, 0.05 * self.s1, fx, down, corner)

    def _face_plates(self, m, p, ends, fx, off, t, w0, h0, grid, chamfer):
        """One plate flush on every column face that takes beams, sized to cover every beam
        end landing there (angled and sloped ends included). Returns used faces and the
        plates' vertical extent relative to p."""
        F = _faces(fx)
        used = {}
        for d, k, s_end, _ in ends:
            used.setdefault(k, []).append((d, s_end))
        zlo, zhi = 0.0, 0.0
        for k, lst in used.items():
            fn = F[k]
            side = Z.cross(fn).normalized()
            xs, zs = [], []
            for d, s_end in lst:
                e = d * s_end
                dh = Vector((d.x, d.y, 0.0))
                ch = max(abs(dh.normalized().dot(fn)), 0.35) if dh.length > 1e-6 else 1.0
                cv = max(dh.length, 0.35)
                hw, hh = w0 / ch / 2, h0 / cv / 2
                xs += [e.dot(side) - hw, e.dot(side) + hw]
                zs += [e.z - hh, e.z + hh]
            x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
            lim = self.half(1 - k // 2)                     # never past the column's edge
            x0, x1 = max(x0, -lim), min(x1, lim)
            zlo, zhi = min(zlo, z0), max(zhi, z1)
            c = p + fn * (self.half(k // 2) + off + t / 2) + side * ((x0 + x1) / 2) + Z * ((z0 + z1) / 2)
            m.plate(rect(x1 - x0, z1 - z0, c=chamfer), c, side, Z, t)
            mg, placed = 0.06 * self.s1, []

            def keep(q, side=side, x0=x0, x1=x1, placed=placed):
                ok = x0 + mg <= (q - p).dot(side) <= x1 - mg and all((q - o).length > 0.13 * self.s1 for o in placed)
                if ok:
                    placed.append(q)
                return ok

            for d, s_end in lst:
                grid(p + d * s_end, fn, side, d, keep)
        return used, zlo, zhi


class Monolith(Style):
    """A: chamfered box sections, node blocks, angular plates only."""
    key, name = "A", "Monolith"
    # Distance from the bay plane to the outer skin of each cheek at unit size.
    large_jaw, skinny_jaw = .18, .1125

    def dims(self):
        s = self.s1
        self.M, self.MB, self.MBD = 0.46 * s, 0.42 * s, 0.36 * s
        self.post_chamfer = self.M * 0.21 if self.bevel_vertical else 0.0
        self.beam_chamfer = 0.075 * s if self.bevel_horizontal else 0.0
        self.plate_out = 0.1 * s           # node block 0.05 + end plate 0.05
        self.face_clear = 0.05 * s

    def fins(self):
        g = self.M / 2 - 0.035 * self.s1   # into the groove
        return g, g

    def column(self, m, p0, p1, fx, cap0=True, cap1=True, foot=False, cut0=None, cut1=None):
        s = self.s1
        m.prism(grooved_box(self.M, self.M, self.post_chamfer, 0.13 * s, 0.035 * s), p0, p1, fx,
                cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1)

    def beam(self, m, p0, p1, cut0=None, cut1=None, cap0=True, cap1=True):
        s = self.s1
        m.prism(grooved_box(self.MBD, self.MB, self.beam_chamfer, 0.12 * s, 0.03 * s), p0, p1, Z,
                cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1)

    def node(self, m, p, ends, fx):
        s = self.s1
        off = 0.05 * s

        def grid(ep, fn, side, d, keep):
            cv = max(Vector((d.x, d.y, 0.0)).length, 0.35)
            m.bolt_grid(ep, fn, side, Z, 2, 2, 0.26 * s, (self.MB + 0.15 * s) / cv, 0.04 * s, keep=keep)

        used, zl, zh = self._face_plates(m, p, ends, fx, off, 0.05 * s, self.MBD + 0.1 * s, self.MB + 0.28 * s,
                                         grid, 0.05 * s)
        h0 = (self.MB + 0.34 * s) / 2
        lo, hi = min(-h0, zl - 0.03 * s), max(h0, zh + 0.03 * s)
        m.prism(offset_convex(rect(self.M, self.M, c=self.post_chamfer), off), p + Z * lo, p + Z * hi, fx)
        f = self.M / 2 + off
        for k, vec in enumerate(_faces(fx)):
            if k not in used:
                m.bolt_grid(p + vec * f + Z * ((lo + hi) / 2), vec, Z.cross(vec), Z, 2, 2, 0.2 * s,
                            self.MB + 0.12 * s, 0.044 * s)
        return lo, hi, True               # the column is cut around this block

    def large_x(self, m, bay):
        s = self.s2
        prof, hd, hw, sl = rect(0.24 * s, 0.2 * s, c=0.045 * s), 0.1 * s, 0.12 * s, 0.03 * s
        sleeve = offset_convex(prof, sl)
        gt = 0.05 * s
        A, B, c, pl1, pl2 = x_pair(bay, 0.17 * s, hw + sl, hw)
        dirs = []
        sin = max(abs(A[2].x * B[2].y - A[2].y * B[2].x), 0.2)
        clear = (hw + sl + hw) / sin + 0.03 * s          # sleeve must end before the other diagonal
        for (st, en, d), split in ((A, False), (B, True)):
            dirs.append(d)
            ln = (en - st).length
            L0 = max(0.0, min(0.42 * s, ln * 0.3, (c - st).dot(d) - clear))
            L1 = max(0.0, min(0.42 * s, ln * 0.3, (en - c).dot(d) - clear))
            a0, a1 = st + d * L0, en - d * L1
            if L0 > 0.05 * s:
                m.prism(sleeve, bay.P(st), bay.P(a0), bay.n)
            else:
                a0 = st
            if L1 > 0.05 * s:
                m.prism(sleeve, bay.P(a1), bay.P(en), bay.n)
            else:
                a1 = en
            for k, length in zip((1,3) if split else (0,2), (L0,L1)):
                face = hd + (sl if length > 0.05 * s else 0.0)
                gusset_pair(m, bay, k, 0.62 * s, gt, -face, face)
            c0, c1 = a0 == st, a1 == en               # no sleeve there: close the end
            if not split:
                m.prism(prof, bay.P(a0), bay.P(a1), bay.n, cap0=c0, cap1=c1)
            else:
                m.prism(prof, bay.P(a0), bay.P(c), bay.n, cap0=c0, cap1=False, cut1=pl1)
                m.prism(prof, bay.P(c), bay.P(a1), bay.n, cap0=False, cap1=c1, cut0=pl2)
        e1 = (dirs[0] + dirs[1]).normalized()        # square diamond, corners on the bisectors
        e2 = Vector((-e1.y, e1.x))
        d1, d2 = bay.w((e1 + e2) / 2), bay.w((e1 - e2) / 2)
        for sgn in (1, -1):
            m.plate(rect(0.5 * s, 0.5 * s), bay.P(c, sgn * (hd + 0.025 * s)), d1, d2, 0.05 * s)
        half = math.asin(min(1.0, sin)) / 2                 # bolts on the two diagonals stay apart
        o = max(0.17 * s, 0.075 * s / max(math.sin(half), 0.1), 0.075 * s / max(math.cos(half), 0.1))
        if o < 0.25 * s:
            for d in dirs:
                for sg in (-1, 1):
                    through_bolt(m, bay, c + d * o * sg, hd + 0.05 * s, -hd - 0.05 * s, 0.042 * s)
        through_bolt(m, bay, c, hd + 0.05 * s, -hd - 0.05 * s, 0.065 * s)

    def skinny_x(self, m, bay):
        s = self.s3
        tg, ts = 0.045 * s, 0.045 * s
        strap = rect(0.2 * s, ts, c=0.012 * s)
        c = bay.center
        g1, g2 = bay.diag(0, 2, 0)[2], bay.diag(1, 3, 0)[2]
        xsin = max(abs(g1.x * g2.y - g1.y * g2.x), 0.2)
        m.ring_plate(bay.P(c), bay.n, 0.3 * s * math.sqrt(2), 0.15 * s * math.sqrt(2), tg, segs=4,
                     rot=math.pi / 4, u=bay.w(g1))      # square ring, edges along the diagonal
        for (i, j), sgn in (((0, 2), 1), ((1, 3), -1)):
            st, en, d = bay.diag(i, j, 0.12 * s, 0.1 * s)
            dep = sgn * (tg / 2 + ts / 2)
            m.prism(strap, bay.P(st, dep), bay.P(en, dep), bay.n)
            face, back = sgn * (tg / 2 + ts), -sgn * tg / 2
            for k in (i,j):
                gusset_pair(m, bay, k, 0.48 * s, tg, dep - ts/2, dep + ts/2)
            o = max(0.22 * s, (0.1 * s + 0.05 * s) / xsin)
            if o < 0.3 * s * math.sqrt(2):
                for sg in (-1, 1):
                    through_bolt(m, bay, c + d * o * sg, face, back, 0.04 * s, lvl=3)

    def z_brace(self, m, bay, flip=False):
        s = self.s2
        prof, zh, hd = rect(0.26 * s, 0.22 * s, c=0.05 * s), 0.26 * s, 0.11 * s
        pts = z_layout(bay, zh, flip, s)
        if pts is None:
            return
        cuts = miters(bay, pts)
        for k in range(3):
            m.prism(prof, bay.P(pts[k]), bay.P(pts[k + 1]), bay.n, cap0=cuts[k] is None,
                    cap1=cuts[k + 1] is None, cut0=cuts[k], cut1=cuts[k + 1])
        for q, dl, dd in z_knuckles(pts):
            poly = knuckle_poly((dl, dd), 0.34 * s, 0.2 * s)
            for sgn in (1, -1):
                m.plate(poly, bay.P(q, sgn * (hd + 0.025 * s)), bay.u, bay.v, 0.05 * s)
            for off in spaced((q, q + dl * 0.16 * s, q + dd * 0.18 * s), 0.14 * s):
                through_bolt(m, bay, off, hd + 0.05 * s, -hd - 0.05 * s, 0.042 * s)
        z_standoffs(m, bay, pts, zh, rect(0.24 * s, 0.2 * s, c=0.03 * s), 0.1 * s, s)


class Flange(Style):
    """B: wide-flange H sections, stiffeners, bolted end plates, lapped gussets, angle irons."""
    key, name = "B", "Flange"
    large_jaw, skinny_jaw = .145, .197

    def dims(self):
        s = self.s1
        self.COL = dict(bf=0.46 * s, d=0.46 * s, tf=0.065 * s, tw=0.045 * s,
                        ch=0.02 * s if self.bevel_vertical else 0.0)
        self.BEAM = dict(bf=0.34 * s, d=0.44 * s, tf=0.05 * s, tw=0.035 * s,
                         ch=0.016 * s if self.bevel_horizontal else 0.0)
        self.chord_skin = self.BEAM["tf"]
        self.M, self.MB, self.MBD = 0.46 * s, 0.44 * s, 0.34 * s
        self.plate_out = 0.045 * s
        self.face_clear = 0.05 * s

    def fins(self):
        return self.M / 2, self.COL["tw"] / 2      # flange face / into the channel onto the web

    def column(self, m, p0, p1, fx, cap0=True, cap1=True, foot=False, cut0=None, cut1=None):
        m.prism(h_prof(**self.COL), p0, p1, fx, cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1)

    def _stiff(self, m, p, axis, up, sec, t):
        w = sec["bf"] / 2 - 0.012 * self.s1 - sec["tw"] / 2
        for sx in (1, -1):
            m.prism(rect(w, sec["d"] - 2 * sec["tf"], cx=sx * (sec["tw"] / 2 + w / 2)),
                    p - axis * t / 2, p + axis * t / 2, up)

    def beam(self, m, p0, p1, cut0=None, cut1=None, cap0=True, cap1=True):
        m.prism(h_prof(**self.BEAM), p0, p1, Z, cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1)
        L = (p1 - p0).length
        t = (p1 - p0).normalized()
        n = max(1, int(L / (1.0 * self.s1)))
        for k in range(1, n):
            self._stiff(m, p0 + t * L * k / n, t, Z, self.BEAM, 0.025 * self.s1)

    def node(self, m, p, ends, fx):
        s = self.s1
        tf = self.BEAM["tf"]
        for dz in (self.MB / 2 - tf / 2, -(self.MB / 2 - tf / 2)):
            self._stiff(m, p + Z * dz, Z, fx, self.COL, 0.025 * s)

        def grid(ep, fn, side, d, keep):
            cv = max(Vector((d.x, d.y, 0.0)).length, 0.35)
            m.bolt_grid(ep, fn, side, Z, 2, 3, 0.23 * s, (self.MB / 2 + 0.065 * s) / cv, 0.038 * s, keep=keep)

        _, zl, zh = self._face_plates(m, p, ends, fx, 0.0, 0.045 * s, self.BEAM["bf"] + 0.1 * s,
                                      self.MB + 0.26 * s, grid, 0.04 * s)
        h = self.MB / 2
        return min(zl, -h), max(zh, h), False

    def large_x(self, m, bay):
        s = self.s2
        sec = dict(bf=0.22 * s, d=0.24 * s, tf=0.035 * s, tw=0.026 * s, ch=0.01 * s)
        gt = 0.035 * s
        back = -(sec["bf"] / 2 + gt / 2)
        for k in range(4):
            gusset_pair(m, bay, k, 0.6 * s, gt, -sec["bf"]/2, sec["bf"]/2)
        A, B, c, pl1, pl2 = x_pair(bay, 0.16 * s, sec["d"] / 2, sec["d"] / 2)
        for (st, en, d), split in ((A, False), (B, True)):
            up = bay.w(Vector((-d.y,d.x)))
            if not split:
                m.prism(h_prof(**sec), bay.P(st), bay.P(en), up)
            else:
                m.prism(h_prof(**sec), bay.P(st), bay.P(c), up, cap1=False, cut1=pl1)
                m.prism(h_prof(**sec), bay.P(c), bay.P(en), up, cap0=False, cut0=pl2)
        for dep in (back, -back):
            m.plate(ngon(0.34 * s, 8, rot=math.pi / 8), bay.P(c, dep), bay.u, bay.v, gt)
        sin = max(abs(A[2].x * B[2].y - A[2].y * B[2].x), 0.2)
        for d, o in ((A[2], 0.2 * s), (B[2], max(0.2 * s, (sec["d"] / 2 + 0.06 * s) / sin))):
            if o > 0.3 * s:
                continue                       # crossing too shallow: the plate edge is the limit
            for sg in (-1, 1):
                through_bolt(m, bay, c + d * o * sg, -back + gt / 2, back - gt / 2, 0.038 * s)

    def skinny_x(self, m, bay):
        s = self.s3
        tg, a, t = 0.035 * s, 0.17 * s, 0.028 * s
        ang = [(-a / 2, 0), (a / 2, 0), (a / 2, t), (-a / 2 + t, t), (-a / 2 + t, a * 0.85), (-a / 2, a * 0.85)]
        c = bay.center
        g1, g2 = bay.diag(0, 2, 0)[2], bay.diag(1, 3, 0)[2]
        xsin = max(abs(g1.x * g2.y - g1.y * g2.x), 0.2)
        d1, d2 = (bay.u + bay.v).normalized(), (bay.v - bay.u).normalized()
        m.plate(rect(0.42 * s, 0.42 * s, c=0.08 * s), bay.P(c), d1, d2, tg)
        for (i, j), sgn in (((0, 2), 1), ((1, 3), -1)):
            st, en, d = bay.diag(i, j, 0.1 * s, a / 2)
            m.prism(ang, bay.P(st, sgn * tg / 2), bay.P(en, sgn * tg / 2), bay.n * sgn)
            face, back = sgn * (tg / 2 + t), -sgn * tg / 2
            side = Vector((-d.y, d.x)) * 0.03 * s * sgn
            foot, lip = sgn * tg/2, sgn * (tg/2 + a*0.85)
            for k in (i,j):
                gusset_pair(m, bay, k, 0.44 * s, tg, min(foot,lip), max(foot,lip))
            o = max(0.13 * s, (a / 2 + 0.05 * s) / xsin)
            if o < 0.27 * s:
                for sg in (-1, 1):
                    through_bolt(m, bay, c + d * o * sg + side, face, back, 0.034 * s, lvl=3)

    def z_brace(self, m, bay, flip=False):
        s = self.s2
        sec = dict(bf=0.2 * s, d=0.26 * s, tf=0.035 * s, tw=0.025 * s, ch=0.01 * s)
        zh, gt = sec["d"], 0.035 * s
        back = -(sec["bf"] / 2 + gt / 2)
        pts = z_layout(bay, zh, flip, s)
        if pts is None:
            return
        cuts = miters(bay, pts)
        for k in range(3):
            m.prism(h_prof(**sec), bay.P(pts[k]), bay.P(pts[k + 1]), bay.v, cap0=cuts[k] is None,
                    cap1=cuts[k + 1] is None, cut0=cuts[k], cut1=cuts[k + 1])
        for q, dl, dd in z_knuckles(pts):
            for dep in (back, -back):
                m.plate(rect(0.5 * s, 0.5 * s, c=0.13 * s), bay.P(q, dep), bay.u, bay.v, gt)
            for off in spaced((q, q + dl * 0.15 * s, q + dd * 0.17 * s), 0.13 * s):
                through_bolt(m, bay, off, -back + gt / 2, back - gt / 2, 0.036 * s)
        z_standoffs(m, bay, pts, zh, h_prof(bf=0.2 * s, d=0.2 * s, tf=0.03 * s, tw=0.025 * s, ch=0.008 * s),
                    0.0125 * s, s, up_axis="u")


class Lattice(Style):
    """C: laced twin-channel members, face gussets, support rings, tie rods."""
    key, name = "C", "Lattice"
    has_frame_bevels = False
    large_jaw, skinny_jaw = .188, .112

    def dims(self):
        s = self.s1
        self.M, self.D, self.MB, self.MBD, self.LT = 0.48 * s, 0.42 * s, 0.44 * s, 0.42 * s, 0.022 * s
        self.chord_skin = 0.035 * s
        self.plate_out = 0.0
        self.side_plate = 0.045 * s

    def half(self, axis):
        return self.M / 2 if axis == 0 else self.D / 2 + self.LT

    def fins(self):
        return self.half(0), self.half(1)

    def beam_side(self, t):
        a = Z.cross(t)
        a = a.normalized() if a.length > 1e-6 else X
        return a, self.MBD / 2 + self.LT

    def column(self, m, p0, p1, fx, cap0=True, cap1=True, foot=False, cut0=None, cut1=None):
        s = self.s1
        laced(m, p0, p1, Z.cross(fx), self.M, self.D, 0.14 * s, 0.03 * s, 0.035 * s, 0.62 * s, 0.07 * s,
              self.LT, 0.28 * s, 0.022 * s, cap0=cap0, cap1=cap1, cut0=cut0, cut1=cut1, rivets=(foot, False))

    def beam(self, m, p0, p1, cut0=None, cut1=None, cap0=True, cap1=True):
        s = self.s1
        up = Z.cross((p1 - p0).normalized())
        if up.length < 1e-6:
            up = X
        laced(m, p0, p1, up, self.MB, self.MBD, 0.13 * s, 0.03 * s, self.chord_skin, 0.6 * s, 0.07 * s, self.LT,
              0.28 * s, 0.022 * s, cut0=cut0, cut1=cut1, cap0=cap0, cap1=cap1,
              rivets=(cut0 is None, cut1 is None))

    def node(self, m, p, ends, fx):
        s = self.s1
        axes = (fx, Z.cross(fx))
        F = _faces(fx)
        aligned, axis_used = True, set()
        free = min((f for *_, f in ends), default=1.0)
        for d, k, s_end, _ in ends:
            dh = Vector((d.x, d.y, 0.0))
            axis_used.add(k // 2)
            if abs(d.z) > 0.02 or dh.length < 1e-6 or abs(dh.normalized().dot(axes[k // 2])) < 0.9986:
                aligned = False
        if aligned and len(axis_used) == 1:
            # straight run through or ending at the column: side plates wrap the column
            ai = axis_used.pop()
            right = any(k == 2 * ai for _, k, _, _ in ends)
            left = any(k == 2 * ai + 1 for _, k, _, _ in ends)
            ref, pdir = axes[ai], axes[1 - ai]
            hw, ext = self.half(ai) + 0.04 * s, self.half(1 - ai)
            L = max(0.1 * s, min(0.56 * s, 0.4 * free - 0.04 * s))   # leave room for the next node
            poly = node_outline(hw, 0.5 * s, self.MB / 2 + 0.04 * s, L, left, right, 0.09 * s)
            for sgn in (1, -1):
                m.plate(poly, p + pdir * sgn * (ext + 0.0175 * s), ref, Z, 0.035 * s)
                face = p + pdir * sgn * (ext + 0.035 * s)
                m.bolt_grid(face, pdir * sgn, ref, Z, 2, 3, self.M * 0.55, 0.42 * s, 0.034 * s)
                for side_sgn, flag in ((1, right), (-1, left)):
                    if flag:
                        m.bolt_grid(face + ref * side_sgn * (hw + L * 0.5), pdir * sgn, ref, Z, 2, 2,
                                    L * 0.5, 0.2 * s, 0.034 * s)
            return -0.5 * s, 0.5 * s, False
        # corners, angled and sloped beams: cheek plates along each beam's sides
        t = 0.035 * s
        L = max(0.1 * s, min(0.56 * s, 0.4 * free - t))
        hh = (self.MB + 0.12 * s) / 2
        lo, hi = -hh, hh
        flat = [Vector((d.x, d.y, 0.0)).normalized() for d, _, _, _ in ends]
        for n_, (d, k, s_end, _) in enumerate(ends):
            a, hwb = self.beam_side(d)
            b = a.cross(d)
            fn = F[k]
            cut = (p + fn * (self.half(k // 2) + t), fn)
            off = hwb + t / 2
            p0, p1 = p + d * s_end, p + d * (s_end + t + L)
            for sgn in (1, -1):
                side = Vector((a.x, a.y, 0.0)).normalized() * sgn
                if any(o_ != n_ and f.dot(side) > 0 and f.dot(flat[n_]) > math.cos(math.radians(75))
                       for o_, f in enumerate(flat)):
                    continue                    # another beam is close on this side
                m.prism(rect(self.MB + 0.12 * s, t, cy=sgn * off), p0, p1, a, cut0=cut)
                m.bolt_grid(p + d * (s_end + t + L / 2) + a * sgn * (off + t / 2), a * sgn, d, b, 2, 2,
                            L * 0.5, 0.2 * s, 0.034 * s)
            for e in (d * s_end, d * (s_end + t + L)):
                lo, hi = min(lo, e.z - hh), max(hi, e.z + hh)
        return lo, hi, False

    def _diag(self, m, p0, p1, bay, s, rivets=(True, True)):
        laced(m, p0, p1, bay.n, 0.3 * s, 0.26 * s, 0.085 * s, 0.022 * s, 0.025 * s, 0.42 * s, 0.05 * s,
              0.018 * s, 0.24 * s, 0.018 * s, rivets=rivets)

    def large_x(self, m, bay):
        s = self.s2
        face, gt, hw = (0.13 + 0.018) * s, 0.04 * s, 0.15 * s
        for k in range(4):
            gusset_pair(m, bay, k, 0.66 * s, gt, -face, face)
        sa, ea, da = bay.diag(0, 2, 0.16 * s, hw)
        sb, eb, db = bay.diag(1, 3, 0.16 * s, hw)
        c = _isect(sa, da, sb, db)
        if c is None:
            c = bay.center
        self._diag(m, bay.P(sa), bay.P(ea), bay, s, (False, False))     # ends under the gussets
        sin, cos = abs(da.x * db.y - da.y * db.x), abs(da.dot(db))
        gap = hw * (1 + cos) / max(sin, 0.2) + 0.01 * s
        if (c - sb).dot(db) > gap + 0.1 * s and (eb - c).dot(db) > gap + 0.1 * s:
            self._diag(m, bay.P(sb), bay.P(c - db * gap), bay, s, (False, False))
            self._diag(m, bay.P(c + db * gap), bay.P(eb), bay, s, (False, False))
        for sgn in (1, -1):                # rings and plates sit on the lacing faces
            m.ring_plate(bay.P(c, sgn * (face + 0.025 * s)), bay.n, 0.56 * s, 0.4 * s, 0.05 * s, segs=16,
                         rot=math.pi / 16)
            m.plate(ngon(0.3 * s, 12, rot=math.pi / 12), bay.P(c, sgn * (face + 0.02 * s)), bay.u, bay.v,
                    0.04 * s)
        half = math.asin(min(1.0, sin)) / 2
        o = max(0.18 * s, 0.07 * s / max(math.sin(half), 0.1), 0.07 * s / max(math.cos(half), 0.1))
        if o < 0.27 * s:
            for d in (da, db):
                for sg in (-1, 1):
                    through_bolt(m, bay, c + d * o * sg, face + 0.04 * s, -face - 0.04 * s, 0.038 * s)
        through_bolt(m, bay, c, face + 0.04 * s, -face - 0.04 * s, 0.07 * s)

    def skinny_x(self, m, bay):
        s = self.s3
        tg = 0.035 * s
        c = bay.center
        m.ring_plate(bay.P(c), bay.n, 0.22 * s, 0.13 * s, tg, segs=12)
        r = 0.032 * s
        rod = ngon(r, 8, rot=math.pi / 8)
        # Seat the octagon on its flat face; its vertex radius leaves a visible gap.
        dep, tb = tg / 2 + r * math.cos(math.pi / 8), 0.18 * s
        for (i, j), sgn in (((0, 2), 1), ((1, 3), -1)):
            st, en, d = bay.diag(i, j, 0.2 * s, r)
            half = (en - st).length / 2
            o = max(half * 0.5, 0.22 * s + tb + 0.03 * s)      # from the centre, outside the ring
            cc = (c - st).dot(d)                                  # crossing, along this rod
            if cc - o - tb < 0.3 * s or cc + o + tb > (en - st).length - 0.3 * s:
                qs = []
            else:
                qs = [st + d * (cc - o), st + d * (cc + o)]
            marks = [st] + [x for q in qs for x in (q - d * tb, q + d * tb)] + [en]
            for k in range(0, len(marks), 2):
                m.prism(rod, bay.P(marks[k], sgn * dep), bay.P(marks[k + 1], sgn * dep), bay.n,
                        cap0=k == 0, cap1=k + 2 == len(marks))
            for q in qs:
                m.prism(ngon(0.058 * s, 6), bay.P(q - d * tb, sgn * dep), bay.P(q + d * tb, sgn * dep), bay.n)
            flat = r * math.cos(math.pi/8)
            for k in (i,j):
                gusset_pair(m, bay, k, 0.4 * s, tg, sgn*dep - flat, sgn*dep + flat)

    def z_brace(self, m, bay, flip=False):
        s = self.s2
        zh, face, hw = 0.3 * s, (0.13 + 0.018) * s, 0.15 * s
        pts = z_layout(bay, zh, flip, s)
        if pts is None:
            return

        def gap(k):
            if k in (0, 3):
                return 0.0
            phi = _ang(pts[k - 1] - pts[k], pts[k + 1] - pts[k])
            return hw / math.tan(max(phi, 0.2) / 2) + 0.01 * s

        for k in range(3):
            a, b = pts[k], pts[k + 1]
            d = (b - a).normalized()
            if (b - a).length > gap(k) + gap(k + 1) + 0.1 * s:
                self._diag(m, bay.P(a + d * gap(k)), bay.P(b - d * gap(k + 1)), bay, s, (k == 0, k == 2))
        for q, _, _ in z_knuckles(pts):
            for sgn in (1, -1):
                m.plate(ngon(0.3 * s, 12, rot=math.pi / 12), bay.P(q, sgn * (face + 0.02 * s)), bay.u, bay.v,
                        0.04 * s)
                for ang in range(4):
                    r = Vector((math.cos(ang * math.pi / 2 + math.pi / 4), math.sin(ang * math.pi / 2 + math.pi / 4)))
                    m.bolt(bay.P(q + r * 0.2 * s, sgn * (face + 0.04 * s)), bay.n * sgn, 0.034 * s)
                m.bolt(bay.P(q, sgn * (face + 0.04 * s)), bay.n * sgn, 0.065 * s)
        z_standoffs(m, bay, pts, zh, rect(0.24 * s, 0.26 * s, c=0.03 * s), 0.13 * s, s)


STYLES = {"A": Monolith(), "B": Flange(), "C": Lattice()}


# ======================================================================================
#  Guide mesh -> scaffold
# ======================================================================================
class Config:
    """Plain copy of the settings so the builder runs without Blender properties."""

    def __init__(self, **kw):
        self.style = "B"
        self.section_w = self.section_h = 3.5
        self.main, self.secondary, self.skinny = 0.46, 0.24, 0.2
        self.bevel_horizontal = self.bevel_vertical = True
        self.bolts, self.bolt_level, self.head_style = True, 1, "hex"
        self.screw_object = None
        self.plinth, self.overrun = 0.9, 0.3
        self.base_plates, self.caps, self.floor_brace = True, True, False
        self.adaptive = False
        self.__dict__.update(kw)

    @classmethod
    def from_props(cls, p):
        return cls(style=p.style, section_w=p.section_w, section_h=p.section_h, main=p.main,
                   secondary=p.secondary, skinny=p.skinny, bevel_horizontal=p.bevel_horizontal,
                   bevel_vertical=p.bevel_vertical, bolts=p.bolts, bolt_level=int(p.bolt_level),
                   head_style=p.head_style, plinth=p.plinth, overrun=p.overrun, base_plates=p.base_plates,
                   caps=p.caps, floor_brace=p.floor_brace, adaptive=p.adaptive, screw_object=p.screw_object)


def newell(pts):
    n = Vector()
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        n.x += (a.y - b.y) * (a.z + b.z)
        n.y += (a.z - b.z) * (a.x + b.x)
        n.z += (a.x - b.x) * (a.y + b.y)
    return n.normalized() if n.length > 1e-12 else Vector((0, 0, 1))


def is_vertical(a, b):
    d = b - a
    return d.length > 1e-9 and abs(d.z) / d.length > 0.85


def wall_order(f, V):
    """A wall quad as (b0, b1, t1, t0): bottom edge b0-b1, top edge t0-t1, upright sides
    b0-t0 and b1-t1. None when neither pair of opposite edges stands upright."""
    def up(a, b):
        d = V[b] - V[a]
        return abs(d.z) / d.length if d.length > 1e-9 else 0.0

    e = [up(f[k], f[(k + 1) % 4]) for k in range(4)]
    g = list(f) if min(e[1], e[3]) >= min(e[0], e[2]) else list(f[1:]) + [f[0]]
    if min(up(g[1], g[2]), up(g[3], g[0])) < 0.85:
        return None
    if V[g[0]].z + V[g[1]].z > V[g[2]].z + V[g[3]].z:
        g = g[2:] + g[:2]
    return g


def build_scaffold(verts, faces, brace, loose, cfg, closed=False):
    st = STYLES[cfg.style]
    st.setup(cfg.main / 0.46, cfg.secondary / 0.24, cfg.skinny / 0.2, cfg.bevel_horizontal, cfg.bevel_vertical)
    bolt_level = cfg.bolt_level if cfg.bolts else 0
    screw = screw_template(cfg.screw_object, closed=closed) if bolt_level else None
    m = Mesh()
    m.closed = closed
    m.bolt_level = bolt_level
    m.head_style = cfg.head_style
    m.screw = screw
    m.pending_bolts = []
    adaptive = getattr(cfg, "adaptive", False)
    V = [Vector(v) for v in verts]

    def ek(a, b):
        return (a, b) if a < b else (b, a)

    # ---- edge rings: opposite edges of every quad share a section count
    parent = {}

    def find(e):
        parent.setdefault(e, e)
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for f in faces:
        es = [ek(f[k], f[(k + 1) % len(f)]) for k in range(len(f))]
        for e in es:
            find(e)
        if len(f) == 4:
            for a, b in ((es[0], es[2]), (es[1], es[3])):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    for a, b in loose:
        find(ek(a, b))
    rings = {}
    for e in list(parent):
        rings.setdefault(find(e), []).append(e)
    count, lvl = {}, {}
    room = st.MB + 0.45 * st.s1                              # height a node's plates need
    p_min = (0.09 * st.s1 + room / 2 + 0.02 * st.s1) if cfg.base_plates else 0.0   # lowest plinth
    min_gap = max(0.35 * cfg.section_h, 2.5 * cfg.main, room + 0.15 * st.s1, p_min + room)
    for es in rings.values():
        lens = [(V[a] - V[b]).length for a, b in es]
        vert = sum(1 for a, b in es if is_vertical(V[a], V[b])) > len(es) / 2
        size = max(cfg.section_h if vert else cfg.section_w, 0.5)
        n = max(1, round(max(lens) / size))
        if vert:
            # adaptive rows fan out under a slope; keep them far enough apart at the short end
            # for beams and plates to fit
            n = max(1, min(n, math.floor(min(lens) / min_gap + 1e-6)))
        for e in es:
            count[e] = n
        if vert:
            # level mode: one row spacing for the whole ring, set by its shortest upright edge,
            # so rows stay at the same heights even where the guide slopes
            dzs = [abs(V[a].z - V[b].z) for a, b in es if abs(V[a].z - V[b].z) > 1e-4]
            short = min(dzs) if dzs else size
            step = short / max(1, round(short / size))
            base = min(min(V[a].z, V[b].z) for a, b in es)
            for e in es:
                lvl[e] = (base, step)

    def levels(e, zb, zt):
        base, step = lvl[e]
        out, k = [], max(1, math.floor((zb - base) / step + 1e-6) + 1)
        while True:
            z = base + k * step
            if zt - z <= 0.5 * step + 1e-6:      # never leave a sliver under the top
                break
            if z > zb + max(0.4 * step, p_min + room):
                out.append(z)
            k += 1
        return out

    # ---- section points, segments and cells
    pts, index = [], {}

    def pid(p):
        k = (round(p.x * 1000), round(p.y * 1000), round(p.z * 1000))
        if k not in index:
            index[k] = len(pts)
            pts.append(Vector(p))
        return index[k]

    segs = set()

    def seg(a, b):
        if a != b:
            segs.add(ek(a, b))

    def line_nodes(pa, pb, e):
        """Node ids along a line of the guide: level heights on upright lines (level mode),
        an even split otherwise."""
        if not adaptive and e in lvl and is_vertical(pa, pb):
            lo, hi = (pa, pb) if pa.z <= pb.z else (pb, pa)
            ids = [pid(lo)]
            for z in levels(e, lo.z, hi.z):
                ids.append(pid(lo.lerp(hi, (z - lo.z) / (hi.z - lo.z))))
            ids.append(pid(hi))
            return ids
        n = count[e]
        return [pid(pa.lerp(pb, t / n)) for t in range(n + 1)]

    def split_edge(a, b):
        ids = line_nodes(V[a], V[b], ek(a, b))
        for x, y in zip(ids, ids[1:]):
            seg(x, y)

    walls, floors = [], []
    for fi, f in enumerate(faces):
        if len(f) < 3:
            continue
        normal = newell([V[i] for i in f])
        if len(f) != 4:
            for k in range(len(f)):
                split_edge(f[k], f[(k + 1) % len(f)])
            continue
        kind = brace[fi] if 0 <= brace[fi] <= 5 else AUTO
        g = wall_order(f, V) if abs(normal.z) < 0.5 else None
        if g is not None:
            b0, b1, t1, t0 = g
            eL, nA = ek(b0, t0), count[ek(b0, b1)]
            cols = [line_nodes(V[b0].lerp(V[b1], i / nA), V[t0].lerp(V[t1], i / nA), eL) for i in range(nA + 1)]
            for ids in cols:
                for x, y in zip(ids, ids[1:]):
                    seg(x, y)
            cells = []
            for i in range(nA):
                A, B = cols[i], cols[i + 1]
                if adaptive and len(A) == len(B):
                    rows = list(zip(A, B))
                else:
                    za = {round(pts[x].z, 3): x for x in A[1:-1]}
                    zb = {round(pts[x].z, 3): x for x in B[1:-1]}
                    rows = [(A[0], B[0])] + [(za[z], zb[z]) for z in sorted(set(za) & set(zb))] + [(A[-1], B[-1])]
                for x, y in rows:
                    seg(x, y)
                for k in range(len(rows) - 1):
                    cells.append((rows[k][0], rows[k][1], rows[k + 1][1], rows[k + 1][0], i, k))
            walls.append((cells, normal, kind))
            continue
        v0, v1, v2, v3 = (V[i] for i in f)
        nA, nB = count[ek(f[0], f[1])], count[ek(f[1], f[2])]
        G = []
        for i in range(nA + 1):
            s = i / nA
            col = []
            for j in range(nB + 1):
                t = j / nB
                col.append(pid(v0 * ((1 - s) * (1 - t)) + v1 * (s * (1 - t)) + v2 * (s * t) + v3 * ((1 - s) * t)))
            G.append(col)
        for i in range(nA + 1):
            for j in range(nB):
                seg(G[i][j], G[i][j + 1])
        for j in range(nB + 1):
            for i in range(nA):
                seg(G[i][j], G[i + 1][j])
        cells = [(G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1], i, j) for i in range(nA) for j in range(nB)]
        (walls if abs(normal.z) < 0.5 else floors).append((cells, normal, kind))
    for a, b in loose:
        split_edge(a, b)
    info = {"sections": sum(len(c) for c, _, _ in walls), "tris": 0}
    if not pts:
        return m, info

    # ---- columns: chains of vertical segments
    vsegs = {s for s in segs if is_vertical(pts[s[0]], pts[s[1]])}
    up, down = {}, {}
    for a, b in sorted(vsegs):
        lo, hi = (a, b) if pts[a].z < pts[b].z else (b, a)
        up.setdefault(lo, []).append(hi)
        down.setdefault(hi, []).append(lo)
    chains = []
    for start, successors in up.items():
        if len(down.get(start, ())) == 1 and len(successors) == 1:
            continue  # An interior vertex belongs to the chain that reaches it.
        for successor in successors:
            chain = [start, successor]
            while len(up.get(chain[-1], ())) == 1 and len(down[chain[-1]]) == 1:
                chain.append(up[chain[-1]][0])
            chains.append(chain)
    chain_of = {i: ci for ci, c in enumerate(chains) for i in c}
    tops = down.keys() - up.keys()

    # ---- ground: anything near the lowest point, so a slightly uneven bottom still counts
    zmin = min(p.z for p in pts)
    tol = max(0.05, 0.3 * cfg.section_h)
    ground = {i for i, p in enumerate(pts) if p.z < zmin + tol and i not in down}
    first = [abs(pts[a].z - pts[b].z) for a, b in vsegs if a in ground or b in ground]
    if first:
        f0 = min(first)
        plinth = max(0.0, min(max(min(cfg.plinth, 0.45 * f0), p_min), f0 - room))
    else:
        plinth = 0.0

    def L(i):
        if i in ground and i in up:
            axis = pts[up[i][0]] - pts[i]
            return pts[i] + axis * (plinth / axis.z)
        return pts[i] + Z * plinth if i in ground else pts[i]

    # ---- beams, and a frame for every column that follows the walls around it
    beams = []
    for a, b in segs:
        if (a, b) in vsegs:
            continue
        pa, pb = L(a), L(b)
        if (pb - pa).length > 1e-6:
            beams.append((a, b, pa, pb, (pb - pa).normalized()))
    acc = [[] for _ in chains]
    for a, b, pa, pb, d in beams:
        h = Vector((d.x, d.y))
        if h.length < 1e-6:
            continue
        ang = math.atan2(h.y, h.x) % (math.pi / 2)        # directions 90 degrees apart fold together
        w = h.length * (1.0 if abs(d.z) < 0.05 else 0.5)   # level beams count most
        for i in (a, b):
            if i in chain_of:
                acc[chain_of[i]].append((ang, w))

    def fold(x):
        x = x % (math.pi / 2)
        return min(x, math.pi / 2 - x)

    frames = []
    for lst in acc:
        th = 0.0
        if lst:
            # the direction most beams agree with, so walls meet the column square on
            tol = math.radians(1.0)
            ref = max(lst, key=lambda q: sum(w for a, w in lst if fold(a - q[0]) < tol))[0]
            near = [((a - ref + math.pi / 4) % (math.pi / 2) - math.pi / 4, w) for a, w in lst
                    if fold(a - ref) < tol]
            th = ref + sum(o * w for o, w in near) / sum(w for _, w in near)
        frames.append(Vector((math.cos(th), math.sin(th), 0.0)))

    # ---- where every beam end stops
    ends_at = {}
    for bi, (a, b, pa, pb, d) in enumerate(beams):
        ends_at.setdefault(a, []).append((bi, 0, d))
        ends_at.setdefault(b, []).append((bi, 1, -d))
    cut, node_ends = {}, {}
    for i, lst in ends_at.items():
        p = L(i)
        if i in chain_of:
            # flush against the column face it lands on, cut to that face's angle
            fx = frames[chain_of[i]]
            F = _faces(fx)
            ks = []
            for bi, sd, d in lst:
                dh = Vector((d.x, d.y, 0.0))
                ks.append(max(range(4), key=lambda q: F[q].dot(dh)) if dh.length > 1e-6 else 0)
            used = set(ks)
            ends = []
            for (bi, sd, d), k in zip(lst, ks):
                fn = F[k]
                off = st.half(k // 2) + st.end_out(k, used)
                s_end = off / max(d.dot(fn), 0.1)
                dh = Vector((d.x, d.y, 0.0)).normalized()
                square = dh.dot(fn) > 0.999                 # square on: the plate covers the end
                cut[(bi, sd)] = (p + d * s_end, (p + fn * off, fn), not square)
                ends.append((d, k, s_end, (beams[bi][3] - beams[bi][2]).length - 2 * s_end))
            node_ends[i] = (ends, fx)
            continue
        if len(lst) == 1:
            cut[(lst[0][0], lst[0][1])] = (p, None, True)
            continue
        # beams meeting without a column: one runs through, the rest stop on its side
        pair, best = None, -0.97
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                dd = lst[x][2].dot(lst[y][2])
                if dd < best:
                    best, pair = dd, (x, y)
        if pair is not None:
            for x in pair:
                cut[(lst[x][0], lst[x][1])] = (p, None, False)
            prim, rest = lst[pair[0]][2], [q for q in range(len(lst)) if q not in pair]
        else:
            prim, rest = lst[0][2], list(range(1, len(lst)))
            reach = max(st.beam_side(lst[q][2])[1] for q in rest)
            cut[(lst[0][0], lst[0][1])] = (p - prim * reach, None, True)
        sn, hw = st.beam_side(prim)
        for q in rest:
            bi, sd, d = lst[q]
            N = sn if d.dot(sn) >= 0 else -sn
            cut[(bi, sd)] = (p + d * (hw / max(d.dot(N), 0.2)), (p + N * hw, N), False)
    for bi, (a, b, pa, pb, d) in enumerate(beams):
        p0, pl0, c0 = cut[(bi, 0)]
        p1, pl1, c1 = cut[(bi, 1)]
        if (p1 - p0).dot(d) > 0.05:
            st.beam(m, p0, p1, pl0, pl1, c0, c1)
    blocks = {}
    for i, (ends, fx) in node_ends.items():
        blocks[i] = st.node(m, L(i), ends, fx)

    # ---- columns, cut around node blocks so nothing is buried inside them
    for ci, chain in enumerate(chains):
        fx = frames[ci]
        bot, top = pts[chain[0]], pts[chain[-1]]
        heights = [pts[i].z for i in chain]
        bends = {}
        for prev, here, nxt in zip(chain, chain[1:], chain[2:]):
            incoming = (pts[here] - pts[prev]).normalized()
            outgoing = (pts[nxt] - pts[here]).normalized()
            if (incoming - outgoing).length > 1e-4:
                bends[pts[here].z] = (pts[here], (incoming + outgoing).normalized())

        def at(z):
            k = max(0, min(len(chain) - 2, bisect_right(heights, z) - 1))
            a, b = pts[chain[k]], pts[chain[k + 1]]
            return a.lerp(b, (z - a.z) / (b.z - a.z))

        def column_span(lo, hi, cap0, cap1, foot):
            cuts = [lo, *(z for z in bends if lo < z < hi), hi]
            for a, b in zip(cuts, cuts[1:]):
                st.column(m, at(a), at(b), fx, cap0 and a == lo, cap1 and b == hi,
                          foot and a == lo, cut0=bends.get(a), cut1=bends.get(b))

        c0 = c1 = True
        ext = sorted((L(i).z + blocks[i][0], L(i).z + blocks[i][1]) for i in chain if i in blocks)
        iv = sorted((L(i).z + blocks[i][0], L(i).z + blocks[i][1]) for i in chain if i in blocks and blocks[i][2])
        if chain[0] in ground:
            start = bot.z
            if cfg.base_plates:
                start += st.base(m, bot, fx, (ext[0][0] - bot.z - 0.01 * st.s1) if ext else None)
                c0 = False
        elif chain[0] not in down:
            start = bot.z - max(cfg.overrun * 0.5, 0.05)
        else:
            start = bot.z
        end = top.z + (cfg.overrun if chain[-1] in tops else 0.0)
        if ext:                                  # caps clear every plate and block on the column
            if chain[-1] in tops:
                end = max(end, max(e[1] for e in ext) + 0.005 * st.s1)
            if chain[0] not in ground and chain[0] not in down:
                start = min(start, min(e[0] for e in ext) - 0.005 * st.s1)
        if chain[0] not in ground and chain[0] not in down and cfg.caps:
            st.cap(m, at(start), fx, down=True)
            c0 = False
        if chain[-1] in tops and cfg.caps:
            st.cap(m, at(end), fx)
            c1 = False
        cur, cap, foot = start, c0, chain[0] in ground
        for lo, hi in iv:
            lo, hi = max(lo, start), min(hi, end)
            if lo - cur > 1e-4:
                column_span(cur, lo, cap, False, foot)
            if hi > cur:
                cur, cap, foot = hi, False, False
        if end - cur > 1e-4:
            column_span(cur, end, cap, c1, foot)

    # ---- braces: every bay is read from its four real corners
    def support(i, u):
        if i not in chain_of:
            return st.beam_side(X)[1]
        fx = frames[chain_of[i]]
        fy = Z.cross(fx)
        body = abs(u.dot(fx)) * st.half(0) + abs(u.dot(fy)) * st.half(1)
        F = _faces(fx)
        k = max(range(4), key=lambda q: F[q].dot(u))   # the face u lands on carries the plate
        side = Z.cross(F[k])
        plate = (st.half(k // 2) + st.plate_out) * abs(F[k].dot(u)) + st.half(1 - k // 2) * abs(side.dot(u))
        return max(body, plate) + st.face_clear

    # Z legs sit on the web. X cheeks seat on the horizontal connector and
    # stop at the outside of their upright, leaving the post profile visible.
    web_seat = 0.006 * st.s1
    chord_seat = st.MB / 2 - 0.10 * st.s1
    wall_mounts = (chord_seat, web_seat, chord_seat, web_seat)
    s2, s3 = st.s2, st.s3

    neighbors = {}
    for cells, surface_normal, kind in walls + (floors if cfg.floor_brace else []):
        if kind == OPEN:
            continue
        for *ids, _, _ in cells:
            corners = [L(i) for i in ids]
            center = sum(corners, Vector()) / 4
            floor_plane = None
            if abs(surface_normal.z) >= 0.5:
                # Bay projects a twisted quad onto the plane through its first
                # oriented corner. The other three guide corners may be off-plane.
                n = newell(corners)
                origin = corners[1] if n.dot(surface_normal) < 0 else corners[0]
                floor_plane = (origin, n)
            for a,b in zip(ids, ids[1:] + ids[:1]):
                along = (L(b) - L(a)).normalized()
                inward = center - L(a)
                inward = (inward - along*inward.dot(along)).normalized()
                neighbors.setdefault(ek(a,b), []).append((tuple(ids), inward, floor_plane))

    def fit_cheeks(bay, ids, normal, floor=False):
        """Share the space around folded guide edges and clip the foot clear of the post."""
        key = tuple(ids)
        if newell([L(i) for i in ids]).dot(normal) < 0:
            ids = [ids[1], ids[0], ids[3], ids[2]]
        folds, planes = [], []
        for k,a in enumerate(ids):
            b = ids[(k+1)%4]
            around = neighbors.get(ek(a,b), ())
            inward = next((v for owner,v,_ in around if owner == key), Vector())
            # The bisector of adjacent bays is a cut-off for both plate skins.
            cosines = [inward.dot(v) for owner,v,plane in around if owner != key and
                       (plane is not None) == floor and inward.dot(v) < .99999]
            cosine = max(-1.0, max(cosines, default=-1.0))
            folds.append(math.sqrt((1 + cosine) / (1 - cosine)))
            if not floor and k in (0,2):
                if st.chord_skin:
                    # Seat inside the bay-facing flange or channel web. Its far face
                    # is exposed inside the member, even on sloping or warped guides.
                    rail = (L(b) - L(a)).normalized()
                    face = (Z - rail * Z.dot(rail)).normalized()
                    if face.dot(inward) < 0:
                        face.negate()
                    # Leave a quarter of the solid skin between the tip and the
                    # open channel; the rest provides bearing on twisted walls.
                    seat = st.MB / 2 - 0.75 * st.chord_skin
                    bay.chord_planes[k].append((L(a) + face * seat, face))
                for _, _, plane in around:
                    if plane is None:
                        continue
                    origin, n = plane
                    n = n if inward.dot(n) >= 0 else -n
                    # A floor cheek can run right up to the beam web. Keep
                    # the wall cheek outside the full thickness of the floor connection.
                    half = min(st.skinny_jaw * s3, st.MB * .44)
                    bay.chord_planes[k].append((origin + n * half, n))
            if a not in chain_of:
                planes.append(None)
                continue
            chain = chains[chain_of[a]]
            pos = chain.index(a)
            other = chain[pos-1] if pos else chain[1]
            if not floor:
                other = ids[(3,2,1,0)[k]]
            t = (pts[other] - pts[a]).normalized()
            ea = (L(ids[(k+1)%4]) - L(a)).normalized()
            eb = (L(ids[k-1]) - L(a)).normalized()
            toward = ea + eb if floor else (ea if k in (0,2) else eb)
            inward = (toward - t*t.dot(toward)).normalized()
            fx = frames[chain_of[a]]
            if isinstance(st, Lattice):
                fy = Z.cross(fx)
                ay = (fy - t*t.dot(fy)).normalized()
                ax = ay.cross(t)
            else:
                ax = (fx - t*t.dot(fx)).normalized()
                ay = t.cross(ax)
            body = abs(inward.dot(ax))*st.half(0) + abs(inward.dot(ay))*st.half(1)
            if floor and isinstance(st, Lattice):
                # Allow the mitered tips of the column lacing to clear the floor plate.
                body += 0.001 * st.s1
            planes.append((pts[a] + inward*body, inward))
        bay.folds, bay.post_planes = folds, planes

    def fitted(bay, w, h, fn, *args):
        """Run a brace builder with its details scaled down to fit a small cell."""
        fit = max(0.45, min(w, h) / 2.0)
        st.s2, st.s3 = min(s2, fit), min(s3, fit)
        try:
            if fn in (st.large_x, st.skinny_x):
                # Both cheek plates must be able to land on the rail. Fit to
                # its face as well as to the space between neighboring nodes.
                room = (.44 * st.MB if abs(bay.n.z) >= .5 else .5 * st.MBD)
                st.s2 = min(st.s2, room / st.large_jaw)
                st.s3 = min(st.s3, room / st.skinny_jaw)
                jaw = st.large_jaw * st.s2 if fn == st.large_x else st.skinny_jaw * st.s3
                bay.ins = tuple(max(old, jaw * fold) for old, fold in zip(bay.ins, bay.folds))
                bay._inner = bay.inset()
                if min(bay.size()) < 0.1 * min(st.s2, st.s3):
                    return
            fn(m, bay, *args)
        finally:
            st.s2, st.s3 = s2, s3

    for cells, normal, kind in walls:
        for c00, c10, c11, c01, i, k in cells:
            C = [L(c00), L(c10), L(c11), L(c01)]
            u = C[1] - C[0]
            u = Vector((u.x, u.y, 0.0))
            u = u.normalized() if u.length > 1e-6 else X
            cc = (C[0] + C[1] + C[2] + C[3]) / 4
            tow = lambda q: u if (cc - q).dot(u) > 0 else -u       # from the column toward the cell
            ins = (st.MB / 2, support(c10, tow(C[1])), st.MB / 2, support(c00, tow(C[0])))
            bay = Bay(C, normal, ins, wall_mounts)
            fit_cheeks(bay, [c00,c10,c11,c01], normal)
            w, h = bay.size()
            if w < st.M + 0.4 or h < st.MB + 0.4:
                continue
            kk = kind
            if kk == AUTO:
                if k == 0 and (c00 in ground or c10 in ground):
                    kk = LARGE_X
                elif c11 in tops and c01 in tops:
                    kk = SKINNY_X
                else:
                    kk = Z_FLIP if (i + k) % 2 else Z_BRACE
            if kk == LARGE_X:
                fitted(bay, w, h, st.large_x)
            elif kk == SKINNY_X:
                fitted(bay, w, h, st.skinny_x)
            elif kk in (Z_BRACE, Z_FLIP):
                fitted(bay, w, h, st.z_brace, kk == Z_FLIP)
    if cfg.floor_brace:
        for cells, normal, kind in floors:
            if kind == OPEN:
                continue
            for c00, c10, c11, c01, i, j in cells:
                ids = (c00, c10, c11, c01)
                C = [L(c) for c in ids]
                ins = []
                for q in range(4):
                    e = C[(q + 1) % 4] - C[q]
                    perp = normal.cross(e)
                    perp = perp.normalized() if perp.length > 1e-9 else X
                    ins.append(max(support(ids[q], perp), support(ids[(q + 1) % 4], perp),
                                   support(ids[q], -perp), support(ids[(q + 1) % 4], -perp),
                                   st.beam_extent(e.normalized(), perp) + st.side_plate))
                bay = Bay(C, normal, ins, (web_seat,) * 4)
                fit_cheeks(bay, list(ids), normal, floor=True)
                w, h = bay.size()
                if w > st.MBD + 0.4 and h > st.MBD + 0.4:
                    fitted(bay, w, h, st.skinny_x)
    m.finish_bolts()
    info["tris"] = m.tris()
    info["columns"], info["nodes"] = len(chains), len(node_ends)
    return m, info


# ======================================================================================
#  Blender side
# ======================================================================================
_pending = {}
_last_info = {}
DELAY = 0.12


def read_guide(obj):
    me = obj.data
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        bm.verts.index_update()
        layer = bm.faces.layers.int.get(ATTR)
        verts = [v.co.copy() for v in bm.verts]
        faces = [[v.index for v in f.verts] for f in bm.faces]
        brace = [f[layer] if layer is not None else 0 for f in bm.faces]
        loose = [(e.verts[0].index, e.verts[1].index) for e in bm.edges if not e.link_faces]
    else:
        verts = [v.co.copy() for v in me.vertices]
        faces = [list(p.vertices) for p in me.polygons]
        attr = me.attributes.get(ATTR)
        if attr is not None and attr.domain == "FACE" and attr.data_type == "INT" and len(attr.data) == len(faces):
            brace = [d.value for d in attr.data]
        else:
            brace = [0] * len(faces)
        used = {k for p in me.polygons for k in p.edge_keys}
        loose = [tuple(e.vertices) for e in me.edges if tuple(sorted(e.vertices)) not in used]
    sc = Vector(abs(a) for a in obj.matrix_world.to_scale())
    verts = [c * sc for c in verts]
    return verts, faces, brace, loose


def _materials():
    out = []
    for name, rgb, metal, rough in (("HS Paint", (0.17, 0.2, 0.23), 0.0, 0.6),
                                    ("HS Steel", (0.025, 0.025, 0.028), 0.8, 0.4)):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            if mat.node_tree is not None:
                b = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if b is not None:
                    b.inputs["Base Color"].default_value = (*rgb, 1.0)
                    b.inputs["Metallic"].default_value = metal
                    b.inputs["Roughness"].default_value = rough
        out.append(mat)
    return out


def ensure_output(guide):
    cfg = guide.heavy_scaffold
    out = cfg.output
    if out is not None and out.parent is not None and out.parent is not guide:
        out = None                     # a duplicated guide: give it its own scaffold
    if out is None or bpy.data.objects.get(out.name) is not out:
        me = bpy.data.meshes.new(guide.name + "_Scaffold")
        out = bpy.data.objects.new(guide.name + "_Scaffold", me)
        colls = guide.users_collection or [bpy.context.scene.collection]
        colls[0].objects.link(out)
        cfg.output = out
    if out.parent is not guide:
        out.parent = guide
    sc = [abs(a) for a in guide.matrix_world.to_scale()]
    inv = Matrix.Diagonal(tuple(1 / max(a, 1e-6) for a in sc) + (1.0,))
    if out.matrix_parent_inverse != inv:
        out.matrix_parent_inverse = inv
    if out.matrix_basis != Matrix.Identity(4):
        out.matrix_basis = Matrix.Identity(4)
    return out


def rebuild(guide):
    if guide is None or not guide.heavy_scaffold.is_guide or guide.type != "MESH":
        return None
    out = ensure_output(guide)
    if out.mode == "EDIT":
        return None
    t = time.perf_counter()
    m, info = build_scaffold(*read_guide(guide), Config.from_props(guide.heavy_scaffold))
    me = out.data
    if me.users > 1:
        me = out.data = me.copy()
    mats = _materials()
    if [mm for mm in me.materials] != mats:
        me.materials.clear()
        for mm in mats:
            me.materials.append(mm)
    m.write(me)
    info["seconds"] = time.perf_counter() - t
    _last_info[guide.name] = info
    return info


def _union_group():
    for ng in bpy.data.node_groups:
        if ng.bl_idname == "GeometryNodeTree" and ng.get("hs_island_union") == 1:
            return ng
    ng = bpy.data.node_groups.new("HS Watertight Union", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    islands = ng.nodes.new("GeometryNodeInputMeshIsland")
    parts = ng.nodes.new("GeometryNodeSplitToInstances")
    parts.domain = "FACE"
    union = ng.nodes.new("GeometryNodeMeshBoolean")
    union.operation = "UNION"
    union.solver = "MANIFOLD"
    ng.links.new(gi.outputs[0], parts.inputs["Geometry"])
    ng.links.new(islands.outputs["Island Index"], parts.inputs["Group ID"])
    ng.links.new(parts.outputs["Instances"], next(s for s in union.inputs if s.identifier == "Mesh 2"))
    ng.links.new(union.outputs[0], go.inputs[0])
    ng["hs_island_union"] = 1
    return ng


def _seat_bake_parts(bm):
    """Give flush and edge-only joints a small overlap before the solid union."""
    if not bm.faces:
        raise ValueError("The scaffold guide has no parts to bake.")
    if any(not e.is_manifold or not e.is_contiguous for e in bm.edges) or any(not v.is_manifold for v in bm.verts):
        raise ValueError("Watertight Bake needs closed, consistently wound parts. Check the guide and custom screw mesh.")
    remaining = set(bm.verts)
    while remaining:
        pending = [remaining.pop()]
        part = []
        while pending:
            vert = pending.pop()
            part.append(vert)
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other in remaining:
                    remaining.remove(other)
                    pending.append(other)

        low = Vector(min(v.co[i] for v in part) for i in range(3))
        high = Vector(max(v.co[i] for v in part) for i in range(3))
        center = (low + high) * 0.5
        # Scale each solid about its own center. Bound displacement at 0.0001
        # Blender units and relative growth at 0.02% even on very small bolts.
        radius = (high - low).length * 0.5
        factor = 1.0 + min(0.0002, 0.0001 / max(radius, 1e-9))
        for v in part:
            v.co = center + (v.co - center) * factor


def _finish_union(me):
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        if not bm.faces:
            raise ValueError("The watertight union produced an empty mesh.")
        bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=list(bm.edges))
        bad_edges = sum(not e.is_manifold or not e.is_contiguous for e in bm.edges)
        bad_verts = sum(not v.is_manifold for v in bm.verts)
        if bad_edges or bad_verts or not bm.faces:
            raise ValueError(f"Watertight Bake could not close this mesh ({bad_edges} invalid edges, {bad_verts} invalid vertices).")
        bm.to_mesh(me)
        me.update()
    finally:
        bm.free()


def bake_watertight(guide):
    """Return the checked solid union, keeping the editable scaffold at its original dimensions."""
    m, _ = build_scaffold(*read_guide(guide), Config.from_props(guide.heavy_scaffold), closed=True)
    try:
        _seat_bake_parts(m.bm)
    except Exception:
        m.bm.free()
        raise
    raw = bpy.data.meshes.new(guide.name + "_raw")
    m.write(raw)
    for mm in _materials():
        raw.materials.append(mm)
    tmp = bpy.data.objects.new("_hs_union_tmp", raw)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("Union", "NODES")
        mod.node_group = _union_group()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        try:
            _finish_union(me)
        except Exception:
            bpy.data.meshes.remove(me)
            raise
    finally:
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(raw)
    me.name = guide.name + "_Baked"
    if not me.materials:
        for mm in _materials():
            me.materials.append(mm)
    return me


def _tick():
    now = time.monotonic()
    for uid, (obj, t) in list(_pending.items()):
        if now - t >= DELAY:
            del _pending[uid]
            try:
                if obj.session_uid == uid:
                    rebuild(obj)
            except ReferenceError:
                pass  # The guide was deleted before its deferred rebuild.
            except Exception as ex:
                print("Heavy Scaffold rebuild failed:", ex)
    return DELAY / 2 if _pending else None


def request_rebuild(guide):
    _pending[guide.session_uid] = (guide, time.monotonic())
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=DELAY)


@persistent
def _on_depsgraph(scene, depsgraph):
    geometry, transforms = set(), set()
    for upd in depsgraph.updates:
        if isinstance(upd.id, (bpy.types.Object, bpy.types.Mesh)):
            if upd.is_updated_geometry:
                geometry.add(upd.id.original)
            if upd.is_updated_transform:
                transforms.add(upd.id.original)
    if not geometry and not transforms:
        return
    for guide in scene.objects:
        if guide.type != "MESH":
            continue
        cfg = guide.heavy_scaffold
        if not cfg.is_guide or not cfg.live:
            continue
        source = cfg.screw_object if cfg.bolts else None
        if (guide in geometry or guide.data in geometry or
                (source and (source in geometry or source.data in geometry or source in transforms))):
            request_rebuild(guide)
        elif guide in transforms and cfg.output:
            # Translation and rotation are handled by parenting. Only resize on a scale change.
            inv = cfg.output.matrix_parent_inverse
            if any(abs(max(abs(a), 1e-6) * inv[k][k] - 1) > 1e-5
                   for k, a in enumerate(guide.matrix_world.to_scale())):
                request_rebuild(guide)


def guide_of(obj):
    if obj is None:
        return None
    if obj.type == "MESH" and obj.heavy_scaffold.is_guide:
        return obj
    par = obj.parent
    if par is not None and par.heavy_scaffold.is_guide and par.heavy_scaffold.output is obj:
        return par
    return None


def brace_attribute_conflict(mesh):
    attr = mesh.attributes.get(ATTR)
    if attr is not None and (attr.data_type != "INT" or attr.domain != "FACE"):
        return f"The scaffold requires a face integer attribute named {ATTR}. Rename the existing attribute first."


def init_guide(obj):
    conflict = brace_attribute_conflict(obj.data)
    if conflict:
        raise ValueError(conflict)
    if obj.data.attributes.get(ATTR) is None:
        obj.data.attributes.new(ATTR, "INT", "FACE")
    obj.heavy_scaffold.is_guide = True
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.hide_render = True


def _changed(self, context):
    guide = self.id_data
    if isinstance(guide, bpy.types.Object) and self.is_guide:
        request_rebuild(guide)


def _screw_poll(self, obj):
    return (obj.type == "MESH" and guide_of(obj) is None and
            obj.name in bpy.context.scene.objects)


STYLE_ITEMS = [("A", "Monolith", "Chamfered box sections, node blocks, angular plates"),
               ("B", "Flange", "Wide-flange H sections, stiffeners, bolted end plates"),
               ("C", "Lattice", "Laced twin channels, face gussets, support rings")]
LEVEL_ITEMS = [("1", "1", "Base plates only"),
               ("2", "2", "Plus structural connections: nodes, end plates, gussets, brace crossings"),
               ("3", "3", "Plus small parts: straps, rods, rings, spacer blocks, rivets")]
HEAD_ITEMS = [("hex", "Hex + Washer", ""), ("hex_plain", "Hex", ""), ("square", "Square", ""),
              ("rivet", "Rivet", "")]
BRACE_ITEMS = [("0", "Auto", "Large X on the bottom row, Skinny X on top, Z between"),
               ("1", "Large X", "Heavy diagonal pair"),
               ("2", "Skinny X", "Light tension bracing"),
               ("3", "Z", "Cranked Z support"),
               ("4", "Z Flipped", "Cranked Z support, mirrored"),
               ("5", "Open", "No brace")]


class HS_Settings(bpy.types.PropertyGroup):
    is_guide: BoolProperty(default=False)
    output: PointerProperty(type=bpy.types.Object)
    style: EnumProperty(name="Style", items=STYLE_ITEMS, default="B", update=_changed)
    section_w: FloatProperty(name="Section Width", default=3.5, min=0.8, soft_max=12.0, unit="LENGTH",
                             update=_changed, description="Target section width; spans snap to whole sections")
    section_h: FloatProperty(name="Section Height", default=3.5, min=0.8, soft_max=12.0, unit="LENGTH",
                             update=_changed, description="Target section height; spans snap to whole sections")
    main: FloatProperty(name="Main", default=0.46, min=0.1, soft_max=1.5, unit="LENGTH", update=_changed,
                        description="Column and beam size")
    secondary: FloatProperty(name="Secondary", default=0.24, min=0.06, soft_max=0.8, unit="LENGTH",
                             update=_changed, description="Large X and Z member size")
    skinny: FloatProperty(name="Skinny", default=0.2, min=0.05, soft_max=0.6, unit="LENGTH", update=_changed,
                          description="Skinny X strap / rod size")
    bevel_horizontal: BoolProperty(name="Beam Bevel", default=True, update=_changed,
                                   description="Chamfer the long edges of horizontal beams, including sloped spans")
    bevel_vertical: BoolProperty(name="Post Bevel", default=True, update=_changed,
                                 description="Chamfer the long edges of vertical posts, including leaning uprights")
    bolts: BoolProperty(name="Bolts", default=True, update=_changed, description="Model bolts at all")
    bolt_level: EnumProperty(name="Bolt Level", items=LEVEL_ITEMS, default="1", update=_changed)
    head_style: EnumProperty(name="Bolt Head", items=HEAD_ITEMS, default="hex", update=_changed)
    screw_object: PointerProperty(name="Custom Screw", type=bpy.types.Object, poll=_screw_poll, update=_changed,
                                  description="Use this mesh at enabled bolt positions, scaled to each bolt's width. "
                                              "Local +Z points outward; put its origin at the mounting surface. "
                                              "Generated screws use the scaffold steel material")
    plinth: FloatProperty(name="Plinth", default=0.9, min=0.0, soft_max=3.0, unit="LENGTH", update=_changed,
                          description="Height of the lowest beams above the base plates")
    overrun: FloatProperty(name="Column Overrun", default=0.3, min=0.0, soft_max=2.0, unit="LENGTH",
                           update=_changed, description="How far columns run past the top beams")
    base_plates: BoolProperty(name="Base Plates", default=True, update=_changed)
    caps: BoolProperty(name="Cap Plates", default=True, update=_changed)
    floor_brace: BoolProperty(name="Brace Floors", default=False, update=_changed,
                              description="Skinny X in the horizontal faces")
    adaptive: BoolProperty(name="Adaptive", default=False, update=_changed,
                           description="On: rows of beams slant to share a sloped edge evenly. "
                                       "Off: rows stay level and only the top or bottom chord slopes")
    watertight: BoolProperty(name="Watertight Bake", default=False,
                             description="Join touching parts into closed solids and verify edges, winding and vertices. "
                                         "Custom screws must be closed meshes")
    live: BoolProperty(name="Live Update", default=True,
                       description="Rebuild a moment after the guide stops changing")


class HS_OT_add(bpy.types.Operator):
    bl_idname = "heavy_scaffold.add"
    bl_label = "Add Scaffold"
    bl_description = "Add a guide box at the 3D cursor. Extrude its faces to grow the scaffold"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT" and context.active_object is not None:
            bpy.ops.object.mode_set(mode="OBJECT")
        w = h = 3.5
        me = bpy.data.meshes.new("ScaffoldGuide")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(w, w, h), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(0.0, 0.0, h / 2), verts=bm.verts)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("ScaffoldGuide", me)
        context.collection.objects.link(ob)
        ob.location = context.scene.cursor.location
        init_guide(ob)
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        context.view_layer.update()
        rebuild(ob)
        return {"FINISHED"}


class HS_OT_make_guide(bpy.types.Operator):
    bl_idname = "heavy_scaffold.make_guide"
    bl_label = "Use as Guide"
    bl_description = "Turn the active mesh into a scaffold guide"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH" and guide_of(ob) is None

    def execute(self, context):
        ob = context.active_object
        conflict = brace_attribute_conflict(ob.data)
        if conflict:
            self.report({"WARNING"}, conflict)
            return {"CANCELLED"}
        if ob.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        init_guide(ob)
        rebuild(ob)
        return {"FINISHED"}


class HS_OT_set_brace(bpy.types.Operator):
    bl_idname = "heavy_scaffold.set_brace"
    bl_label = "Set Brace"
    bl_description = "Brace type for the selected guide faces"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(items=BRACE_ITEMS, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return ob is not None and ob.type == "MESH" and ob.heavy_scaffold.is_guide

    def execute(self, context):
        ob = context.edit_object
        conflict = brace_attribute_conflict(ob.data)
        if conflict:
            self.report({"WARNING"}, conflict)
            return {"CANCELLED"}
        bm = bmesh.from_edit_mesh(ob.data)
        layer = bm.faces.layers.int.get(ATTR) or bm.faces.layers.int.new(ATTR)
        n = 0
        for f in bm.faces:
            if f.select:
                f[layer] = int(self.kind)
                n += 1
        bmesh.update_edit_mesh(ob.data)
        request_rebuild(ob)
        self.report({"INFO"}, f"{n} face(s) set")
        return {"FINISHED"}


class HS_OT_rebuild(bpy.types.Operator):
    bl_idname = "heavy_scaffold.rebuild"
    bl_label = "Rebuild"
    bl_description = "Rebuild the scaffold from the guide now"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        info = rebuild(guide_of(context.active_object))
        if info:
            self.report({"INFO"}, f"Scaffold: {info['tris']:,} tris in {info['seconds']:.2f}s")
        return {"FINISHED"}


class HS_OT_bake(bpy.types.Operator):
    bl_idname = "heavy_scaffold.bake"
    bl_label = "Bake to Mesh"
    bl_description = "Copy the scaffold into a plain mesh object for export and hide the guide"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return guide_of(context.active_object) is not None

    def execute(self, context):
        guide = guide_of(context.active_object)
        if context.mode == "EDIT_MESH":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.update()
        if rebuild(guide) is None:
            self.report({"WARNING"}, "The live scaffold could not be rebuilt for baking.")
            return {"CANCELLED"}
        context.view_layer.update()
        out = guide.heavy_scaffold.output
        if guide.heavy_scaffold.watertight:
            t = time.perf_counter()
            try:
                me = bake_watertight(guide)
            except ValueError as error:
                self.report({"WARNING"}, str(error))
                return {"CANCELLED"}
            self.report({"INFO"}, f"Watertight bake: {len(me.polygons):,} faces in {time.perf_counter() - t:.1f}s")
        else:
            me = out.data.copy()
        baked = bpy.data.objects.new(guide.name + "_Baked", me)
        (guide.users_collection or [context.scene.collection])[0].objects.link(baked)
        baked.matrix_world = out.matrix_world.copy()
        guide.hide_set(True)
        out.hide_set(True)
        out.hide_render = True
        for o in context.selected_objects:
            o.select_set(False)
        baked.select_set(True)
        context.view_layer.objects.active = baked
        return {"FINISHED"}


class HS_OT_edit_guide(bpy.types.Operator):
    bl_idname = "heavy_scaffold.edit_guide"
    bl_label = "Edit Guide"
    bl_description = "Select the guide and switch to Edit Mode"
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


class HS_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = DEFAULT_TAB
    bl_label = "Heavy Scaffold"

    def draw(self, context):
        layout = self.layout
        guide = guide_of(context.active_object)
        if guide is None:
            col = layout.column(align=True)
            col.scale_y = 1.4
            col.operator("heavy_scaffold.add", icon="MESH_CUBE")
            ob = context.active_object
            if ob is not None and ob.type == "MESH":
                layout.operator("heavy_scaffold.make_guide", icon="MESH_GRID")
            return
        cfg = guide.heavy_scaffold
        layout.row(align=True).prop(cfg, "style", expand=True)
        row = layout.row(align=True)
        row.prop(cfg, "section_w", text="W")
        row.prop(cfg, "section_h", text="H")
        layout.prop(cfg, "adaptive", icon="MOD_LATTICE" if cfg.adaptive else "ALIGN_JUSTIFY")
        row = layout.row(align=True)
        row.prop(cfg, "main")
        row.prop(cfg, "secondary", text="Sec")
        row.prop(cfg, "skinny", text="Skinny")
        if STYLES[cfg.style].has_frame_bevels:
            row = layout.row(align=True)
            row.prop(cfg, "bevel_horizontal")
            row.prop(cfg, "bevel_vertical")
        row = layout.row(align=True)
        row.prop(cfg, "bolts", text="Bolts", toggle=True, icon="CHECKBOX_HLT" if cfg.bolts else "CHECKBOX_DEHLT")
        sub = row.row(align=True)
        sub.active = cfg.bolts
        sub.prop(cfg, "bolt_level", expand=True)
        if cfg.bolts:
            layout.prop(cfg, "screw_object")
        header, body = layout.panel("HS_PT_braces", default_closed=False)
        header.label(text="Brace Selected Faces")
        if body:
            grid = body.grid_flow(row_major=True, columns=3, even_columns=True, align=True)
            for key, label, _ in BRACE_ITEMS:
                grid.operator("heavy_scaffold.set_brace", text=label).kind = key
            if context.mode != "EDIT_MESH":
                body.label(text="Tab into the guide to pick faces", icon="INFO")
        header, body = layout.panel("HS_PT_options", default_closed=True)
        header.label(text="Options")
        if body:
            col = body.column(align=True)
            col.prop(cfg, "plinth")
            col.prop(cfg, "overrun")
            if cfg.screw_object is None:
                body.prop(cfg, "head_style", text="Head")
            grid = body.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            grid.prop(cfg, "base_plates")
            grid.prop(cfg, "caps")
            grid.prop(cfg, "floor_brace")
            grid.prop(cfg, "live")
        row = layout.row(align=True)
        row.operator("heavy_scaffold.rebuild", icon="FILE_REFRESH")
        row.operator("heavy_scaffold.bake", icon="CHECKMARK")
        layout.prop(cfg, "watertight")
        if context.active_object is not guide:
            layout.operator("heavy_scaffold.edit_guide", icon="EDITMODE_HLT")
        info = _last_info.get(guide.name)
        if info:
            layout.label(text=f"{info['tris']:,} tris · {info['sections']} sections", icon="MESH_DATA")


def _register_panel(tab):
    try:
        bpy.utils.unregister_class(HS_PT_main)
    except RuntimeError:
        pass
    HS_PT_main.bl_category = (tab or "").strip() or DEFAULT_TAB
    bpy.utils.register_class(HS_PT_main)


def _tab_changed(self, context):
    _register_panel(self.tab_name)


class HS_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    tab_name: StringProperty(name="Sidebar Tab", default=DEFAULT_TAB, update=_tab_changed,
                             description="Sidebar (N) tab the panel lives in. Type an existing tab to share it")

    def draw(self, context):
        self.layout.prop(self, "tab_name")


CLASSES = (HS_Preferences, HS_Settings, HS_OT_add, HS_OT_make_guide, HS_OT_set_brace, HS_OT_rebuild,
           HS_OT_bake, HS_OT_edit_guide)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.heavy_scaffold = PointerProperty(type=HS_Settings)
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
    _last_info.clear()
    try:
        bpy.utils.unregister_class(HS_PT_main)
    except RuntimeError:
        pass
    del bpy.types.Object.heavy_scaffold
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
