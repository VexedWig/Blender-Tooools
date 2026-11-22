"""Dense UV pipeline: unwrap, straighten, equalize, stack, then pack with a small search.

Everything runs in one multi-object Edit Mode session. Blender's own operators do the
unwrapping and the packing; this module decides what they act on (face selection) and
does the island work in between with bmesh.
"""
import math
import time
from array import array
from collections import defaultdict, deque
from functools import lru_cache

import bmesh
import bpy
from bpy_extras.bmesh_utils import bmesh_linked_uv_islands
from mathutils import Matrix, Vector
from mathutils.geometry import delaunay_2d_cdt
from mathutils.kdtree import KDTree

TINY = 1e-12
# Gap the unwrappers leave between islands, so islands never share UV coordinates by accident.
UNWRAP_MARGIN = 0.002

PACK_VARIANTS = {
    'FAST': [('CONCAVE', 'ANY')],
    'BALANCED': [('CONCAVE', 'ANY'), ('CONCAVE', 'AXIS_ALIGNED'), ('CONCAVE', 'CARDINAL')],
    # More packing strategies never beat these three in testing; Best spends its time on
    # trying each object with and without straightening instead.
    'BEST': [('CONCAVE', 'ANY'), ('CONCAVE', 'AXIS_ALIGNED'), ('CONCAVE', 'CARDINAL')],
}
PACK_SHAPES_NO_ROTATION = {
    'FAST': ['CONCAVE'],
    'BALANCED': ['CONCAVE', 'CONVEX'],
    'BEST': ['CONCAVE', 'CONVEX', 'AABB'],
}


class DenseUVError(Exception):
    pass


# --------------------------------------------------------------------------- helpers

def view3d_override(context):
    """Context override for a 3D Viewport, or {} when there is none (background mode)."""
    wm = context.window_manager
    windows = ([context.window] if context.window else []) + list(wm.windows if wm else [])
    for win in windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region:
                    return {'window': win, 'screen': win.screen, 'area': area, 'region': region}
    return {}


def signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        a += p[0] * q[1] - q[0] * p[1]
    return 0.5 * a


def newell_area(cos):
    nx = ny = nz = 0.0
    n = len(cos)
    for i in range(n):
        a, b = cos[i], cos[(i + 1) % n]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def face_uv_area(f, uv):
    return signed_area([l[uv].uv for l in f.loops])


def uv_key(l, uv):
    """Identity of a UV vertex: mesh vertex plus its UV position (split at seams)."""
    p = l[uv].uv
    return (l.vert.index, round(p.x, 5), round(p.y, 5))


def bounds(faces, uv):
    umin = vmin = math.inf
    umax = vmax = -math.inf
    for f in faces:
        for l in f.loops:
            p = l[uv].uv
            if p.x < umin: umin = p.x
            if p.x > umax: umax = p.x
            if p.y < vmin: vmin = p.y
            if p.y > vmax: vmax = p.y
    return umin, vmin, umax, vmax


def transform(faces, uv, fn):
    for f in faces:
        for l in f.loops:
            l[uv].uv = fn(l[uv].uv)


def distortion(faces, uv, areas):
    """Area-weighted stretch (0 = none) and the fraction of 3D area whose UVs are flipped."""
    data = []
    t3 = tu = total = 0.0
    for f in faces:
        a3 = areas[f.index]
        au = face_uv_area(f, uv)
        data.append((a3, au))
        t3 += a3
        tu += abs(au)
        total += au
    if t3 <= TINY:
        return 0.0, 0.0
    if tu <= TINY:
        return 99.0, 0.0
    k = t3 / tu
    sign = 1.0 if total >= 0.0 else -1.0
    d = flipped = 0.0
    for a3, au in data:
        if a3 <= t3 * 1e-7:
            continue
        if au * sign < 0.0:
            flipped += a3
        r = abs(au) * k / a3
        d += a3 * (min(4.0, max(r, 1.0 / r) - 1.0) if r > 1e-9 else 4.0)
    return d / t3, flipped / t3


# --------------------------------------------------------------------------- data

class Job:
    """One mesh in Edit Mode, with everything needed to put its state back afterwards."""

    def __init__(self, obj):
        self.obj = obj
        self.me = obj.data
        self.had_uvs = len(obj.data.uv_layers) > 0
        self.bm = bmesh.from_edit_mesh(self.me)
        bm = self.bm
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
        bm.faces.ensure_lookup_table()
        self.uv = bm.loops.layers.uv.verify()
        # A flat buffer keeps the rollback copy to eight bytes per face corner.
        self.original_uvs = (array('f', (value for face in bm.faces for loop in face.loops
                                         for value in loop[self.uv].uv)) if self.had_uvs else None)

        self.saved_sel = ([v.select for v in bm.verts], [e.select for e in bm.edges],
                          [f.select for f in bm.faces])
        self.saved_history = list(bm.select_history)
        self.active_face = bm.faces.active
        self.saved_hide = ([v.hide for v in bm.verts], [e.hide for e in bm.edges],
                           [f.hide for f in bm.faces])
        self.saved_seams = [e.seam for e in bm.edges]
        self.saved_select_mode = set(bm.select_mode)
        for f in bm.faces:
            f.hide_set(False)
        self.has_sel = True

        mw = obj.matrix_world
        self.wco = [mw @ v.co for v in bm.verts]
        self.area_w = [newell_area([self.wco[v.index] for v in f.verts]) for f in bm.faces]

        # Loose parts (vertex-connected pieces) and a size signature per part. Only islands on
        # identical copies of a part may stack.
        part_of_vert = [-1] * len(bm.verts)
        n_parts = 0
        for v in bm.verts:
            if part_of_vert[v.index] >= 0:
                continue
            part_of_vert[v.index] = n_parts
            todo = [v]
            while todo:
                x = todo.pop()
                for e in x.link_edges:
                    y = e.other_vert(x)
                    if part_of_vert[y.index] < 0:
                        part_of_vert[y.index] = n_parts
                        todo.append(y)
            n_parts += 1
        self.part_of_face = [part_of_vert[f.verts[0].index] for f in bm.faces]
        self.part_of_vert = part_of_vert
        self.part_shapes = None
        faces_in = [0] * n_parts
        verts_in = [0] * n_parts
        area_in = [0.0] * n_parts
        for f in bm.faces:
            p = self.part_of_face[f.index]
            faces_in[p] += 1
            area_in[p] += self.area_w[f.index]
        for p in part_of_vert:
            verts_in[p] += 1
        self.part_sig = [(faces_in[p], verts_in[p], f"{area_in[p]:.4g}") for p in range(n_parts)]

        self.mode = 'KEEP'
        self.islands = []
        self.stats = defaultdict(int)

    def restore(self, keep_seams, completed):
        bm = self.bm
        if not completed:
            if self.original_uvs is None:
                bm.loops.layers.uv.remove(self.uv)
            else:
                coords = iter(self.original_uvs)
                for face in bm.faces:
                    for loop in face.loops:
                        loop[self.uv].uv = (next(coords), next(coords))
        vs, es, fs = self.saved_hide
        for v, h in zip(bm.verts, vs): v.hide = h
        for e, h in zip(bm.edges, es): e.hide = h
        for f, h in zip(bm.faces, fs): f.hide = h
        vs, es, fs = self.saved_sel
        for v, s in zip(bm.verts, vs): v.select = s
        for e, s in zip(bm.edges, es): e.select = s
        for f, s in zip(bm.faces, fs): f.select = s
        if not completed or not keep_seams:
            for e, s in zip(bm.edges, self.saved_seams): e.seam = s
        bm.select_mode = self.saved_select_mode
        bm.select_history.clear()
        for element in self.saved_history:
            if element.is_valid and element.select:
                bm.select_history.add(element)
        bm.faces.active = self.active_face
        if hasattr(bm, "uv_select_sync_valid"):
            bm.uv_select_sync_valid = False
        bmesh.update_edit_mesh(self.me, loop_triangles=False, destructive=False)


class Island:
    __slots__ = ("job", "faces", "area_w", "dup_of", "pairs", "pts", "reps", "frame", "kd")

    def __init__(self, job, faces):
        self.job = job
        self.faces = faces
        self.area_w = sum(job.area_w[f.index] for f in faces)
        self.dup_of = None
        self.pairs = None

    def uv_area(self):
        uv = self.job.uv
        return sum(abs(face_uv_area(f, uv)) for f in self.faces)


def select_only(jobs, faces_by_job):
    """Make exactly these faces selected (sync selection) across all jobs."""
    for j in jobs:
        want = faces_by_job.get(j)
        if want is None and not j.has_sel:
            continue
        bm = j.bm
        for v in bm.verts: v.select = False
        for e in bm.edges: e.select = False
        for f in bm.faces: f.select = False
        j.has_sel = False
        if want:
            for f in want:
                f.select_set(True)
            j.has_sel = True
        if hasattr(bm, "uv_select_sync_valid"):
            bm.uv_select_sync_valid = False
        bmesh.update_edit_mesh(j.me, loop_triangles=False, destructive=False)


# --------------------------------------------------------------------------- unwrap

def mark_hybrid_seams(job, s):
    edges = job.bm.edges
    # FBX round trips often mark every edge sharp; then the marks carry no information.
    use_sharp = s.seams_from_sharp and sum(1 for e in edges if not e.smooth) < 0.9 * len(edges)
    for e in edges:
        lf = e.link_faces
        if len(lf) != 2:
            e.seam = len(lf) > 2
            continue
        e.seam = ((s.keep_existing_seams and e.seam)
                  or (use_sharp and not e.smooth)
                  or (s.seams_from_materials and lf[0].material_index != lf[1].material_index)
                  or e.calc_face_angle(0.0) > s.seam_angle)


def unwrap_phase(jobs, s, ov):
    # Blender unwraps in local space. On objects with unapplied non-uniform scale that bakes
    # the stretch into the UVs, so unwrap a temporarily world-proportioned copy of the shape.
    shaped = {}
    try:
        for j in jobs:
            if j.mode == 'KEEP':
                continue
            m = j.obj.matrix_world.to_3x3()
            sc = [abs(x) for x in m.to_scale()]
            if max(sc) <= min(sc) * 1.0005:
                continue  # uniform scale, negative or not, changes no proportions
            if m.determinant() < 0.0:
                # A mirroring transform would flip every island. Unwrap a mirror image with
                # the same proportions instead, so the UVs keep the mesh's own winding.
                m = m @ Matrix.Diagonal((-1.0, 1.0, 1.0))
            shaped[j] = [v.co.copy() for v in j.bm.verts]
            for v in j.bm.verts:
                v.co = m @ v.co
            bmesh.update_edit_mesh(j.me, loop_triangles=False, destructive=False)
        _unwrap(jobs, s, ov)
    finally:
        for j, cos in shaped.items():
            for v, co in zip(j.bm.verts, cos):
                v.co = co
            bmesh.update_edit_mesh(j.me, loop_triangles=False, destructive=False)


def _unwrap(jobs, s, ov):
    hybrid = [j for j in jobs if j.mode == 'HYBRID']
    smart = [j for j in jobs if j.mode == 'SMART']
    redo = {}

    if hybrid:
        for j in hybrid:
            mark_hybrid_seams(j, s)
        select_only(jobs, {j: list(j.bm.faces) for j in hybrid})
        with bpy.context.temp_override(**ov):
            bpy.ops.uv.unwrap(method=s.unwrap_method, fill_holes=True, correct_aspect=False,
                              margin_method='FRACTION', margin=UNWRAP_MARGIN)
        # Overlap check: give every object its own far-apart UV tile first, so only
        # self-overlaps inside one object's fresh unwrap get flagged.
        overlapping = {}
        saved = {j: [l[j.uv].uv.copy() for f in j.bm.faces for l in f.loops] for j in jobs}
        for n, j in enumerate(jobs):
            off = Vector((4.0 * (n % 8), 4.0 * (n // 8))) if j.mode == 'HYBRID' else Vector((-500.0, -500.0))
            transform(j.bm.faces, j.uv, lambda p, off=off: p + off)
        try:
            with bpy.context.temp_override(**ov):
                bpy.ops.uv.select_overlap(extend=False)
            for j in hybrid:
                overlapping[j] = {f.index for f in j.bm.faces if f.select}
        except RuntimeError:
            pass
        for j in jobs:
            j.has_sel = True
            for (l, p) in zip((l for f in j.bm.faces for l in f.loops), saved[j]):
                l[j.uv].uv = p
        del saved
        for j in hybrid:
            for e, seam in zip(j.bm.edges, j.saved_seams):
                e.seam = seam
            bad = []
            for faces in bmesh_linked_uv_islands(j.bm, j.uv):
                d, flipped = distortion(faces, j.uv, j.area_w)
                over = overlapping.get(j)
                if (d > s.max_distortion or flipped > 0.002
                        or (over and any(f.index in over for f in faces))):
                    bad.extend(faces)
            if bad:
                redo[j] = bad
                j.stats["fallback_faces"] = len(bad)

    for j in smart:
        redo[j] = list(j.bm.faces)
    if redo:
        select_only(jobs, redo)
        with bpy.context.temp_override(**ov):
            bpy.ops.uv.smart_project(angle_limit=s.smart_angle, margin_method='FRACTION',
                                     island_margin=UNWRAP_MARGIN, area_weight=0.0,
                                     correct_aspect=False, scale_to_bounds=False)


# --------------------------------------------------------------------------- rectify

def oriented_aspect(pts):
    """Length over width of a point set, measured along its principal axes."""
    if len(pts) < 3:
        return 1.0
    cx, cy, ang, _ = _frame([(p.x, p.y) for p in pts])
    c, s = math.cos(ang), math.sin(ang)
    xs = [(p.x - cx) * c + (p.y - cy) * s for p in pts]
    ys = [-(p.x - cx) * s + (p.y - cy) * c for p in pts]
    a, b = max(xs) - min(xs), max(ys) - min(ys)
    lo, hi = min(a, b), max(a, b)
    return hi / lo if lo > TINY else 1e6


def quad_corner_deviation(faces, wco):
    """Mean and max deviation (radians) of the 3D quad corners from a right angle."""
    total = 0.0
    worst = 0.0
    n = 0
    for f in faces:
        vs = [wco[v.index] for v in f.verts]
        for i in range(4):
            a = vs[i - 1] - vs[i]
            b = vs[(i + 1) % 4] - vs[i]
            if a.length <= TINY or b.length <= TINY:
                return math.pi, math.pi
            dev = abs(a.angle(b) - math.pi / 2)
            total += dev
            worst = max(worst, dev)
            n += 1
    return total / n, worst


def rectify_island(isl, tolerance):
    """Lay an all-quad island out as a grid of rectangles (edge rings keep their 3D length).

    Gives up on anything that is not a clean grid. Returns True when the island changed.
    """
    job = isl.job
    uv, wco = job.uv, job.wco
    faces = isl.faces
    if any(len(f.loops) != 4 for f in faces):
        return False
    mean_dev, max_dev = quad_corner_deviation(faces, wco)
    if mean_dev > tolerance or max_dev > 2.0 * tolerance:
        return False

    # Edge rings: opposite edges of every quad share a ring and get the ring's average length.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for f in faces:
        e = [l.edge.index for l in f.loops]
        for a, b in ((e[0], e[2]), (e[1], e[3])):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    sums = defaultdict(float)
    counts = defaultdict(int)
    seen = set()
    for f in faces:
        for l in f.loops:
            ed = l.edge
            if ed.index in seen:
                continue
            seen.add(ed.index)
            r = find(ed.index)
            sums[r] += (wco[ed.verts[0].index] - wco[ed.verts[1].index]).length
            counts[r] += 1
    ring = {r: sums[r] / counts[r] for r in sums}

    def length(edge):
        return ring[find(edge.index)]

    keys = {l: uv_key(l, uv) for f in faces for l in f.loops}
    fset = set(faces)
    start = max(faces, key=lambda f: job.area_w[f.index])
    sl = start.loops
    w, h = length(sl[0].edge), length(sl[1].edge)
    if w <= TINY or h <= TINY:
        return False
    tol = 1e-3 * (sum(ring.values()) / len(ring))
    pos = {}

    def assign(k, p):
        q = pos.get(k)
        if q is None:
            pos[k] = p
            return True
        return (q - p).length <= tol

    corners = {start: [Vector((0.0, 0.0)), Vector((w, 0.0)), Vector((w, h)), Vector((0.0, h))]}
    for l, p in zip(sl, corners[start]):
        assign(keys[l], p)
    visited = {start}
    queue = deque([start])
    while queue:
        f = queue.popleft()
        fl = f.loops
        c = corners[f]
        centre = (c[0] + c[1] + c[2] + c[3]) * 0.25
        for i in range(4):
            la, lb = fl[i], fl[(i + 1) % 4]
            for g in la.edge.link_faces:
                if g is f or g in visited or g not in fset:
                    continue
                gl = g.loops
                ga = gb = None
                for j in range(4):
                    if gl[j].vert is la.vert: ga = j
                    elif gl[j].vert is lb.vert: gb = j
                if ga is None or gb is None:
                    continue
                if keys[gl[ga]] != keys[la] or keys[gl[gb]] != keys[lb]:
                    continue  # UV seam between f and g inside the island
                pa, pb = c[i], c[(i + 1) % 4]
                d = pb - pa
                n = Vector((-d.y, d.x))
                if n.length <= TINY:
                    return False
                n.normalize()
                if n.dot(centre - (pa + pb) * 0.5) > 0.0:
                    n.negate()
                if (ga + 1) % 4 == gb:
                    nb, na = (gb + 1) % 4, (ga + 3) % 4
                else:
                    nb, na = (gb + 3) % 4, (ga + 1) % 4
                side = gl[gb].edge if nb == (gb + 1) % 4 else gl[nb].edge
                step = n * length(side)
                gc = [None] * 4
                gc[ga], gc[gb] = pa.copy(), pb.copy()
                gc[nb], gc[na] = pb + step, pa + step
                for j in range(4):
                    if not assign(keys[gl[j]], gc[j]):
                        return False
                corners[g] = gc
                visited.add(g)
                queue.append(g)
    if len(visited) != len(faces):
        return False

    old = [(l, l[uv].uv.copy()) for f in faces for l in f.loops]
    d0, f0 = distortion(faces, uv, job.area_w)
    a0 = oriented_aspect([p for _, p in old])
    # Re-centre on the old island so straightened neighbours never share coordinates.
    def centre(points):
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5))
    shift = centre([p for _, p in old]) - centre(list(pos.values()))
    for k in pos:
        pos[k] = pos[k] + shift
    for l, _ in old:
        l[uv].uv = pos[keys[l]]
    d1, f1 = distortion(faces, uv, job.area_w)
    a1 = oriented_aspect(list(pos.values()))
    # A curved strip (cable, bent pipe) straightened becomes a long thin bar, and the longest
    # island caps how far the whole layout can scale up. Keep the curve in that case.
    if d1 > d0 + 0.08 or f1 > f0 + 1e-6 or a1 > max(3.0, a0 * 1.15):
        for l, p in old:
            l[uv].uv = p
        return False
    return True


# --------------------------------------------------------------------------- equalize

def equalize(islands):
    """Scale every island so UV area per world-space square metre is the same."""
    for isl in islands:
        au = isl.uv_area()
        if isl.area_w <= TINY or au <= TINY:
            continue
        k = math.sqrt(isl.area_w / au)
        u0, v0, u1, v1 = bounds(isl.faces, isl.job.uv)
        cx, cy = (u0 + u1) * 0.5, (v0 + v1) * 0.5
        transform(isl.faces, isl.job.uv,
                  lambda p: Vector(((p.x - cx) * k + cx, (p.y - cy) * k + cy)))


# --------------------------------------------------------------------------- stacking

def _frame(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    sxx = syy = sxy = 0.0
    for x, y in pts:
        dx, dy = x - cx, y - cy
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    trace = sxx + syy
    aniso = math.sqrt((sxx - syy) ** 2 + 4.0 * sxy * sxy) / trace if trace > TINY else 0.0
    return cx, cy, 0.5 * math.atan2(2.0 * sxy, sxx - syy), aniso


def _prepare_shape(isl):
    uv = isl.job.uv
    pts, reps, seen = [], [], set()
    for f in isl.faces:
        for l in f.loops:
            k = uv_key(l, uv)
            if k not in seen:
                seen.add(k)
                pts.append((l[uv].uv.x, l[uv].uv.y))
                reps.append(l)
    isl.pts, isl.reps = pts, reps
    isl.frame = _frame(pts)
    isl.kd = None


def _candidate_angles(prim, pts, frame):
    cx, cy, ang, aniso = frame
    if min(prim.frame[3], aniso) > 0.05:
        base = prim.frame[2] - ang
        return [base, base + math.pi]
    # Nearly isotropic (squares, round caps): align the farthest points instead.
    def far(points, c):
        d = [math.hypot(x - c[0], y - c[1]) for x, y in points]
        m = max(d)
        return [math.atan2(points[i][1] - c[1], points[i][0] - c[0])
                for i in range(len(points)) if d[i] >= m * 0.999]
    target = far(prim.pts, prim.frame)[0]
    return [target - a for a in far(pts, frame)[:16]]


def _match(prim, dup, tol, mirror):
    pts = [(-x, y) for x, y in dup.pts] if mirror else dup.pts
    frame = _frame(pts) if mirror else dup.frame
    if prim.kd is None:
        prim.kd = KDTree(len(prim.pts))
        for i, (x, y) in enumerate(prim.pts):
            prim.kd.insert((x, y, 0.0), i)
        prim.kd.balance()
    pcx, pcy = prim.frame[0], prim.frame[1]
    cx, cy = frame[0], frame[1]
    for th in _candidate_angles(prim, pts, frame):
        c, s = math.cos(th), math.sin(th)
        mapping = []
        for x, y in pts:
            x0, y0 = x - cx, y - cy
            _, idx, dist = prim.kd.find((c * x0 - s * y0 + pcx, s * x0 + c * y0 + pcy, 0.0))
            if dist > tol:
                break
            mapping.append(idx)
        else:
            if len(set(mapping)) == len(mapping):
                return mapping
    return None


def _cycle(indices):
    """A face boundary independent of starting corner and winding."""
    i = indices.index(min(indices))
    seq = indices[i:] + indices[:i]
    return min(seq, seq[:1] + seq[:0:-1])


class PartShape:
    """World-space geometry and connectivity used to verify a candidate copy."""

    def __init__(self):
        self.points = {}
        self.edges = set()
        self.faces = set()
        self.tree = None

    def corners(self):
        for face in sorted(self.faces):
            for i, vertex in enumerate(face):
                origin = self.points[vertex]
                yield (len(face), origin, self.points[face[i - 1]] - origin,
                       self.points[face[(i + 1) % len(face)]] - origin)

    def matches(self, other, mirrored):
        if len(self.edges) != len(other.edges):
            return False
        # A non-collinear corner determines a rigid coordinate frame. Try its
        # possible counterparts, then verify every vertex, edge and face.
        n, origin, left, right = max(self.corners(), key=lambda c: c[2].cross(c[3]).length_squared)
        normal = left.cross(right)
        if normal.length_squared == 0:
            return False
        x, z = left.normalized(), normal.normalized()
        y = z.cross(x)
        local = {}
        for i, p in self.points.items():
            delta = p - origin
            local[i] = (delta.dot(x), delta.dot(y), delta.dot(z))
        tol = max(1e-8, 1e-5 * max((p - origin).length for p in self.points.values()))
        if other.tree is None:
            other.tree = KDTree(len(other.points))
            for i, p in other.points.items():
                other.tree.insert(p, i)
            other.tree.balance()
        for face_size, target, a, b in other.corners():
            if face_size != n:
                continue
            for u, v in ((a, b), (b, a)):
                if abs(u.length - left.length) > tol or abs(v.length - right.length) > tol:
                    continue
                tz = u.cross(v).normalized()
                tx = u.normalized()
                ty = tz.cross(tx)
                if abs(v.dot(tx) - right.dot(x)) > tol:
                    continue
                for sign in ((1, -1) if mirrored else (1,)):
                    mapping = {}
                    for i, (px, py, pz) in local.items():
                        _, idx, distance = other.tree.find(target + tx * px + ty * py + tz * (sign * pz))
                        if distance > tol:
                            break
                        mapping[i] = idx
                    else:
                        if (len(set(mapping.values())) == len(mapping)
                                and all(tuple(sorted((mapping[i], mapping[j]))) in other.edges for i, j in self.edges)
                                and all(_cycle(tuple(mapping[i] for i in face)) in other.faces for face in self.faces)):
                            return True
        return False


def part_shape(job, part):
    if job.part_shapes is None:
        shapes = defaultdict(PartShape)
        for i, point in enumerate(job.wco):
            shapes[job.part_of_vert[i]].points[i] = point
        for edge in job.bm.edges:
            a, b = (v.index for v in edge.verts)
            shapes[job.part_of_vert[a]].edges.add(tuple(sorted((a, b))))
        for face in job.bm.faces:
            shapes[job.part_of_face[face.index]].faces.add(_cycle(tuple(v.index for v in face.verts)))
        job.part_shapes = shapes
    return job.part_shapes[part]


def stack_similar(islands, allow_mirrored, copies_only=True):
    """Find islands with the same shape and size; mark all but one as copies.

    copies_only: stack only islands on duplicated loose parts, never two faces of one part.
    """
    @lru_cache(maxsize=None)
    def same_shape(a, b):
        return part_shape(*a).matches(part_shape(*b), allow_mirrored)

    buckets = defaultdict(list)
    for isl in islands:
        if isl.area_w <= TINY:
            continue
        nq = sum(1 for f in isl.faces if len(f.loops) == 4)
        nt = sum(1 for f in isl.faces if len(f.loops) == 3)
        key = (len(isl.faces), nq, nt)
        if copies_only:
            # Same-shaped faces of different parts are not copies: compare only islands whose
            # loose parts match (face count, vertex count, surface area).
            key += (isl.job.part_sig[isl.job.part_of_face[isl.faces[0].index]],)
        buckets[key].append(isl)
    stacked = 0
    for group in buckets.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda i: i.area_w)
        prims = []
        # In Copies Only, a stack may contain at most one island from each loose part.
        occupied_parts = defaultdict(set)
        for isl in group:
            _prepare_shape(isl)
            tol = 2e-3 * math.sqrt(isl.uv_area()) + 1e-7
            part = (isl.job, isl.job.part_of_face[isl.faces[0].index])
            for prim in reversed(prims):
                if isl.area_w > prim.area_w * 1.002:
                    break
                if len(prim.pts) != len(isl.pts):
                    continue
                if copies_only:
                    source = (prim.job, prim.job.part_of_face[prim.faces[0].index])
                    if part in occupied_parts[prim] or not same_shape(source, part):
                        continue
                mapping = _match(prim, isl, tol, False)
                if mapping is None and allow_mirrored:
                    mapping = _match(prim, isl, tol, True)
                if mapping is None:
                    continue
                prim_uv, uv = prim.job.uv, isl.job.uv
                by_key = {uv_key(l, uv): prim.reps[m] for l, m in zip(isl.reps, mapping)}
                isl.pairs = [(l, by_key[uv_key(l, uv)]) for f in isl.faces for l in f.loops]
                isl.dup_of = prim
                if copies_only:
                    occupied_parts[prim].add(part)
                stacked += 1
                break
            if isl.dup_of is None:
                prims.append(isl)
                if copies_only:
                    occupied_parts[isl].add(part)
    return stacked


# --------------------------------------------------------------------------- packing

def _measure(face_list):
    tot = 0.0
    umin = vmin = math.inf
    umax = vmax = -math.inf
    for f, uv in face_list:
        pts = [l[uv].uv for l in f.loops]
        tot += abs(signed_area(pts))
        for p in pts:
            if p.x < umin: umin = p.x
            if p.x > umax: umax = p.x
            if p.y < vmin: vmin = p.y
            if p.y > vmax: vmax = p.y
    return tot, (umin, vmin, umax, vmax)


def _island_spans(prims, layout):
    """Per island: (first loop index, loop count, centre x, y, principal angle, length, width)."""
    spans = []
    idx = 0
    for isl in prims:
        n = sum(len(f.loops) for f in isl.faces)
        pts = layout[idx:idx + n]
        cx, cy, ang, _ = _frame([(p.x, p.y) for p in pts])
        c, s = math.cos(ang), math.sin(ang)
        xs = [(p.x - cx) * c + (p.y - cy) * s for p in pts]
        ys = [-(p.x - cx) * s + (p.y - cy) * c for p in pts]
        spans.append((idx, n, cx, cy, ang, max(xs) - min(xs), max(ys) - min(ys)))
        idx += n
    return spans


def _diagonal_layout(prims, layout):
    """Copy of `layout` with the long thin islands turned to 45 degrees, or None.

    The longest island caps how far a layout can scale. Laid along the tile's diagonal it
    gets ~41% more length, but Blender's packer rarely tries that by itself. Packing this
    layout with 90-degree steps keeps those islands diagonal.
    """
    spans = _island_spans(prims, layout)
    longest = max(sp[5] for sp in spans)
    targets = [sp for sp in spans if sp[5] >= 0.5 * longest and sp[5] > 6.0 * max(sp[6], TINY)]
    if not targets:
        return None
    out = list(layout)
    for idx, n, cx, cy, ang, _, _ in targets:
        th = math.pi / 4 - ang
        c, s = math.cos(th), math.sin(th)
        for i in range(idx, idx + n):
            p = layout[i]
            x, y = p.x - cx, p.y - cy
            out[i] = Vector((c * x - s * y + cx, s * x + c * y + cy))
    return out


def pack_scope(jobs, islands, s, ov):
    """Pack the primary islands of one texture sheet. Returns the filled fraction (0..1)."""
    prims = [i for i in islands if i.dup_of is None]
    if not prims:
        return 0.0
    aspect = int(s.texture_width) / int(s.texture_height)
    square = aspect == 1.0

    # Starting layout, in square-texel space inside a box of aspect x 1. Blender's packer does
    # far worse when its input islands are tiny, so scale them to ~70% of the box area. They
    # may overlap (the packer finds islands by connectivity, not position); scattering them
    # keeps neighbours that were cut apart from sharing UV coordinates and fusing.
    face_list = [(f, i.job.uv) for i in prims for f in i.faces]
    loops = [(l, uv) for f, uv in face_list for l in f.loops]
    boxes = [bounds(i.faces, i.job.uv) for i in prims]
    total = sum(i.uv_area() for i in prims)
    k = math.sqrt(0.7 * aspect / total) if total > TINY else 1.0
    if not square:
        # ORIGINAL_AABB needs every island inside the box. (Square sheets skip this: on long
        # strips the clamp would shrink everything and the packer copes badly.)
        widest = max(b[2] - b[0] for b in boxes)
        tallest = max(b[3] - b[1] for b in boxes)
        if widest > TINY:
            k = min(k, 0.98 * aspect / widest)
        if tallest > TINY:
            k = min(k, 0.98 / tallest)
    placed = []
    for idx, (isl, b) in enumerate(zip(prims, boxes)):
        w, h = (b[2] - b[0]) * k, (b[3] - b[1]) * k
        # R2 low-discrepancy sequence: even, deterministic scatter.
        ox = ((0.5 + idx * 0.7548776662) % 1.0) * (aspect - w)
        oy = ((0.5 + idx * 0.5698402910) % 1.0) * (1.0 - h)
        transform(isl.faces, isl.job.uv,
                  lambda p, b=b, ox=ox, oy=oy: Vector(((p.x - b[0]) * k + ox, (p.y - b[1]) * k + oy)))
        placed.append((ox, oy, ox + w, oy + h))
    if not square:
        # The packer fills the islands' starting bounding box; make it exactly aspect x 1.
        for axis, edge, target in ((0, 0, 0.0), (1, 1, 0.0), (0, 2, aspect), (1, 3, 1.0)):
            pick = (min if edge < 2 else max)(range(len(prims)), key=lambda i: placed[i][edge])
            delta = target - placed[pick][edge]
            shift = Vector((delta, 0.0)) if axis == 0 else Vector((0.0, delta))
            transform(prims[pick].faces, prims[pick].job.uv, lambda p, d=shift: p + d)
            x0, y0, x1, y1 = placed[pick]
            placed[pick] = (x0 + shift.x, y0 + shift.y, x1 + shift.x, y1 + shift.y)

    start = [l[uv].uv.copy() for l, uv in loops]
    faces_by_job = defaultdict(list)
    for i in prims:
        faces_by_job[i.job].extend(i.faces)
    select_only(jobs, faces_by_job)
    margin = s.padding / (2.0 * int(s.texture_height))

    def attempt(shape, rot, layout):
        """Pack from `layout`; return (filled area, resulting UVs) or None."""
        for (l, uv), p in zip(loops, layout):
            l[uv].uv = p
        for j in faces_by_job:
            bmesh.update_edit_mesh(j.me, loop_triangles=False, destructive=False)
        with bpy.context.temp_override(**ov):
            # 'ADD' gives the same exact margin as 'FRACTION' and runs ~15x faster.
            bpy.ops.uv.pack_islands(udim_source='CLOSEST_UDIM' if square else 'ORIGINAL_AABB',
                                    rotate=rot is not None, rotate_method=rot or 'ANY',
                                    scale=True, merge_overlap=False, margin_method='ADD',
                                    margin=margin, pin=False, shape_method=shape)
        area, (u0, v0, u1, v1) = _measure(face_list)
        eps = 1e-4
        if square:
            # Oversized starting islands can make the packer pick another UDIM tile.
            tu, tv = math.floor(u0 + eps), math.floor(v0 + eps)
            if tu or tv:
                for l, uv in loops:
                    p = l[uv].uv
                    l[uv].uv = (p.x - tu, p.y - tv)
                u0, u1, v0, v1 = u0 - tu, u1 - tu, v0 - tv, v1 - tv
        if u0 < -eps or v0 < -eps or u1 > aspect + eps or v1 > 1.0 + eps:
            # Islands rotated to odd angles can poke slightly past the border. Shrink the
            # layout to fit inside the padding instead of throwing the result away.
            f = min((aspect - 2.0 * margin) / (u1 - u0), (1.0 - 2.0 * margin) / (v1 - v0))
            if f < 0.85:
                return None
            for l, uv in loops:
                p = l[uv].uv
                l[uv].uv = ((p.x - u0) * f + margin, (p.y - v0) * f + margin)
            area *= f * f
        return area, [l[uv].uv.copy() for l, uv in loops]

    if s.allow_rotation:
        variants = PACK_VARIANTS[s.quality]
    else:
        variants = [(shape, None) for shape in PACK_SHAPES_NO_ROTATION[s.quality]]
    best = None
    for shape, rot in variants:
        result = attempt(shape, rot, start)
        if result and (best is None or result[0] > best[0]):
            best = result
    if best and square and s.allow_rotation and s.quality != 'FAST':
        # When the longest island spans the tile it caps the scale; try it on the diagonal.
        if max(sp[5] for sp in _island_spans(prims, best[1])) > 0.85:
            diagonal = _diagonal_layout(prims, start)
            result = attempt('CONCAVE', 'CARDINAL', diagonal) if diagonal else None
            if result and result[0] > best[0]:
                best = result
    if best is None:
        raise DenseUVError("The packer left islands outside the texture; try another quality")
    for (l, uv), p in zip(loops, best[1]):
        l[uv].uv = (p.x / aspect, p.y)
    return best[0] / aspect


def place_duplicates(islands, offset):
    for isl in islands:
        if isl.dup_of is None:
            continue
        uv, prim_uv = isl.job.uv, isl.dup_of.job.uv
        for l, src in isl.pairs:
            p = src[prim_uv].uv
            l[uv].uv = (p.x + offset, p.y)


# --------------------------------------------------------------------------- entry points

def process(jobs, s, ov, rectify, tick=lambda: None):
    """One full pass with each job's current mode. Leaves a summary in job.result."""
    for j in jobs:
        j.stats = defaultdict(int)
    unwrap_phase(jobs, s, ov)

    aspect = int(s.texture_width) / int(s.texture_height)
    for j in jobs:
        j.islands = [Island(j, faces) for faces in bmesh_linked_uv_islands(j.bm, j.uv)]
        # Existing layouts are already normalized to the rectangular texture.
        # Fresh unwraps are in square-texel space until pack_scope normalizes them.
        if j.mode == 'KEEP' and aspect != 1.0:
            transform(j.bm.faces, j.uv, lambda p: (p.x * aspect, p.y))
        if rectify:
            j.stats["rectified"] = sum(rectify_island(i, s.rectify_tolerance) for i in j.islands)
    all_islands = [i for j in jobs for i in j.islands]
    if s.equalize:
        equalize(all_islands)

    coverage = {}
    for scope in ([all_islands] if s.shared else [j.islands for j in jobs]):
        if s.stack:
            stack_similar(scope, s.stack_mirrored, s.stack_mode == 'COPIES')
        cov = pack_scope(jobs, scope, s, ov)
        place_duplicates(scope, 1.0 if s.stack_offset else 0.0)
        for j in {i.job for i in scope}:
            coverage[j] = cov
        tick()

    w, h = int(s.texture_width), int(s.texture_height)
    for j in jobs:
        prims = [i for i in j.islands if i.dup_of is None]
        a3 = sum(i.area_w for i in prims)
        au = sum(i.uv_area() for i in prims)
        stretch = sum(distortion(i.faces, j.uv, j.area_w)[0] * i.area_w for i in prims)
        stretch = stretch / a3 if a3 > TINY else 0.0
        density = math.sqrt(au * w * h / a3) if a3 > TINY else 0.0
        j.result = {
            "coverage": coverage.get(j, 0.0),
            "density": density,
            "stretch": stretch,
            # Stretched UVs starve some faces of pixels, so they must not win on raw px/m.
            "score": density / (1.0 + stretch),
            "rectify": rectify,
            "islands": len(prims),
            "stacked": len(j.islands) - len(prims),
            "rectified": j.stats["rectified"],
            "fallback_faces": j.stats["fallback_faces"],
            "mode": j.mode,
        }


def target_objects(context):
    vl = context.view_layer
    if context.mode == 'EDIT_MESH':
        objs = [o for o in vl.objects if o.mode == 'EDIT' and o.type == 'MESH']
    else:
        objs = [o for o in vl.objects.selected if o.type == 'MESH']
    unique, seen = [], set()
    for o in objs:
        if o.data not in seen and len(o.data.polygons):
            seen.add(o.data)
            unique.append(o)
    return unique


def run(context, s, pack_only=False):
    t0 = time.perf_counter()
    targets = target_objects(context)
    if not targets:
        raise DenseUVError("Select one or more mesh objects")
    vl = context.view_layer
    ts = context.scene.tool_settings
    start_mode = context.mode
    prev_sel = list(vl.objects.selected)
    prev_active = vl.objects.active
    prev_sync = ts.use_uv_select_sync
    prev_select_mode = tuple(ts.mesh_select_mode)
    ov = view3d_override(context)
    jobs = []
    lines = []

    completed = False
    try:
        with context.temp_override(**ov):
            if start_mode != 'EDIT_MESH':
                for o in prev_sel:
                    o.select_set(False)
                for o in targets:
                    o.select_set(True)
                vl.objects.active = targets[0]
                bpy.ops.object.mode_set(mode='EDIT')
        for obj in targets:
            jobs.append(Job(obj))
        ts.use_uv_select_sync = True
        ts.mesh_select_mode = (False, False, True)
        for j in jobs:
            j.bm.select_mode = {'FACE'}

        # Neither unwrap wins everywhere, nor does straightening: try the combinations the
        # quality setting allows and keep the best per object.
        if pack_only or s.unwrap_mode == 'KEEP':
            unwraps = ['KEEP']
        elif s.unwrap_mode == 'HYBRID' and s.quality != 'FAST':
            unwraps = ['HYBRID', 'SMART']
        else:
            unwraps = [s.unwrap_mode]
        if unwraps == ['KEEP']:
            rectify_options = [False]  # Pack Only keeps every island's shape.
        else:
            rectify_options = [True, False] if s.rectify and s.quality == 'BEST' else [s.rectify]
        candidates = [(u, r) for u in unwraps for r in rectify_options]
        wm = context.window_manager
        steps = len(candidates) * (1 if s.shared else len(jobs))
        done = [0]

        def tick():
            done[0] += 1
            wm.progress_update(done[0])

        wm.progress_begin(0, steps)
        # Score = texel density over (1 + stretch), not fill %: stacking frees space without
        # raising the fill, and stretch hides lost pixels.
        best = {}
        for cand, rectify in candidates:
            for j in jobs:
                j.mode = 'SMART' if cand == 'KEEP' and not j.had_uvs else cand
            process(jobs, s, ov, rectify, tick)
            if s.shared:
                better = not best or jobs[0].result["score"] > best[jobs[0]][0]["score"] * 1.0001
                winners = jobs if better else []
            else:
                winners = [j for j in jobs
                           if j not in best or j.result["score"] > best[j][0]["score"] * 1.0001]
            for j in winners:
                snap = [l[j.uv].uv.copy() for f in j.bm.faces for l in f.loops] if len(candidates) > 1 else None
                best[j] = (j.result, snap)
        if len(candidates) > 1:
            for j in jobs:
                result, snap = best[j]
                for l, p in zip((l for f in j.bm.faces for l in f.loops), snap):
                    l[j.uv].uv = p
                j.result = result

        if s.write_seams:
            select_only(jobs, {j: list(j.bm.faces) for j in jobs})
            with bpy.context.temp_override(**ov):
                bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)

        for j in jobs:
            r = j.result
            parts = [f"{j.obj.name}: {r['coverage'] * 100:.1f}% filled", f"{r['density']:.0f} px/m",
                     f"{r['islands']} islands"]
            if r["stacked"]:
                parts.append(f"{r['stacked']} stacked")
            if r["rectified"]:
                parts.append(f"{r['rectified']} straightened")
            if r["fallback_faces"]:
                parts.append(f"{r['fallback_faces']} faces re-projected")
            if len(unwraps) > 1:
                parts.append(f"{r['mode'].title()} unwrap won")
            if len(rectify_options) > 1 and not r["rectify"]:
                parts.append("unstraightened won")
            lines.append(", ".join(parts))
        completed = True
    finally:
        context.window_manager.progress_end()
        ts.mesh_select_mode = prev_select_mode
        ts.use_uv_select_sync = prev_sync
        for j in jobs:
            try:
                j.restore(keep_seams=s.write_seams, completed=completed)
            except ReferenceError:
                pass
        with context.temp_override(**ov):
            if start_mode != 'EDIT_MESH':
                if context.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                for o in vl.objects.selected:
                    o.select_set(False)
                for o in prev_sel:
                    o.select_set(True)
                vl.objects.active = prev_active

    covs = [j.result["coverage"] for j in jobs]
    summary = (f"Dense UV: {len(jobs)} object{'s' if len(jobs) != 1 else ''}, "
               f"{sum(covs) / len(covs) * 100:.1f}% filled on average, "
               f"{time.perf_counter() - t0:.1f} s")
    return [summary] + lines


def tile_coverage(bm, uv):
    """Area of the geometric UV union intersected with tile 1001."""
    # A constrained triangulation splits crossings into disjoint triangles and
    # records all source faces covering each result. Add the tile as one of
    # those faces so the same operation also clips islands at its boundary.
    coords = []
    index = {}
    faces = []
    seen = set()
    def vertex(p):
        if p not in index:
            index[p] = len(coords)
            coords.append(Vector(p))
        return index[p]

    for tri in bm.calc_loop_triangles():
        pts = [tuple(loop[uv].uv) for loop in tri]
        if any(max(p[axis] for p in pts) <= 0 or min(p[axis] for p in pts) >= 1 for axis in range(2)):
            continue
        area = signed_area(pts)
        if area == 0:
            continue
        ids = tuple(vertex(p) for p in pts)
        key = tuple(sorted(ids))
        if key not in seen:
            faces.append(ids if area > 0 else ids[::-1])
            seen.add(key)
    if not faces:
        return 0.0
    tile = len(faces)
    faces.append(tuple(vertex(p) for p in ((0, 0), (1, 0), (1, 1), (0, 1))))
    coords, _, triangles, _, _, sources = delaunay_2d_cdt(coords, [], faces, 1, 1e-8)
    return sum(abs(signed_area([coords[i] for i in tri])) for tri, ids in zip(triangles, sources)
               if tile in ids and len(ids) > 1)


def analyze(context, s):
    """Report fill, texel density and stretch of the current UVs, without changing anything."""
    targets = target_objects(context)
    if not targets:
        raise DenseUVError("Select one or more mesh objects")
    w, h = int(s.texture_width), int(s.texture_height)
    lines = []
    for obj in targets:
        if obj.mode == 'EDIT':
            obj.update_from_editmode()
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        try:
            uv = bm.loops.layers.uv.active
            if uv is None:
                lines.append(f"{obj.name}: no UV map")
                continue
            bm.faces.index_update()
            mw = obj.matrix_world
            areas = [newell_area([mw @ v.co for v in f.verts]) for f in bm.faces]
            islands = bmesh_linked_uv_islands(bm, uv)
            mapped_area = stretch = 0.0
            outside = 0
            for faces in islands:
                u0, v0, u1, v1 = bounds(faces, uv)
                fa = sum(areas[f.index] for f in faces)
                d, _ = distortion(faces, uv, areas)
                stretch += d * fa
                if min(u0, v0) < -1e-7 or max(u1, v1) > 1.0 + 1e-7:
                    outside += 1
                mapped_area += sum(abs(face_uv_area(f, uv)) for f in faces)
            total3 = sum(areas) or 1.0
            # Stacked and offset copies each receive the same texel density as
            # their primary. Only tile occupancy uses the deduplicated union.
            density = math.sqrt(mapped_area * w * h / total3)
            used = tile_coverage(bm, uv)
            line = (f"{obj.name}: {used * 100:.1f}% filled, {density:.0f} px/m, "
                    f"{len(islands)} islands, stretch {stretch / total3:.3f}")
            if outside:
                line += f", {outside} outside 0-1"
            lines.append(line)
        finally:
            bm.free()
    return lines
