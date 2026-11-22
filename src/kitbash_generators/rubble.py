# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
"""
Rubble Pile generator  -  Blender 5.1+
======================================

Run from Blender's Scripting tab (Text Editor > Run Script), or use the add-on.

Builds a sedan-sized pile (about 4.5 x 2 m footprint) of broken building concrete:
floor slabs and wall pieces with chunks missing, broken column stubs, fractured
chunks, bent low-poly rebar (square profile with length-dependent segments) and
small rocks sitting on the ground around the base. The pieces are dropped with a
rigid-body simulation so they settle naturally, then joined into two objects:

    RubblePile         concrete pieces + core mound + ground rocks
    RubblePile_Rebar   the bars

No materials and no export setup (that happens later for Substance Painter).
Both objects get a quick Smart UV Project map. Origin = ground at the pile centre,
1 unit = 1 m. Change SEED or the values in P and run again for a new pile.
"""

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

SEED = 47
MAX_REBAR_LENGTH = 1.50  # Longest exposed path in the original generator, in metres.

P = {
    # pile shape (a low-poly core mound the chunks settle on; it also fills the gaps)
    "mound_half_len": 1.55,
    "mound_half_wid": 0.56,
    "mound_height": 0.64,
    "fence_half_len": 2.0,   # temporary ring that keeps the pile compact while it settles
    "fence_half_wid": 0.82,
    # pieces dropped onto the mound
    "slabs": 3,            # broken floor slabs with chunks missing and rebar mats
    "walls": 2,            # wall pieces, one with a clean window-corner cut
    "columns": 2,          # broken column stubs with 4 corner bars
    "chunks": 30,          # fractured chunks
    "small_chunks": 20,    # small chunks that end up on / around the pile
    "ground_rocks": 40,    # small rocks on the ground around the base
    # rebar
    "rebar_radius": 0.012,  # half width of the square (4-sided) bar, ~24 mm bar
    "rebar_length": 0.35,
    "rebar_amount": 0.0,
    "additional_pieces": 0,
    "max_bars": 16,
    "sharp_angle_deg": 35.0,
}

COLL_NAME = "Rubble Pile"
DEBUG = {}
ASSET_NAME = "RubblePile"


# --------------------------------------------------------------------------------------
#  bmesh helpers
# --------------------------------------------------------------------------------------
def _frac_layer(bm):
    lay = bm.faces.layers.int.get("fracture")
    return lay if lay is not None else bm.faces.layers.int.new("fracture")


def box_bm(sx, sy, sz):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    _frac_layer(bm)
    return bm


def cut(bm, n, keep):
    """Chop off the +n side of a convex mesh. keep = share of the extent along n that stays."""
    n = n.normalized()
    hi = max(v.co.dot(n) for v in bm.verts)
    lo = min(v.co.dot(n) for v in bm.verts)
    d = lo + (hi - lo) * keep
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=n * d, plane_no=n, clear_outer=True)
    edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge) and e.is_boundary]
    if edges:
        out = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
        lay = _frac_layer(bm)
        for f in out["faces"]:
            f[lay] = 1


def rand_dir(rng, zscale=1.0):
    while True:
        v = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1) * zscale))
        if 0.1 < v.length <= 1.0:
            return v.normalized()


def chunk_bm(rng, size, cuts, keep=(0.6, 0.9)):
    """Keep cast faces on the original block and mark the new cuts as fractures."""
    bm = box_bm(*size)
    for _ in range(cuts):
        cut(bm, rand_dir(rng), rng.uniform(*keep))
    return bm


def bm_to_object(bm, name, coll):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def boolean_diff(host, cutter_bm, matrix, coll, fracture=True):
    """A bite exposes broken concrete on every cut wall. A formed window exposes cast concrete."""
    tag = _frac_layer(cutter_bm)
    for face in cutter_bm.faces:
        face[tag] = int(fracture)
    cut_ob = bm_to_object(cutter_bm, host.name + "_cutter", coll)
    cut_ob.matrix_world = matrix
    mod = host.modifiers.new("bite", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cut_ob
    dg = bpy.context.evaluated_depsgraph_get()
    new_me = bpy.data.meshes.new_from_object(host.evaluated_get(dg))
    host.modifiers.remove(mod)
    old = host.data
    host.data = new_me
    bpy.data.meshes.remove(old)
    me = cut_ob.data
    bpy.data.objects.remove(cut_ob)
    bpy.data.meshes.remove(me)


def mesh_volume(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    v = abs(bm.calc_volume(signed=False))
    bm.free()
    return max(v, 1e-4)


def local_bvh(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    lay = bm.faces.layers.int.get("fracture")
    frac = [f[lay] if lay else 1 for f in bm.faces]
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree, frac


def bar_exits(ob, lines, need_fracture=True):
    """For lines (origin inside, dir) find where a bar would leave the piece (local space)."""
    tree, frac = local_bvh(ob)
    out = []
    for o, d in lines:
        hit, nrm, idx, dist = tree.ray_cast(o, d, 50.0)
        if hit is None or nrm.dot(d) <= 0.2:      # started outside, or grazing exit
            continue
        if need_fracture and not frac[idx]:
            continue
        out.append((idx, hit.copy(), d.normalized()))
    return out


# --------------------------------------------------------------------------------------
#  Pieces
# --------------------------------------------------------------------------------------
def make_slab(rng, coll, name, L, W, T, bites, wall_window=False):
    bm = box_bm(L, W, T)
    for _ in range(rng.randint(6, 8)):                       # broken outline
        th = rng.uniform(0, 2 * math.pi)
        cut(bm, Vector((math.cos(th), math.sin(th), rng.uniform(-0.35, 0.35))), rng.uniform(0.8, 0.95))
    for _ in range(2):                                       # spalled top / bottom edges
        th = rng.uniform(0, 2 * math.pi)
        cut(bm, Vector((math.cos(th), math.sin(th), rng.choice((-1, 1)) * rng.uniform(0.7, 1.3))),
            rng.uniform(0.88, 0.96))
    ob = bm_to_object(bm, name, coll)
    for _ in range(bites):                                   # chunks of the slab missing
        th = rng.uniform(0, 2 * math.pi)
        reach = 0.5 * (abs(math.cos(th)) * L + abs(math.sin(th)) * W) * rng.uniform(0.8, 1.0)
        size = (rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.55), T * 3.0)
        cb = chunk_bm(rng, size, 5, (0.65, 0.9))
        m = Matrix.Translation((math.cos(th) * reach, math.sin(th) * reach, 0.0)) @ \
            Matrix.Rotation(rng.uniform(0, math.pi), 4, "Z")
        boolean_diff(ob, cb, m, coll)
    if wall_window:                                          # clean window-corner cut (cast faces)
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        win = box_bm(0.9, 0.7, T * 3.0)
        boolean_diff(ob, win, Matrix.Translation((sx * L * 0.5, sy * W * 0.5, 0.0)), coll, fracture=False)
    # rebar candidates: two mats (top / bottom), bars every 0.2 m both ways
    lines = []
    for z in (T * 0.22, -T * 0.22):
        for y in [k * 0.2 for k in range(-int(W / 0.4), int(W / 0.4) + 1)]:
            lines += [(Vector((0, y, z)), Vector((1, 0, 0))), (Vector((0, y, z)), Vector((-1, 0, 0)))]
        for x in [k * 0.2 for k in range(-int(L / 0.4), int(L / 0.4) + 1)]:
            lines += [(Vector((x, 0, z)), Vector((0, 1, 0))), (Vector((x, 0, z)), Vector((0, -1, 0)))]
    return ob, bar_exits(ob, lines)


def make_column(rng, coll, name):
    S = rng.uniform(0.3, 0.38)
    H = rng.uniform(0.9, 1.35)
    bm = box_bm(S, S, H)
    cut(bm, Vector((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0)), rng.uniform(0.86, 0.95))
    both = rng.random() < 0.6
    if both:
        cut(bm, Vector((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), -1.0)), rng.uniform(0.88, 0.96))
    for _ in range(rng.randint(2, 3)):                       # spalled corners near the ends
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        cut(bm, Vector((sx, sy, rng.choice((-1, 1)) * rng.uniform(0.6, 1.4))), rng.uniform(0.88, 0.95))
    ob = bm_to_object(bm, name, coll)
    c = S * 0.5 - 0.05
    lines = [(Vector((sx * c, sy * c, 0.0)), Vector((0, 0, 1))) for sx in (-1, 1) for sy in (-1, 1)]
    if both:
        lines += [(Vector((sx * c, sy * c, 0.0)), Vector((0, 0, -1))) for sx in (-1, 1) for sy in (-1, 1)]
    exits = bar_exits(ob, lines)
    return ob, exits


def make_chunk(rng, coll, name, smin, smax, cuts, bar_chance):
    size = (rng.uniform(smin, smax), rng.uniform(smin, smax), rng.uniform(smin * 0.6, smax * 0.8))
    ob = bm_to_object(chunk_bm(rng, size, cuts), name, coll)
    exits = []
    if rng.random() < bar_chance:
        d = rand_dir(rng, 0.3)
        exits = bar_exits(ob, [(Vector((0, 0, 0)), d)])
    return ob, exits


def mound_height(x, y):
    a, b, h = P["mound_half_len"], P["mound_half_wid"], P["mound_height"]
    return h * math.sqrt(max(0.0, 1.0 - (x / a) ** 2 - (y / b) ** 2))


def make_mound(rng, coll):
    """Low-poly dome of crushed concrete under the pieces (about 70 faces)."""
    a, b, h = P["mound_half_len"], P["mound_half_wid"], P["mound_height"]
    seg, rings = 14, 5
    bm = bmesh.new()
    top = bm.verts.new((rng.uniform(-0.15, 0.15), 0.0, h * rng.uniform(0.95, 1.08)))
    rows = []
    for r in range(1, rings + 1):
        phi = r / rings * math.pi / 2
        row = []
        for k in range(seg):
            th = 2 * math.pi * (k + rng.uniform(-0.2, 0.2)) / seg
            j = rng.uniform(0.85, 1.12)
            z = h * math.cos(phi) * rng.uniform(0.78, 1.15) if r < rings else -0.03
            row.append(bm.verts.new((a * math.sin(phi) * math.cos(th) * j, b * math.sin(phi) * math.sin(th) * j, z)))
        rows.append(row)
    for k in range(seg):
        bm.faces.new((top, rows[0][k], rows[0][(k + 1) % seg]))
    for r in range(rings - 1):
        for k in range(seg):
            bm.faces.new((rows[r][k], rows[r + 1][k], rows[r + 1][(k + 1) % seg], rows[r][(k + 1) % seg]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    lay = _frac_layer(bm)
    for f in bm.faces:
        f[lay] = 0
    return bm_to_object(bm, "mound", coll)


# --------------------------------------------------------------------------------------
#  Rigid body settle
# --------------------------------------------------------------------------------------
def make_fence(coll):
    """Temporary elliptical ring that keeps the pile compact while it settles."""
    a, b = P["fence_half_len"], P["fence_half_wid"]
    bm = bmesh.new()
    seg = 40
    lo = [bm.verts.new((a * math.cos(2 * math.pi * k / seg), b * math.sin(2 * math.pi * k / seg), -0.1))
          for k in range(seg)]
    hi = [bm.verts.new((v.co.x, v.co.y, 2.5)) for v in lo]
    for k in range(seg):
        bm.faces.new((lo[k], lo[(k + 1) % seg], hi[(k + 1) % seg], hi[k]))
    return bm_to_object(bm, "_fence", coll)


def settle(rng, pieces, coll, mound, frames=340, fence_drop=(160, 200)):
    sc = bpy.context.scene
    created_world = sc.rigidbody_world is None
    ground = bm_to_object(box_bm(30, 30, 1), "_ground", coll)
    ground.location.z = -0.5       # BOX collision is centred on the object origin
    fence = make_fence(coll)
    for col, shape in ((ground, "BOX"), (mound, "CONVEX_HULL"), (fence, "MESH")):
        with bpy.context.temp_override(object=col, active_object=col, selected_objects=[col]):
            bpy.ops.rigidbody.object_add(type="PASSIVE")
        col.rigid_body.friction = 1.0
        col.rigid_body.collision_shape = shape
        col.rigid_body.use_margin = True
        col.rigid_body.collision_margin = 0.002
    # the fence sinks into the ground part way through, so the edges slump naturally
    fence.rigid_body.kinematic = True
    fence.location.z = 0.0
    fence.keyframe_insert("location", index=2, frame=fence_drop[0])
    fence.location.z = -3.0
    fence.keyframe_insert("location", index=2, frame=fence_drop[1])

    # stack the pieces into a loose mound with a coarse height map so nothing starts overlapping
    cell = 0.2
    heights = {}

    def ground_h(x, y, r):
        h = mound_height(x, y)
        n = int(r / cell) + 1
        ix, iy = int(round(x / cell)), int(round(y / cell))
        for i in range(ix - n, ix + n + 1):
            for j in range(iy - n, iy + n + 1):
                h = max(h, heights.get((i, j), 0.0))
        return h

    def raise_h(x, y, r, top):
        n = int(r / cell) + 1
        ix, iy = int(round(x / cell)), int(round(y / cell))
        for i in range(ix - n, ix + n + 1):
            for j in range(iy - n, iy + n + 1):
                if (i - ix) ** 2 + (j - iy) ** 2 <= (n + 0.5) ** 2:
                    heights[(i, j)] = max(heights.get((i, j), 0.0), top)

    a, b, hgt = P["mound_half_len"], P["mound_half_wid"], P["mound_height"]
    for pc in pieces:
        ob = pc["ob"]
        r = max(v.co.length for v in ob.data.vertices)
        # spread pieces over the whole mound (flanks too), slabs / walls mostly on the flanks
        lo, hi = (0.35, 0.8) if pc["kind"] in ("slab", "wall") else (0.0, 0.85)
        rad_f = math.sqrt(rng.uniform(lo * lo, hi * hi))
        th = rng.uniform(0, 2 * math.pi)
        x, y = math.cos(th) * P["fence_half_len"] * rad_f, math.sin(th) * P["fence_half_wid"] * rad_f
        zs = mound_height(x, y)
        n = Vector((2 * x / a ** 2, 2 * y / b ** 2, 2 * max(zs, 0.05) / hgt ** 2)).normalized()
        if pc["kind"] in ("slab", "wall"):
            # lean against the mound: local Z roughly along the surface normal
            rot = Vector((0, 0, 1)).rotation_difference(n)
            rot = rot @ Quaternion((0, 0, 1), rng.uniform(0, 2 * math.pi))
            rot = Quaternion(rand_dir(rng, 0.0), math.radians(rng.uniform(-15, 15))) @ rot
        else:
            rot = Quaternion(rand_dir(rng), rng.uniform(0, math.pi))
        z = ground_h(x, y, r * 0.7) + r + 0.02
        ob.location = (x, y, z)
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = rot
        raise_h(x, y, r * 0.7, z + r * 0.8)
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = ob.rigid_body
        rb.collision_shape = "CONVEX_HULL"
        pc["volume"] = mesh_volume(ob)
        rb.mass = pc["volume"] * 2400.0
        rb.friction = 1.0
        rb.restitution = 0.0
        rb.linear_damping = 0.55
        rb.angular_damping = 0.85
        rb.use_margin = True
        rb.collision_margin = 0.002

    rw = sc.rigidbody_world
    rw.substeps_per_frame = 12
    rw.solver_iterations = 24
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = frames
    sc.frame_start, sc.frame_end = 1, frames
    for f in range(1, frames + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for pc in pieces:
        pc["matrix"] = pc["ob"].evaluated_get(dg).matrix_world.copy()
    # clean up the simulation
    for pc in pieces:
        with bpy.context.temp_override(object=pc["ob"], active_object=pc["ob"], selected_objects=[pc["ob"]]):
            bpy.ops.rigidbody.object_remove()
    for col in (ground, mound, fence):
        with bpy.context.temp_override(object=col, active_object=col, selected_objects=[col]):
            bpy.ops.rigidbody.object_remove()
    fence_action = fence.animation_data.action
    sim_collection = rw.collection
    for tmp in (ground, fence):
        me = tmp.data
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(me)
    if sc.rigidbody_world:
        bpy.ops.rigidbody.world_remove()
    if fence_action and not fence_action.users:
        bpy.data.actions.remove(fence_action)
    if created_world and sim_collection and not sim_collection.objects and not sim_collection.children:
        bpy.data.collections.remove(sim_collection)
    sc.frame_set(1)


# --------------------------------------------------------------------------------------
#  Rebar
# --------------------------------------------------------------------------------------
def _perp(d, rng):
    a = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
    p = d.cross(a).normalized()
    return Quaternion(d, rng.uniform(0, 2 * math.pi)) @ p


def _bar_frame(direction):
    axis = Vector((0, 0, 1)) if abs(direction.z) < 0.9 else Vector((1, 0, 0))
    u = direction.cross(axis).normalized()
    return u, direction.cross(u).normalized()


def _embed_depth(rad):
    return min(0.003, rad * 0.35)


def _collar_length(rad):
    # Beyond this straight collar the entire bar fits in an outward swept sphere.
    return max(0.04, rad * 2)


class _BarMount:
    """Intersect a straight square shaft's four side edges with its fracture plane."""
    def __init__(self, axis, normal, rad):
        self.axis = axis
        alignment = axis.dot(normal)
        u, v = _bar_frame(axis)
        self.square = [rad * c for c in (u + v, -u + v, -u - v, u - v)]
        self.shear = [-corner.dot(normal) / alignment for corner in self.square]
        self.overhang = max(self.shear)
        # The next perpendicular section also clears the surface by a corner radius.
        self.collar = max(_collar_length(rad), math.sqrt(2) * rad / alignment + 0.0002)
        self.ramp = self.collar + _embed_depth(rad)

    def ring(self, center, distance):
        # For a short stub, ease out the axial shear before closing its tip.
        # Every corner remains on its own straight edge, even below collar length.
        fraction = max(0.0, min(1.0, (self.ramp - distance) / self.collar))
        return [center + corner + self.axis * (skew * fraction)
                for corner, skew in zip(self.square, self.shear)]


class _BarPath(list):
    """A centerline and its face-cut foot, retained when the visible tip is trimmed."""
    def __init__(self, points, mount):
        super().__init__(points)
        self.mount = mount


def _seat_anchor(point, normal, world, rad, source_faces, support_faces, axis=None):
    """Snap the foot to the source plane and clear the full oblique mounting collar."""
    tolerance = 0.0001
    hit, face_normal, face_id, _ = world.ray_cast(point + normal * tolerance, -normal, 2 * tolerance)
    if hit is None:
        # A twisted n-gon can tessellate along a different diagonal in a BVH.
        snap = min(0.04, rad * 3)
        hit, face_normal, face_id, _ = world.ray_cast(point + normal * snap, -normal, 2 * snap)
    if hit is None or face_normal.dot(normal) < 0.9:
        return None
    if source_faces is not None and face_id not in source_faces:
        return None
    if support_faces is None:
        support_faces = (face_id,)
    point, normal = hit, face_normal.normalized()
    axis = normal if axis is None else axis.normalized()
    alignment = axis.dot(normal)
    if alignment < 0.65:
        return None    # A grazing bar would require an oversized, nearly edge-on foot.
    mount = _BarMount(axis, normal, rad)
    embed = _embed_depth(rad)
    for corner in mount.ring(point, 0.0):
        # Even on a mildly twisted face every corner must sit inside the root cap.
        inset = embed * alignment * 0.8
        contact, _, idx, _ = world.ray_cast(corner + normal * inset, -normal, 2 * inset)
        if contact is None or idx not in support_faces:
            return None    # Part of the root hangs off the edge or sits on another piece.
    # A nearby enclosing solid can contain the whole collar without crossing its sides.
    if hasattr(world, "contains") and world.contains(point + normal * tolerance):
        return None
    collar = mount.collar
    verts = mount.ring(point - axis * embed, 0.0) + mount.ring(point + axis * collar, mount.ramp)
    if min(p.z for p in verts) < 0.0001:
        return None
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    faces.extend((k, (k+1) % 4, (k+1) % 4 + 4, k+4) for k in range(4))
    body = BVHTree.FromPolygons(verts, faces)
    if any(idx not in support_faces for _, idx in body.overlap(getattr(world, "surface", world))):
        return None    # Includes the back of a very thin piece and obstacles beside the axis.
    # Catch a small obstacle entirely enclosed by the collar as well.
    center = point + axis * ((collar - embed) * 0.5)
    search = max((p - center).length for p in verts) + tolerance
    u, v = _bar_frame(axis)
    for closest, _, idx, _ in world.find_nearest_range(center, search):
        q = closest - point
        if (idx not in support_faces and q.dot(normal) >= -embed * alignment and q.dot(axis) <= collar
                and abs(q.dot(u)) <= rad and abs(q.dot(v)) <= rad):
            return None
    return point, normal


def _clear_run(world, start, direction, length, rad):
    """Advance conservatively using distance to the mesh and the bar's corner radius."""
    radius = math.sqrt(2) * rad + 0.0001
    distance = 0.0
    # Unsigned surface distance is 1-Lipschitz. Each step certifies a whole
    # interval, including thin obstacles beside or between the centerline samples.
    for _ in range(64):
        point = start + direction * distance
        _, _, _, clearance = world.find_nearest(point, length - distance + radius)
        if clearance is None:
            return length
        step = clearance - radius
        if step <= 0.00001:
            break
        distance += step
        if distance >= length:
            return length
    return distance


class _BarClearance:
    """Spatial grid of reserved shaft segments. The corner radius bounds the square sweep."""
    def __init__(self, rad):
        self.separation = 2 * math.sqrt(2) * rad + 0.0001
        self.cell = max(0.2, 3 * self.separation)
        self.grid = {}
        self.segments = []

    def _cells(self, a, b, margin):
        axes = [range(math.floor((min(a[i], b[i]) - margin) / self.cell),
                      math.floor((max(a[i], b[i]) + margin) / self.cell) + 1) for i in range(3)]
        return ((x, y, z) for x in axes[0] for y in axes[1] for z in axes[2])

    def reserve(self, path):
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if i == 0 and getattr(path, "mount", None):
                mount = path.mount
                if len(path) == 2:
                    # A clipped tip within the collar still has an oblique leading corner.
                    fraction = max(0.0, (mount.ramp - (b - a).length) / mount.collar)
                    b = b + mount.axis * (mount.overhang * fraction)
                a = a - mount.axis * mount.overhang
            delta = b - a
            length = delta.length
            if length <= 1e-8:
                continue
            idx = len(self.segments)
            self.segments.append((a, delta / length, length))
            for cell in self._cells(a, b, self.separation):
                self.grid.setdefault(cell, []).append(idx)

    def clear_run(self, a, d, length):
        neighbors = {idx for cell in self._cells(a, a + d * length, 0) for idx in self.grid.get(cell, ())}
        if not neighbors:
            return length
        distance = 0.0
        for _ in range(64):
            p = a + d * distance
            step = length - distance
            for idx in neighbors:
                b, axis, extent = self.segments[idx]
                offset = min(extent, max(0.0, (p - b).dot(axis)))
                step = min(step, (p - (b + axis * offset)).length - self.separation)
            if step <= 0.00001:
                break
            distance += step
            if distance >= length:
                return length
        return distance


def _bar_plan(rng, p_exit, d, world_bvh, rad, splay, reserved, face_normal=None):
    """Grow out of the verified mounting collar, trimming bends at the first obstacle."""
    mount = _BarMount(d, face_normal, rad) if face_normal is not None else None
    collar = mount.collar if mount else _collar_length(rad)
    pts = [p_exit]
    stub = max(rng.uniform(0.08, 0.28), collar + rad * math.sqrt(2))
    cur = p_exit + d * stub
    pts.append(cur)
    direction = d.copy()
    bends = 2 if rng.random() < 0.45 else 1
    for b in range(bends):
        axis = _perp(direction, rng)
        want = None
        if splay is not None and b == 0:
            want = splay                                   # column bars peel outward
        elif rng.random() < 0.35:
            want = Vector((0, 0, -1))                      # some bars sag
        if want is not None and abs(direction.dot(want)) < 0.95:
            w = (want - direction * direction.dot(want)).normalized()
            axis = direction.cross(w).normalized()
        ang = math.radians(rng.uniform(25, 70) if b == 0 else rng.uniform(15, 45))
        mid = Quaternion(axis, ang * 0.5) @ direction
        new_dir = Quaternion(axis, ang) @ direction
        cur = cur + mid * rng.uniform(0.04, 0.08)
        pts.append(cur)                                    # one ring in the middle of the bend
        seg = rng.uniform(0.28, 0.65) if b == 0 else rng.uniform(0.15, 0.4)
        nxt = cur + new_dir * seg + _perp(new_dir, rng) * rng.uniform(0.0, 0.015)   # not perfectly straight
        direction = (nxt - cur).normalized()
        cur = nxt
        pts.append(cur)
    # Use the same random path at every setting. A longer bar extends the tip
    # without moving its anchor, changing its bends, or rerolling the next bar.
    full_length = sum((b - a).length for a, b in zip(pts, pts[1:]))
    remaining = MAX_REBAR_LENGTH
    out = _BarPath([p_exit - d * _embed_depth(rad)], mount)
    floor = math.sqrt(2) * rad + 0.0001
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        seg = b - a
        L = seg.length
        if L < 1e-6:
            continue
        direction = seg / L
        t_end = min(L, remaining)
        # The first collar was tested as a square prism, down to the buried cap.
        offset = min(collar, t_end) if i == 0 else 0.0
        if t_end > offset:
            t_end = offset + _clear_run(world_bvh, a + direction * offset, direction, t_end - offset, rad)
        if direction.z < 0:
            t_end = min(t_end, max(offset, (a.z - floor) / -direction.z))
        if reserved is not None:
            embed = (_embed_depth(rad) + (mount.overhang if mount else 0.0)) if i == 0 else 0.0
            clear = reserved.clear_run(a - direction * embed, direction, t_end + embed) - embed
            if i == 0 and mount and clear < collar:
                # Model the sheared leading corner when another bar clips this collar.
                clear = (clear - mount.overhang) / (1.0 - mount.overhang / collar)
            t_end = min(t_end, clear)
        if t_end <= 1e-5:
            break
        # Keep an axis-perpendicular ring at the end of the checked collar. Otherwise
        # a later bend can tilt the sides of the shaft back into the mounting area.
        ring_start = 0.0
        if i == 0 and collar + 1e-5 < t_end:
            out.append(a + direction * collar)
            ring_start = collar
        for step in range(1, math.ceil((t_end - ring_start - 1e-5) / 0.15)):
            out.append(a + direction * (ring_start + step * 0.15))
        out.append(a + direction * t_end)
        remaining -= t_end
        if t_end < L or remaining <= 1e-5:
            break
    return out, min(1.0, full_length / MAX_REBAR_LENGTH)


def _visible_path(path, length):
    result = _BarPath(path[:1], getattr(path, "mount", None))
    for a, b in zip(path, path[1:]):
        edge = b - a
        if length < edge.length:
            if length > 1e-6:
                result.append(a + edge.normalized() * length)
            break
        result.append(b)
        length -= edge.length
    return result


def rebar_path(rng, p_exit, d, world_bvh, rad, splay=None):
    path, factor = _bar_plan(rng, p_exit, d, world_bvh, rad, splay, None)
    return _visible_path(path, _embed_depth(rad) + P["rebar_length"] * factor)


class RebarSide:
    def __init__(self, normal, point, area):
        self.normal = normal
        self.point = point
        self.area = area
        self.anchors = []
        self.faces = set()
        self.support_faces = set()


def _fracture_patches(mesh, areas, neighbors, radius):
    """Rank connected planar surfaces, independent of how they were triangulated."""
    tags = mesh.attributes.get("fracture")
    if not areas or tags is None or tags.domain != "FACE" or tags.data_type != "INT":
        return []
    polygons = mesh.polygons
    remaining = set(range(len(polygons)))
    patches = []
    for face in polygons:
        if face.index not in remaining:
            continue
        remaining.remove(face.index)
        indices = [face.index]
        # Compare every candidate to the starting plane to prevent gradual drift
        # across a curved row of small facets.
        for idx in indices:
            for other in neighbors[idx]:
                if other not in remaining:
                    continue
                poly = polygons[other]
                if face.normal.dot(poly.normal) < 0.999999:
                    continue
                if any(abs(face.normal.dot(mesh.vertices[v].co - face.center)) > 1e-5 for v in poly.vertices):
                    continue
                remaining.remove(other)
                indices.append(other)
        patches.append((sum(areas[i] for i in indices), indices))

    largest_patch = max(area for area, _ in patches)
    largest_poly = max(areas)
    # A usable break needs room for material around the full square bar foot.
    # Relative area also keeps incidental chips off large pieces of concrete.
    minimum = max(4 * (2 * radius) ** 2, 0.03 * largest_patch)
    return sorted((area, indices) for area, indices in patches
                  if minimum <= area < largest_patch * (1 - 1e-5)
                  and all(tags.data[i].value and areas[i] < largest_poly * (1 - 1e-5) for i in indices))


def exposed_sides(piece, shift, con_bvh):
    """Find seated mounts on suitably sized, exposed fractures of a settled piece."""
    mesh = piece["ob"].data
    transform = Matrix.Translation(shift) @ piece["matrix"]
    normal_matrix = transform.to_3x3().inverted_safe().transposed()
    normals = [(normal_matrix @ face.normal).normalized() for face in mesh.polygons]
    radius = P["rebar_radius"]
    spacing = max(0.07, radius * 4)
    min_height = max(0.045, radius * 1.5)
    points = [transform @ v.co for v in mesh.vertices]
    triangles = [[] for _ in mesh.polygons]
    areas = [0.0] * len(mesh.polygons)
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        a, b, c = (points[i] for i in tri.vertices)
        area = (b - a).cross(c - a).length * 0.5
        triangles[tri.polygon_index].append((a, b, c, area))
        areas[tri.polygon_index] += area

    edge_faces = {}
    for face in mesh.polygons:
        for edge in face.edge_keys:
            edge_faces.setdefault(edge, []).append(face.index)
    neighbors = [{other for edge in face.edge_keys for other in edge_faces[edge] if other != face.index}
                 for face in mesh.polygons]
    patches = _fracture_patches(mesh, areas, neighbors, radius)
    eligible = {i for _, indices in patches for i in indices}
    exits = {}
    for idx, point, _ in piece.get("exits", []):
        if idx in eligible:
            exits.setdefault(idx, []).append(transform @ point)

    collision_faces = piece.get("collision_faces")
    if collision_faces is None:
        # Standalone pieces can be tested against a BVH shared with other pieces.
        # Identify its face IDs geometrically when no joined-pile mapping was supplied.
        collision_faces = [{hit[2] for a, b, c, _ in group
                            if (hit := con_bvh.find_nearest((a + b + c) / 3, 1e-5))[0] is not None}
                           for group in triangles]
    sides = []
    for area, indices in patches:
        first = mesh.polygons[indices[0]]
        normal = normals[first.index]
        if normal.z < -0.45:
            continue
        side = RebarSide(normal, transform @ first.center, area)
        for idx in indices:
            source_faces = collision_faces[idx]
            support_faces = set(source_faces)
            # A shallow adjoining fracture on this same piece may support a corner.
            for other in neighbors[idx] & eligible:
                if normals[idx].dot(normals[other]) > 0.96:
                    support_faces.update(collision_faces[other])
            side.faces.update(source_faces)
            side.support_faces.update(support_faces)
            samples = exits.get(idx, []).copy()
            for a, b, c, triangle_area in triangles[idx]:
                divisions = max(1, min(12, math.ceil(math.sqrt(triangle_area) / spacing)))
                for i in range(divisions):
                    for j in range(divisions - i):
                        u, v = (i + 1 / 3) / divisions, (j + 1 / 3) / divisions
                        samples.append(a + (b - a) * u + (c - a) * v)
            for point in samples:
                if point.z < min_height or any((point - p).length_squared < spacing ** 2 for p, _ in side.anchors):
                    continue
                mount = _seat_anchor(point, normal, con_bvh, radius, source_faces, support_faces)
                if mount is not None:
                    side.anchors.append(mount)
        if side.anchors:
            side.point = sum((p for p, _ in side.anchors), Vector()) / len(side.anchors)
            side.anchors.sort(key=lambda a: (a[0] - side.point).length_squared)
            sides.append(side)
    return sides


def _opposing_side(first, others):
    """A parallel-looking break is only an opposite if it is across the concrete."""
    best, best_score = None, -1.0
    for side in others:
        alignment = -first.normal.dot(side.normal)
        across = (side.point - first.point).normalized()
        depth = min(-first.normal.dot(across), side.normal.dot(across))
        if alignment >= 0.75 and depth >= 0.45:
            score = alignment + 0.5 * depth
            if score > best_score:
                best, best_score = side, score
    return best


def _staged_sides(sides, pitch, cap, column, choose_opposite):
    """Assign Amount thresholds to slots on one main run and at most one transverse run."""
    primary = sides[0]
    opposite = choose_opposite(primary, sides[1:])
    main = [primary, opposite] if opposite else [primary]
    capacity = {s: min(len(s.anchors), max(1, int(s.area / pitch**2))) for s in sides}
    transverse = []
    if opposite and not column:
        candidates = [s for s in sides if capacity[s] >= 2 and
                      all(abs(s.normal.dot(end.normal)) <= 0.4 for end in main)]
        if candidates:
            lead = candidates[0]
            transverse.append(lead)
            other = choose_opposite(lead, candidates[1:])
            if other:
                transverse.append(other)

    slots = []
    count = dict.fromkeys(main + transverse, 0)

    def add(threshold, side):
        if len(slots) < cap and count[side] < capacity[side]:
            slots.append((threshold, side))
            count[side] += 1

    cluster = min(2, max(1, cap // 3))
    for i in range(cluster):
        add(0.0, primary)
    if opposite:
        for i in range(cluster):
            add(0.25 + 0.09 * i, opposite)
    for row in range(2):
        for end, side in enumerate(main):
            add(round(0.47 + 0.12 * row + 0.06 * end, 2), side)

    # Tertiary faces only appear near the end of the slider. Surfaces that
    # cannot fit more rods may stop growing below the artist's upper limit.
    early = len(slots)
    for _ in range(cap - early):
        for side in transverse + main:
            add(1.0, side)
        if len(slots) >= cap or all(count[s] >= capacity[s] for s in count):
            break
    late = len(slots) - early
    for i in range(late):
        slots[early + i] = (0.75 + 0.25 * i / max(1, late - 1), slots[early + i][1])
    return main, transverse, slots


def _row_axis(side, slab_up):
    if slab_up is not None:
        return slab_up.cross(side.normal).normalized()
    # Place a row along the longest available direction on a broken face.
    u, v = _bar_frame(side.normal)
    xx = xy = yy = 0.0
    for p, _ in side.anchors:
        offset = p - side.point
        x, y = offset.dot(u), offset.dot(v)
        xx += x*x
        xy += x*y
        yy += y*y
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    return u * math.cos(angle) + v * math.sin(angle)


def _run_directions(ends, up, column, transverse_to=None):
    """One outward axis at the first end of the entire row of reinforcing bars."""
    lead = ends[0]
    directions = [lead.normal]
    if len(ends) == 2:
        mate = ends[1]
        directions = [lead.point - mate.point, lead.normal - mate.normal]
        # The patch centers need not line up across a concavity. Try a few
        # nearby chords while keeping every rod in the chosen row parallel.
        directions.extend(a[0] - b[0] for a in lead.anchors[:3] for b in mate.anchors[:3])
    tried = []
    for direction in directions:
        axis = direction.copy()
        if column:
            axis = up if lead.normal.dot(up) > 0 else -up
        elif up is not None:
            axis -= up * axis.dot(up)
        if transverse_to is not None:
            axis -= transverse_to * axis.dot(transverse_to)
        axis = axis.normalized()
        if lead.normal.dot(axis) >= 0.65 and (len(ends) == 1 or ends[1].normal.dot(-axis) >= 0.65):
            if not any(axis.dot(previous) > .9998 for previous in tried):
                tried.append(axis)
                yield axis


def _next_fracture(point, inward, target_faces, world):
    # Start just inside the host. Its very next boundary must be the far end.
    # A ray through a window, another rock or an unrelated face cannot be paired.
    hit, normal, index, _ = world.ray_cast(point + inward * 0.0001, inward)
    if hit is not None and index in target_faces and normal.dot(inward) >= 0.65:
        return hit, normal
    return None


def _clear_chord(start, end, axis, world, radius, allowed_faces):
    """Check the sides of the buried square bar against notches as well as its centerline."""
    u, v = _bar_frame(axis)
    corners = [radius * offset for offset in (u+v, -u+v, -u-v, u-v)]
    verts = []
    for point, normal in (start, end):
        # Intersect each corner's longitudinal line with the actual fracture plane.
        verts.extend(point + offset - axis * (normal.dot(offset) / normal.dot(axis)) for offset in corners)
    faces = [(i, (i+1) % 4, (i+1) % 4 + 4, i+4) for i in range(4)]
    chord = BVHTree.FromPolygons(verts, faces)
    surface = getattr(world, "surface", world)
    return all(idx in allowed_faces for _, idx in chord.overlap(surface))


class _RebarRun:
    """Plan the lead ends once, then expose only their actual partners across the rock."""
    def __init__(self, ends, axis, world, radius):
        self.ends = ends
        self.axis = axis
        self.mates = []
        self.candidates = []
        self.checked = []
        self.unchecked = self._mounts(world, radius)
        if axis is None:
            return
        lead = ends[0]
        singles = list(lead.anchors)
        if len(ends) == 2:
            mate = ends[1]
            # Sampling both ends finds overlapping regions even on dissimilar breaks.
            samples = singles.copy()
            for point, _ in mate.anchors:
                near = _next_fracture(point, axis, lead.faces, world)
                if near is not None:
                    samples.append(near)
            for near in samples:
                far = _next_fracture(near[0], -axis, mate.faces, world)
                if far is not None:
                    self.candidates.append((near, far))
            self.candidates.sort(key=lambda pair: (pair[0][0] - lead.point).length_squared)
        self.candidates.extend((near, None) for near in singles)

    def _mounts(self, world, radius):
        lead = self.ends[0]
        for near, far in self.candidates:
            point, normal = near
            near = _seat_anchor(point, normal, world, radius, lead.faces, lead.support_faces, axis=self.axis)
            if near is None:
                continue
            if far is not None:
                mate = self.ends[1]
                # Project from the final snapped near root, never from a nearby sample.
                far = _next_fracture(near[0], -self.axis, mate.faces, world)
                if far is None:
                    continue
                far = _seat_anchor(*far, world, radius, mate.faces, mate.support_faces, axis=-self.axis)
                if far is None or not _clear_chord(near, far, self.axis, world, radius,
                                                    lead.support_faces | mate.support_faces):
                    continue
            yield near, far

    def available(self, side):
        if side is not self.ends[0]:
            yield from ((mate, None) for mate in self.mates)
            return
        # Each projected mount pays for its footprint tests once, even at a high cap.
        yield from self.checked
        for candidate in self.unchecked:
            self.checked.append(candidate)
            yield candidate


def _choose_run(lead, others, world, rad, up, column, transverse_to):
    remaining = list(others)
    while remaining:
        mate = _opposing_side(lead, remaining)
        if mate is None:
            break
        remaining.remove(mate)
        ends = [lead, mate]
        for axis in _run_directions(ends, up, column, transverse_to):
            run = _RebarRun(ends, axis, world, rad)
            if any(far is not None for _, far in run.candidates):
                # Paired candidates always precede singles in this iterator.
                if next(run.available(lead), (None, None))[1] is not None:
                    return run
    axis = next(_run_directions([lead], up, column, transverse_to), None)
    return _RebarRun([lead], axis, world, rad)


def _piece_bars(index, piece, sides, shift, con_bvh, seed, reserved):
    rad = P["rebar_radius"]
    pitch = max(0.11, 9 * rad)
    amount = round(P["rebar_amount"], 6)
    cap = P["max_bars"]
    column = piece["kind"] == "column"
    slab = piece["kind"] in {"slab", "wall"}
    up = piece["matrix"].col[2].to_3d().normalized()
    if column:
        sides = [s for s in sides if abs(s.normal.dot(up)) >= 0.75]
    elif slab:
        sides = [s for s in sides if abs(s.normal.dot(up)) <= 0.5]
    sides = sorted(sides, key=lambda s: s.area)
    while sides:
        member_axis = up if slab or column else None
        runs = {}

        def choose_opposite(lead, others):
            transverse_to = runs[sides[0]].axis if runs else None
            run = _choose_run(lead, others, con_bvh, rad, member_axis, column, transverse_to)
            runs.update(dict.fromkeys(run.ends, run))
            return run.ends[1] if len(run.ends) == 2 else None

        main, transverse, slots = _staged_sides(sides, pitch, cap, column, choose_opposite)
        placed = {s: [] for s in runs}
        rows = {s: _row_axis(s, up if slab else None) for s in placed}
        all_roots = []
        buried = _BarClearance(rad)
        for ordinal, (threshold, side) in enumerate(slots):
            if side not in runs or (side in transverse and not placed[main[1]]):
                continue
            row = placed[side]
            run = runs[side]
            if run.axis is None:
                continue
            first = side is run.ends[0]
            direction = run.axis if first else -run.axis
            for (point, normal), mate in run.available(side):
                if any((point - other).length_squared < pitch**2 for other in all_roots):
                    continue
                # The near-side row defines the row across the rock. A slanted
                # opposite face may show exits at different heights along that row.
                if first and row and abs((point - row[0]).dot(side.normal.cross(rows[side]))) > 0.45 * pitch:
                    continue
                if first:
                    # A secondary mat needs its own depth in the concrete. For
                    # a single visible end, reserve steel up to the next boundary.
                    back = mate[0] if mate else con_bvh.ray_cast(point - direction * .0001, -direction)[0]
                    if back is None:
                        continue
                    depth = (point - back).dot(direction)
                    if depth <= 0 or buried.clear_run(point, -direction, depth) < depth - 1e-6:
                        continue
                splay = None
                if column:
                    radial = point - (shift + piece["matrix"].translation)
                    radial -= direction * radial.dot(direction)
                    if radial.length > 1e-4:
                        splay = radial.normalized()
                rng = random.Random(f"{seed}/{index}/{ordinal}")
                path, factor = _bar_plan(rng, point, direction, con_bvh, rad, splay, reserved, face_normal=normal)
                if len(path) < 2 or (path[1] - point).dot(direction) < min(0.02, rad):
                    continue
                reserved.reserve(path)
                if first:
                    buried.reserve([point, back])
                row.append(point)
                all_roots.append(point)
                if mate is not None:
                    run.mates.append(mate)
                if amount >= threshold:
                    length = _embed_depth(rad) + P["rebar_length"] * factor
                    yield index, ordinal, point, direction, splay, _visible_path(path, length)
                break
            if not all_roots:
                break
        if all_roots:
            return
        # The primary has no free shaft despite its seated roots. Recompute
        # both axes around the next-smallest break before spending this piece.
        sides.pop(0)


def pick_bars(kept, shift, con_bvh, seed, *, with_paths=False):
    """Allocate rods largest-piece first, preserving the layout as sliders increase."""
    if not P["max_bars"]:
        return
    candidates = [(i, p) for i, p in enumerate(kept) if p["kind"] in {"slab", "wall", "column", "chunk"}]
    candidates.sort(key=lambda item: item[1]["volume"] if "volume" in item[1] else mesh_volume(item[1]["ob"]),
                    reverse=True)
    remaining = 1 + P["additional_pieces"]
    reserved = _BarClearance(P["rebar_radius"])
    for index, piece in candidates:
        sides = exposed_sides(piece, shift, con_bvh)
        used = False
        for item in _piece_bars(index, piece, sides, shift, con_bvh, seed, reserved):
            used = True
            yield item if with_paths else item[:5]
        if used:
            remaining -= 1
            if remaining <= 0:
                break


def rebar_bm(paths, rad):
    """Square (4-sided) profile swept along each path; smooth sides, sharp caps."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for pts in paths:
        if len(pts) < 2:
            continue
        rings = []
        mount = getattr(pts, "mount", None)
        tangent_prev = mount.axis if mount else (pts[1] - pts[0]).normalized()
        nrm, _ = _bar_frame(tangent_prev)
        length = 0.0
        vs = []
        for i, p in enumerate(pts):
            if i:
                length += (p - pts[i - 1]).length
            vs.append(length)
            if i == 0:
                t = tangent_prev
            elif i == len(pts) - 1:
                t = (pts[i] - pts[i - 1]).normalized()
            else:
                t = ((pts[i] - pts[i - 1]).normalized() + (pts[i + 1] - pts[i]).normalized()).normalized()
            rot = tangent_prev.rotation_difference(t)          # parallel transport
            nrm = (rot @ nrm).normalized()
            tangent_prev = t
            bin_ = t.cross(nrm).normalized()
            if mount and length <= mount.ramp + 1e-6:
                corners = mount.ring(p, length)
            else:
                corners = [p + (nrm * cx + bin_ * cy) * rad
                           for cx, cy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            rings.append([bm.verts.new(co) for co in corners])
        for i in range(len(rings) - 1):
            for k in range(4):
                f = bm.faces.new((rings[i][k], rings[i][(k + 1) % 4], rings[i + 1][(k + 1) % 4], rings[i + 1][k]))
                f.smooth = True
                us = (k / 4.0, (k + 1) / 4.0, (k + 1) / 4.0, k / 4.0)
                vv = (vs[i], vs[i], vs[i + 1], vs[i + 1])
                for loop, uu, v in zip(f.loops, us, vv):
                    loop[uv].uv = (uu, v / (8 * rad))
        for ring, flip in ((rings[0], True), (rings[-1], False)):
            f = bm.faces.new(list(reversed(ring)) if flip else ring)
            f.smooth = True
            for e in f.edges:
                e.smooth = False
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


# --------------------------------------------------------------------------------------
#  Ground rocks
# --------------------------------------------------------------------------------------
def place_rocks(rng, count, pile_bvh, half_len, half_wid):
    rocks = []
    tries = 0
    while len(rocks) < count and tries < count * 30:
        tries += 1
        ang = rng.uniform(0, 2 * math.pi)
        rr = 0.93 + abs(rng.gauss(0.0, 0.11))
        x = math.cos(ang) * half_len * rr
        y = math.sin(ang) * half_wid * rr
        hit = pile_bvh.ray_cast(Vector((x, y, 5.0)), Vector((0, 0, -1)), 10.0)
        if hit[0] is not None and hit[0].z > 0.03:
            continue                       # under the pile edge, skip
        s = rng.uniform(0.05, 0.22) * (1.3 if rng.random() < 0.15 else 1.0)
        bm = chunk_bm(rng, (s, s * rng.uniform(0.7, 1.2), s * rng.uniform(0.45, 0.8)),
                      rng.randint(4, 6), (0.62, 0.88))
        bm.transform(Quaternion(rand_dir(rng), rng.uniform(0, math.pi)).to_matrix().to_4x4())
        zmin = min(v.co.z for v in bm.verts)
        bmesh.ops.translate(bm, vec=(x, y, -zmin - s * rng.uniform(0.05, 0.25)), verts=bm.verts)  # sit / sink
        # don't let rocks overlap each other too much
        c = Vector((x, y))
        if any((c - q).length < (s + qs) * 0.55 for q, qs in rocks):
            bm.free()
            continue
        rocks.append((c, s))
        yield bm


# --------------------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------------------
def _clear_previous(scene=None):
    """Remove only this scene's remembered output, even if artists rename or duplicate it."""
    scene = scene or bpy.context.scene
    record = scene.get("rcg_last_pile")
    if record is None:
        # Older files track only a reusable collection name. Preserve ambiguous artwork.
        return
    coll = record.get("collection")
    parts = (record.get("concrete"), record.get("rebar"))
    del scene["rcg_last_pile"]  # Release Blender ID users before testing for shared data.
    if (not isinstance(coll, bpy.types.Collection) or not coll.get("rcg_pile")
            or scene.collection.children.get(coll.name) != coll):
        return
    if coll.users - int(coll.use_fake_user) > 1:
        # Another scene or collection instance still needs the original parts.
        # Edit a shallow collection copy so that unrelated members stay linked.
        local = coll.copy()
        local.use_fake_user = False
        scene.collection.children.link(local)
        scene.collection.children.unlink(coll)
        coll = local
    for ob in parts:
        if (not isinstance(ob, bpy.types.Object) or ob.type != "MESH"
                or not ob.get("rcg_rubble_part") or coll.objects.get(ob.name) != ob):
            continue
        mesh = ob.data
        coll.objects.unlink(ob)
        if not ob.users:
            bpy.data.objects.remove(ob)
        if not mesh.users:
            bpy.data.meshes.remove(mesh)
    if not coll.objects and not coll.children and not coll.use_fake_user:
        bpy.data.collections.remove(coll)


def _scene_blockers(scene, layer, origin, ignored, low, high):
    """Copy nearby evaluated mesh obstacles into pile coordinates, including instances."""
    obstacles = []
    to_pile = Matrix.Translation(-Vector(origin))
    with bpy.context.temp_override(scene=scene, view_layer=layer):
        graph = bpy.context.evaluated_depsgraph_get()
        for instance in graph.object_instances:
            obj = instance.object
            if obj.type != "MESH" or not instance.show_self:
                continue
            if not instance.is_instance and (obj.original in ignored or not obj.original.visible_get(view_layer=layer)):
                continue
            transform = to_pile @ instance.matrix_world
            bounds = [transform @ Vector(corner) for corner in obj.bound_box]
            if any(max(v[i] for v in bounds) < low[i] or min(v[i] for v in bounds) > high[i] for i in range(3)):
                continue
            mesh = obj.data
            if not mesh.polygons:
                continue
            vertices = [transform @ v.co for v in mesh.vertices]
            # A reflected instance reverses geometric winding in pile coordinates.
            winding = -1 if transform.to_3x3().determinant() < 0 else 1
            mesh.calc_loop_triangles()
            faces = [tri.vertices[:][::winding] for tri in mesh.loop_triangles]
            obstacles.append((vertices, faces))
    return obstacles


class _SolidShell:
    """A closed surface used only for testing whether an entire bar root is buried."""
    direction = Vector((0.387119, 0.814285, 0.431721)).normalized()

    def __init__(self, vertices, triangles):
        self.low = tuple(min(v[i] for v in vertices) for i in range(3))
        self.high = tuple(max(v[i] for v in vertices) for i in range(3))
        self.tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)

    def winding_at(self, point):
        if not (self.low[0] < point.x < self.high[0] and self.low[1] < point.y < self.high[1]
                and self.low[2] < point.z < self.high[2]):
            return 0
        # Count crossings per shell. For nested surfaces the outer wall and
        # inward-facing cavity cancel. A distant open wall never enters this test.
        winding = 0
        while True:
            hit, normal, _, _ = self.tree.ray_cast(point, self.direction)
            if hit is None:
                return winding
            winding += 1 if normal.dot(self.direction) > 0 else -1
            advance = max(1e-7, max(abs(x) for x in hit) * 4e-7)
            point = hit + self.direction * advance


def _closed_shells(vertices, triangles, cap_mound=False):
    """Separate edge-connected components so loose open geometry remains a surface blocker."""
    parent = list(range(len(triangles)))

    def component(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = {}
    for i, (a, b, c) in enumerate(triangles):
        for start, end in ((a,b), (b,c), (c,a)):
            key = (min(start, end), max(start, end))
            if key in edges:
                first, count = edges[key]
                parent[component(i)] = component(first)
                edges[key] = (first, count + 1)
            else:
                edges[key] = (i, 1)
    groups, unclosed = {}, {}
    for i, tri in enumerate(triangles):
        groups.setdefault(component(i), []).append(tri)
    for edge, (i, count) in edges.items():
        if count != 2:
            unclosed.setdefault(component(i), []).append((edge, count))

    shells = []
    for group, faces in groups.items():
        boundary = unclosed.get(group, [])
        if boundary and not cap_mound:
            continue
        ids = {i for tri in faces for i in tri}
        remap = {old: new for new, old in enumerate(ids)}
        coords = [vertices[i] for i in ids]
        local = [tuple(remap[i] for i in face) for face in faces]
        if boundary:
            # The generated mound is a dome with one flat, radial lower rim.
            # Close that rim only in the containment copy.
            bottom = min(v.z for v in coords)
            if any(n != 1 or any(abs(vertices[i].z - bottom) > 1e-5 for i in edge)
                   for edge, n in boundary):
                continue
            rim = {remap[i] for edge, _ in boundary for i in edge}
            center = len(coords)
            coords.append(sum((coords[i] for i in rim), Vector()) / len(rim))
            for (a,b), _ in boundary:
                a, b = remap[a], remap[b]
                if (coords[a] - coords[center]).cross(coords[b] - coords[center]).z > 0:
                    a, b = b, a
                local.append((center, a, b))
        shells.append(_SolidShell(coords, local))
    return shells


class _RebarColliders:
    def __init__(self, surface, solids):
        self.surface = surface
        self.solids = solids
        self.ray_cast = surface.ray_cast
        self.find_nearest = surface.find_nearest
        self.find_nearest_range = surface.find_nearest_range

    def contains(self, point):
        return any(sum(shell.winding_at(point) for shell in solid) != 0 for solid in self.solids)


def _rebar_collision_tree(mesh, kept, blockers):
    """Use Blender's final displayed triangles for both mounting and collisions."""
    vertices = [v.co.copy() for v in mesh.vertices]
    mesh.calc_loop_triangles()
    faces = [t.vertices[:] for t in mesh.loop_triangles]
    source_ids = mesh.attributes["_rebar_source_face"].data
    by_id = {}
    for idx, tri in enumerate(mesh.loop_triangles):
        by_id.setdefault(source_ids[tri.polygon_index].value, []).append(idx)
    solids = []
    for piece in kept:
        ids = [frozenset(by_id.get(i, ())) for i in piece.pop("source_ids")]
        piece["collision_faces"] = ids
        triangles = [faces[i] for indices in ids for i in indices]
        solids.append(_closed_shells(vertices, triangles, cap_mound=piece["kind"] == "mound"))
    # Loose ground rocks share source ID zero, but each one is a separate solid.
    solids.extend([shell] for shell in _closed_shells(vertices, [faces[i] for i in by_id.get(0, ())]))
    if blockers:
        # The growth plan is shared by every length setting, including its longest tips.
        margin = MAX_REBAR_LENGTH + math.sqrt(2) * P["rebar_radius"]
        low = [min(v[i] for v in vertices) - margin for i in range(3)]
        high = [max(v[i] for v in vertices) + margin for i in range(3)]
        for other_verts, other_faces in blockers(low, high):
            solids.append(_closed_shells(other_verts, other_faces))
            start = len(vertices)
            vertices.extend(other_verts)
            faces.extend(tuple(start + i for i in face) for face in other_faces)
    return _RebarColliders(BVHTree.FromPolygons(vertices, faces, all_triangles=True), solids)


def _finish_pile_object(bm, name, collection):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if "fracture" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["fracture"])
    obj = bpy.data.objects.new(name, mesh)
    obj["rcg_rubble_part"] = True
    collection.objects.link(obj)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    if mesh.polygons:
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = obj
        with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj], edit_object=obj):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def build(seed=SEED, replace=True, blockers=None):
    """Build the pile in the current scene. Use generate() to keep it out of your rigid-body world."""
    rng = random.Random(seed)
    if replace:
        _clear_previous()
    coll = bpy.data.collections.new(COLL_NAME)
    coll["rcg_pile"] = True
    bpy.context.scene.collection.children.link(coll)
    work = bpy.data.collections.new("_rubble_work")
    work["rcg_work"] = True
    bpy.context.scene.collection.children.link(work)

    pieces = []

    def add(ob, exits, kind):
        pieces.append({"ob": ob, "exits": exits, "kind": kind})

    big_chunks = 6
    for i in range(big_chunks):
        add(*make_chunk(rng, work, f"chunk_big{i}", 0.45, 0.8, rng.randint(6, 9), 0.35),
            "chunk")
    for i in range(P["slabs"]):
        L, W, T = rng.uniform(1.1, 1.5), rng.uniform(0.75, 1.05), rng.uniform(0.16, 0.22)
        add(*make_slab(rng, work, f"slab{i}", L, W, T, bites=rng.randint(1, 2)), "slab")
    for i in range(P["walls"]):
        L, W, T = rng.uniform(0.95, 1.25), rng.uniform(0.7, 0.95), rng.uniform(0.18, 0.24)
        add(*make_slab(rng, work, f"wall{i}", L, W, T, bites=1, wall_window=(i == 0)), "wall")
    for i in range(P["columns"]):
        add(*make_column(rng, work, f"column{i}"), "column")
    for i in range(P["chunks"] - big_chunks):
        add(*make_chunk(rng, work, f"chunk{i}", 0.25, 0.55, rng.randint(6, 9), 0.15),
            "chunk")
    for i in range(P["small_chunks"]):
        add(*make_chunk(rng, work, f"small{i}", 0.1, 0.25, rng.randint(5, 7), 0.0), "small")

    # big pieces first -> they end up lower; some chunks first so slabs lean on them
    order = [p for p in pieces if p["kind"] == "chunk"][:big_chunks]
    order += [p for p in pieces if p["kind"] in ("slab", "wall")]
    order += [p for p in pieces if p["kind"] == "column"]
    order += [p for p in pieces if p["kind"] == "chunk"][big_chunks:]
    order += [p for p in pieces if p["kind"] == "small"]
    mound = make_mound(rng, work)
    settle(rng, order, work, mound)
    pieces.append({"ob": mound, "exits": [], "kind": "mound", "matrix": Matrix.Identity(4)})

    # drop anything that rolled away
    def inside(p):
        t = p["matrix"].translation
        e = math.hypot(t.x / P["fence_half_len"], t.y / P["fence_half_wid"])
        return p["kind"] == "mound" or e < (1.8 if p["kind"] == "small" else 1.45)
    keep = [p for p in pieces if inside(p)]

    # ---- concrete: join settled pieces
    con = bmesh.new()
    if P["max_bars"]:
        con.faces.layers.int.new("_rebar_source_face")
    next_id = 1
    for p in keep:
        me = p["ob"].data.copy()
        if P["max_bars"]:
            ids = range(next_id, next_id + len(me.polygons))
            me.attributes.new("_rebar_source_face", "INT", "FACE").data.foreach_set("value", ids)
            p["source_ids"] = ids
            next_id = ids.stop
        me.transform(p["matrix"])
        con.from_mesh(me)
        bpy.data.meshes.remove(me)
    # recentre on the ground: xy centre of the pile, ground stays at z = 0
    xs = [v.co.x for v in con.verts]
    ys = [v.co.y for v in con.verts]
    shift = Vector((-(min(xs) + max(xs)) / 2, -(min(ys) + max(ys)) / 2, 0.0))
    bmesh.ops.translate(con, vec=shift, verts=con.verts)
    con_bvh = BVHTree.FromBMesh(con)
    half_len = (max(xs) - min(xs)) / 2
    half_wid = (max(ys) - min(ys)) / 2

    # ---- ground rocks around the base
    pile = con
    for rbm in place_rocks(rng, P["ground_rocks"], con_bvh, half_len, half_wid):
        me = bpy.data.meshes.new("_rock")
        rbm.to_mesh(me)
        rbm.free()
        pile.from_mesh(me)
        bpy.data.meshes.remove(me)

    # ---- two objects, no materials (assigned later for Substance Painter):
    #      RubblePile = concrete pieces + core + ground rocks,  RubblePile_Rebar = bars
    sharp = math.radians(P["sharp_angle_deg"])
    for f in pile.faces:
        f.smooth = True
    for e in pile.edges:
        if len(e.link_faces) == 2:
            e.smooth = e.calc_face_angle(math.pi) < sharp      # hard edges on the breaks
    bmesh.ops.remove_doubles(pile, verts=pile.verts[:], dist=1e-5)

    # UV projection goes through Edit Mode. Sample the finished mesh so that
    # nonplanar faces use the same triangles for drawing and collision.
    concrete = _finish_pile_object(pile, ASSET_NAME, coll)
    rad = P["rebar_radius"]
    paths = []
    if P["max_bars"]:
        con_bvh = _rebar_collision_tree(concrete.data, keep, blockers)
        concrete.data.attributes.remove(concrete.data.attributes["_rebar_source_face"])
        paths = [path for *_, path in pick_bars(keep, shift, con_bvh, seed, with_paths=True)]
    steel = _finish_pile_object(rebar_bm(paths, rad), ASSET_NAME + "_Rebar", coll)
    DEBUG.update(bars=len(paths), kept=len(keep), dropped=len(pieces) - len(keep),
                 pile_half=(round(half_len, 2), round(half_wid, 2)))

    # ---- tidy up working objects
    for o in list(work.objects):
        m = o.data
        bpy.data.objects.remove(o)
        if m and m.users == 0:
            bpy.data.meshes.remove(m)
    bpy.data.collections.remove(work)
    return [concrete, steel]


def stats(objs):
    out = {}
    for ob in objs:
        me = ob.data
        out[ob.name] = {"verts": len(me.vertices), "tris": sum(len(p.vertices) - 2 for p in me.polygons)}
    vs = [v.co for ob in objs for v in ob.data.vertices]
    out["size_m"] = (round(max(v.x for v in vs) - min(v.x for v in vs), 2),
                     round(max(v.y for v in vs) - min(v.y for v in vs), 2), round(max(v.z for v in vs), 2))
    return out


def generate(seed=SEED, location=(0.0, 0.0, 0.0), replace=False, params=None):
    """Build a pile in a throw-away scene (so the scene's own rigid-body world is never
    touched), then link it into the current scene at `location`."""
    user_scene = bpy.context.scene
    layer = bpy.context.view_layer
    previous = user_scene.get("rcg_last_pile") if replace else None
    ignored = set()
    previous_collection = previous.get("collection") if previous else None
    if (isinstance(previous_collection, bpy.types.Collection) and previous_collection.get("rcg_pile")
            and user_scene.collection.children.get(previous_collection.name) == previous_collection):
        # Objects also linked elsewhere in this scene, and any other visible
        # instances of the old pile, stay in place when the original is replaced.
        scene_collections = set(user_scene.collection.children_recursive) | {user_scene.collection}
        for key in ("concrete", "rebar"):
            obj = previous.get(key)
            if (isinstance(obj, bpy.types.Object) and obj.get("rcg_rubble_part")
                    and set(obj.users_collection) & scene_collections == {previous_collection}):
                ignored.add(obj)
    def blockers(low, high):
        return _scene_blockers(user_scene, layer, location, ignored, low, high)

    sim = bpy.data.scenes.new("_rubble_sim")
    original_settings = P.copy()
    try:
        if params:
            P.update(params)
        with bpy.context.temp_override(scene=sim, view_layer=sim.view_layers[0]):
            objs = build(seed, replace=False, blockers=blockers)
        for ob in objs:
            ob.location = location
        coll = objs[0].users_collection[0]
        user_scene.collection.children.link(coll)
        if replace:
            _clear_previous(user_scene)
        user_scene["rubble_last"] = coll.name
        user_scene["rcg_last_pile"] = {"collection": coll, "concrete": objs[0], "rebar": objs[1]}
        sim.collection.children.unlink(coll)
    finally:
        P.clear()
        P.update(original_settings)
        # A failed build can leave temporary objects in the isolated simulation.
        for collection in list(sim.collection.children):
            if not (collection.get("rcg_work") or collection.get("rcg_pile")):
                continue
            for obj in list(collection.objects):
                mesh = obj.data
                bpy.data.objects.remove(obj)
                if isinstance(mesh, bpy.types.Mesh) and not mesh.users:
                    bpy.data.meshes.remove(mesh)
            bpy.data.collections.remove(collection)
        bpy.data.scenes.remove(sim)
    return objs


def main():
    objs = generate(SEED, replace=True)
    print("Rubble pile:", stats(objs))
    return objs


if __name__ == "__main__":
    main()
