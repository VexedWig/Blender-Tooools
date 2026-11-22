"""Dense UV: unwrap and pack selected mesh objects to fill as much texture space as possible.

Sidebar (N) > Dense UV, in the 3D Viewport and the UV Editor.

    Unwrap & Pack   re-unwrap, straighten, equalize, stack and pack
    Pack Only       keep the current unwrap, optimize only the layout
    Analyze         report fill %, texel density and stretch of the current UVs
"""
import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty

from . import core

SIZES = [(str(n), str(n), f"{n} px") for n in (256, 512, 1024, 2048, 4096, 8192)]


class DenseUVSettings(bpy.types.PropertyGroup):
    unwrap_mode: EnumProperty(
        name="Unwrap",
        items=[
            ('HYBRID', "Hybrid", "Cut seams at hard angles, sharp edges and material borders, then "
             "unwrap with low stretch. Islands that come out distorted or overlapping are "
             "re-projected with Smart UV Project"),
            ('SMART', "Smart UV", "Smart UV Project on everything"),
            ('KEEP', "Keep", "Keep the current unwrap and only optimize the layout"),
        ],
        default='HYBRID',
    )
    unwrap_method: EnumProperty(
        name="Method",
        items=[
            ('ANGLE_BASED', "Angle Based", "Fast, good for most hard-surface parts"),
            ('MINIMUM_STRETCH', "Minimum Stretch", "Least area stretch, slower"),
            ('CONFORMAL', "Conformal", "Preserves angles, more area stretch"),
        ],
        default='ANGLE_BASED',
    )
    seam_angle: FloatProperty(
        name="Seam Angle", subtype='ANGLE', default=math.radians(50.0),
        min=math.radians(1.0), max=math.radians(180.0),
        description="Edges bending more than this become seams",
    )
    seams_from_sharp: BoolProperty(name="Sharp Edges", default=False,
                                   description="Edges marked sharp become seams (ignored on meshes "
                                               "where nearly every edge is sharp)")
    keep_existing_seams: BoolProperty(name="Existing Seams", default=True,
                                      description="Keep the seams already marked on the mesh")
    seams_from_materials: BoolProperty(name="Material Borders", default=True,
                                       description="Cut where the material changes")
    max_distortion: FloatProperty(
        name="Max Stretch", default=0.3, min=0.0, max=4.0,
        description="Islands stretched more than this (average area error) are re-projected with "
                    "Smart UV Project",
    )
    smart_angle: FloatProperty(
        name="Smart Angle", subtype='ANGLE', default=math.radians(66.0),
        min=math.radians(1.0), max=math.radians(89.0),
        description="Angle limit for Smart UV Project (Smart mode and re-projected islands)",
    )
    write_seams: BoolProperty(name="Mark Seams", default=False,
                              description="Mark the final island borders as seams")

    rectify: BoolProperty(
        name="Straighten Quads", default=True,
        description="Lay out all-quad islands as rectangle grids when that adds no stretch. "
                    "Straight islands pack tighter",
    )
    rectify_tolerance: FloatProperty(
        name="Corner Tolerance", subtype='ANGLE', default=math.radians(15.0),
        min=0.0, max=math.radians(45.0),
        description="Only straighten islands whose 3D quad corners are this close to 90 degrees "
                    "on average",
    )
    equalize: BoolProperty(
        name="Equal Texel Density", default=True,
        description="Scale islands so every square metre gets the same texture area "
                    "(world space, object scale included)",
    )
    stack: BoolProperty(
        name="Stack Identical", default=True,
        description="Overlap islands with identical shape and size so they share texture space",
    )
    stack_mode: EnumProperty(
        name="Stack",
        items=[
            ('COPIES', "Copies Only", "Stack islands only on duplicated parts (identical loose pieces, "
             "such as repeated bolts). Faces of one part never share pixels"),
            ('SHAPES', "Any Identical Shape", "Stack every island with the same shape and size, even "
             "faces of one part (the four sides of a post). Saves more space; those faces show the "
             "same texture"),
        ],
        default='COPIES',
    )
    stack_offset: BoolProperty(
        name="Offset Copies +1 U", default=True,
        description="Move stacked copies one UV tile right. They sample the same pixels in a "
                    "repeating texture, but bakers like Substance Painter only bake the original",
    )
    stack_mirrored: BoolProperty(
        name="Allow Mirrored", default=False,
        description="Also stack mirror images (flips the texture on the copy; bad for normal maps)",
    )

    texture_width: EnumProperty(name="Width", items=SIZES, default='2048')
    texture_height: EnumProperty(name="Height", items=SIZES, default='2048')
    padding: FloatProperty(
        name="Padding", default=8.0, min=0.0, max=128.0, precision=1,
        description="Gap between islands in pixels (half of it at the texture border). "
                    "8 px at 2K keeps mipmaps clean",
    )
    allow_rotation: BoolProperty(name="Rotate Islands", default=True)
    quality: EnumProperty(
        name="Search",
        items=[
            ('FAST', "Fast", "One unwrap, one packing pass"),
            ('BALANCED', "Balanced", "Try Hybrid and Smart unwraps with 3 rotation strategies (plus a "
             "diagonal pass when a long island caps the scale); keep the best per object"),
            ('BEST', "Best", "As Balanced, and also try every object with straightening off. "
             "About twice as slow"),
        ],
        default='BALANCED',
    )
    shared: BoolProperty(
        name="One Shared Texture", default=False,
        description="Pack all selected objects into one 0-1 space (one texture set). "
                    "Off: each object fills its own 0-1 space",
    )
    last_report: StringProperty(default="")


def _run(op, context, fn):
    s = context.scene.dense_uv
    try:
        lines = fn(context, s)
    except core.DenseUVError as e:
        op.report({'ERROR'}, str(e))
        return {'CANCELLED'}
    s.last_report = "\n".join(lines)
    op.report({'INFO'}, lines[0])
    return {'FINISHED'}


class UV_OT_dense_unwrap_pack(bpy.types.Operator):
    """Unwrap and pack the selected mesh objects for maximum texture-space usage"""
    bl_idname = "uv.dense_unwrap_pack"
    bl_label = "Dense UV Unwrap & Pack"
    bl_options = {'REGISTER', 'UNDO'}

    pack_only: BoolProperty(name="Pack Only", default=False,
                            description="Keep the current unwrap and only optimize the layout",
                            options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'} and bool(core.target_objects(context))

    def execute(self, context):
        return _run(self, context, lambda c, s: core.run(c, s, self.pack_only))


class UV_OT_dense_analyze(bpy.types.Operator):
    """Report fill %, texel density and stretch of the selected objects' current UVs"""
    bl_idname = "uv.dense_analyze"
    bl_label = "Analyze UVs"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'} and bool(core.target_objects(context))

    def execute(self, context):
        return _run(self, context, core.analyze)


def draw_panel(layout, context):
    s = context.scene.dense_uv
    col = layout.column(align=True)
    col.scale_y = 1.4
    col.operator(UV_OT_dense_unwrap_pack.bl_idname, text="Unwrap & Pack", icon='UV')
    row = layout.row(align=True)
    row.operator(UV_OT_dense_unwrap_pack.bl_idname, text="Pack Only", icon='STICKY_UVS_LOC').pack_only = True
    row.operator(UV_OT_dense_analyze.bl_idname, text="Analyze", icon='VIEWZOOM')

    box = layout.box()
    box.label(text="Unwrap", icon='MOD_UVPROJECT')
    box.row().prop(s, "unwrap_mode", expand=True)
    if s.unwrap_mode == 'HYBRID':
        col = box.column(align=True)
        col.prop(s, "seam_angle")
        col.prop(s, "unwrap_method", text="")
        row = box.row(align=True)
        row.prop(s, "seams_from_sharp", toggle=True)
        row.prop(s, "keep_existing_seams", toggle=True)
        box.prop(s, "seams_from_materials")
        col = box.column(align=True)
        col.prop(s, "max_distortion")
        col.prop(s, "smart_angle")
    elif s.unwrap_mode == 'SMART':
        box.prop(s, "smart_angle")
    if s.unwrap_mode != 'KEEP':
        box.prop(s, "write_seams")

    box = layout.box()
    box.label(text="Islands", icon='UV_ISLANDSEL')
    row = box.row(align=True)
    row.prop(s, "rectify")
    sub = row.row()
    sub.active = s.rectify
    sub.prop(s, "rectify_tolerance", text="")
    box.prop(s, "equalize")
    box.prop(s, "stack")
    if s.stack:
        col = box.column(align=True)
        col.prop(s, "stack_mode", text="")
        col.prop(s, "stack_offset")
        col.prop(s, "stack_mirrored")

    box = layout.box()
    box.label(text="Pack", icon='PACKAGE')
    row = box.row(align=True)
    row.prop(s, "texture_width", text="")
    row.label(text="", icon='X')
    row.prop(s, "texture_height", text="")
    box.prop(s, "padding", text="Padding (px)")
    box.prop(s, "allow_rotation")
    box.prop(s, "quality")
    box.prop(s, "shared")

    if s.last_report:
        box = layout.box()
        lines = s.last_report.split("\n")
        box.label(text=lines[0], icon='INFO')
        col = box.column(align=True)
        for line in lines[1:]:
            col.label(text=line)


class VIEW3D_PT_dense_uv(bpy.types.Panel):
    bl_label = "Dense UV"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Dense UV"

    def draw(self, context):
        draw_panel(self.layout, context)


class IMAGE_PT_dense_uv(bpy.types.Panel):
    bl_label = "Dense UV"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Dense UV"

    def draw(self, context):
        draw_panel(self.layout, context)


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(UV_OT_dense_unwrap_pack.bl_idname, text="Dense UV Unwrap & Pack")


classes = (DenseUVSettings, UV_OT_dense_unwrap_pack, UV_OT_dense_analyze,
           VIEW3D_PT_dense_uv, IMAGE_PT_dense_uv)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dense_uv = PointerProperty(type=DenseUVSettings)
    bpy.types.VIEW3D_MT_uv_map.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_uv_map.remove(menu_func)
    del bpy.types.Scene.dense_uv
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
