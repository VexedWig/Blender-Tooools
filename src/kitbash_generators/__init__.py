# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
bl_info = {
    "name": "Rubble + Cable Generator",
    "author": "Aidan Hart",
    "version": (2, 3, 1),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Generators",
    "description": "Rubble pile and cable / pipe run generators in one compact sidebar panel",
    "category": "Add Mesh",
}

"""
Rubble + Cable Generator
------------------------
One sidebar panel ("Generators" tab) with a Rubble / Cable switcher at the top.

Rubble   Size (L / W / H), a collapsible Pieces section (slabs, walls, columns, chunks,
         rocks, rebar), seed, Replace Last, Generate. A new pile is built at the 3D cursor
         (RubblePile + RubblePile_Rebar, no materials). The settling simulation runs in a
         throw-away scene, so your own rigid-body world is untouched.
Cable    Add Cable Run puts the Cable Generator object with Start / End empties at the
         3D cursor (cursor rotation is used, so +Y = out of the wall). New runs share
         a generator-owned node group. Refresh rebuilds that group. Older untagged
         runs continue to use their original node trees.

The sidebar tab name can be changed in the add-on's preferences.
This replaces kitbash_generators.py (same two tools, two panels); turn that one off if it
is installed.

The cable and rubble modules share the panel and settings below.
"""

import random

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty


from . import cable, rubble

DEFAULT_TAB = "Generators"


# --------------------------------------------------------------------------------------
#  Settings
# --------------------------------------------------------------------------------------
def _get_add_rebar(settings):
    # Previous versions stored a zero bar count when rebar was disabled.
    return settings.get("add_rebar", settings.bars > 0)


def _set_add_rebar(settings, enabled):
    settings["add_rebar"] = enabled
    if enabled and settings.bars == 0:
        settings.bars = 16


class RCG_RubbleProps(bpy.types.PropertyGroup):
    seed: IntProperty(name="Seed", default=47, min=0)
    length: FloatProperty(name="Length", default=4.6, min=1.0, soft_max=12.0, unit="LENGTH",
                          description="Rough pile length (a sedan is about 4.6 m)")
    width: FloatProperty(name="Width", default=1.9, min=0.5, soft_max=6.0, unit="LENGTH",
                         description="Rough pile width")
    height: FloatProperty(name="Height", default=1.4, min=0.3, soft_max=4.0, unit="LENGTH",
                          description="Rough pile height")
    slabs: IntProperty(name="Slabs", default=3, min=0, soft_max=12, description="Broken floor slabs")
    walls: IntProperty(name="Wall Pieces", default=2, min=0, soft_max=12)
    columns: IntProperty(name="Columns", default=2, min=0, soft_max=8, description="Broken column stubs")
    chunks: IntProperty(name="Chunks", default=30, min=6, soft_max=120)
    small_chunks: IntProperty(name="Small Chunks", default=20, min=0, soft_max=120)
    ground_rocks: IntProperty(name="Ground Rocks", default=40, min=0, soft_max=200)
    add_rebar: BoolProperty(name="Add Rebar", get=_get_add_rebar, set=_set_add_rebar)
    rebar_amount: FloatProperty(name="Amount", subtype="FACTOR", default=0.0, min=0.0, max=1.0,
                                description="Start on a small break, expose aligned opposite ends, then fill the row. Higher values can add a transverse row")
    additional_pieces: IntProperty(name="Additional Pieces", default=0, min=0, soft_max=32,
                                   description="Number of extra concrete pieces to reinforce, largest first. Zero reinforces only the largest eligible piece")
    bars: IntProperty(name="Bar Limit per Piece", default=16, min=0, soft_max=60,
                      description="Maximum bars on each piece. Available fracture area, bar width and clearance determine how many fit")
    bar_length: FloatProperty(name="Rebar Length", default=0.35, min=0.02, max=rubble.MAX_REBAR_LENGTH,
                                unit="LENGTH", description="Maximum exposed length. Bars grow in segments and stop at obstructions")
    bar_thickness: FloatProperty(name="Rebar Width", default=0.024, min=0.004, soft_max=0.06,
                                 unit="LENGTH", description="Width of the square bar. Its full base must fit the concrete surface")
    replace: BoolProperty(name="Replace Last Pile", default=False,
                          description="Delete the most recent pile before building the new one")


class RCG_CableProps(bpy.types.PropertyGroup):
    demo_surfaces: BoolProperty(name="Demo Floor + Wall", default=False,
                                description="Also add a 6 x 4 m floor and a 3 m wall to see it against")


class RCG_UIProps(bpy.types.PropertyGroup):
    tool: EnumProperty(name="Tool", default="RUBBLE", items=[
        ("RUBBLE", "Rubble", "Rubble pile generator", "MOD_EXPLODE", 0),
        ("CABLE", "Cable", "Cable / pipe run generator", "CURVE_PATH", 1),
    ])


def _rubble_params(p):
    return {
        "mound_half_len": 0.33 * p.length, "mound_half_wid": 0.29 * p.width, "mound_height": 0.46 * p.height,
        "fence_half_len": 0.43 * p.length, "fence_half_wid": 0.43 * p.width,
        "slabs": p.slabs, "walls": p.walls, "columns": p.columns, "chunks": p.chunks,
        "small_chunks": p.small_chunks, "ground_rocks": p.ground_rocks,
        "max_bars": p.bars if p.add_rebar else 0,
        "rebar_amount": p.rebar_amount, "additional_pieces": p.additional_pieces,
        "rebar_length": p.bar_length, "rebar_radius": p.bar_thickness / 2.0,
    }


# --------------------------------------------------------------------------------------
#  Operators
# --------------------------------------------------------------------------------------
class RCG_OT_rubble_generate(bpy.types.Operator):
    bl_idname = "rcg.rubble_generate"
    bl_label = "Generate Rubble Pile"
    bl_description = "Build a rubble pile at the 3D cursor (takes a few seconds)"
    bl_options = {"REGISTER", "UNDO"}

    new_seed: BoolProperty(name="New Seed", default=False, options={"SKIP_SAVE", "HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        if properties.new_seed:
            return "Pick a random seed and build a rubble pile at the 3D cursor"
        return cls.bl_description

    def execute(self, context):
        p = context.scene.rcg_rubble
        if self.new_seed:
            p.seed = random.randint(0, 99999)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        objs = rubble.generate(p.seed, location=context.scene.cursor.location.copy(), replace=p.replace,
                               params=_rubble_params(p))
        for o in context.selected_objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        context.view_layer.objects.active = objs[0]
        s = rubble.stats(objs)
        tris = sum(v["tris"] for k, v in s.items() if k != "size_m")
        self.report({"INFO"}, f"Rubble pile: {tris} tris, {s['size_m'][0]} x {s['size_m'][1]} m")
        return {"FINISHED"}


class RCG_OT_cable_add(bpy.types.Operator):
    bl_idname = "rcg.cable_add"
    bl_label = "Add Cable Run"
    bl_description = ("Add a Cable Generator object at the 3D cursor. Its origin is the wall/floor corner: "
                      "+Y points out of the wall, +Z up")
    bl_options = {"REGISTER", "UNDO"}

    rebuild: BoolProperty(name="Rebuild Node Group", default=False, options={"SKIP_SAVE", "HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        if properties.rebuild:
            return ("Rebuild the Cable Generator node group from the add-on, then add a cable run. "
                    "Existing cable runs keep their settings")
        return cls.bl_description

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            gen = cable.create_run(context.scene.cursor.matrix.copy(),
                                   context.scene.rcg_cable.demo_surfaces, refresh=self.rebuild)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self.report({"ERROR"}, f"Could not create cable run: {exc}")
            return {"CANCELLED"}
        for o in context.selected_objects:
            o.select_set(False)
        gen.select_set(True)
        context.view_layer.objects.active = gen
        return {"FINISHED"}


# --------------------------------------------------------------------------------------
#  Panel (one panel, tool switcher at the top)
# --------------------------------------------------------------------------------------
class RCG_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = DEFAULT_TAB
    bl_label = "Rubble + Cable"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        ui = scene.rcg_ui
        layout.row(align=True).prop(ui, "tool", expand=True)
        if ui.tool == "RUBBLE":
            self.draw_rubble(layout, scene.rcg_rubble)
        else:
            self.draw_cable(layout, scene.rcg_cable)

    @staticmethod
    def draw_rubble(layout, p):
        row = layout.row(align=True)
        row.prop(p, "length", text="L")
        row.prop(p, "width", text="W")
        row.prop(p, "height", text="H")

        header, body = layout.panel("RCG_PT_rubble_pieces", default_closed=True)
        header.label(text="Pieces")
        if body:
            grid = body.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            grid.prop(p, "slabs")
            grid.prop(p, "walls", text="Walls")
            grid.prop(p, "columns")
            grid.prop(p, "chunks")
            grid.prop(p, "small_chunks", text="Small")
            grid.prop(p, "ground_rocks", text="Rocks")

        layout.prop(p, "add_rebar")
        if p.add_rebar:
            controls = layout.column(align=True)
            controls.prop(p, "rebar_amount", slider=True)
            controls.prop(p, "additional_pieces", slider=True)
            controls.prop(p, "bar_length")
            controls.prop(p, "bar_thickness")
            controls.prop(p, "bars")

        row = layout.row(align=True)
        row.prop(p, "seed")
        row.prop(p, "replace", text="Replace Last", toggle=True)
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("rcg.rubble_generate", text="Generate", icon="MOD_EXPLODE")
        row.operator("rcg.rubble_generate", text="", icon="FILE_REFRESH").new_seed = True

    @staticmethod
    def draw_cable(layout, c):
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("rcg.cable_add", text="Add Cable Run", icon="CURVE_PATH")
        row.operator("rcg.cable_add", text="", icon="FILE_REFRESH").rebuild = True
        layout.prop(c, "demo_surfaces")
        layout.label(text="Origin = wall / floor corner", icon="INFO")


# --------------------------------------------------------------------------------------
#  Preferences (sidebar tab name)
# --------------------------------------------------------------------------------------
def _register_panel(tab):
    try:
        bpy.utils.unregister_class(RCG_PT_main)
    except RuntimeError:
        pass
    RCG_PT_main.bl_category = (tab or "").strip() or DEFAULT_TAB
    bpy.utils.register_class(RCG_PT_main)


def _tab_changed(self, context):
    _register_panel(self.tab_name)


class RCG_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    tab_name: StringProperty(name="Sidebar Tab", default=DEFAULT_TAB, update=_tab_changed,
                             description="Sidebar (N) tab the panel lives in. "
                                         "Type the name of an existing tab to share it")

    def draw(self, context):
        self.layout.prop(self, "tab_name")


CLASSES = (RCG_Preferences, RCG_RubbleProps, RCG_CableProps, RCG_UIProps,
           RCG_OT_rubble_generate, RCG_OT_cable_add)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.rcg_rubble = PointerProperty(type=RCG_RubbleProps)
    bpy.types.Scene.rcg_cable = PointerProperty(type=RCG_CableProps)
    bpy.types.Scene.rcg_ui = PointerProperty(type=RCG_UIProps)
    addon = bpy.context.preferences.addons.get(__name__)
    _register_panel(addon.preferences.tab_name if addon and addon.preferences else DEFAULT_TAB)


def unregister():
    try:
        bpy.utils.unregister_class(RCG_PT_main)
    except RuntimeError:
        pass
    del bpy.types.Scene.rcg_ui
    del bpy.types.Scene.rcg_cable
    del bpy.types.Scene.rcg_rubble
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
