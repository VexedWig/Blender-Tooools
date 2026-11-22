bl_info = {
    "name": "N-Fix Addon Panel Optimizer",
    "author": "Aidan Hart",
    "version": (3, 0, 10),
    "blender": (4, 5, 0),
    "location": "3D Viewport > top centre, N key",
    "description": "Organize Sidebar tabs and open their expanded controls in a persistent dock",
    "category": "Interface",
}

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Aidan Hart

"""
N-Fix Addon Panel Optimizer
---------------
The currently available Sidebar (N panel) tabs of the 3D viewport as a stack of buttons centred
at the top of the viewport. Selected controls open in an opaque dock at the right.

  Favourites box         right-click any button to pin it (10 at most); pinned buttons live in a
                         box at the very top that sizes itself to fit and stays centred
  Arrow under the box    unfolds the rest of the stack below it (points up while open)
  Arrow under the stack  folds it away again; N does the same while the viewport has keyboard focus
  Click a tab            shows its expanded panels in the right-hand dock, keeping the stack open
                         (the title pin adds a favourite and X closes the dock)
  Folder button         opens the user add-ons folder; Shift opens the extension's repository

The dock stays open on outside clicks. Press N over the viewport to close it and fold the stack.
Original add-on draw callbacks are reused with their Sidebar formatting.

Rows alternate long and short (10 / 8 by default, pref "Long Row"): the row under the favourites
is short, the next long, and so on. The stack is centred on the whole viewport, so nothing moves
when windows open or N is pressed. In a narrow viewport the rows shorten; in a very narrow one the
buttons form a single column of equal width; when even that won't fit, the overflow goes into a
"+N" button that lists the rest. Buttons never overlap, and quad view shows one stack, top right.

The tab list is rebuilt at startup, after every file load and every few seconds, so newly
installed or enabled add-ons appear on their own. Pins are kept in config/npanel_tab_bar_pins.json.
The stack lives in the gizmo layer, so the viewport's Show Gizmo toggle hides it too.
"""

import inspect
import json
import math
import os
import sys
import time
import types
from collections import deque

import addon_utils
import blf
import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from gpu_extras.batch import batch_for_shader

FONT = 0
HEADER_TYPES = {"HEADER", "TOOL_HEADER", "ASSET_SHELF", "ASSET_SHELF_HEADER", "FOOTER"}
ANIM_SECONDS = 0.16
RESCAN_SECONDS = 3.0
BUTTON_SCALE = 0.85 * 1.20   # V2 centre-list buttons: 20% larger than the previous edition
MAX_PINS = 10
BOX_MAX_WIDTH = 600.0    # the favourites box wraps into more rows past this width (pixels at scale 1)
PINS_FILE = "npanel_tab_bar_pins.json"
_registered = False
_pending_clicks = set()


def prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def stack_enabled():
    settings = prefs()
    return settings is None or settings.enabled


def ui_scale(context):
    """Blender's effective interface scale (display DPI times the Resolution Scale preference)."""
    try:
        return max(0.5, context.preferences.system.ui_scale or 1.0)
    except AttributeError:
        return 1.0


def widget_points(context):
    try:
        return context.preferences.ui_styles[0].widget.points
    except (AttributeError, IndexError):
        return 11.0


def view3d_areas():
    try:
        wm = bpy.context.window_manager
    except AttributeError:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                yield win, area


def redraw_all():
    for _, area in view3d_areas():
        area.tag_redraw()


def primary_region(area):
    """The viewport region that shows the stack: the only one, or the top-right quadrant in quad view."""
    best = None
    for r in area.regions:
        if r.type == "WINDOW" and (best is None or (r.y + r.height, r.x + r.width) > (best.y + best.height,
                                                                                           best.x + best.width)):
            best = r
    return best


_draw_errors = set()


def _report(kind, ex):
    msg = f"{type(ex).__name__}: {ex}"
    if msg not in _draw_errors:
        _draw_errors.add(msg)
        print(f"N-Fix Addon Panel Optimizer {kind} error:", msg)


# ======================================================================================
#  Registry: registered Sidebar panels of the 3D viewport, grouped by tab
# ======================================================================================
class Registry:
    roots = {}          # category -> [panel classes without a parent], in draw order
    children = {}       # parent idname -> [panel classes]
    signature = None


_owner_paths = ()
_known_addon_roots = ()


def user_addon_dirs():
    """Known installation roots, used only to reject orphaned add-on panels."""
    paths = list(addon_utils.paths())
    paths.append(bpy.utils.user_resource("EXTENSIONS"))
    paths.extend(repo.directory for repo in bpy.context.preferences.extensions.repos if repo.directory)
    return tuple(dict.fromkeys(path for path in paths if path))


def _norm(path):
    return os.path.normcase(os.path.normpath(path)) if path else ""


def _refresh_owner_paths():
    global _owner_paths, _known_addon_roots
    enabled = bpy.context.preferences.addons
    owners = []
    for name, mod in tuple(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue
        data = vars(mod)
        if "__addon_enabled__" not in data and name not in enabled:
            continue
        path = _norm(data.get("__file__", ""))
        if path:
            package = "__path__" in data
            root = os.path.dirname(path) if package else path
            owners.append((root, package, name, mod))
    _owner_paths = tuple(sorted(owners, key=lambda item: len(item[0]), reverse=True))
    _known_addon_roots = tuple(_norm(path) for path in user_addon_dirs())


def panel_owner_enabled(cls):
    """Reject known disabled add-ons even if they left registered panels behind.

    Walk full module prefixes: extension owners include bl_ext.<repository>.<id>.
    Native Blender panels and registered panels made by scripts are still eligible.
    """
    if not alive(cls):
        return False
    module_name = cls.__dict__.get("__module__", "")
    enabled = bpy.context.preferences.addons
    parts = module_name.split(".")
    for length in range(len(parts), 0, -1):
        name = ".".join(parts[:length])
        mod = sys.modules.get(name)
        state = vars(mod).get("__addon_enabled__") if mod is not None else None
        if state is not None:
            return bool(state)
        if name in enabled:
            return True
    mod = sys.modules.get(module_name)
    path = _norm(vars(mod).get("__file__", "")) if mod is not None else ""
    if path:
        # Some add-ons import a helper as a top-level module rather than under
        # their package name. Match the longest owning source directory too.
        for root, package, owner_name, owner in _owner_paths:
            if path == root or (package and path.startswith(root + os.sep)):
                if sys.modules.get(owner_name) is not owner:
                    return False
                data = vars(owner)
                return bool(data["__addon_enabled__"]) if "__addon_enabled__" in data else owner_name in enabled
        for root in _known_addon_roots:
            if root and (path == root or path.startswith(root + os.sep)):
                return False
    if module_name.startswith("bl_ext."):
        return False
    return True


def _panel_classes():
    """All Panel subclasses, breadth first so earlier-defined classes come first."""
    out, seen, queue = [], set(), deque(bpy.types.Panel.__subclasses__())
    while queue:
        c = queue.popleft()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        queue.extend(c.__subclasses__())
    return out


def panel_id(cls):
    return getattr(cls, "bl_idname", "") or cls.__name__


def alive(cls):
    """True while Blender still has the class registered. Checked through the class __dict__ only:
    an unregistered class keeps its Python object but its RNA is freed, and an attribute lookup that
    falls through to RNA can crash Blender."""
    try:
        return "bl_rna" in cls.__dict__
    except Exception:
        return False


def rescan(force=False):
    """Rebuild the tab list from the registered panels. Cheap; returns True when it changed."""
    _refresh_owner_paths()
    found = []
    for c in _panel_classes():
        if c is NTB_PT_window or not alive(c):
            continue
        try:
            if getattr(c, "bl_space_type", "") == "VIEW_3D" and getattr(c, "bl_region_type", "") == "UI":
                if not getattr(c, "bl_parent_id", "") and not getattr(c, "bl_category", "").strip():
                    continue
                if panel_owner_enabled(c):
                    found.append(c)
        except (AttributeError, ReferenceError):
            continue
    sig = (tuple((id(c), panel_id(c), getattr(c, "bl_category", ""),
                  getattr(c, "bl_parent_id", ""), getattr(c, "bl_order", 0),
                  getattr(c, "bl_context", "")) for c in found),
           tuple(bpy.context.preferences.addons.keys()))
    if not force and sig == Registry.signature:
        return False
    roots, children = {}, {}
    for i, c in enumerate(found):
        key = (getattr(c, "bl_order", 0), i)
        parent = getattr(c, "bl_parent_id", "")
        if parent:
            children.setdefault(parent, []).append((key, c))
        else:
            roots.setdefault(c.bl_category, []).append((key, c))
    Registry.roots = {k: [c for _, c in sorted(v, key=lambda t: t[0])] for k, v in roots.items()}
    Registry.children = {k: [c for _, c in sorted(v, key=lambda t: t[0])] for k, v in children.items()}
    Registry.signature = sig
    redraw_all()
    return True


def sidebar(area):
    for r in area.regions:
        if r.type == "UI":
            return r
    return None


def sidebar_context(area):
    """The context used by real Sidebar polls, without opening the Sidebar."""
    if area is None:
        return None
    win = next((win for win, candidate in view3d_areas() if candidate == area), None)
    ui = sidebar(area)
    if win is None or ui is None:
        return None
    return bpy.context.temp_override(window=win, area=area, region=ui)


def sidebar_contents(area):
    """Only live root panels visible in this viewport's current Sidebar context."""
    override = sidebar_context(area)
    if override is None:
        return {}, {}
    with override:
        context = bpy.context
        contexts = mode_contexts(context)
        roots = {}
        for name, classes in Registry.roots.items():
            visible = [cls for cls in classes if panel_applies(cls, context, contexts)
                       and getattr(cls, "bl_category", "") == name]
            if visible:
                roots[name] = visible
        # Item/Transform is native C code, so it has a narrow explicit adapter.
        extras = {name: extra for name, extra in BUILTIN_EXTRA.items() if extra[1](context)}
    return roots, extras


def all_categories(area):
    roots, extras = sidebar_contents(area)
    return set(roots) | set(extras)


def grid_items(area):
    """Current Sidebar tabs plus the dedicated add-on-folder utility."""
    tabs = all_categories(area)
    items = [(n, "tab", n) for n in tabs]
    items.sort(key=lambda it: (it[0].casefold(), it[0]))
    items.append(("Open Add-on Folder", "folder", ""))
    return items


# --- which panels apply right now ------------------------------------------------------
MODE_CONTEXT = {
    "OBJECT": "objectmode", "EDIT_MESH": "mesh_edit", "EDIT_CURVE": "curve_edit",
    "EDIT_SURFACE": "surface_edit", "EDIT_CURVES": "curves_edit", "EDIT_TEXT": "text_edit",
    "EDIT_ARMATURE": "armature_edit", "EDIT_METABALL": "mball_edit", "EDIT_LATTICE": "lattice_edit",
    "EDIT_GREASE_PENCIL": "grease_pencil_edit", "EDIT_POINT_CLOUD": "pointcloud_edit", "POSE": "posemode",
    "SCULPT": "sculpt_mode", "PAINT_WEIGHT": "weightpaint", "PAINT_VERTEX": "vertexpaint",
    "PAINT_TEXTURE": "imagepaint", "PARTICLE": "particlemode", "SCULPT_CURVES": "curves_sculpt",
    "PAINT_GREASE_PENCIL": "grease_pencil_paint", "SCULPT_GREASE_PENCIL": "grease_pencil_sculpt",
    "WEIGHT_GREASE_PENCIL": "greasepencil_weight", "VERTEX_GREASE_PENCIL": "greasepencil_vertex",
}
PAINT_COMMON = {"SCULPT", "PAINT_WEIGHT", "PAINT_VERTEX", "PAINT_TEXTURE", "PARTICLE", "SCULPT_CURVES"}


def mode_contexts(context):
    mode = getattr(context, "mode", "OBJECT")
    names = {MODE_CONTEXT.get(mode, mode.lower())}
    if mode in PAINT_COMMON:
        names.add("paint_common")
    return names | {"." + n for n in names}


def _has_own_poll(cls):
    for k in cls.__mro__:
        if k is bpy.types.Panel:
            return False
        if "poll" in k.__dict__:
            return True
    return False


def panel_applies(cls, context, contexts):
    if not panel_owner_enabled(cls):  # its add-on may have been disabled since the last rescan
        return False
    bc = getattr(cls, "bl_context", "")
    if bc and bc not in contexts:
        return False
    if _has_own_poll(cls):
        try:
            return bool(cls.poll(context))
        except Exception:
            return False
    return True


# ======================================================================================
#  Pins
# ======================================================================================
_pins = set()


def pin_key(kind, payload):
    return f"{kind}:{payload or ''}"


def _pins_path():
    return bpy.utils.user_resource("CONFIG", path=PINS_FILE)


def load_pins():
    _pins.clear()
    try:
        with open(_pins_path(), encoding="utf-8") as f:
            _pins.update(str(k) for k in json.load(f))
    except (OSError, ValueError, TypeError):
        pass


def save_pins():
    try:
        path = _pins_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(_pins), f, indent=1)
    except OSError as ex:
        _report("pin save", ex)
    for st in _states.values():          # the rows re-sort, so an old hover index points elsewhere
        st.hover = -1
    redraw_all()


def toggle_pin(kind, payload, present=None):
    """Pin or unpin a button; saved straight away so pins survive without saving preferences.
    `present` is the set of pin keys on screen right now: pins for add-ons that are gone don't count
    toward the limit (and are dropped if they'd push the file past it).
    Returns the new state, or None when MAX_PINS visible buttons are already pinned."""
    key = pin_key(kind, payload)
    if key in _pins:
        _pins.discard(key)
    else:
        live = _pins & present if present is not None else _pins
        if len(live) >= MAX_PINS:
            return None
        _pins.add(key)
        if present is not None:
            for stale in sorted(_pins - present - {key}):
                if len(_pins) <= MAX_PINS:
                    break
                _pins.discard(stale)
    save_pins()
    return key in _pins


# ======================================================================================
#  The add-on window: one tab's panels hosted in the viewport's persistent sidebar
# ======================================================================================
_INSTANCE_DEFAULTS = {"is_popover": False, "use_pin": False, "custom_data": None, "bl_space_type": "VIEW_3D",
                      "bl_region_type": "UI"}


class ExpandedLayout:
    """Draw inline panels open in the organizer, independently of their saved Sidebar state."""

    __slots__ = ("_layout",)
    _sublayouts = {"row", "column", "column_flow", "grid_flow", "box", "split", "menu_pie"}

    def __init__(self, layout):
        object.__setattr__(self, "_layout", layout)

    def __getattr__(self, name):
        value = getattr(self._layout, name)
        if name in self._sublayouts:
            def make_sublayout(*args, **kwargs):
                return ExpandedLayout(value(*args, **kwargs))
            return make_sublayout
        return value

    def __setattr__(self, name, value):
        setattr(self._layout, name, value)

    def _open_section(self):
        section = self.column()
        return section.row(align=True), section.column()

    def panel(self, idname, *, default_closed=False):
        return self._open_section()

    def panel_prop(self, data, property):
        # This property belongs to the originating add-on; display the body
        # without changing its value or triggering its update callback.
        return self._open_section()

    def template_light_linking_collection(self, context_layout, data, property):
        # This RNA template takes another actual UILayout as an argument.
        if isinstance(context_layout, ExpandedLayout):
            context_layout = context_layout._layout
        return self._layout.template_light_linking_collection(context_layout, data, property)


class PanelProxy:
    """Stands in for a Panel instance so another panel's draw() can run inside our window:
    self.layout is ours, everything else comes from the panel class."""

    def __init__(self, cls, layout):
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "layout", ExpandedLayout(layout))

    def __getattr__(self, name):
        cls = object.__getattribute__(self, "_cls")
        if name == "text":
            return getattr(cls, "bl_label", "")
        try:
            raw = inspect.getattr_static(cls, name)
        except AttributeError:
            if name in _INSTANCE_DEFAULTS:
                return _INSTANCE_DEFAULTS[name]
            raise
        if isinstance(raw, (staticmethod, classmethod)):
            return getattr(cls, name)
        if inspect.isfunction(raw):
            return types.MethodType(raw, self)
        if type(raw).__name__ in {"getset_descriptor", "member_descriptor"}:   # bpy instance property
            if name in _INSTANCE_DEFAULTS:
                return _INSTANCE_DEFAULTS[name]
        return getattr(cls, name)


def _call(cls, method, layout, context):
    fn = inspect.getattr_static(cls, method, None)
    if fn is None or not inspect.isfunction(fn):
        return False
    fn(PanelProxy(cls, layout), context)
    return True


def _sidebar_column(layout):
    """Give each original draw callback the same clean layout defaults as a Sidebar panel."""
    column = layout.column()
    column.use_property_split = False
    column.use_property_decorate = True
    column.scale_x = column.scale_y = 1.0
    column.active = column.enabled = True
    return column


def draw_panel(layout, context, cls, contexts, show_title=True):
    if not panel_owner_enabled(cls):
        return
    label = getattr(cls, "bl_label", "") or cls.__name__
    options = getattr(cls, "bl_options", set()) or set()
    try:
        section = _sidebar_column(layout)
        if "HIDE_HEADER" not in options and (show_title or
                inspect.isfunction(inspect.getattr_static(cls, "draw_header", None)) or
                inspect.isfunction(inspect.getattr_static(cls, "draw_header_preset", None))):
            header = _sidebar_column(section)
            row = header.row(align=True)
            _call(cls, "draw_header", row, context)
            if show_title:
                row.label(text=label)
            sub = header.row(align=True)
            sub.alignment = "RIGHT"
            _call(cls, "draw_header_preset", sub, context)
        # Never add V2 foldouts around an add-on's panels. DEFAULT_CLOSED
        # describes the native panel container, not the add-on's body controls.
        body = _sidebar_column(section)
        _call(cls, "draw", body, context)
        for child in Registry.children.get(panel_id(cls), ()):
            if panel_applies(child, context, contexts):
                # Native child panels have their own layout; a parent's draw()
                # may disable its body or change property formatting locally.
                draw_panel(section, context, child, contexts)
    except Exception as ex:
        err = layout.column()
        err.alert = True
        err.label(text=f"{label}: {type(ex).__name__}", icon="ERROR")
        _report(f"panel {panel_id(cls)}", ex)
        print(f"N-Fix Addon Panel Optimizer: couldn't draw {panel_id(cls)} here: {type(ex).__name__}: {ex}")


# --- tabs whose main panel Blender draws in C, so there's no draw() to borrow --------------
def _item_poll(context):
    return getattr(context, "active_object", None) is not None


def _item_draw(layout, context):
    ob = context.active_object
    body = _sidebar_column(layout)
    body.label(text="Transform")
    if context.mode != "OBJECT":
        body.label(text="Transform values show in Object Mode", icon="INFO")
        return
    col = body.column()

    def vec(prop, lock, title):
        row = col.row(align=True)
        row.column().prop(ob, prop, text=title)
        if lock:
            locks = row.column(align=True)
            locks.label(text="")
            for i in range(3):
                locks.prop(ob, lock, index=i, text="", emboss=False, icon_only=True,
                           icon="LOCKED" if getattr(ob, lock)[i] else "UNLOCKED")

    vec("location", "lock_location", "Location")
    mode = ob.rotation_mode
    if mode == "QUATERNION":
        vec("rotation_quaternion", None, "Rotation")
    elif mode == "AXIS_ANGLE":
        vec("rotation_axis_angle", None, "Rotation")
    else:
        vec("rotation_euler", "lock_rotation", "Rotation")
    col.prop(ob, "rotation_mode", text="")
    vec("scale", "lock_scale", "Scale")
    col.separator()
    col.column().prop(ob, "dimensions")


BUILTIN_EXTRA = {"Item": (_item_draw, _item_poll)}

HOST_CATEGORY = "Organizer"
_hosts = {}  # Area pointer -> selected tab and sidebar state before the dock opened.


def host_for(area):
    if area is None or area.type != "VIEW_3D":
        return None
    host = _hosts.get(area.as_pointer())
    return host if host and host["space"] == area.spaces.active else None


def host_is_visible(area):
    if not host_for(area) or not area.spaces.active.show_region_ui:
        return False
    ui = sidebar(area)
    return ui is not None and ui.width > 1 and ui.active_panel_category == HOST_CATEGORY


def close_window(area):
    host = host_for(area)
    if not host:
        return
    ui = sidebar(area)
    # Leave a different sidebar tab alone if the user has since switched to it.
    if host_is_visible(area) or host["deadline"]:
        if not host["was_open"]:
            area.spaces.active.show_region_ui = False
        elif ui and host["previous"] and host["previous"] != HOST_CATEGORY:
            try:
                ui.active_panel_category = host["previous"]
            except (AttributeError, TypeError, ValueError):
                pass  # The previously selected add-on may have been disabled.
    _hosts.pop(area.as_pointer(), None)
    area.tag_redraw()


class NTB_OT_close_window(bpy.types.Operator):
    bl_idname = "npanel_tabs.close_window"
    bl_label = "Close Add-on Controls"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        close_window(context.area)
        return {"FINISHED"}


class NTB_OT_window_pin(bpy.types.Operator):
    bl_idname = "npanel_tabs.window_pin"
    bl_label = "Pin to Favourites"
    bl_description = "Pin this add-on to the favourites box, or unpin it"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        present = {pin_key(kind, payload) for _, kind, payload in grid_items(context.area)}
        host = host_for(context.area)
        if not host or pin_key("tab", host["tab"]) not in present:
            return {"CANCELLED"}
        if toggle_pin("tab", host["tab"], present) is None:
            self.report({"WARNING"}, f"{MAX_PINS} favourites at most: right-click one to unpin it first")
        return {"FINISHED"}


class NTB_PT_window(bpy.types.Panel):
    bl_idname = "NTB_PT_window"
    bl_label = ""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = HOST_CATEGORY
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context):
        return host_for(context.area) is not None

    def draw_header(self, context):
        layout = self.layout
        host = host_for(context.area)
        if not host:
            return
        name = host["tab"]
        row = layout.row(align=True)
        row.label(text=name)
        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        pinned = pin_key("tab", name) in _pins
        sub.operator(NTB_OT_window_pin.bl_idname, text="", emboss=False, depress=pinned,
                     icon="PINNED" if pinned else "UNPINNED")
        sub.operator(NTB_OT_close_window.bl_idname, text="", emboss=False, icon="X")

    def draw(self, context):
        layout = self.layout
        name = host_for(context.area)["tab"]
        roots, extras = sidebar_contents(context.area)
        extra = extras.get(name)
        panels = roots.get(name, ())
        if extra:
            extra[0](layout, context)
        for cls in panels:
            draw_panel(layout, context, cls, mode_contexts(context), show_title=len(panels) > 1 or bool(extra))
        if not (extra or panels):
            layout.label(text="This tab is unavailable in the current context", icon="INFO")


def frame_edges(area, region, space):
    """Toolbar, header and footer limits in viewport pixels. The dock is handled separately."""
    left, bottom, top = 0.0, 0.0, float(region.height)
    rx0, ry0 = region.x, region.y
    rx1, ry1 = rx0 + region.width, ry0 + region.height
    for r in area.regions:
        if r.type in {"WINDOW", "UI", "HUD"} or r.width <= 1 or r.height <= 1:
            continue
        if r.type == "TOOLS" and space is not None and not space.show_region_toolbar:
            continue
        ax0, ay0, ax1, ay1 = r.x, r.y, r.x + r.width, r.y + r.height
        if ax1 <= rx0 or ax0 >= rx1 or ay1 <= ry0 or ay0 >= ry1:
            continue                                     # not over this viewport
        if r.type == "TOOLS" or r.alignment == "LEFT":
            left = max(left, ax1 - rx0)
        elif r.alignment == "RIGHT":
            continue
        elif r.type in HEADER_TYPES or r.alignment in {"TOP", "BOTTOM"}:
            if r.alignment == "TOP" or (r.alignment != "BOTTOM" and (ay0 + ay1) > (ry0 + ry1)):
                top = min(top, ay0 - ry0)
            else:
                bottom = max(bottom, ay1 - ry0)
    return left, top, bottom


def window_overlap(area, region):
    """Visible dock rectangle in this WINDOW region's pixels, including individual quad views."""
    if not host_is_visible(area):
        return None
    ui = sidebar(area)
    if ui is None or ui.width <= 1 or ui.height <= 1:
        return None
    x0, y0 = ui.x - region.x, ui.y - region.y
    x1, y1 = min(region.width, x0 + ui.width), min(region.height, y0 + ui.height)
    x0, y0 = max(0, x0), max(0, y0)
    if x0 < x1 and y0 < y1:
        return x0, y0, x1, y1
    return None


def nav_gizmo_size(context, space, sc):
    """Width and height the navigation gizmo takes at the top right."""
    if space is None or not (space.show_gizmo and space.show_gizmo_navigate):
        return 0.0, 0.0
    view = context.preferences.view
    if view.mini_axis_type == "GIZMO":
        size = (view.gizmo_size_navigate_v3d + 10.0) * sc
        return size, size
    if view.show_navigate_ui:
        return 40.0 * sc, 40.0 * sc
    return 0.0, 0.0


def nav_buttons_size(context, space, sc):
    """Width and height of the column of navigation buttons (zoom, pan, camera, view) under the gizmo."""
    if space is None or not (space.show_gizmo and space.show_gizmo_navigate):
        return 0.0, 0.0
    if not context.preferences.view.show_navigate_ui:
        return 0.0, 0.0
    return 36.0 * sc, 122.0 * sc


def _select_pending_hosts():
    """Sidebar category enums are populated by Blender on its next region draw."""
    pending = False
    for win, area in view3d_areas():
        host = host_for(area)
        if not host or not host["deadline"]:
            continue
        ui = sidebar(area)
        if ui and ui.width > 1:
            with bpy.context.temp_override(window=win, area=area, region=ui):
                try:
                    ui.active_panel_category = HOST_CATEGORY
                    host["deadline"] = 0
                    area.tag_redraw()
                    continue
                except (AttributeError, TypeError, ValueError):
                    pass
        if time.monotonic() >= host["deadline"]:
            close_window(area)
            _report("dock", RuntimeError("Blender did not expose the Organizer sidebar tab"))
        else:
            area.tag_redraw()
            pending = True
    return 0.05 if pending else None


def dock_window(area, payload):
    ui = sidebar(area)
    if ui is None:
        return
    space = area.spaces.active
    host = host_for(area)
    if not host or (not host["deadline"] and not host_is_visible(area)):
        host = {"space": space, "was_open": space.show_region_ui,
                "previous": ui.active_panel_category if space.show_region_ui and ui.width > 1 else None}
        _hosts[area.as_pointer()] = host
    host.update(tab=payload, deadline=time.monotonic() + 5.0)
    space.show_region_ui = True
    area.tag_redraw()
    if not bpy.app.timers.is_registered(_select_pending_hosts):
        bpy.app.timers.register(_select_pending_hosts, first_interval=0.03)


def addon_folder(extensions=False):
    """Legacy user add-ons, or the repository holding this extension (Shift)."""
    if extensions:
        parts = __package__.split(".")
        repository = parts[1] if len(parts) >= 3 and parts[0] == "bl_ext" else "user_default"
        for repo in bpy.context.preferences.extensions.repos:
            if repo.module == repository and repo.directory:
                return repo.directory
        return os.path.join(bpy.utils.user_resource("EXTENSIONS"), "user_default")
    return os.path.join(bpy.utils.user_resource("SCRIPTS"), "addons")


def open_addon_folder(extensions=False):
    """Open File Explorer after gizmo dispatch; Shift selects the extensions repository."""
    path = addon_folder(extensions)
    os.makedirs(path, exist_ok=True)
    bpy.ops.wm.path_open(filepath=path)


def close_sidebars():
    if not _registered or not stack_enabled():
        return
    for _, area in view3d_areas():
        space = area.spaces.active
        if getattr(space, "show_region_ui", False) and not host_for(area):
            space.show_region_ui = False
            area.tag_redraw()


def run_after_click(fn, window):
    """Run a callback after click dispatch, cancelling it if the add-on is disabled."""
    def tick():
        _pending_clicks.discard(tick)
        if not _registered:
            return None
        try:
            wins = list(bpy.context.window_manager.windows)
            win = window if window in wins else (wins[0] if wins else None)
            if win is not None:
                with bpy.context.temp_override(window=win):
                    fn()
        except Exception as ex:
            _report("click", ex)
        return None
    if _registered:
        _pending_clicks.add(tick)
        bpy.app.timers.register(tick, first_interval=0.0)


def activate(context, area, kind, payload, label, mouse_xy, shift=False):
    """What a click on a stack button does."""
    if area is None:
        return
    if kind == "toggle":
        set_expanded(area, not is_expanded(area))
        return
    if kind == "more":
        _overflow["items"] = list(state_for(primary_region(area)).overflow)
        _overflow["area"] = area.as_pointer()
        bpy.ops.wm.call_panel(name=NTB_PT_overflow.bl_idname, keep_open=False)
        return
    if kind == "folder" and payload in {None, ""}:
        run_after_click(lambda: open_addon_folder(extensions=shift), context.window)
        return
    if kind != "tab":
        return
    rescan(force=True)
    if payload not in all_categories(area):
        redraw_all()
        return
    set_expanded(area, True)
    dock_window(area, payload)


_overflow = {"items": [], "area": 0}


class NTB_PT_overflow(bpy.types.Panel):
    bl_idname = "NTB_PT_overflow"
    bl_label = "More"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 18

    def draw(self, context):
        col = self.layout.column(align=True)
        valid = set(grid_items(context.area))
        sc = ui_scale(context)
        # Popovers may be clipped by the application window on small displays.
        # Account for both the button margins and an optional folder icon.
        text_room = min(18 * 20 * sc, context.window.width - 16 * sc) - 48 * sc
        for label, kind, payload in _overflow["items"]:
            if (label, kind, payload) not in valid:
                continue
            for line in wrap_label(label, widget_points(context) * sc, text_room):
                op = col.operator(NTB_OT_activate.bl_idname, text=line,
                                  icon="FILE_FOLDER" if kind == "folder" else "NONE")
                op.kind, op.payload, op.label = kind, payload or "", label


# ======================================================================================
#  Stack state (per viewport region) and the fold / unfold animation
# ======================================================================================
_expanded = {}       # area pointer -> stack unfolded


class BarState:
    __slots__ = ("t", "stamp", "hover", "pressed", "layout", "overflow")

    def __init__(self):
        now = time.monotonic()
        self.t, self.stamp, self.hover, self.pressed = 0.0, now, -1, -1
        self.layout, self.overflow = None, []


_states = {}


def state_for(region):
    key = region.as_pointer()
    st = _states.get(key)
    if st is None:
        st = _states[key] = BarState()
    return st


def is_expanded(area):
    return _expanded.get(area.as_pointer(), False)


_anim_deadline = [0.0]


def set_expanded(area, value):
    _expanded[area.as_pointer()] = value
    now = time.monotonic()
    for r in area.regions:              # restart the clock, or a long idle makes the slide jump to the end
        st = _states.get(r.as_pointer())
        if st is not None:
            st.stamp = now
    _anim_deadline[0] = now + ANIM_SECONDS + 0.5
    if not bpy.app.timers.is_registered(_anim_tick):
        bpy.app.timers.register(_anim_tick, first_interval=0.0)
    area.tag_redraw()


def _anim_tick():
    """Keeps viewports redrawing while a stack slides open or shut. Gives up shortly after the slide
    should have finished, so a viewport that isn't redrawing (minimised) can't keep it spinning."""
    if not _registered or time.monotonic() > _anim_deadline[0]:
        return None
    busy = False
    for _, area in view3d_areas():
        target = 1.0 if is_expanded(area) else 0.0
        region = primary_region(area)
        st = _states.get(region.as_pointer()) if region else None
        if st is not None and abs(st.t - target) > 1e-3:
            busy = True
            area.tag_redraw()
    return 1.0 / 60.0 if busy else None


def step_anim(st, target):
    now = time.monotonic()
    dt, st.stamp = now - st.stamp, now
    step = min(1.0, dt / ANIM_SECONDS)
    st.t = min(target, st.t + step) if target > st.t else max(target, st.t - step)
    return st.t


# ======================================================================================
#  Layout (pure geometry, region pixels, y up)
# ======================================================================================
_width_cache = {}


def text_width(text, px):
    key = (text, round(px, 2))
    w = _width_cache.get(key)
    if w is None:
        if len(_width_cache) > 4096:
            _width_cache.clear()
        blf.size(FONT, px)
        w = _width_cache[key] = blf.dimensions(FONT, text)[0]
    return w


def wrap_label(text, px, room):
    """Keep the full label, breaking at a word boundary or inside an oversized word."""
    lines = []
    while text and text_width(text, px) > room:
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if text_width(text[:mid], px) <= room:
                lo = mid
            else:
                hi = mid - 1
        space = text.rfind(" ", 1, lo + 1)
        cut = space if space > 0 else lo
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        lines.append(text)
    return lines


def text_info_width(context, space, sc):
    """Width of the view name / object / statistics text in the top-left corner."""
    try:
        ov = space.overlay
        if not (ov.show_overlays and ov.show_text):
            return 0.0
        stats = ov.show_stats
    except AttributeError:
        return 0.0
    px = widget_points(context) * sc
    lines = ["User Orthographic (Local)"]
    try:
        line = f"({context.scene.frame_current}) {context.view_layer.active_layer_collection.collection.name}"
        ob = context.active_object
        if ob is not None:
            line += f" | {ob.name}"
        lines.append(line)
    except AttributeError:
        pass
    if stats:
        lines.append("Triangles   9,999,999 / 9,999,999")
    return 10.0 * sc + max(text_width(t, px) for t in lines)


def text_info_height(context, space, sc):
    """Height of the view name / object / statistics text in the top-left corner (generous: the
    statistics block changes with the mode)."""
    try:
        ov = space.overlay
        if not (ov.show_overlays and ov.show_text):
            return 0.0
        lines = 2 + (6 if ov.show_stats else 0)
    except AttributeError:
        return 0.0
    return (1.45 * widget_points(context) * lines + 12.0) * sc


def fill_rows(idx, widths, width_for_row, gap, cap_for_row):
    """Fill each row up to its cap (or the width available at that row) before starting the next."""
    rows, row, row_w = [], [], 0.0
    for i in idx:
        w = widths[i]
        need = w if not row else row_w + gap + w
        if row and (len(row) >= cap_for_row(len(rows)) or need > width_for_row(len(rows))):
            rows.append(row)
            row, need = [], w
        row.append(i)
        row_w = need
    if row:
        rows.append(row)
    return rows


def balance_rows(rows, widths, width_for_row, gap):
    """The same buttons in the same number of rows, spread as evenly as the widths allow (5 + 5
    rather than 9 + 1)."""
    k = len(rows)
    if k < 2:
        return rows
    flat = [i for row in rows for i in row]
    out, start = [], 0
    for r in range(k):
        take = -(-(len(flat) - start) // (k - r))       # what's left, shared over the rows left
        row = flat[start:start + take]
        if sum(widths[i] for i in row) + gap * (len(row) - 1) > width_for_row(r):
            return rows
        out.append(row)
        start += take
    return out


class Btn:
    __slots__ = ("kind", "payload", "label", "text", "x0", "y0", "x1", "y1", "alpha", "layer")

    def __init__(self, kind, payload, label, text, x0, y0, x1, y1, alpha, layer):
        self.kind, self.payload, self.label, self.text = kind, payload, label, text
        self.x0, self.y0, self.x1, self.y1 = round(x0), round(y0), round(x1), round(y1)
        self.alpha, self.layer = alpha, layer        # layer 0: unfolding rows (drawn first), 1: box + arrow


class Layout:
    __slots__ = ("buttons", "box", "box_pad", "font_px", "cap_h", "radius", "line", "chevron", "chevron_size", "s")


def build_layout(context):
    region, area, space = context.region, context.area, context.space_data
    if region is None or area is None or region.type != "WINDOW":
        return None
    p = prefs()
    st = state_for(region)
    if primary_region(area) != region:              # quad view: one stack, top right
        st.layout = None
        return None
    sc = ui_scale(context)
    s = sc * BUTTON_SCALE
    font_px = widget_points(context) * s
    blf.size(FONT, font_px)
    cap_h = blf.dimensions(FONT, "H")[1]
    # every size is a whole number of pixels, so equal buttons come out exactly equal
    h = round((p.bar_height if p else 24) * s * 0.85)
    pad_x = 7.0 * s
    gap = max(2, round(3 * s))                       # between buttons, across and down
    vgap = gap
    group_gap = gap                                  # box, arrow and rows keep the same rhythm
    min_w = round(44 * s)
    box_pad = round(4 * s)
    hw, hh = round(30 * s), 2 * round(6.5 * s)       # the arrows; an even height centres the chevron
    margin = 8.0 * sc

    # --- frame: centred on the whole viewport, clear of the toolbar, the info text and the gizmo
    left, top_edge, bottom = frame_edges(area, region, space)
    right = region.width
    cx = right // 2
    dock = window_overlap(area, region)
    if dock:
        if sidebar(area).alignment == "LEFT":
            left = max(left, dock[2])
        else:
            right = dock[0]
    top = math.floor(top_edge - (p.top_margin if p else 6) * sc)
    bottom += margin
    zones = []                                       # (x0, x1, y0): kept clear from y0 up to the top edge
    info_w = text_info_width(context, space, sc)
    if info_w:
        zones.append((left, left + info_w, top_edge - text_info_height(context, space, sc)))
    nav_w, nav_h = nav_gizmo_size(context, space, sc)
    if nav_w:
        zones.append((right - nav_w, right, top_edge - nav_h))
    nb_w, nb_h = nav_buttons_size(context, space, sc)
    if nb_w:
        zones.append((right - nb_w, right, top_edge - nav_h - nb_h))

    def room(y_top):
        """Width available, centred on cx, for a band whose top edge is at y_top."""
        lo, hi = left + margin, right - margin
        for zx0, zx1, zy0 in zones:
            if y_top > zy0:
                if zx1 <= cx:
                    lo = max(lo, zx1 + margin)
                else:
                    hi = min(hi, zx0 - margin)
        return 2.0 * max(0.0, min(cx - lo, hi - cx))

    free_w = room(-1.0e9)                            # below every zone
    if free_w < min_w + 2 * box_pad:
        cx = round((left + right) / 2.0)
        free_w = room(-1.0e9)
    if free_w < min_w + 2 * box_pad or top - bottom < 2 * hh:
        st.layout = None
        return None

    items = grid_items(area)
    n = len(items)
    pinned = [pin_key(kind, payload) in _pins for _, kind, payload in items]
    widths = [round(h * 1.5) if kind == "folder" else math.ceil(max(min_w, text_width(lab, font_px) + 2 * pad_x))
              for lab, kind, _ in items]
    if not n:
        st.layout = None
        return None
    typical = sorted(widths)[n // 2]
    column = free_w < 3 * typical + 2 * gap          # too tight for rows: one column of equal-width buttons
    rtl = p.right_to_left if p else False
    buttons = []
    # A button keeps its measured width. A label wider than the whole viewport
    # goes into the overflow menu where it can use as many lines as it needs.
    visible = [i for i in range(n) if widths[i] + (2 * box_pad if pinned[i] else 0) <= free_w]
    pin_idx = [i for i in visible if pinned[i]]
    rest = [i for i in visible if not pinned[i]]
    too_wide = [i for i in range(n) if i not in visible]
    if column and visible:
        col_w = max(widths[i] for i in visible)
        for i in visible:
            widths[i] = min(col_w, math.floor(free_w) - (2 * box_pad if pinned[i] else 0))
    long_row = p.per_row if p else 10
    first = 1 if pin_idx else 0                       # the row under the favourites is a short one

    def caps(r):
        return long_row if (first + r) % 2 == 0 else max(1, long_row - 2)

    def span(row):
        return sum(widths[i] for i in row) + gap * (len(row) - 1)

    def place(row, y_top, alpha, layer):
        row_w = span(row)
        x, y_top = cx - row_w // 2, round(y_top)
        for i in (reversed(row) if rtl else row):
            lab, kind, payload = items[i] if i < n else (f"+{len(st.overflow)}", "more", None)
            text = "" if kind == "folder" else lab
            buttons.append(Btn(kind, payload, lab, text, x, y_top - h, x + widths[i], y_top, alpha, layer))
            x += widths[i] + gap
        return row_w

    def rows_height(k):
        return k * h + max(0, k - 1) * vgap

    # --- where the stack starts: directly under the header, unless the info text or the gizmo leave
    # too little room beside them; then under whichever is in the way (narrow viewports only)
    widest_pin = max((widths[i] + 2 * box_pad for i in pin_idx), default=0)
    box_max = max(BOX_MAX_WIDTH * s, widest_pin)
    need = max(hw, min(span(pin_idx) + 2 * box_pad, box_max) if pin_idx else 0,
               min(span(rest[:caps(0)]), 0.5 * free_w), max((widths[i] for i in rest), default=0))
    y = top
    for zy0 in sorted({z[2] for z in zones}, reverse=True):
        if room(y) >= need:
            break
        y = min(y, math.floor(zy0 - margin))
    if y - hh < bottom:
        st.layout = None
        return None

    # --- favourites box at the very top: as wide as its content, at most box_max
    box = None
    lost = list(too_wide)
    if pin_idx:
        y_box = y

        def box_room(r):
            return min(box_max, room(y_box - r * (h + vgap))) - 2 * box_pad

        if column:
            prow = [[i] for i in pin_idx]
        else:
            prow = balance_rows(fill_rows(pin_idx, widths, box_room, gap, lambda r: MAX_PINS), widths, box_room, gap)
        fit = max(0, int((y - bottom - hh - group_gap - 2 * box_pad + vgap) // (h + vgap)))
        lost.extend(i for row in prow[fit:] for i in row)
        prow = prow[:fit]
        y_row, box_w = y - box_pad, 0
        for row in prow:
            box_w = max(box_w, place(row, y_row, 1.0, 1))
            y_row -= h + vgap
        if prow:
            box_bottom = y_row + vgap - box_pad
            box = (cx - box_w // 2 - box_pad, box_bottom, cx - box_w // 2 + box_w + box_pad, y)
            y = box_bottom - group_gap

    # --- the arrow under the box: fixed, it never moves when the stack unfolds
    t = step_anim(st, 1.0 if is_expanded(area) else 0.0)
    ease = 1.0 - (1.0 - t) ** 3
    buttons.append(Btn("toggle", None, "", "", cx - hw // 2, y - hh, cx - hw // 2 + hw, y, 1.0, 1))
    rows_top = y - hh - group_gap

    # --- everything else, unfolding below the arrow
    first_room = room(rows_top)
    # Rows may widen below the info text and the gizmo, but only when the first row is already a full
    # one; otherwise every row keeps the first row's width: a tidy block rather than a funnel.
    widen = column or not rest or span(rest[:caps(0)]) <= first_room

    def rows_room(r):
        return room(rows_top - r * (h + vgap)) if widen else first_room

    budget = rows_top - bottom - hh - group_gap       # leave room for the arrow at the bottom
    if column:
        rows = [[i] for i in rest]
    else:
        rows = fill_rows(rest, widths, rows_room, gap, caps)
        if rows_height(len(rows)) > budget:           # condense: fill rows to the width, no caps
            rows = fill_rows(rest, widths, rows_room, gap, lambda r: 1 << 30)
    st.overflow = []
    if lost or rows_height(len(rows)) > budget:
        fit = max(0, int((budget + vgap) // (h + vgap)))
        hidden = [i for row in rows[fit:] for i in row]
        rows = rows[:fit]
        if fit:
            chip_w = math.ceil(max(min_w, text_width(f"+{n}", font_px) + 2 * pad_x))
            widths.append(chip_w)
            if (not rows or column or span(rows[-1]) + gap + chip_w > rows_room(len(rows) - 1)) and len(rows) < fit:
                rows.append([])
            last = rows[-1]
            while last and (column or span(last) + gap + chip_w > rows_room(len(rows) - 1)):
                hidden.insert(0, last.pop())
            last.append(n)
        st.overflow = [items[i] for i in lost + hidden]
    if t > 0.0:
        for r, row in enumerate(rows):
            y_final = rows_top - r * (h + vgap)
            place(row, rows_top + h + (y_final - rows_top - h) * ease, ease, 0)
        y_end = rows_top - rows_height(len(rows)) - group_gap if rows else rows_top
        y_b = round(rows_top + (y_end - rows_top) * ease)
        buttons.append(Btn("toggle", None, "", "", cx - hw // 2, y_b - hh, cx - hw // 2 + hw, y_b, ease, 0))
    if not rows and st.overflow:                      # nothing fits: the top arrow opens the list
        buttons = [b if b.kind != "toggle" or b.layer != 1 else
                   Btn("more", None, "", "", b.x0, b.y0, b.x1, b.y1, 1.0, 1) for b in buttons if b.layer == 1]

    lay = Layout()
    lay.buttons, lay.box, lay.box_pad, lay.font_px, lay.cap_h, lay.s = buttons, box, box_pad, font_px, cap_h, s
    try:
        roundness = context.preferences.themes[0].user_interface.wcol_tab.roundness
    except (AttributeError, IndexError):
        roundness = 0.4
    lay.radius = max(1.0, min(roundness * 0.5 * h, h / 2.0))
    lay.line = max(1.0, round(sc))
    lay.chevron = is_expanded(area)
    lay.chevron_size = (4.5 * s, 2.25 * s, max(1.25, 1.5 * s))     # half width, half height, stroke
    st.layout = lay
    return lay


def hit_test(lay, x, y):
    """Index into lay.buttons under (x, y), or -1. Top layer first; rows still sliding in don't count."""
    if lay is None:
        return -1
    for k in sorted(range(len(lay.buttons)), key=lambda k: -lay.buttons[k].layer):
        b = lay.buttons[k]
        if b.alpha > 0.6 and b.x0 <= x < b.x1 and b.y0 <= y < b.y1:
            return k
    return -1


# ======================================================================================
#  Drawing
# ======================================================================================
# Every shape is one triangle running a distance-field shader, the way Blender draws its own widgets:
# exact pixel-aligned 1 px outlines and smooth corners. Plain polygon fans don't survive the gizmo pass,
# which draws with polygon smoothing on (their inner seams come out half transparent).
_COLOR_LIB = """
vec4 framebuffer_color(vec4 c)
{
  /* theme colours are sRGB; Blender's own shaders convert them like this for sRGB targets */
  if (srgbTarget) {
    vec3 v = max(c.rgb, vec3(0.0));
    c.rgb = mix(v / 12.92, pow((v + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), v));
  }
  return c;
}
"""

_BOX_FRAG = _COLOR_LIB + """
void main()
{
  vec2 half_size = 0.5 * (rect.zw - rect.xy);
  vec2 p = co - 0.5 * (rect.xy + rect.zw);
  float r = min(shape.x, min(half_size.x, half_size.y));
  vec2 q = abs(p) - half_size + r;
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;   /* signed distance to the edge */
  float cover = clamp(0.5 - d, 0.0, 1.0);
  float inside = clamp(0.5 - d - shape.y, 0.0, 1.0);             /* inside the outline */
  vec4 col = mix(line_color, fill_color, inside);
  fragColor = framebuffer_color(vec4(col.rgb, col.a * cover));
}
"""

_CHEVRON_FRAG = _COLOR_LIB + """
float segment(vec2 p, vec2 a, vec2 b)
{
  vec2 pa = p - a, ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

void main()
{
  vec2 tip = chevron.xy + vec2(0.0, chevron.w);
  float d = min(segment(co, chevron.xy - chevron.zw, tip),
                segment(co, tip, chevron.xy + vec2(chevron.z, -chevron.w))) - 0.5 * stroke;
  fragColor = framebuffer_color(vec4(color.rgb, color.a * clamp(0.5 - d, 0.0, 1.0)));
}
"""

_shaders = {}


def _shader(kind):
    """(shader, batch) for "box" or "chevron", built on first use."""
    cached = _shaders.get(kind)
    if cached is not None:
        return cached
    # push constants in this order stay within the 128 bytes Vulkan guarantees
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "ModelViewProjectionMatrix")
    if kind == "box":
        info.push_constant("VEC4", "rect")
        info.push_constant("VEC4", "fill_color")
        info.push_constant("VEC4", "line_color")
        info.push_constant("BOOL", "srgbTarget")               # Blender sets it on bind
        info.push_constant("VEC2", "shape")                    # corner radius, outline width
        corner, extent, frag = "rect.xy - 1.0", "rect.zw - rect.xy + 2.0", _BOX_FRAG
    else:
        info.push_constant("VEC4", "chevron")                  # centre, half width, half height (signed)
        info.push_constant("VEC4", "color")
        info.push_constant("BOOL", "srgbTarget")
        info.push_constant("FLOAT", "stroke")
        corner = "chevron.xy - vec2(chevron.z, abs(chevron.w)) - stroke - 1.0"
        extent = "2.0 * (vec2(chevron.z, abs(chevron.w)) + stroke + 1.0)"
        frag = _CHEVRON_FRAG
    info.vertex_in(0, "VEC2", "unit")
    iface = gpu.types.GPUStageInterfaceInfo(f"ntb_{kind}_iface")
    iface.smooth("VEC2", "co")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")
    info.vertex_source(f"void main() {{ co = {corner} + unit * ({extent}); "
                       "gl_Position = ModelViewProjectionMatrix * vec4(co, 0.0, 1.0); }")
    info.fragment_source(frag)
    shader = gpu.shader.create_from_info(info)
    # one triangle twice the size of the shape's square: no inner edge to smooth into a seam
    batch = batch_for_shader(shader, "TRIS", {"unit": [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]})
    _shaders[kind] = shader, batch
    return shader, batch


def draw_box(x0, y0, x1, y1, radius, fill, line=None, width=0.0):
    shader, batch = _shader("box")
    shader.bind()
    shader.uniform_float("rect", (x0, y0, x1, y1))
    shader.uniform_float("fill_color", fill)
    shader.uniform_float("line_color", fill if line is None else line)
    shader.uniform_float("shape", (radius, 0.0 if line is None else width))
    gpu.state.blend_set("ALPHA")                 # every blf.draw() switches blending off again
    batch.draw(shader)


def draw_chevron(cx, cy, half_w, half_h, color, stroke):
    """A ^ (half_h > 0) or v (half_h < 0) centred on (cx, cy)."""
    shader, batch = _shader("chevron")
    shader.bind()
    shader.uniform_float("chevron", (cx, cy, half_w, half_h))
    shader.uniform_float("color", color)
    shader.uniform_float("stroke", stroke)
    gpu.state.blend_set("ALPHA")
    batch.draw(shader)


def _rgba(c):
    c = tuple(c)
    return c if len(c) == 4 else (*c, 1.0)


def _mix(a, b, f):
    return tuple(a[i] + (b[i] - a[i]) * f for i in range(4))


def _luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


RIM = 0.16          # outline mixed this far from the fill toward the text, when the theme's is invisible


def palette(context, opacity):
    """One look for every button, from the theme's Sidebar tab colours (hover: the active tab's shade;
    pressed: the blue of a pressed button); the favourites box uses the popup-menu background.
    Where the theme's outline matches the fill, a light rim takes its place, so buttons stay distinct
    over dark scenes; over light or saturated ones the dark fill does that."""
    try:
        ui = context.preferences.themes[0].user_interface
        tab, menu = ui.wcol_tab, ui.wcol_menu_back
        inner, sel, outline = _rgba(tab.inner), _rgba(tab.inner_sel), _rgba(tab.outline)
        text, press = _rgba(tab.text_sel), _rgba(ui.wcol_tool.inner_sel)
        box, box_line = _rgba(menu.inner), _rgba(menu.outline)
    except (AttributeError, IndexError):
        inner, sel, outline = (0.11, 0.11, 0.11, 1.0), (0.19, 0.19, 0.19, 1.0), (0.11, 0.11, 0.11, 1.0)
        text, press = (1.0, 1.0, 1.0, 1.0), (0.28, 0.45, 0.70, 1.0)
        box, box_line = (0.09, 0.09, 0.09, 1.0), (0.14, 0.14, 0.14, 1.0)
    if abs(_luma(outline) - _luma(inner)) < 0.06:
        outline = _mix(inner, text, RIM)
    if abs(_luma(box_line) - _luma(box)) < 0.06:
        box_line = _mix(box, text, RIM)
    # Blender blends in linear light, where the 8 % left of a bright background shows far more than
    # 8 %; this makes the preference read as the opacity you see
    alpha = 1.0 - (1.0 - opacity) ** 1.5

    def a(c, f=1.0):
        return (c[0], c[1], c[2], alpha * f)

    return dict(fill=a(inner), hover=a(sel), press=a(press), outline=a(outline), text=text,
                box=a(box, 0.75), box_line=a(box_line))


def draw_layout(lay, hover, pressed, colors):
    def faded(c, alpha):
        return (c[0], c[1], c[2], c[3] * alpha)

    def draw_box_frame():
        if lay.box is not None:                      # corners concentric with the buttons inside
            draw_box(*lay.box, lay.radius + lay.box_pad, colors["box"], colors["box_line"], lay.line)

    old_blend = gpu.state.blend_get()
    s = lay.s
    half_w, half_h, stroke = lay.chevron_size
    order = [k for k in range(len(lay.buttons)) if lay.buttons[k].layer == 0]
    order += [k for k in range(len(lay.buttons)) if lay.buttons[k].layer == 1]
    box_drawn = False
    for k in order:
        b = lay.buttons[k]
        if b.layer == 1 and not box_drawn:           # box goes over the unfolding rows, under its buttons
            box_drawn = True
            draw_box_frame()
        col = colors["press"] if k == pressed else colors["hover"] if k == hover else colors["fill"]
        draw_box(b.x0, b.y0, b.x1, b.y1, lay.radius, faded(col, b.alpha), faded(colors["outline"], b.alpha),
                 lay.line)
        tc = faded(colors["text"], b.alpha)
        mx, my = (b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0
        if b.kind == "toggle":                       # chevron: down while folded, up while open
            up = lay.chevron if b.layer == 1 else True
            draw_chevron(mx, my, half_w, half_h if up else -half_h, tc, stroke)
            continue
        if b.kind == "more" and not b.text:          # top arrow standing in for the overflow list
            r = 1.25 * s
            for dx in (-4.0 * s, 0.0, 4.0 * s):
                draw_box(mx + dx - r, my - r, mx + dx + r, my + r, r, tc)
            continue
        if b.kind == "folder":                       # folder silhouette: tab and body
            ih = round((b.y1 - b.y0) * 0.42)
            iw = round(ih * 1.3)
            x0, y0 = round(mx - iw / 2.0), round(my - ih / 2.0)
            r = max(1.0, 1.2 * s)
            draw_box(x0, y0 + ih - max(2, round(ih * 0.3)), x0 + round(iw * 0.45), y0 + ih, r, tc)
            draw_box(x0, y0, x0 + iw, y0 + ih - max(1, round(ih * 0.18)), r, tc)
            continue
        blf.size(FONT, lay.font_px)
        w = text_width(b.text, lay.font_px)
        blf.color(FONT, *tc)
        blf.position(FONT, round(mx - w / 2.0), round(my - lay.cap_h / 2.0), 0.0)
        blf.draw(FONT, b.text)
    if not box_drawn:
        draw_box_frame()
    gpu.state.blend_set(old_blend)


# ======================================================================================
#  Gizmo: draws the stack and takes left clicks on it
# ======================================================================================
class NTB_GT_tab_bar(bpy.types.Gizmo):
    bl_idname = "NTB_GT_tab_bar"
    __slots__ = ("press_part", "press_shift", "press_xy", "press_over", "press_key")

    def setup(self):
        self.press_part, self.press_shift, self.press_xy, self.press_over = -1, False, (0, 0), False
        self.press_key = None

    def draw(self, context):
        try:
            st = state_for(context.region)
            lay = st.layout or build_layout(context)
            if lay is not None:
                p = prefs()
                hover = st.hover if self.is_highlight else -1     # mouse left the viewport from a button
                draw_layout(lay, hover, st.pressed, palette(context, p.button_opacity if p else 0.92))
        except Exception as ex:
            _report("draw", ex)

    def test_select(self, context, location):
        st = state_for(context.region)
        part = hit_test(st.layout, location[0], location[1])
        if part != st.hover:
            st.hover = part
            context.region.tag_redraw()
        return part

    def invoke(self, context, event):
        st = state_for(context.region)
        self.press_part = hit_test(st.layout, event.mouse_region_x, event.mouse_region_y)
        self.press_shift = event.shift
        self.press_xy = (event.mouse_x, event.mouse_y)
        self.press_over = self.press_part >= 0
        button = st.layout.buttons[self.press_part] if self.press_over and st.layout else None
        self.press_key = (button.kind, button.payload) if button else None
        st.pressed = self.press_part
        context.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        # like any Blender button: it stays pressed only while the mouse is on it, and letting go
        # anywhere else cancels the click
        st = state_for(context.region)
        over = (st.layout is not None and 0 <= self.press_part < len(st.layout.buttons)
                and hit_test(st.layout, event.mouse_region_x, event.mouse_region_y) == self.press_part
                and (st.layout.buttons[self.press_part].kind,
                     st.layout.buttons[self.press_part].payload) == self.press_key)
        if over != self.press_over:
            self.press_over = over
            st.pressed = self.press_part if over else -1
            context.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        st = state_for(context.region)
        st.pressed = -1
        lay = st.layout
        part = self.press_part
        context.region.tag_redraw()
        if cancel or not self.press_over or lay is None or not (0 <= part < len(lay.buttons)):
            return
        b = lay.buttons[part]
        if (b.kind, b.payload) != self.press_key:
            return
        try:
            activate(context, context.area, b.kind, b.payload, b.label, self.press_xy, self.press_shift)
        except Exception as ex:
            _report("click", ex)


class NTB_GGT_tab_bar(bpy.types.GizmoGroup):
    bl_idname = "NTB_GGT_tab_bar"
    bl_label = "N-Fix Addon Panel Optimizer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"PERSISTENT"}

    @classmethod
    def poll(cls, context):
        return stack_enabled()

    def setup(self, context):
        gz = self.gizmos.new(NTB_GT_tab_bar.bl_idname)
        gz.use_draw_modal = True
        # Blender 4.5 has no Gizmo.use_undo; this UI-only gizmo changes no scene data.
        if hasattr(gz, "use_undo"):
            gz.use_undo = False
        gz.use_tooltip = False
        self.bar = gz

    def draw_prepare(self, context):
        try:
            lay = build_layout(context)
        except Exception as ex:
            lay = None
            _report("layout", ex)
        self.bar.hide = lay is None


# ======================================================================================
#  Operators
# ======================================================================================
class NTB_OT_toggle(bpy.types.Operator):
    bl_idname = "npanel_tabs.toggle"
    bl_label = "Fold / Unfold Add-on Stack"
    bl_description = "Unfold the add-on stack at the top of the viewport, or fold it away"

    @classmethod
    def poll(cls, context):
        return stack_enabled() and context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        area = context.area
        if host_for(area):
            close_window(area)
            set_expanded(area, False)
            return {"FINISHED"}
        set_expanded(area, not is_expanded(area))
        space = context.space_data
        if space is not None and space.show_region_ui:
            space.show_region_ui = False             # N never leaves Blender's Sidebar showing
        return {"FINISHED"}


class NTB_OT_activate(bpy.types.Operator):
    bl_idname = "npanel_tabs.activate"
    bl_label = "Open Add-on"
    bl_description = "Open this add-on"
    bl_options = {"INTERNAL"}

    kind: StringProperty(options={"SKIP_SAVE"})
    payload: StringProperty(options={"SKIP_SAVE"})
    label: StringProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        area = context.area if context.area and context.area.type == "VIEW_3D" else None
        if area is None:
            area = next((a for _, a in view3d_areas() if a.as_pointer() == _overflow["area"]), None)
        activate(context, area, self.kind, self.payload or None, self.label, (event.mouse_x, event.mouse_y),
                 event.shift)
        return {"FINISHED"}


class NTB_OT_pin(bpy.types.Operator):
    bl_idname = "npanel_tabs.pin"
    bl_label = "Pin to Favourites"
    bl_description = "Right-click a stack button to pin it to the favourites box, or unpin it"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (stack_enabled() and context.area is not None and context.area.type == "VIEW_3D"
                and context.region is not None and context.region.type == "WINDOW")

    def invoke(self, context, event):
        lay = state_for(context.region).layout
        k = hit_test(lay, event.mouse_region_x, event.mouse_region_y)
        if k < 0:
            return {"PASS_THROUGH"}                 # not over a button: normal right-click menu
        b = lay.buttons[k]
        if b.kind in {"toggle", "more"}:
            return {"FINISHED"}                     # nothing to pin, but no context menu over the stack
        present = {pin_key(kind, payload) for _, kind, payload in grid_items(context.area)}
        state = toggle_pin(b.kind, b.payload, present)
        if state is None:
            self.report({"WARNING"}, f"{MAX_PINS} favourites at most: right-click one to unpin it first")
        else:
            self.report({"INFO"}, f"{'Pinned' if state else 'Unpinned'} {b.label}")
        return {"FINISHED"}


class NTB_OT_clear_pins(bpy.types.Operator):
    bl_idname = "npanel_tabs.clear_pins"
    bl_label = "Clear Favourites"
    bl_description = "Unpin every button, including pins for add-ons that are no longer installed"

    def execute(self, context):
        _pins.clear()
        save_pins()
        return {"FINISHED"}


class NTB_OT_rescan(bpy.types.Operator):
    bl_idname = "npanel_tabs.rescan"
    bl_label = "Rescan Add-ons"
    bl_description = "Rebuild the button list from the add-ons registered right now"

    def execute(self, context):
        rescan(force=True)
        self.report({"INFO"}, f"N-Fix Addon Panel Optimizer: {len(Registry.roots)} Sidebar categories")
        return {"FINISHED"}


# ======================================================================================
#  Preferences
# ======================================================================================
def _redraw(self, context):
    _width_cache.clear()
    if not self.enabled:
        _close_all_hosts()
    redraw_all()


def _hide_sidebar_update(self, context):
    if self.hide_sidebar:
        close_sidebars()


class NTB_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    enabled: BoolProperty(name="Show Add-on Stack", default=True, update=_redraw,
                          description="Show the add-on buttons at the top of each 3D viewport")
    # Retained only so old saved preferences can load; V2 always keeps the list open.
    collapse_after_pick: BoolProperty(name="Fold After Picking", default=False, options={"HIDDEN"})
    hide_sidebar: BoolProperty(name="Start with the Sidebar Closed", default=True, update=_hide_sidebar_update,
                               description="Close Blender's right-hand Sidebar when a file loads")
    per_row: IntProperty(name="Long Row", default=10, min=3, max=40, update=_redraw,
                         description="Buttons in a long row; short rows hold two fewer. Rows alternate, "
                                     "starting with a short row under the favourites")
    right_to_left: BoolProperty(name="Right to Left", default=False, update=_redraw,
                                description="Alphabetical order runs right to left within each row")
    top_margin: IntProperty(name="Top Margin", default=6, min=0, max=400, subtype="PIXEL", update=_redraw,
                            description="Space between the viewport header and the add-on stack")
    bar_height: IntProperty(name="Button Height", default=24, min=16, max=64, subtype="PIXEL", update=_redraw,
                            description="Height of each button before interface scaling")
    button_opacity: FloatProperty(name="Button Opacity", default=0.92, min=0.3, max=1.0, subtype="FACTOR",
                                  update=_redraw, description="Opacity of the button backgrounds")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        col = layout.column()
        col.prop(self, "enabled")
        col.prop(self, "hide_sidebar")
        col = layout.column(heading="Layout")
        col.prop(self, "per_row")
        col.prop(self, "right_to_left")
        col.prop(self, "top_margin")
        col.prop(self, "bar_height")
        col.prop(self, "button_opacity")
        row = layout.row()
        row.label(text=f"{len(Registry.roots)} registered Sidebar categories")
        row.operator(NTB_OT_rescan.bl_idname, icon="FILE_REFRESH")
        row = layout.row()
        visible_pins = {pin_key(kind, payload) for _, area in view3d_areas()
                        for _, kind, payload in grid_items(area)}
        row.label(text=f"{len(_pins & visible_pins)} of {MAX_PINS} visible favourites (right-click to pin)")
        row.operator(NTB_OT_clear_pins.bl_idname, icon="X")


# ======================================================================================
#  Registration
# ======================================================================================
def _rescan_tick():
    if not _registered:
        return None
    try:
        rescan()
        if _hosts or _expanded or _states:
            # Blender can reuse area and region addresses after joining viewports.
            areas = [a for s in bpy.data.screens for a in s.areas if a.type == "VIEW_3D"]
            area_keys = {a.as_pointer() for a in areas}
            region_keys = {r.as_pointer() for a in areas for r in a.regions if r.type == "WINDOW"}
            for cache, valid in ((_hosts, area_keys), (_expanded, area_keys), (_states, region_keys)):
                for key in cache.keys() - valid:
                    del cache[key]
    except Exception as ex:
        _report("rescan", ex)
    return RESCAN_SECONDS


@persistent
def _on_load(_dummy):
    if not _registered:
        return
    # a new file brings new screens: per-area state keyed by old pointers could be inherited by
    # whatever area reuses the address, so drop it all
    _expanded.clear()
    _states.clear()
    _hosts.clear()
    _pending_clicks.clear()  # Blender discards the non-persistent click timers on file load.
    load_pins()                         # pick up pins another Blender window may have changed
    rescan(force=True)
    p = prefs()
    if p is None or p.hide_sidebar:
        if not bpy.app.timers.is_registered(close_sidebars):
            bpy.app.timers.register(close_sidebars, first_interval=0.1)


def _startup():
    if _registered:
        rescan(force=True)
        p = prefs()
        if p is None or p.hide_sidebar:
            close_sidebars()


def _close_all_hosts():
    if _hosts:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if host_for(area):
                    close_window(area)
        _hosts.clear()


CLASSES = (NTB_Preferences, NTB_PT_window, NTB_PT_overflow, NTB_GT_tab_bar, NTB_GGT_tab_bar, NTB_OT_toggle,
           NTB_OT_activate, NTB_OT_pin, NTB_OT_window_pin, NTB_OT_close_window, NTB_OT_clear_pins, NTB_OT_rescan)

# Right-click goes through every mode keymap: they're checked before "3D View", and the mode keymaps
# hold the right-click context menus. Add-on items run first and pass through off the buttons.
PIN_KEYMAPS = (("3D View", "VIEW_3D"),) + tuple((name, "EMPTY") for name in (
    "Object Mode", "Mesh", "Curve", "Curves", "Armature", "Metaball", "Lattice", "Font", "Pose",
    "Vertex Paint", "Weight Paint", "Image Paint", "Sculpt", "Sculpt Curves", "Particle", "Point Cloud",
    "Grease Pencil Edit Mode", "Grease Pencil Paint Mode", "Grease Pencil Sculpt Mode"))
_keymaps = []


def register():
    global _registered
    if __package__ != "npanel_tab_bar" and bpy.context.preferences.addons.get("npanel_tab_bar") is not None:
        raise RuntimeError("Disable the other copy of N-Fix Addon Panel Optimizer before enabling this copy")
    load_pins()
    for c in CLASSES:
        bpy.utils.register_class(c)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        km = kc.keymaps.new(name="3D View Generic", space_type="VIEW_3D")
        _keymaps.append((km, km.keymap_items.new(NTB_OT_toggle.bl_idname, "N", "PRESS")))
        for name, space in PIN_KEYMAPS:
            km = kc.keymaps.new(name=name, space_type=space, region_type="WINDOW")
            _keymaps.append((km, km.keymap_items.new(NTB_OT_pin.bl_idname, "RIGHTMOUSE", "PRESS")))
    _registered = True
    bpy.app.handlers.load_post.append(_on_load)
    # other add-ons may register after this one at startup: scan once they're in, then keep watching
    bpy.app.timers.register(_rescan_tick, first_interval=0.5, persistent=True)
    bpy.app.timers.register(_startup, first_interval=0.6)


def unregister():
    global _registered
    _registered = False
    _close_all_hosts()
    for fn in (_rescan_tick, _anim_tick, _startup, _select_pending_hosts, close_sidebars, *tuple(_pending_clicks)):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    _pending_clicks.clear()
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except (ReferenceError, RuntimeError):
            pass
    _keymaps.clear()
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    _states.clear()
    _expanded.clear()
    _shaders.clear()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
