# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aidan Hart
# The MIT License
# ---------------
#
# Copyright (c) 2026 Aidan Hart
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
bl_info = {
    "name": "Universal Export Folder Opener",
    "author": "Aidan Hart",
    "version": (1, 0, 2),
    "blender": (4, 0, 0),
    "location": "Runs automatically in background",
    "description": "Select completed exports from Python exporters such as FBX and glTF in Windows Explorer",
    "category": "Import-Export",
}

import os
import subprocess
from functools import wraps

import bpy

# A timer coalesces nested exporters writing the same file. Once it runs, exporting
# to the same filename again opens Explorer again.
_pending = {}
_wrapped = {}
_enabled = False


def queue_export(filepath, batch=False):
    path = os.path.normpath(bpy.path.abspath(filepath))
    key = (os.path.normcase(path), batch)
    if key in _pending:
        return

    def show_file():
        _pending.pop(key, None)
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            if not batch and os.path.isfile(path):
                subprocess.Popen(f'explorer.exe /select,"{path}"')
            elif os.path.isdir(folder):
                subprocess.Popen(["explorer.exe", folder])
        except OSError as exc:
            print(f"[Export Folder Opener] {exc}")

    _pending[key] = show_file
    bpy.app.timers.register(show_file, first_interval=0.4, persistent=True)


def wrap_operator(cls):
    original = cls.execute
    if getattr(original, "_export_folder_hook", False):
        return
    had_own_execute = "execute" in vars(cls)

    @wraps(original)
    def execute(self, context):
        result = original(self, context)
        if _enabled and "FINISHED" in result:
            filepath = getattr(self, "filepath", "")
            if filepath:
                is_fbx_batch = self.bl_rna.identifier == "EXPORT_SCENE_OT_fbx" and self.batch_mode != "OFF"
                queue_export(filepath, batch=is_fbx_batch)
        return result

    execute._export_folder_hook = True
    cls.execute = execute
    _wrapped[cls] = (original, execute, had_own_execute)


def scan_exporters():
    # Registered exporters may inherit a shared Operator base. Native operators
    # (e.g. OBJ and STL) have no Python execute method to hook.
    candidates = list(bpy.types.Operator.__subclasses__())
    seen = set()
    while candidates:
        cls = candidates.pop()
        if cls in seen:
            continue
        seen.add(cls)
        candidates.extend(cls.__subclasses__())
        if ("bl_rna" in vars(cls) and "export" in getattr(cls, "bl_idname", "").lower()
                and hasattr(cls, "execute") and cls not in _wrapped):
            wrap_operator(cls)
    return 15.0


def register():
    global _enabled
    _enabled = True
    if os.name == "nt" and not bpy.app.timers.is_registered(scan_exporters):
        bpy.app.timers.register(scan_exporters, first_interval=1.5, persistent=True)


def unregister():
    global _enabled
    _enabled = False
    for callback in (scan_exporters, *_pending.values()):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    _pending.clear()
    for cls, (original, wrapper, had_own_execute) in _wrapped.items():
        if cls.__dict__.get("execute") is wrapper:
            if had_own_execute:
                cls.execute = original
            else:
                del cls.execute
    _wrapped.clear()


if __name__ == "__main__":
    register()
