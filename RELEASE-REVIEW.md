# Local release review

Six installer ZIPs were exercised together in a fresh Blender 5.1.2 profile on Windows. Blender loaded each module from that profile. All six passed a disable and re-enable cycle.

## Finding fixed for this release

**Dense UV 1.1.1.** Blender rejected the installed 1.1.0 manifest because its tagline was 70 characters. The release shortens the tagline and increments the patch version. Blender’s extension validator accepted the finished ZIP, and Blender’s extension installer loaded it successfully. The UV algorithms are the same as 1.1.0.

## Functional checks

| Tool | What was exercised | Result |
| --- | --- | --- |
| Heavy Scaffold | Convert the scaled guide, then watertight-bake all three styles. Inspect every edge for manifold incidence and consistent winding. | Monolith 6,265 faces, Flange 8,505, Lattice 29,445. Zero failing edges in all three. |
| Rubble | Generate a pile with rebar. Exercise five controlled fractures, including rotated, concave, and short-bar cases, at four Amount settings. | Generated steel on the pile. Opposed bar axes in the fixtures aligned within 0.004 mm. |
| Floor cable | Add and refresh a run, inject failures during graph construction and commit, reopen the file, then add another run. | Socket references, material overrides, and unrelated node trees survived. The reopened run produced geometry. |
| Hanging cable | Build two spans in Row, Bundle, and Stack layouts, then bake. | All three layouts generated. The bake hid the live output from rendering. |
| Dense UV | Unwrap, pack and analyze a rectangular sheet. Repeat packing on wide, tall, and square sheets. Check stacked and partially overlapping islands and Edit Mode selection history. | Coverage and geometry assertions passed, including the 12,800-triangle analysis and selection restoration after an injected failure. |
| N-Fix Addon Panel Optimizer | Discover viewport sidebar categories and reload the add-on. | Discovered Blender’s Item category and registered again successfully. |
| Universal Export Folder Opener | Write an actual FBX with spaces in the filename and capture the Explorer invocation. | One select-file command pointed to the exported file. Explorer itself was intercepted for this test. |

## Coverage limits

The final package tests ran on Windows with Blender 5.1.2. The minimum Blender versions in the README are the versions declared by the add-ons. Interactive drawing gestures and every third-party sidebar combination were outside this pass.

The earlier geometry review measured brace-to-frame gaps of about 11 mm and 16 mm on a deliberately twisted Monolith guide. The successful scaled-guide bakes above cover a different fixture.
