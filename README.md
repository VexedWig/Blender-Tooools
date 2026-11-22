# Blender Artist Tools

Blender add-ons by **Aidan Hart** for building game environments. Generate scaffold from a guide mesh, expose rebar along concrete fractures, draw hanging cables between surfaces, and route cable bundles between movable sockets. Three workflow tools handle UV packing, sidebar organization, and finding exported files. [Aidan Hart’s portfolio](https://aidan-hart.com)

![Monolith, Flange, and Lattice scaffolds generated from the same guide, alongside the Heavy Scaffold panel](showcase/01%20Structural%20styles.png)

## Install

1. Choose a ZIP from the table below. Each ZIP installs independently. Rubble and the floor cable tool share one ZIP.
2. In Blender, open **Edit → Preferences → Add-ons**. Open the upper-right menu, choose **Install from Disk**, and select the ZIP directly.
3. Search for the add-on by name and enable its checkbox if needed. In the 3D Viewport, press **N** to open the sidebar for the generators and Dense UV. The export opener works in the background.

All six packages were installed and exercised in a clean **Blender 5.1.2** profile on Windows. The minimums below come from the add-ons’ own metadata. [Blender installation guide](https://docs.blender.org/manual/en/5.0/editors/preferences/extensions.html)

| Add-on | Install ZIP | Blender minimum | Where to find it |
| --- | --- | --- | --- |
| Heavy Scaffold Generator | [1.2.5](packages/heavy_scaffold-1.2.5.zip) | 5.1 | 3D Viewport → Generators → Heavy Scaffold |
| Rubble + Cable Generator | [2.3.1](packages/kitbash_generators-2.3.1.zip) | 5.1 | 3D Viewport → Generators → Rubble + Cable |
| Hanging Cable Generator | [1.0.2](packages/hanging_cables-1.0.2.zip) | 5.1 | 3D Viewport → Generators → Hanging Cables |
| Dense UV | [1.1.1](packages/dense_uv-1.1.1.zip) | 4.2 | 3D Viewport or UV Editor → sidebar → Dense UV |
| N-Fix Addon Panel Optimizer | [3.0.10](packages/n-fix-addon-panel-optimizer-3.0.10.zip) | 4.5 | Top center of the 3D Viewport |
| Universal Export Folder Opener | [1.0.2](packages/universal_export_folder_opener-1.0.2.zip) | 4.0 | Windows Explorer after a supported export |

## Heavy Scaffold

Choose **Add Scaffold**, enter Edit Mode on the guide, and extrude a face to extend the structure. Switch between **Monolith**, **Flange**, and **Lattice** on that same guide. **W** and **H** set the target bay dimensions. **Adaptive** follows the guide’s slope, and **Brace Selected Faces** gives individual faces their own bracing pattern.

Pick a scene mesh in the screw-object field to use a custom fastener. When the structure is ready, choose **Bake to Mesh**. Enable **Watertight** before baking when you need a closed mesh. For strongly twisted guide faces, inspect the brace-to-frame contact at the corners.

<details>
<summary>See the other nine scaffold sheets</summary>

![Heavy Scaffold — Face bracing](showcase/02%20Face%20bracing.png)

![Heavy Scaffold — Bay spacing](showcase/03%20Bay%20spacing.png)

![Heavy Scaffold — Sloped guides](showcase/04%20Sloped%20guides.png)

![Heavy Scaffold — Return deck](showcase/05%20Return%20deck.png)

![Heavy Scaffold — Member sizes](showcase/06%20Member%20sizes.png)

![Heavy Scaffold — Foot and crown](showcase/07%20Foot%20and%20crown.png)

![Heavy Scaffold — Fastener library](showcase/08%20Fastener%20library.png)

![Heavy Scaffold — Brace-end joints](showcase/09%20Brace-end%20joints.png)

![Heavy Scaffold — From guide to mesh](showcase/10%20From%20guide%20to%20mesh.png)

</details>

## Rubble Generator

Pick **Rubble**, set the pile dimensions and piece counts, enable **Add Rebar**, and choose **Generate**. **Amount** adds exposed ends on each reinforced chunk. **Rebar Length** limits how far each end can extend beyond the fracture. **Additional Pieces** reinforces more chunks, starting with the largest eligible pieces. Keep the same seed to compare settings on the same pile.

![The same rubble pile comparing rebar amount, exposed length, and the number of reinforced pieces](showcase/Rubble/01%20Rubble%20Generator.png)

## Cable Generator

Switch to **Cable** and choose **Add Cable Run**. Move the **Start** and **End** empties to position the sockets. Select the generated mesh and open its Geometry Nodes modifier. **Count** controls the number of runs, **Slack** changes the length available for the route, and **Pipe Ratio** changes the share of pipes in the bundle. To make the ends follow scene props, parent each socket empty to its prop.

![Cable Generator modifier controls alongside the eight-cable run they produce](showcase/Cable/01%20Cable%20Generator%20overview.png)

<details>
<summary>Compare count, slack, and pipe ratio, and see the anchors move</summary>

![Cable count, slack and pipe ratio compared between the same anchors](showcase/Cable/02%20Cable%20Generator%20settings.png)

![Animated example of the cable following two moving anchors](showcase/Cable/Moving%20the%20primitives.webp)

</details>

## Hanging Cable Generator

Point and click cable generator. Choose **Draw Cables**, click two surfaces to place the first span, and keep clicking to extend the run. **Shift-click** starts a separate run. Press **Enter** to finish. **Sag** sets the depth of the curve and **Weight Bias** shifts its low point toward one end. Choose **Edit Points** to reposition anchors or **Bake to Mesh** for a mesh copy.

![Hanging cable comparisons for sag, weight bias, count, thickness, layout, and segments](showcase/Hanging%20Cables/01%20Hanging%20Cable%20Generator.png)

## Dense UV

Select your mesh objects and open **Dense UV**. Set the target texture width, height, and padding in pixels. Choose **Unwrap & Pack** for a fresh layout or **Pack Only** to rearrange existing islands. **Analyze** reports coverage, texel density, and stretch. For a rectangular atlas, enter both dimensions before packing.

The Radio Tower center console has 1,342 faces and closely spaced support loops around its recessed openings. With no marked seams or sharp edges, the same mesh took one click with Dense UV and six clicks in the manual Smart UV workflow.

![The radio tower console and its UV layouts after one Dense UV click and six manual workflow clicks](showcase/Dense%20UV/02%20Console%20unwrap%20workflow.png)

**How it works.** Hybrid marks seams from the angle between faces and the enabled seam rules. It unwraps the mesh, checks each island for stretch, flipped faces, and overlaps, then runs Smart UV Project on the islands that fail. Equal Texel Density rescales each island by the square root of its 3D area divided by its UV area. Doubling an island’s width and height gives it four times the texture area. The packer fits the resulting islands into the requested texture dimensions with a gap measured in pixels.

<details>
<summary>See the Dense UV panel and settings</summary>

![Dense UV panel with its unwrap, island and packing controls explained](showcase/Dense%20UV/01%20Dense%20UV%20overview.png)

</details>

## N-Fix Addon Panel Optimizer

Click a tab at the top of the 3D Viewport to open its panels in the right-hand dock. Right-click a tab to pin it. Use the arrows to unfold or collapse the stack, or press **N** with the pointer over the viewport. If the entire bar is hidden, check **Show Gizmos** in the viewport and **Show Add-on Stack** in the add-on preferences.

![N-Fix Addon Panel Optimizer in the Blender viewport](showcase/N-Fix%20Addon%20Panel%20Optimizer/01%20N-Fix%20Addon%20Panel%20Optimizer%20overview.png)

<details>
<summary>See how favorites and the expanded menu work</summary>

![The menu open and folded with zero, three, and nine favorite tabs](showcase/N-Fix%20Addon%20Panel%20Optimizer/02%20Favorites%20and%20the%20add-on%20menu.png)

</details>

## Universal Export Folder Opener

Enable the add-on, then export an FBX or glTF file. After a successful export, Windows Explorer selects the file. An FBX batch export opens the containing folder. The opener hooks Python exporters. Blender’s native OBJ, STL, and USD exporters use a different path.

![Blender exports Ventilation Elbow.fbx and File Explorer opens with that file selected](showcase/Universal%20Export%20Folder%20Opener/01%20Universal%20Export%20Folder%20Opener.png)

## Source and licensing

The editable Python for each installer is in [src](src/). Aidan Hart’s original source for Heavy Scaffold, Rubble + Cable, Hanging Cable, and Universal Export Folder Opener is available under **MIT**. If you redistribute copies or substantial portions of that code, include the copyright and permission notice. Dense UV and N-Fix Addon Panel Optimizer use **GPL-3.0-or-later**. Each installer carries its applicable license. [MIT notice](src/heavy_scaffold/LICENSE.txt) · [Dense UV license](src/dense_uv/LICENSE.txt) · [N-Fix Addon Panel Optimizer license](src/npanel_tab_bar/LICENSE.txt)

## A note from Aidan

I make these tools available for free. You’re welcome to use them in personal projects and paid work, edit the code, and build on it. If a tool helps you ship something, I’d appreciate a line in your README, game credits, or project acknowledgments. **Blender Artist Tools by Aidan Hart** is plenty.

If you’re sharing an unchanged add-on, please point people to the free original or offer your copy for free. If you’ve substantially reworked it and sell your version, please credit the original and tell people what you changed. These are requests to the community. The licenses above govern redistribution, including paid copies. [Blender licensing FAQ](https://www.blender.org/support/faq/)
