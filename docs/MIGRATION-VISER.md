# Migrating from the PyVista renderer to the Viser viewer

The PyVista backend (`tsr.viz`) is **deprecated** and will be removed in **sstsr 3.0**.
The recommended visualization path is the interactive Viser viewer, `tsr.viser`.

Nothing breaks in 2.x. `tsr.viz` keeps working, the `[viz]` extra still installs
PyVista, and constructing `TSRVisualizer` emits a `DeprecationWarning` pointing here.
Merely importing `tsr.viz` does not warn, so a library that imports it cannot warn on
its users' behalf.

```bash
pip install "sstsr[viser]"        # recommended
pip install "sstsr[viz]"          # PyVista, deprecated, unchanged through 2.x
```

## The shape of the change

PyVista renders a **picture of samples**: you pick poses, draw them, and save a PNG.
Viser serves a **scene you interrogate**: you pick a template family and scrub its
continuous freedoms while watching one grasp move. That is the point of the switch —
a static image renders a continuous freedom as clutter — so the APIs are deliberately
not parallel.

| PyVista workflow | Viser replacement |
|---|---|
| `TSRVisualizer(...).render(ref, subject, poses, out=...)` | `show_templates(templates, cylinder=..., gripper=..., seed=...)` — samples and draws the family |
| `TSRVisualizer(...).render_multi(ref, subjects, out=...)` | `show_templates(...)` per family into one shared `server`, each with its own `name=` |
| Inspecting a region by rendering many sampled poses | `explore_templates(...)` — a slider per free `Bw` coordinate, driving one grasp |
| `cylinder_renderer`, `box_renderer`, `sphere_renderer`, `torus_renderer` (the reference object) | `scene.add_cylinder` / `add_box` / `add_icosphere` and Viser's own primitives; for the common case pass `cylinder=(radius, height)` to `show_templates` and it is drawn for you |
| `parallel_jaw_renderer(...)` | drawn for you from the template's own preshape; the raw segments are `gripper_segments(finger_length, aperture)` |
| `placed_box_renderer`, `placed_cylinder_renderer`, `placed_sphere_renderer`, `placed_torus_renderer` (a placed object at a pose) | `scene.add_box` / `add_cylinder` / `add_icosphere` with `position=` and `wxyz=`; for many poses of one shape use a batched node (`add_batched_meshes_simple`) |
| `plasma_colors(n)` | jaws are coloured by depth index automatically; pass your own `colors` to `add_line_segments` when drawing directly |
| `table_surface_renderer(...)` | `scene.add_grid(...)` or `scene.add_box(...)` |
| **Headless PNG** (`out="x.png"` with no display) | **No replacement — see below** |

## Headless PNG rendering

This capability goes away with PyVista, deliberately and with eyes open.

Viser's pixels come from the browser's WebGL context. Confirmed against viser 1.1.1:

- `get_render(...)` is a method on `ClientHandle`; with no browser attached
  `server.get_clients()` is empty, so there is no handle to call it on;
- no offscreen, headless or screenshot API exists on `viser` or `ViserServer`;
- `get_scene_serializer().serialize()` produces a `.viser` scene file and `as_html()` a
  standalone interactive page — neither is an image.

So capture runs through a browser:

```python
server = show_templates(templates, cylinder=(0.03, 0.12), gripper=gripper, seed=0)
# open http://localhost:8080 in a browser, then:
client = next(iter(server.get_clients().values()))
image = client.get_render(height=900, width=1400, transport_format="png")  # numpy RGBA
```

If you need a picture with no browser in the loop at all, stay on `tsr.viz` for as long
as 2.x is supported, and keep the generated PNG.

The three images in this repository's README are already produced this way, by
`scripts/render_readme_figures.py`, so they no longer depend on PyVista.

`as_html()` is worth knowing about for documentation sites: it writes a self-contained
interactive page, which is strictly more useful than a static image where a browser is
available anyway.

## Two habits worth carrying over

**Seed your views.** `show_templates(..., seed=0)` and `sample_poses(..., seed=0)` make
a view reproducible; without a seed or an explicit `rng` the sampling is deliberately
nondeterministic.

**Let the template describe itself.** `free_coordinates(template)` returns the
non-degenerate `Bw` rows, which is where the explorer's sliders come from. A cylinder
side grasp offers `z` and `yaw`; a cap grasp offers `yaw` alone. Nothing in the viewer
invents a freedom the template does not have.

## Stability

The `tsr.viser` API — `show_templates`, `explore_templates`, `free_coordinates`,
`sample_poses`, `gripper_segments` — is stable for 2.x under the usual semantic
versioning rules. Viser stays optional and out of `[project.dependencies]`; importing
core sstsr imports neither backend and never starts a server.
