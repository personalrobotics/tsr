# Experimental interactive viewer (Viser) — evaluation

`tsr.viser` is an **experiment** (#77), not a supported backend and not a replacement
for `tsr.viz`. It exists to answer one question: is a browser-based interactive view a
better way to inspect grasp templates than the current static PyVista renderer?

PyVista is unchanged. Adoption, deprecation and removal are separately gated in #139
and #140.

Visualization is explanatory and diagnostic. Nothing rendered here is evidence that a
template is correct — that is the analytic oracle's job (see ARCHITECTURE.md).

## Install and run

```bash
pip install "sstsr[viser]"          # or: uv sync --extra viser
uv run python examples/viser_cylinder_grasps.py
```

Then open <http://localhost:8080>. `Ctrl-C` stops the server.

```python
from tsr.hands import ParallelJawGripper
from tsr.viser import show_templates

gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)
templates = gripper.grasp_cylinder_side(0.03, 0.12)

server = show_templates(templates, cylinder=(0.03, 0.12), gripper=gripper, seed=0)
...                                  # draw more into the same server if you like
server.stop()
```

`show_templates` either takes a `viser.ViserServer` or creates one and returns it, so
the server's lifetime is always explicit and visible at the call site. Sampling takes a
`seed` or an `rng`, so a view can be reproduced exactly.

## Scrubbing the region with sliders

A static cloud of sampled poses shows a region as clutter. `explore_templates` instead
drives **one** grasp from a slider per free `Bw` coordinate, so a continuous freedom is
scrubbed rather than sampled:

```bash
uv run python examples/viser_cylinder_grasps.py --explore
```

```python
from tsr.viser import explore_templates, free_coordinates

free_coordinates(templates[0])
# [(2, 'z [m]', -0.057, 0.057), (5, 'yaw [rad]', 0.0, 6.283)]

server = explore_templates(templates, cylinder=(0.03, 0.12), gripper=gripper)
```

The sliders come from the template's own non-degenerate `Bw` rows, so a template is
never given a slider for a coordinate it does not actually have: a cylinder side grasp
offers `z` and `yaw`, a cap grasp offers `yaw` alone. A dropdown selects which template
(so the discrete depth levels are steps, not a slider), a checkbox overlays the sampled
cloud for comparison, and a read-only field shows the current coordinates.

Moving a slider re-evaluates `tsr.to_transform(ξ)` and assigns the node's `position` and
`wxyz`. The jaw geometry is uploaded once per template, not per frame, so scrubbing
sends two transforms rather than re-sending geometry.

## Over SSH

Viser serves HTTP and a websocket on one port; forward it and browse locally:

```bash
ssh -L 8080:localhost:8080 user@host
uv run python examples/viser_cylinder_grasps.py --port 8080   # on the remote host
```

Open <http://localhost:8080> on your own machine. Nothing needs to be installed
locally, and no X server, VTK or OpenGL is involved on the remote side — which is the
main practical difference from the PyVista path.

`server.request_share_url()` exists for sharing without a tunnel; it relays through a
third-party service, so it is a deliberate choice rather than the documented default.

## What it draws

For each sampled pose: the canonical end-effector axes (x red, y green, **z blue =
approach**) and the idealized jaw geometry at the template's own preshape, plus the
reference object in the template's frame. Approach direction, jaw orientation and
grasp depth are therefore readable directly, and orbiting shows the continuous yaw
freedom as a band of poses rather than a single picture.

## Measured behaviour

On this machine (M-series laptop, viser 1.1.1), server-side scene construction:

| poses | line segments | build time |
|---|---|---|
| 12 | 48 | 0.003 s |
| 240 | 960 | 0.016 s |
| 1200 | 4800 | 0.072 s |
| 4800 | 19200 | 0.261 s |

Server startup plus a first scene is ~0.1 s. All poses share **two** scene nodes — one
batched-axes node and one line-segments node — so the browser updates two objects
rather than N, which is why this scales linearly rather than collapsing at a few
hundred poses. Browser-side frame rate has to be judged interactively; these numbers
only bound the Python side.

## Image capture

Viser captures images through a **connected browser client**:

```python
client = next(iter(server.get_clients().values()))    # requires an open browser
image = client.get_render(height=720, width=1280)     # numpy array
```

This is not browser-free: with no client connected there is nothing to render from, so
it does not replace `tsr.viz`'s headless PNG generation for docs or CI. Whether we
intend to keep a browser-free path is one of the open questions in #77.

## Constraints this respects

- Viser is not in `[project.dependencies]`; it lives in the `viser` extra.
- `import tsr` imports neither Viser nor this module, and never starts a server
  (enforced in `tests/tsr/test_architecture.py`).
- The default test suite does not require the extra: the backend's tests skip when it
  is absent, and none of them start a server or open a port.
- There is no backend-neutral scene abstraction: this calls Viser's own mesh, frame and
  line-segment APIs directly, and passes plain NumPy arrays at the sstsr boundary.
- The GUI is Viser's own: sliders and dropdowns are created through `server.gui`, and no
  widget abstraction is introduced for one experimental backend.
