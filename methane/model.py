"""Pure numerical model for isotope vibrations, tracked geometry and clocks.

All functions use the generated/frozen public JSON schema. Model time and video
time are distinct; physical femtoseconds are derived from the harmonic phase.
The model imports NumPy only and can be checked without a graphical renderer.
"""

from pathlib import Path
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("METHANE_DATA_FILE", ROOT / "data/methane_modes.json"))


def load_dataset(path):
    """Read and validate the public numerical schema before constructing a scene.

    A custom E2E dataset must contain exactly two five-atom isotopologues,
    nine positive physical modes each, two selected comparisons, and a proper
    camera rotation. Shapes, finite values and frequency-ratio captions are
    checked here so malformed input cannot silently produce plausible graphics.
    """
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1 or set(dataset["isotopologues"]) != {"CH4", "CH3D"}:
        raise ValueError("Expected the version-1 methane isotope dataset")
    for record in dataset["isotopologues"].values():
        for key, shape in [
            ("coordinates_angstrom", (5, 3)),
            ("masses_amu", (5,)),
            ("frequencies_cm1", (9,)),
            ("cartesian_modes", (9, 5, 3)),
        ]:
            values = np.asarray(record[key], dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid numerical field: {key}")
        if min(record["masses_amu"]) <= 0 or min(record["frequencies_cm1"]) <= 0:
            raise ValueError("Masses and physical frequencies must be positive")
    if len(dataset["comparisons"]) != 2:
        raise ValueError("The scene expects one stretch and one bend comparison")
    for comparison in dataset["comparisons"]:
        for name in ["CH4", "CH3D"]:
            case = comparison["cases"][name]
            vector = np.asarray(case["cartesian_mode_unit_norm"], dtype=float)
            if (
                vector.shape != (5, 3)
                or not np.all(np.isfinite(vector))
                or np.linalg.norm(vector) == 0
            ):
                raise ValueError(
                    "Selected displacement vectors must be finite, nonzero (5,3) arrays"
                )
        ratio = (
            comparison["cases"]["CH3D"]["frequency_cm1"]
            / comparison["cases"]["CH4"]["frequency_cm1"]
        )
        if not np.isclose(comparison["shift_percent"], 100 * (1 - ratio), atol=1e-9):
            raise ValueError("Comparison caption disagrees with its frequency ratio")
    orientation = np.asarray(dataset["selected_orientation"], dtype=float)
    if (
        orientation.shape != (3, 3)
        or not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-10)
        or not np.isclose(np.linalg.det(orientation), 1)
    ):
        raise ValueError("Camera orientation must be a proper 3D rotation")
    return dataset


DATA = load_dataset(DATA_PATH)
PAIRS = DATA["comparisons"]
CASES = {
    name: dict(record, frequency_cm1=PAIRS[0]["cases"][name]["frequency_cm1"])
    for name, record in DATA["isotopologues"].items()
}
SELECTED_ROTATION = np.array(DATA["selected_orientation"])
WHITE = "#EDF0F3"
GOLD = "#E4C58A"
GRAY = "#C5CCD6"
FAINT = "#283440"
BG = "#05070A"
MODE_COLORS = ("#66D9E8", "#F2A6BE")
FPS = 60
FREF = CASES["CH4"]["frequency_cm1"]
FS_PER_Q = 1e15 / (2 * FREF * 2.99792458e10)
SELECT_START = 2.0
LENS_START = 2.6
LENS_FULL = 3.4
FOCUS_START = 13.4
FOCUS_END = 16.0
NUCLEUS_START = 5.4
NUCLEUS_FULL = 6.6
NUCLEUS_CLOSE = 11.2
NUCLEUS_END = 12.4
SLOW_START = 16.0
SLOW_END = 24.0
BEND_START = 24.0
STRETCH_START = 28.0
GUIDE_END = 32.0
SPLIT_START = 32.0
PLOTS_START = 34.0
MODE_INTRO_TEXT = ("Two selected vibrations", "Both shift in frequency when H is replaced by D")
NUCLEUS_OFFSET = np.array([0.0, 2.3, 0.0])
INITIAL_RATE = 24.0
FINAL_RATE = 1.0
Q_SLOW_START = INITIAL_RATE * SLOW_START
Q_SLOW_END = Q_SLOW_START + (INITIAL_RATE + FINAL_RATE) / 2 * (SLOW_END - SLOW_START)
GRAPH_FOCUS_FS = 650.0
READING_SECONDS = 9.0
GRAPH_FOCUS_START = (
    SLOW_END + (GRAPH_FOCUS_FS / FS_PER_Q - (Q_SLOW_END - Q_SLOW_START)) / FINAL_RATE
)
GRAPH_FOCUS_END = GRAPH_FOCUS_START + 2.0
END_TIME = float(np.ceil((GRAPH_FOCUS_END + 0.4 + READING_SECONDS) * FPS) / FPS)
COMPARISON_EXTRA_SECONDS = 2.0
VIDEO_END_TIME = END_TIME + COMPARISON_EXTRA_SECONDS
HOLD = 0.0
FIELD_SCALE = 0.14
LENS_MAG = 4.0
LENS_RADIUS = 1.95
FOCUS_SCALE = 0.65
FINAL_SCALE = 0.55
FINAL_TEXT = ("For these two vibrations, CH₃D completes fewer cycles", "in the same time")
FINAL_TEXT_SIZE = 32
FINAL_TEXT_Y = (-2.75, -3.30)
PLOT_Y = (-2.80, -3.60)
PLOT_AMPLITUDE = 0.20
TARGET_START = np.array([-2.4, 0.7])
TARGET_VELOCITY = np.array([0.22, -0.085])
TARGET_AXIS = np.array([0.32, -0.43, 0.38])
TARGET_AXIS /= np.linalg.norm(TARGET_AXIS)
TARGET_OMEGA = 0.60
LOCK_TAU_END = FOCUS_START + (FOCUS_END - FOCUS_START) / 2

# Cartesian modes are rescaled for display by their largest atomic vector.
# This does not change the eigenfrequencies or imply physical thermal amplitudes.
VECTORS = {}
PURE = {}
FREQUENCIES = {}
for name in CASES:
    raw = DATA["isotopologues"][name]
    u = np.array(raw["cartesian_modes"])
    u /= np.linalg.norm(u, axis=2).max(axis=1)[:, None, None]
    for k in [0, 1]:
        p = np.array(PAIRS[k]["cases"][name]["cartesian_mode_unit_norm"])
        p /= np.linalg.norm(p, axis=1).max()
        PURE[k, name] = p
    u[5] = PURE[0, name]
    VECTORS[name] = u
    FREQUENCIES[name] = np.array(raw["frequencies_cm1"]) / FREF


def ease(x):
    """Return a clamped quintic smootherstep for scalar or array input.

    The first two derivatives vanish at each endpoint. This avoids position,
    velocity and acceleration jumps in optical and presentation transitions."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def comparison_model_time(video_time):
    """Map video seconds onto the original model-time coordinate.

    Exactly two moving seconds are added after the completed split and before
    graph enlargement. Outside that interval the map is identity or v-2.
    The same map drives atoms, plots and the displayed physical clock."""
    duration = GRAPH_FOCUS_START + COMPARISON_EXTRA_SECONDS - PLOTS_START
    return float(
        video_time - COMPARISON_EXTRA_SECONDS * ease((video_time - PLOTS_START) / duration)
    )


def comparison_playback_rate(video_time):
    """Return d(model_time)/d(video_time), a positive dimensionless rate.

    This is the analytic derivative of comparison_model_time. It returns to
    one at both joins; it changes presentation speed, not molecular frequency."""
    duration = GRAPH_FOCUS_START + COMPARISON_EXTRA_SECONDS - PLOTS_START
    s = float(np.clip((video_time - PLOTS_START) / duration, 0.0, 1.0))
    return 1 - COMPARISON_EXTRA_SECONDS * 30 * s * s * (1 - s) * (1 - s) / duration


def phase_rate(v):
    """Return the derivative of the dimensionless phase clock.

    Model time v is in seconds. A cosine ramp changes the graphical rate
    from 24 to 1 between SLOW_START and SLOW_END, without a phase reset."""
    v = np.asarray(v, dtype=float)
    u = np.clip((v - SLOW_START) / (SLOW_END - SLOW_START), 0, 1)
    return FINAL_RATE + (INITIAL_RATE - FINAL_RATE) * (1 + np.cos(np.pi * u)) / 2


def phase_clock(v):
    """Integrate the graphical playback rate into one continuous phase coordinate.

    The analytic integral is linear before/after a cosine slowdown. Each
    harmonic normal coordinate uses sin(pi * frequency_ratio * q), so two
    q units correspond to one CH4 reference stretching period."""
    v = np.asarray(v, dtype=float)
    x = np.clip(v - SLOW_START, 0, SLOW_END - SLOW_START)
    duration = SLOW_END - SLOW_START
    # Integral of the cosine rate ramp: continuity is analytic, not frame-fitted.
    decel = (
        Q_SLOW_START
        + (INITIAL_RATE + FINAL_RATE) * x / 2
        + (INITIAL_RATE - FINAL_RATE) * duration / (2 * np.pi) * np.sin(np.pi * x / duration)
    )
    return np.where(
        v <= SLOW_START,
        INITIAL_RATE * v,
        np.where(v < SLOW_END, decel, Q_SLOW_END + FINAL_RATE * (v - SLOW_END)),
    )


def time_fs(v):
    """Return elapsed physical femtoseconds measured from slowdown onset.

    The phase clock supplies both atomic motion and this timer. Values before
    SLOW_START are zero because the timer has not yet been introduced."""
    return FS_PER_Q * (phase_clock(np.maximum(v, SLOW_START)) - Q_SLOW_START)


def focus(v):
    """Return optical-zoom progress in [0,1] from model seconds v."""
    return float(ease((v - FOCUS_START) / (FOCUS_END - FOCUS_START)))


def separation(v):
    """Return progress of the post-zoom split into stretch and bend views."""
    return float(ease((v - SPLIT_START) / (PLOTS_START - SPLIT_START)))


def lens_growth(v):
    """Return progress of the first tracked molecular magnifier."""
    return float(ease((v - LENS_START) / (LENS_FULL - LENS_START)))


def text_opacity(v):
    """Return opacity of the Hydrogen/Deuterium first-lens captions."""
    return float(ease((v - LENS_FULL) / 0.4)) * (1 - float(ease((v - (FOCUS_START - 0.4)) / 0.4)))


def selection_opacity(v):
    """Return the short opening selection-marker opacity."""
    return float(ease((v - SELECT_START) / 0.2)) * (1 - float(ease((v - LENS_START) / 0.2)))


def ring_opacity(v):
    """Return first-lens rim opacity while the selected molecule zooms."""
    return lens_growth(v) * (1 - float(ease((v - FOCUS_START) / 1.4)))


def background_opacity(v):
    """Return the fading background-field opacity during optical zoom."""
    return 1 - float(ease((v - FOCUS_START) / 2.0))


def plots_visible(v):
    """Return binary plot visibility from completed split onward.

    The plot and timer appear only when the selected view is ready, rather
    than covering the opening molecular field."""
    return float(v + 1e-09 >= PLOTS_START)


def graph_focus(v):
    """Return progress of the final two-second plot enlargement."""
    return float(ease((v - GRAPH_FOCUS_START) / 2.0))


def scene_visibility(v):
    """Return opacity of non-plot scene objects during final enlargement."""
    return 1 - float(ease((v - GRAPH_FOCUS_START) / 0.9))


def final_text_opacity(v):
    """Fade the accepted two-line explanation after graph enlargement."""
    return float(ease((v - GRAPH_FOCUS_END) / 0.4))


def timer_opacity(v):
    """Show the physical timer until enlargement starts at exactly 650 fs."""
    return float(SLOW_START <= v < GRAPH_FOCUS_START)


def plot_geometry(v, k):
    """Return horizontal scale, row centre and amplitude for plot row k.

    Scene lengths are graphical units. The final display amplifies both
    normal-coordinate plots equally without changing their phase or frequency."""
    u = graph_focus(v)
    return (
        1 + 0.05 * u,
        (1 - u) * PLOT_Y[k] + u * (1.05 if k == 0 else -1.05),
        PLOT_AMPLITUDE * (1 + 2 * u),
    )


def nucleus_growth(v):
    """Return growth/retraction progress of the tracked nuclear inset."""
    return float(ease((v - NUCLEUS_START) / (NUCLEUS_FULL - NUCLEUS_START))) * (
        1 - float(ease((v - NUCLEUS_CLOSE) / (NUCLEUS_END - NUCLEUS_CLOSE)))
    )


def mode_intro_opacity(v):
    """Return the brief two-selected-vibrations introduction opacity."""
    return float(ease((v - (SLOW_START + 0.2)) / 0.6)) * (
        1 - float(ease((v - (SLOW_END - 0.8)) / 0.6))
    )


def rigid_tau(v):
    """Integrate rigid-body motion until optical focus fixes the molecule.

    The cosine velocity ramp has zero endpoint acceleration. Internal
    vibrations continue on their own shared phase clock during this lock."""
    if v <= FOCUS_START:
        return float(v)
    if v >= FOCUS_END:
        return LOCK_TAU_END
    x = v - FOCUS_START
    duration = FOCUS_END - FOCUS_START
    return FOCUS_START + 0.5 * (x + duration / np.pi * np.sin(np.pi * x / duration))


def rotation(axis, angle):
    """Return the Rodrigues 3x3 rotation matrix for a unit axis and radian angle."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * K
    )


def phases(index):
    """Return deterministic phases for nine modes of a field molecule.

    The fixed seed makes the illustration repeatable. These phases and the
    display amplitudes are artistic choices, not a sampled thermal ensemble."""
    p = np.random.default_rng(59801 + index).uniform(0, 2 * np.pi, 9)
    if index == 0:
        p[5] = 0
    return p


def local_atoms(name, v, index=0, branch=0):
    """Return (5,3) instantaneous Cartesian coordinates in angstrom.

    Initially all nine harmonic vectors are superposed. The selected molecule
    then isolates the two tracked components, followed by separate branches.
    branch=1 selects stretching, -1 bending, and 0 the unsplit native view."""
    q = float(phase_clock(v))
    eq = np.array(CASES[name]["coordinates_angstrom"])
    # Superpose all nine modes with deterministic, non-thermal display phases.
    mixed = 0.055 * np.einsum(
        "k,kij->ij", np.sin(np.pi * FREQUENCIES[name] * q + phases(index)), VECTORS[name]
    )
    if index != 0:
        return eq + mixed
    pair = sum(
        (
            0.15
            * np.sin(np.pi * PAIRS[k]["cases"][name]["frequency_cm1"] / FREF * q)
            * PURE[k, name]
            for k in [0, 1]
        )
    )
    isolate_pair = float(ease((v - (FOCUS_END - 1.5)) / 1.5))
    combined = (1 - isolate_pair) * mixed + isolate_pair * pair
    if branch == 0:
        return eq + combined
    k = 0 if branch == 1 else 1
    split = separation(v)
    pure = 0.3 * np.sin(np.pi * PAIRS[k]["cases"][name]["frequency_cm1"] / FREF * q) * PURE[k, name]
    return eq + (1 - split) * combined + split * pure


def selected_anchor(v):
    """Return the selected molecular centre in graphical field coordinates."""
    return TARGET_START + TARGET_VELOCITY * rigid_tau(v)


def selected_orientation(v):
    """Return the rigid orientation shared by both selected isotope views.

    The smooth compensation preserves the initial pose and the accepted final
    comparison orientation after the shortened nuclear hold."""
    return (
        rotation(TARGET_AXIS, TARGET_OMEGA * (rigid_tau(v) - 16.7 + 2 * focus(v)))
        @ SELECTED_ROTATION
    )


def projected_vectors(local, orientation, depth=0.0):
    """Perspective-project Cartesian vectors into two graphical dimensions.

    Input is (N,3), orientation is 3x3, and depth is an illustrative offset.
    The 2.15 scale and focal length 12 control appearance, not physical size."""
    xyz = 2.15 * local @ orientation.T
    p = 12 / (12 + depth + xyz[:, 2])
    return xyz[:, :2] * p[:, None]


def native_selected(name, v):
    """Return (5,2) coordinates of the native selected field molecule."""
    return FIELD_SCALE * projected_vectors(
        local_atoms(name, v), selected_orientation(v)
    ) + selected_anchor(v)


def selected_positions(name, v, branch=1):
    """Return physical local coordinates and (5,3) displayed scene points.

    The optical correction keeps the magnifier attached to the live carbon.
    After focus, the same object splits continuously into two selected modes;
    no second molecule appears at an unrelated phase or position."""
    offset = -4 if name == "CH4" else 4
    g = lens_growth(v)
    f = focus(v)
    s = separation(v)
    local = local_atoms(name, v, branch=branch)
    projected = projected_vectors(local, selected_orientation(v))
    anchor = (1 - f) * selected_anchor(v) + f * np.array([0, 0.65])
    optical = FIELD_SCALE * (1 + (LENS_MAG - 1) * g)
    scale = (1 - f) * optical + f * FOCUS_SCALE
    scale = (1 - s) * scale + s * FINAL_SCALE
    # Remove optical displacement of the carbon so the live magnifier stays attached.
    correction = (1 - f) * (optical - FIELD_SCALE) * projected[0]
    xy = anchor + scale * projected - correction + [offset, (1 if branch == 1 else -1) * 1.4 * s]
    return (local, np.column_stack((xy, np.zeros(5))))


def lens_state(name, v):
    """Return native points, displayed points, lens centre/radius and label centre.

    All positions are evaluated at the same model time, so the first lens
    follows the moving source without an independently animated substitute."""
    offset = -4 if name == "CH4" else 4
    native = native_selected(name, v) + [offset, 0]
    displayed = selected_positions(name, v)[1]
    centre = displayed[0]
    g = lens_growth(v)
    f = focus(v)
    radius = (0.52 + (LENS_RADIUS - 0.52) * g) * (
        1 - f + f * FOCUS_SCALE / (FIELD_SCALE * LENS_MAG)
    )
    x = np.clip(centre[0], 0.9, 4.65) if name == "CH3D" else np.clip(centre[0], -4.65, -0.9)
    label = np.array([x, centre[1] - radius - 0.7, 0.0])
    return (native, displayed, centre, radius, label)


def field_definitions():
    """Generate sixteen deterministic noninteracting field-molecule poses.

    Translations, rotations and spacing are illustrative. They do not model
    a solvent, collisions, diffusion coefficients or a temperature."""
    rng = np.random.default_rng(20261110)
    definitions = []
    while len(definitions) < 16:
        c = np.array([rng.uniform(-3.7, 3.7), rng.uniform(-3.4, 3.4)])
        if np.linalg.norm(c - TARGET_START) < 1.2 or any(
            (np.linalg.norm(c - d["centre"]) < 1.05 for d in definitions)
        ):
            continue
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.23, 0.38)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        initial = rotation(axis, float(rng.uniform(-np.pi, np.pi)))
        definitions.append(
            {
                "index": len(definitions) + 1,
                "centre": c,
                "velocity": speed * np.array([np.cos(angle), np.sin(angle)]),
                "axis": axis,
                "omega": float(rng.uniform(0.4, 0.65)),
                "initial_orientation": initial,
                "depth": float(rng.uniform(-0.5, 1.5)),
            }
        )
    return definitions


def field_positions(name, v, d):
    """Return one background molecule's projected (5,2) scene positions.

    Periodic wrapping keeps the initial field populated; individual bond
    segments are clipped to the correct isotope panel by the renderer."""
    anchor = (d["centre"] + d["velocity"] * v + 4.5) % 9.0 - 4.5
    orientation = rotation(d["axis"], d["omega"] * v) @ d["initial_orientation"]
    local = local_atoms(name, v, d["index"])
    xy = anchor + FIELD_SCALE * projected_vectors(local, orientation, d["depth"])
    f = focus(v)
    a = selected_anchor(v)
    xy = a + (1 + 0.45 * f) * (xy - a)
    xy[:, 0] += -4 if name == "CH4" else 4
    return xy


def nucleus_state(name, v):
    """Return source atom, inset centre, radius and growth for H/D magnification.

    Atom 1 is H in CH4 and D in CH3D. The full inset follows carbon with a
    constant vertical offset; opening and retraction target the live H/D atom."""
    points = selected_positions(name, v, 1)[1]
    source = points[1]
    destination = points[0] + NUCLEUS_OFFSET
    g = nucleus_growth(v)
    return (source, (1 - g) * source + g * destination, 0.16 + (1.27 - 0.16) * g, g)


def guide_weights(v):
    """Return stretch/bend annotation opacities without modifying motion.

    Bending is introduced first, then stretching. The existing superposed
    trajectory is unchanged while these arrows and labels appear."""

    def window(start, end):
        """Return a short smootherstep fade-in/fade-out visibility window."""
        return float(ease((v - start) / 0.4)) * (1 - float(ease((v - (end - 0.4)) / 0.4)))

    return (window(STRETCH_START, GUIDE_END), window(BEND_START, STRETCH_START))
