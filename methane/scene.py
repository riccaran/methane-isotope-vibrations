"""Manim view layer for the numerical methane-isotope model.

Render through ``python tools/render.py``. Every scene object is updated from
one shared, continuously retimed clock; drawing never changes scientific data.
The preferred reference fonts are Segoe UI/Consolas, with open-font fallbacks
for Linux/macOS. See README for physical units and illustrative conventions.
"""

from pathlib import Path
import hashlib
import json
import os
import numpy as np
import manimpango
from manim import (
    Scene,
    VGroup,
    ValueTracker,
    Text,
    Line,
    Circle,
    Dot,
    RoundedRectangle,
    Rectangle,
    VMobject,
    config,
    LEFT,
    RIGHT,
    DOWN,
    UP,
    ORIGIN,
)
from methane import model as physics

_FONTS = set(manimpango.list_fonts())
TEXT_FONT = "Segoe UI" if "Segoe UI" in _FONTS else "DejaVu Sans"
MONO_FONT = "Consolas" if "Consolas" in _FONTS else "DejaVu Sans Mono"
METADATA_PATH = Path(os.environ.get("METHANE_METADATA", Path(config.media_dir) / "metadata.json"))
config.frame_width = 16
config.frame_height = 9
config.background_color = physics.BG
config.disable_caching = True


def clip_line(line, lo, hi):
    """Clip a drawn bond segment to the horizontal isotope-panel bounds.

    This mutates only the Manim line's endpoints/opacity, never molecular data."""
    a, b = (line.get_start().copy(), line.get_end().copy())
    dx = b[0] - a[0]
    if a[0] < lo and b[0] < lo or (a[0] > hi and b[0] > hi):
        line.set_stroke(opacity=0)
        return
    if abs(dx) < 1e-12:
        return
    x0 = max(0, min((lo - a[0]) / dx, (hi - a[0]) / dx))
    x1 = min(1, max((lo - a[0]) / dx, (hi - a[0]) / dx))
    if x1 > x0:
        line.put_start_and_end_on(a + x0 * (b - a), a + x1 * (b - a))
    else:
        line.set_stroke(opacity=0)


class FieldMolecule(VGroup):
    """Render one small background molecule in the opening field.

    Only bonds are drawn at this scale; its motion is illustrative while
    internal displacement uses the nine supplied harmonic coordinates."""

    def __init__(self, name, d):
        """Create bonds for isotope name and deterministic field-definition dictionary d.

        name is CH4 or CH3D. d stores the seeded pose, velocity and rotation;
        the only coloured isotope bond is the bond from carbon to atom 1."""
        super().__init__()
        self.name = name
        self.d = d
        self.bonds = VGroup(
            *[
                Line(
                    LEFT,
                    RIGHT,
                    stroke_color=physics.GOLD if name == "CH3D" and i == 1 else physics.WHITE,
                    stroke_width=1.6,
                )
                for i in range(1, 5)
            ]
        )
        self.add(self.bonds)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        opacity = physics.background_opacity(v)
        points = physics.field_positions(self.name, v, self.d)
        lo, hi = (-8, 0) if self.name == "CH4" else (0, 8)
        for i, line in enumerate(self.bonds, 1):
            line.put_start_and_end_on(np.r_[points[0], 0], np.r_[points[i], 0]).set_stroke(
                opacity=0.85 * opacity
            )
            clip_line(line, lo, hi)
        return self


class SelectedMolecule(VGroup):
    """Render the selected isotope as bonds, nodes and readable atom labels.

    A branch identifies the later stretch or bend view. The bend copy stays
    hidden until the post-zoom split starts."""

    def __init__(self, name, branch):
        """Create one isotope view with branch 1 (stretch) or -1 (bend).

        Both branches share the same pre-split source position and phase.
        Bonds, small nodes and atom labels are reused rather than recreated."""
        super().__init__()
        self.name = name
        self.branch = branch
        self.bonds = VGroup(
            *[
                Line(
                    LEFT,
                    RIGHT,
                    stroke_color=physics.GOLD if name == "CH3D" and i == 1 else physics.WHITE,
                    stroke_width=2.0,
                )
                for i in range(1, 5)
            ]
        )
        self.nodes = VGroup(
            *[
                Dot(
                    radius=0.022, color=physics.GOLD if name == "CH3D" and i == 1 else physics.WHITE
                )
                for i in range(5)
            ]
        )
        self.atoms = VGroup(
            *[
                Text(
                    "C" if i == 0 else "D" if name == "CH3D" and i == 1 else "H",
                    font=MONO_FONT,
                    font_size=24,
                    color=physics.GOLD if name == "CH3D" and i == 1 else physics.WHITE,
                )
                for i in range(5)
            ]
        )
        self.heights = [m.height for m in self.atoms]
        self.add(self.bonds, self.nodes, self.atoms)
        self.set_z_index(5)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        if self.branch == -1 and v < physics.SPLIT_START:
            self.set_opacity(0)
            return self
        _, points = physics.selected_positions(self.name, v, self.branch)
        g = physics.lens_growth(v)
        visible = physics.scene_visibility(v)
        size = (1 - physics.focus(v)) * (0.25 + 0.75 * g) + physics.focus(v) * (
            1 - 0.15 * physics.separation(v)
        )
        for i, line in enumerate(self.bonds, 1):
            vector = points[i] - points[0]
            unit = vector / np.linalg.norm(vector)
            gap = 0.115 * size * g
            line.put_start_and_end_on(points[0] + gap * unit, points[i] - gap * unit).set_stroke(
                opacity=visible, width=1.6 + 0.6 * g
            )
        for node, p in zip(self.nodes, points):
            node.move_to(p).set_opacity((1 - g) * visible)
        for atom, p, height in zip(self.atoms, points, self.heights):
            atom.scale_to_fit_height(height * size).move_to(p).set_opacity(g * visible)
        return self


class LensContext(VGroup):
    """Render the moving first magnifier and its Hydrogen/Deuterium label."""

    def __init__(self, name):
        """Create the first magnifier for isotope name (CH4 or CH3D).

        Its opaque disc, caption plate and element/isotope name are separate
        layers, so the moving lens never reveals unrelated background bonds."""
        super().__init__()
        self.name = name
        self.colour = physics.GOLD if name == "CH3D" else physics.WHITE
        self.disc = Circle(
            radius=1,
            stroke_color=self.colour,
            stroke_width=2.5,
            fill_color=physics.BG,
            fill_opacity=1,
        )
        self.caption = Text(
            "Deuterium (D)" if name == "CH3D" else "Hydrogen (H)",
            font=TEXT_FONT,
            font_size=34,
            color=self.colour,
        )
        self.plate = RoundedRectangle(
            corner_radius=0.07,
            width=self.caption.width + 0.24,
            height=self.caption.height + 0.16,
            stroke_width=0,
            fill_color=physics.BG,
            fill_opacity=1,
        )
        self.add(self.disc, self.plate, self.caption)
        self.set_z_index(3)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        native, points, c, r, label = physics.lens_state(self.name, v)
        g = physics.lens_growth(v)
        a = physics.text_opacity(v)
        self.disc.scale_to_fit_width(2 * r).move_to(c).set_stroke(
            opacity=physics.ring_opacity(v)
        ).set_fill(physics.BG, opacity=g * (1 - physics.focus(v)))
        name_centre = label + [0, 0.25, 0]
        self.caption.move_to(name_centre).set_opacity(a)
        self.plate.move_to(name_centre).set_fill(physics.BG, opacity=a)
        return self


class Selection(VGroup):
    """Render the brief opening halo and corner markers on the selected molecule."""

    def __init__(self, name):
        """Create a short-lived selection halo and corner marks for isotope name.

        The marker follows the native molecular carbon before magnification."""
        super().__init__()
        self.name = name
        self.colour = physics.GOLD if name == "CH3D" else physics.WHITE
        self.halo = Circle(radius=0.56, stroke_color=self.colour, stroke_width=7, fill_opacity=0)
        self.corners = VGroup(*[VGroup(Line(LEFT, RIGHT), Line(DOWN, UP)) for _ in range(4)])
        self.add(self.halo, self.corners)
        self.set_z_index(6)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        c = np.r_[
            physics.native_selected(self.name, v)[0] + [-4 if self.name == "CH4" else 4, 0], 0
        ]
        a = physics.selection_opacity(v)
        self.halo.move_to(c).set_stroke(opacity=0.18 * a)
        for corner, (sx, sy) in zip(self.corners, [(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            p = c + [sx * 0.56, sy * 0.56, 0]
            corner[0].put_start_and_end_on(p, p + [-sx * 0.18, 0, 0])
            corner[1].put_start_and_end_on(p, p + [0, -sy * 0.18, 0])
            corner.set_stroke(color=self.colour, width=2.3, opacity=a)
        return self


class NuclearMagnifier(VGroup):
    """Render a tracked diagrammatic H/D nucleus with the correct p/n count.

    The inset follows the source molecular motion during opening and closure.
    It illustrates isotope composition rather than resolved nuclear structure."""

    def __init__(self, name):
        """Create the second tracked magnifier for isotope name (CH4 or CH3D).

        Both isotopes receive one proton; only CH3D receives one neutron.
        Reusable diagram templates are scaled and translated every frame."""
        super().__init__()
        self.name = name
        self.colour = physics.WHITE if name == "CH4" else physics.GOLD
        self.halo = Circle(radius=0.19, stroke_color=self.colour, stroke_width=2.5)
        self.leader = Line(LEFT, RIGHT, stroke_color=self.colour, stroke_width=1.7)
        self.disc = Circle(
            radius=1,
            stroke_color=self.colour,
            stroke_width=2.5,
            fill_color=physics.BG,
            fill_opacity=1,
        )
        self.content = VGroup()
        self.content.add(
            Text(
                "H nucleus" if name == "CH4" else "D nucleus",
                font=TEXT_FONT,
                font_size=25,
                color=self.colour,
            ).move_to([0, 0.9, 0])
        )

        def nucleon(label, colour, x):
            """Build a flat labelled p/n disc for the diagrammatic nuclear inset.

            Its radius and placement communicate particle counts only; they are not
            a nuclear-wavefunction calculation or a physical length scale."""
            particle = VGroup(
                Circle(
                    radius=0.38,
                    stroke_color=colour,
                    stroke_width=1.3,
                    fill_color=colour,
                    fill_opacity=1,
                ),
                Text(label, font=TEXT_FONT, font_size=25, color="#14212C"),
            )
            return particle.move_to([x, 0.04, 0])

        self.content.add(nucleon("p", "#EF9A8C", 0 if name == "CH4" else -0.36))
        if name == "CH3D":
            self.content.add(nucleon("n", "#8DCDE8", 0.36))
        count = VGroup(Text("1 proton", font=TEXT_FONT, font_size=22, color=physics.WHITE))
        if name == "CH3D":
            count.add(Text("1 neutron", font=TEXT_FONT, font_size=22, color=physics.WHITE))
        count.arrange(DOWN, buff=0.06).move_to([0, -0.72, 0])
        self.content.add(count)
        self.template = self.content.copy()
        self.add(self.halo, self.leader, self.disc, self.content)
        self.set_z_index(12)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        source, centre, r, g = physics.nucleus_state(self.name, v)
        visible = float(physics.NUCLEUS_START < v < physics.NUCLEUS_END)
        self.halo.move_to(source).set_stroke(opacity=0.85 * visible)
        delta = centre - source
        distance = np.linalg.norm(delta)
        if distance > r + 0.22:
            unit = delta / distance
            self.leader.put_start_and_end_on(source + 0.2 * unit, centre - r * unit).set_stroke(
                opacity=0.7 * g
            )
        else:
            self.leader.set_stroke(opacity=0)
        self.disc.scale_to_fit_width(2 * r).move_to(centre).set_stroke(opacity=visible).set_fill(
            physics.BG, opacity=visible
        )
        detail_alpha = float(physics.ease((g - 0.15) / 0.5)) * visible
        self.content.become(self.template.copy().scale(r / 1.27).shift(centre)).set_opacity(
            detail_alpha
        ).set_z_index(12)
        return self


class ComponentGuide(VGroup):
    """Annotate one of the two superposed modes without changing its trajectory."""

    def __init__(self, name, k):
        """Create annotation arrows for isotope name and mode index k.

        k=0 identifies stretching and k=1 bending. These objects annotate
        displacement without changing model coordinates or phase."""
        super().__init__()
        self.name = name
        self.k = k
        self.colour = physics.MODE_COLORS[k]
        self.arrows = VGroup(
            *[
                VGroup(
                    *[
                        Line(LEFT, RIGHT, stroke_color=self.colour, stroke_width=2.6)
                        for _ in range(3)
                    ]
                )
                for _ in range(4)
            ]
        )
        self.label = VGroup(
            Text(
                "Stretching" if k == 0 else "Bending",
                font=TEXT_FONT,
                font_size=29,
                color=self.colour,
            ),
            Text(
                "Bond lengths change" if k == 0 else "Bond angles change",
                font=TEXT_FONT,
                font_size=21,
                color=physics.GRAY,
            ),
        ).arrange(DOWN, buff=0.1)
        self.label.move_to([-4 if name == "CH4" else 4, 3.05 if k == 0 else -1.85, 0])
        self.add(self.arrows, self.label)
        self.set_z_index(9)

    def draw(self, v):
        """Update this object in place from model seconds v.

        All positions and opacities are evaluated analytically at the supplied
        time; no independent per-object phase or accumulated numerical drift is used.
        Return self for Manim updater chaining."""
        alpha = physics.guide_weights(v)[self.k]
        self.label.set_opacity(alpha)
        local, points = physics.selected_positions(self.name, v, 1)
        disp = (
            0.15
            * np.sin(
                np.pi
                * physics.PAIRS[self.k]["cases"][self.name]["frequency_cm1"]
                / physics.FREF
                * float(physics.phase_clock(v))
            )
            * physics.PURE[self.k, self.name]
        )
        tip_local = local + 2.2 * disp
        tip_xy = physics.FOCUS_SCALE * physics.projected_vectors(
            tip_local, physics.selected_orientation(v)
        ) + [-4 if self.name == "CH4" else 4, 0.65]
        for atom, arrow in enumerate(self.arrows, 1):
            base = points[atom]
            end = np.r_[tip_xy[atom], 0]
            delta = end - base
            length = np.linalg.norm(delta)
            if length < 0.16:
                arrow.set_opacity(0)
                continue
            unit = delta / length
            normal = np.array([-unit[1], unit[0], 0])
            arrow[0].put_start_and_end_on(base + 0.12 * unit, end)
            arrow[1].put_start_and_end_on(end, end - 0.09 * unit + 0.045 * normal)
            arrow[2].put_start_and_end_on(end, end - 0.09 * unit - 0.045 * normal)
            arrow.set_stroke(color=self.colour, width=2.6, opacity=alpha)
        return self


def mode_comparison_caption(k, y):
    """Build a mode title and the accepted two-line CH3D/CH4 comparison.

    The full sentence explicitly identifies which isotopologue is slower.
    Percentages come from the selected calculated frequency ratios."""
    return VGroup(
        Text(
            "Stretching" if k == 0 else "Bending",
            font=TEXT_FONT,
            font_size=30,
            color=physics.MODE_COLORS[k],
        ).move_to([0, y + 0.02, 0]),
        Text(
            f"CH₃D is {physics.PAIRS[k]['shift_percent']:.2f}% slower",
            font=TEXT_FONT,
            font_size=24,
            color=physics.GRAY,
        ).move_to([0, y - 0.45, 0]),
        Text("than CH₄", font=TEXT_FONT, font_size=20, color=physics.GRAY).move_to(
            [0, y - 0.79, 0]
        ),
    )


class MethaneComparison(Scene):
    """Render the complete approved 55.58-second comparison at 60 fps.

    A single ValueTracker drives all molecules, waveforms, legends and timer.
    The numerical model is independent of these Manim objects; metadata records
    its state at every encoded frame for independent verification."""

    def construct(self):
        """Assemble the full scene, animate the common clock and export frame metadata.

        Closures capture their row/isotope parameters explicitly where needed,
        preventing late-binding mistakes in per-frame Manim updaters."""
        if config.frame_rate != physics.FPS:
            raise ValueError("Use --fps 60 to preserve the frame-grid timeline")
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        timer = ValueTracker(0)
        definitions = physics.field_definitions()
        for name in ["CH4", "CH3D"]:
            for d in definitions:
                m = FieldMolecule(name, d)
                m.add_updater(lambda obj: obj.draw(timer.get_value()))
                m.draw(0)
                self.add(m)
            lens = LensContext(name)
            lens.add_updater(lambda obj: obj.draw(timer.get_value()))
            lens.draw(0)
            self.add(lens)
            for branch in [1, -1]:
                m = SelectedMolecule(name, branch)
                m.add_updater(lambda obj: obj.draw(timer.get_value()))
                m.draw(0)
                self.add(m)
            marker = Selection(name)
            marker.add_updater(lambda obj: obj.draw(timer.get_value()))
            marker.draw(0)
            self.add(marker)
            nuclear = NuclearMagnifier(name)
            nuclear.add_updater(lambda obj: obj.draw(timer.get_value()))
            nuclear.draw(0)
            self.add(nuclear)
            for k in [0, 1]:
                guide = ComponentGuide(name, k)
                guide.add_updater(lambda obj: obj.draw(timer.get_value()))
                guide.draw(0)
                self.add(guide)
            header = (
                Text(
                    "CH₄" if name == "CH4" else "CH₃D",
                    font=TEXT_FONT,
                    font_size=40,
                    color=physics.WHITE if name == "CH4" else physics.GOLD,
                )
                .move_to([-4 if name == "CH4" else 4, 4, 0])
                .set_z_index(20)
            )
            header.add_updater(
                lambda obj: obj.set_opacity(physics.scene_visibility(timer.get_value()))
            )
            self.add(header)
        divider = Line([0, -3.65, 0], [0, 3.65, 0], stroke_color=physics.GRAY, stroke_width=1.4)
        divider.add_updater(
            lambda m: m.set_stroke(opacity=0.55 * (1 - physics.focus(timer.get_value())))
        )
        self.add(divider)
        intro = (
            VGroup(
                Text(physics.MODE_INTRO_TEXT[0], font=TEXT_FONT, font_size=31, color=physics.WHITE),
                Text(physics.MODE_INTRO_TEXT[1], font=TEXT_FONT, font_size=27, color=physics.GRAY),
            )
            .arrange(DOWN, buff=0.16)
            .move_to([0, 2.95, 0])
            .set_z_index(16)
        )
        intro.add_updater(lambda m: m.set_opacity(physics.mode_intro_opacity(timer.get_value())))
        intro.set_opacity(0)
        self.add(intro)
        grid = VGroup(
            Line([-7.4, 0.65, 0], [7.4, 0.65, 0], stroke_color=physics.FAINT, stroke_width=1)
        )
        grid.add_updater(
            lambda m: m.set_opacity(
                physics.separation(timer.get_value()) * physics.scene_visibility(timer.get_value())
            )
        )
        self.add(grid)
        # Both mode rows share captions and a single isotope legend.
        rows = VGroup()
        for k, y in [(0, 2.05), (1, -0.75)]:
            rows.add(mode_comparison_caption(k, y))
        rows.add_updater(
            lambda m: m.set_opacity(
                physics.separation(timer.get_value()) * physics.scene_visibility(timer.get_value())
            )
        )
        self.add(rows)
        plot_bg = (
            Rectangle(width=16, height=2.25, stroke_width=0, fill_color=physics.BG)
            .move_to([0, -3.375, 0])
            .set_z_index(10)
        )

        def background(m):
            """Resize and move the plot backdrop during final graph enlargement."""
            v = timer.get_value()
            u = physics.graph_focus(v)
            m.stretch_to_fit_height(2.25 + (9 - 2.25) * u).move_to(
                [0, -3.375 * (1 - u), 0]
            ).set_fill(opacity=physics.plots_visible(v))

        plot_bg.add_updater(background)
        self.add(plot_bg)
        for k, y in enumerate(physics.PLOT_Y):
            line = Line(
                [-6.9, y, 0], [6.9, y, 0], stroke_color=physics.FAINT, stroke_width=1
            ).set_z_index(11)

            def baseline(m, k=k):
                """Update one plot baseline from the shared graphical geometry."""
                v = timer.get_value()
                sx, cy, amp = physics.plot_geometry(v, k)
                m.put_start_and_end_on([-6.9 * sx, cy, 0], [6.9 * sx, cy, 0]).set_stroke(
                    opacity=physics.plots_visible(v)
                )

            line.add_updater(baseline)
            self.add(line)
            mode_label = Text(
                "Stretching" if k == 0 else "Bending",
                font=TEXT_FONT,
                font_size=22,
                color=physics.MODE_COLORS[k],
            ).set_z_index(12)
            h = mode_label.height

            def label_position(m, k=k, h=h):
                """Position the mode name above its own plot row as that row enlarges."""
                v = timer.get_value()
                u = physics.graph_focus(v)
                sx, cy, amp = physics.plot_geometry(v, k)
                m.scale_to_fit_height(h * (1 + 0.25 * u)).move_to(
                    [-6.9 * sx, cy + 0.4 + 0.45 * u, 0], aligned_edge=LEFT
                ).set_opacity(physics.plots_visible(v))

            mode_label.add_updater(label_position)
            label_position(mode_label)
            self.add(mode_label)
            for name, color, x in [("CH4", physics.WHITE, 4.0), ("CH3D", physics.GOLD, 6.0)]:
                r = physics.PAIRS[k]["cases"][name]["frequency_cm1"] / physics.FREF
                curve = VMobject(
                    stroke_color=color, stroke_width=2.8 if k == 0 else 2.6
                ).set_z_index(11)
                dot = Dot(radius=0.032, color=color).set_z_index(12)

                def trace(m, k=k, r=r, dot=dot):
                    """Draw a normal-coordinate waveform and its current-phase endpoint.

                    The last eight q units form a uniform physical-time window. Every curve
                    point uses the same clock and frequency as the associated atomic motion."""
                    v = min(timer.get_value(), physics.END_TIME)
                    q = float(physics.phase_clock(v))
                    qvalues = np.linspace(q - 8, q, 700)
                    sx, cy, amp = physics.plot_geometry(v, k)
                    u = physics.graph_focus(v)
                    points = np.column_stack(
                        (
                            np.linspace(-6.9 * sx, 6.9 * sx, 700),
                            cy + amp * np.sin(np.pi * r * qvalues),
                            np.zeros(700),
                        )
                    )
                    m.set_points_as_corners(points).set_stroke(
                        opacity=physics.plots_visible(v),
                        width=(2.8 if k == 0 else 2.6) * (1 + 0.4 * u),
                    )
                    dot.scale_to_fit_width(0.064 * (1 + 0.4 * u)).move_to(
                        [6.9 * sx, cy + amp * np.sin(np.pi * r * q), 0]
                    ).set_opacity(physics.plots_visible(v))

                curve.add_updater(trace)
                trace(curve)
                self.add(curve, dot)
        entries = VGroup()
        for text, color in [("CH₄", physics.WHITE), ("CH₃D", physics.GOLD)]:
            sample = Line(LEFT * 0.225, RIGHT * 0.225, stroke_color=color, stroke_width=2.8)
            label = Text(text, font=TEXT_FONT, font_size=22, color=color)
            entries.add(VGroup(sample, label).arrange(RIGHT, buff=0.12))
        entries.arrange(RIGHT, buff=0.7).move_to(ORIGIN)
        plate = RoundedRectangle(
            corner_radius=0.05,
            width=entries.width + 0.28,
            height=entries.height + 0.16,
            stroke_width=0,
            fill_color=physics.BG,
            fill_opacity=1,
        )
        legend_template = VGroup(plate, entries)
        legend = legend_template.copy().set_z_index(13)

        def colour_key(m):
            """Keep exactly one shared isotope legend visible above both plot rows."""
            v = timer.get_value()
            u = physics.graph_focus(v)
            m.become(
                legend_template.copy()
                .scale(1 + 0.2 * u)
                .move_to([5.55 * (1 - u), -2.1 * (1 - u) + 2.85 * u, 0])
            ).set_opacity(physics.plots_visible(v)).set_z_index(13)

        legend.add_updater(colour_key)
        colour_key(legend)
        self.add(legend)
        explanation = VGroup(
            *[
                Text(
                    line, font=TEXT_FONT, font_size=physics.FINAL_TEXT_SIZE, color=physics.GRAY
                ).move_to([0, y, 0])
                for line, y in zip(physics.FINAL_TEXT, physics.FINAL_TEXT_Y)
            ]
        ).set_z_index(16)
        explanation.add_updater(
            lambda m: m.set_opacity(physics.final_text_opacity(timer.get_value()))
        )
        explanation.set_opacity(0)
        self.add(explanation)
        # Prebuild digit banks once; frame updates select glyph opacity only.
        numeric = VGroup()
        banks = []
        y = -4.08
        numeric.add(Text("t =", font=MONO_FONT, font_size=28, color=physics.GRAY))
        for x in [-0.48, -0.26, -0.04, 0.4, 0.62]:
            bank = VGroup(
                *[
                    Text(str(n), font=MONO_FONT, font_size=28, color=physics.WHITE)
                    .move_to([x, y, 0])
                    .set_opacity(0)
                    for n in range(10)
                ]
            )
            numeric.add(bank)
            banks.append(bank)
        numeric.add(
            Text(".", font=MONO_FONT, font_size=28, color=physics.WHITE).move_to(
                [0.18, y - 0.105, 0]
            ),
            Text("fs", font=MONO_FONT, font_size=28, color=physics.GRAY).move_to([1.05, y, 0]),
        )

        def tick(m):
            """Update prebuilt fixed-width glyph banks with elapsed femtoseconds.

            No text object is regenerated every frame. The integer has no leading
            zero, and the full timer disappears when graph enlargement begins."""
            v = min(timer.get_value(), physics.END_TIME)
            alpha = physics.timer_opacity(v)
            integer, fraction = f"{float(physics.time_fs(v)):.2f}".split(".")
            chars = integer.rjust(3) + fraction
            m[0].move_to([[-0.48, -0.26, -0.04][3 - len(integer)] - 0.61, y, 0]).set_opacity(alpha)
            for bank, char in zip(banks, chars):
                for n, glyph in enumerate(bank):
                    glyph.set_opacity(alpha if char != " " and n == int(char) else 0)
            for obj in m[-2:]:
                obj.set_opacity(alpha)

        numeric.set_z_index(15).add_updater(tick)
        tick(numeric)
        self.add(numeric)
        limit = getattr(self, "preview_end", physics.VIDEO_END_TIME)
        model_limit = physics.comparison_model_time(limit)
        self.play(
            timer.animate.set_value(model_limit),
            run_time=limit,
            rate_func=lambda alpha: physics.comparison_model_time(alpha * limit) / model_limit,
        )
        if hasattr(self, "preview_end"):
            return
        if physics.HOLD > 0:
            self.wait(physics.HOLD)
        for m in self.mobjects:
            m.clear_updaters()
        # Metadata records video time, model time and physical phase independently.
        frames = []
        for wall_time in np.arange(round(physics.VIDEO_END_TIME * physics.FPS)) / physics.FPS:
            v = physics.comparison_model_time(wall_time)
            row = {
                "v": float(wall_time),
                "model_time": v,
                "comparison_playback_rate": physics.comparison_playback_rate(wall_time),
                "q": float(physics.phase_clock(v)),
                "rate": float(physics.phase_rate(v)) * physics.comparison_playback_rate(wall_time),
                "time_fs": float(physics.time_fs(v)),
                "focus": physics.focus(v),
                "separation": physics.separation(v),
                "background_opacity": physics.background_opacity(v),
                "ring_opacity": physics.ring_opacity(v),
                "caption_opacity": physics.text_opacity(v),
                "nucleus_growth": physics.nucleus_growth(v),
                "mode_intro_opacity": physics.mode_intro_opacity(v),
                "graph_focus": physics.graph_focus(v),
                "nonplot_opacity": physics.scene_visibility(v),
                "final_text_opacity": physics.final_text_opacity(v),
                "timer_opacity": physics.timer_opacity(v),
                "guide_opacities_stretch_bend": list(physics.guide_weights(v)),
                "cases": {},
            }
            for name in ["CH4", "CH3D"]:
                native, displayed, c, r, label = physics.lens_state(name, float(v))
                row["cases"][name] = {
                    "native_points": native.tolist(),
                    "lens_centre": c.tolist(),
                    "lens_radius": r,
                    "label_centre": label.tolist(),
                }
                source, nc, nr, ng = physics.nucleus_state(name, float(v))
                row["cases"][name]["nucleus"] = {
                    "source_atom": 1,
                    "source_scene": source.tolist(),
                    "centre_scene": nc.tolist(),
                    "radius_scene": nr,
                    "growth": ng,
                    "protons": 1,
                    "neutrons": 0 if name == "CH4" else 1,
                }
                for branch, key in [(1, "stretch"), (-1, "bend")]:
                    local, points = physics.selected_positions(name, float(v), branch)
                    row["cases"][name][key] = {
                        "local_angstrom": local.tolist(),
                        "projected_scene": points.tolist(),
                    }
            frames.append(row)
        serialized = [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}
            for d in definitions
        ]
        meta = {
            "fps": physics.FPS,
            "width": 1920,
            "height": 1080,
            "active_seconds": physics.VIDEO_END_TIME,
            "model_end_seconds": physics.END_TIME,
            "comparison_extra_seconds": physics.COMPARISON_EXTRA_SECONDS,
            "comparison_original_interval": [physics.PLOTS_START, physics.GRAPH_FOCUS_START],
            "comparison_video_interval": [
                physics.PLOTS_START,
                physics.GRAPH_FOCUS_START + physics.COMPARISON_EXTRA_SECONDS,
            ],
            "comparison_retiming": "v - 2*smootherstep((v-34)/(original_graph_zoom+2-34)); unit speed at both joins",
            "hold_seconds": physics.HOLD,
            "scene_duration_seconds": self.time,
            "source_data_sha256": hashlib.sha256(physics.DATA_PATH.read_bytes()).hexdigest(),
            "captions_full_opacity_interval": [3.8, physics.FOCUS_START - 0.4],
            "captions_full_seconds": physics.FOCUS_START - 0.4 - 3.8,
            "molecular_lens_names": ["Hydrogen (H)", "Deuterium (D)"],
            "redundant_explanation_lines_removed": True,
            "lens_interval": [physics.LENS_START, physics.FOCUS_START],
            "focus_interval": [physics.FOCUS_START, physics.FOCUS_END],
            "nuclear_inset_interval": [physics.NUCLEUS_START, physics.NUCLEUS_END],
            "nuclear_full_interval": [physics.NUCLEUS_FULL, physics.NUCLEUS_CLOSE],
            "nuclear_diagrammatic": True,
            "nuclear_frame_offset_from_carbon": physics.NUCLEUS_OFFSET.tolist(),
            "nucleon_labels": ["p", "n"],
            "nucleon_highlights": False,
            "mode_intro_text": list(physics.MODE_INTRO_TEXT),
            "mode_comparison_captions": [
                [m.original_text for m in mode_comparison_caption(k, y)]
                for k, y in [(0, 2.05), (1, -0.75)]
            ],
            "mode_intro_full_interval": [physics.SLOW_START + 0.8, physics.SLOW_END - 0.8],
            "nucleus_delay_after_molecular_lens_full_s": physics.NUCLEUS_START - physics.LENS_FULL,
            "focus_delay_after_nucleus_closed_s": physics.FOCUS_START - physics.NUCLEUS_END,
            "rendered_label_strings": sorted(
                set(
                    (
                        m.text
                        for root in self.mobjects
                        for m in root.get_family()
                        if isinstance(m, Text) and m.text != "."
                    )
                )
            ),
            "slowdown_interval": [physics.SLOW_START, physics.SLOW_END],
            "phase_rate_start": physics.INITIAL_RATE,
            "phase_rate_end": physics.FINAL_RATE,
            "guide_bend_interval": [physics.BEND_START, physics.STRETCH_START],
            "guide_stretch_interval": [physics.STRETCH_START, physics.GUIDE_END],
            "guide_changes_molecular_motion": False,
            "no_reacceleration": False,
            "original_slowdown_preserved": True,
            "timer_starts_s": physics.SLOW_START,
            "split_interval": [physics.SPLIT_START, physics.PLOTS_START],
            "plots_start_s": physics.PLOTS_START,
            "final_timer_text": None,
            "timer_hidden_from_fs": physics.GRAPH_FOCUS_FS,
            "timer_hide_s": physics.GRAPH_FOCUS_START + physics.COMPARISON_EXTRA_SECONDS,
            "physical_clock_continues_to_video_end": True,
            "last_encoded_elapsed_fs": float(
                physics.time_fs(
                    physics.comparison_model_time(physics.VIDEO_END_TIME - 1 / physics.FPS)
                )
            ),
            "plot_window_q": 8.0,
            "plot_window_fs": 8 * physics.FS_PER_Q,
            "initial_field_velocity": physics.TARGET_VELOCITY.tolist(),
            "selected_axis": physics.TARGET_AXIS.tolist(),
            "graph_focus_interval": [
                physics.GRAPH_FOCUS_START + physics.COMPARISON_EXTRA_SECONDS,
                physics.GRAPH_FOCUS_END + physics.COMPARISON_EXTRA_SECONDS,
            ],
            "graph_focus_start_fs": physics.GRAPH_FOCUS_FS,
            "nonplot_fade_end_s": physics.GRAPH_FOCUS_START
            + physics.COMPARISON_EXTRA_SECONDS
            + 0.9,
            "final_text": list(physics.FINAL_TEXT),
            "final_text_font_size": physics.FINAL_TEXT_SIZE,
            "final_text_y": list(physics.FINAL_TEXT_Y),
            "removed_final_headline": True,
            "nuclear_full_seconds": physics.NUCLEUS_CLOSE - physics.NUCLEUS_FULL,
            "nuclear_hold_reduction_seconds": 0.0,
            "final_reading_reduction_seconds": 0.0,
            "reading_seconds_requested": physics.READING_SECONDS,
            "final_text_full_interval": [
                physics.GRAPH_FOCUS_END + physics.COMPARISON_EXTRA_SECONDS + 0.4,
                physics.VIDEO_END_TIME + physics.HOLD,
            ],
            "shared_legend_count": 1,
            "shared_legend_labels": [
                m.original_text for m in legend.get_family() if isinstance(m, Text)
            ],
            "final_plot_amplitude_scale": 3.0,
            "final_plot_row_centres": [1.05, -1.05],
            "selected_omega": physics.TARGET_OMEGA,
            "field_definitions": serialized,
            "frames": frames,
            "new_quantum_jobs": int(os.environ.get("METHANE_NEW_QUANTUM_JOBS", "0")),
        }
        METADATA_PATH.write_text(json.dumps(meta), encoding="utf-8")


class ComparisonStill(MethaneComparison):
    """Render a quick last-frame preview of the split molecular comparison."""

    preview_end = 37.0


class FinalFrame(MethaneComparison):
    """Render the final frame without paying for the full encoded movie."""

    preview_end = physics.VIDEO_END_TIME
