"""Scene + camera specifications for Blender-rendered gsplat training
data. gs_experiment.blender_render (which needs `bpy`, and so can only
run inside a Blender process) consumes these as JSON; the specs
themselves live here as plain dataclasses over numpy/CameraPose, so
they're constructible and testable without Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from gs_experiment.camera import CameraPose, translate_cameras, turntable_arc, turntable_camera, turntable_ring


@dataclass
class ObjectSpec:
    type: str  # "sphere" | "cube" | "cylinder" | "plane"
    location: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    color: Tuple[float, float, float]
    rotation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RenderSceneSpec:
    objects: List[ObjectSpec]
    cameras: List[CameraPose]
    resolution: int = 400
    fov_deg: float = 50.0
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.05)

    def to_json_dict(self) -> dict:
        return {
            "objects": [
                {
                    "type": o.type,
                    "location": list(o.location),
                    "scale": list(o.scale),
                    "color": list(o.color),
                    "rotation_euler": list(o.rotation_euler),
                }
                for o in self.objects
            ],
            "cameras": [
                {"center": c.center.tolist(), "forward": c.forward.tolist(), "up": c.up.tolist()}
                for c in self.cameras
            ],
            "resolution": self.resolution,
            "fov_deg": self.fov_deg,
            "background_color": list(self.background_color),
        }


def quick_validation_scene(n_views: int = 24, radius: float = 6.0) -> RenderSceneSpec:
    """A handful of simple colored primitives shot from one turntable
    ring -- just enough to get a first real gsplat checkpoint to validate
    gs_experiment.splat_scene.load_from_gsplat_checkpoint against. Not the
    differentiation-experiment scene (see differentiation_scene below) --
    no controlled thin-structure or narrow-viewpoint zone here, just
    "does the real training+loading pipeline work end to end."
    """
    objects = [
        ObjectSpec("sphere", (0.8, 0.0, 0.0), (0.6, 0.6, 0.6), (0.9, 0.2, 0.2)),
        ObjectSpec("cube", (-0.8, 0.5, 0.0), (0.5, 0.5, 0.5), (0.2, 0.6, 0.9)),
        ObjectSpec("cylinder", (0.0, -0.8, 0.2), (0.4, 0.4, 0.7), (0.3, 0.9, 0.3)),
        ObjectSpec("sphere", (0.3, 0.6, -0.6), (0.35, 0.35, 0.35), (0.9, 0.8, 0.2)),
    ]
    cameras = turntable_ring(radius=radius, n_views=n_views, phi_deg=35.0)
    return RenderSceneSpec(objects=objects, cameras=cameras)


def thin_rod_cluster(rng: np.random.Generator, center, n_rods: int, spread: float) -> List[ObjectSpec]:
    """A cluster of thin cylinders ("rods") scattered around `center` --
    the fine/thin geometry both differentiation_scene and
    nbv_test_scene use, factored out so both build it identically rather
    than maintaining two copies."""
    objs = []
    for _ in range(n_rods):
        offset = rng.uniform(-spread, spread, size=3)
        offset[2] = rng.uniform(-0.3, 0.3)
        loc = np.array(center) + offset
        color = rng.uniform(0.3, 0.9, size=3)
        objs.append(ObjectSpec("cylinder", tuple(loc), (0.04, 0.04, 0.5), tuple(color)))
    return objs


def differentiation_scene(
    n_ring_views: int = 40,
    n_arc_views: int = 10,
    ring_radius: float = 6.5,
    arc_radius: float = 6.5,
    arc_half_width_deg: float = 12.0,
    separation: float = 18.0,
    fov_deg: float = 40.0,
) -> RenderSceneSpec:
    """The real (non-mock) differentiation-experiment scene: ROADMAP.md's
    milestone-2 go/no-go setup, ported to an actual rendered/trained
    scene. Two zones, both built from identical thin/fine geometry (a
    cluster of thin rods) so the *only* deliberately-varied factor
    between them is camera coverage, not geometric detail -- the same
    "hold spatial density/geometry equal, vary only viewing coverage"
    control the toy directional experiment used
    (bq_splat/results/FINDINGS.md section 9) to avoid confounding the two
    effects:

    - a "wide" rod cluster, shot from a full turntable ring: well-
      observed by a visibility proxy (many, angularly diverse views), but
      the rods are thin enough that BQ's quadrature variance may still
      flag it as numerically under-resolved -- this is the load-bearing
      comparison, per ROADMAP.md's "differentiation experiment" section.
    - a "narrow" rod cluster, identical in construction, shot only from a
      narrow arc of views: under-observed directionally, which both a
      visibility proxy and BQ's directional-kernel variance should catch
      -- a sanity check that the effect direction is right, not the
      load-bearing claim.

    Each cluster gets its own dedicated camera rig, aimed at that
    cluster's own center (via camera.translate_cameras, which exploits
    forward/up being direction vectors unaffected by translation) rather
    than a single shared rig aimed at a shared origin. Combined with
    `separation` between the two cluster centers, this means real
    geometric visibility attribution (gs_experiment.visibility_attribution
    .attribute_observations) will -- from actual frustum geometry, not an
    assignment rule -- naturally find that ring cameras only see the wide
    cluster and arc cameras only see the narrow cluster, the same way
    make_occluder_scene lets real occlusion geometry (rather than fiat)
    produce the mock scene's observed_camera_idx. That also means the
    *trained* splats reflect real optimization dynamics under each
    coverage pattern, not just bookkeeping: gsplat sees each camera's
    image only contains its own cluster, exactly like photographing two
    separate stages in one large room from two disjoint camera positions.
    """
    rng = np.random.default_rng(0)

    wide_center = np.array([0.0, 0.0, 0.0])
    narrow_center = np.array([separation, 0.0, 0.0])
    objects = thin_rod_cluster(rng, wide_center, n_rods=14, spread=0.8) + thin_rod_cluster(
        rng, narrow_center, n_rods=14, spread=0.8
    )

    ring_cameras = turntable_ring(radius=ring_radius, n_views=n_ring_views, phi_deg=35.0)
    arc_cameras = turntable_arc(
        radius=arc_radius, n_views=n_arc_views, theta_center_deg=200.0, half_width_deg=arc_half_width_deg, phi_deg=35.0
    )
    arc_cameras = translate_cameras(arc_cameras, narrow_center)

    cameras = ring_cameras + arc_cameras
    return RenderSceneSpec(objects=objects, cameras=cameras, resolution=400, fov_deg=fov_deg)


def gradient_scene(
    n_zones: int = 5,
    n_views_per_zone: int = 12,
    zone_spacing: float = 18.0,
    radius: float = 6.5,
    fov_deg: float = 40.0,
    theta_center_deg: float = 200.0,
    min_half_width_deg: float = 8.0,
    max_half_width_deg: float = 180.0,
):
    """A realistic view-direction *uncertainty gradient*, not the binary
    wide-vs-narrow split `differentiation_scene`/`nbv_test_scene` use.
    Every prior directional-uncertainty result in this project (toy scale:
    bq_splat/results/FINDINGS.md section 9; real scale: gs_experiment/
    results/FINDINGS.md sections 17-19, 22) compares exactly two
    conditions. A robot scanning past a shelf, or a SLAM system that
    revisits some parts of a scene more than others, doesn't produce two
    discrete coverage buckets -- it produces a continuum. This scene is
    built to have one.

    `n_zones` identical thin-rod clusters (same construction, same rod
    count/spread, as `differentiation_scene`'s zones -- geometry and local
    spatial density are held exactly equal across zones, isolating the
    directional effect the same way `validate_directional_combined.py`
    and `differentiation_scene` already do), spaced `zone_spacing` apart
    along the x-axis. Each zone gets its own turntable-arc camera rig,
    all centered on the *same* absolute azimuth (`theta_center_deg`) --
    only the arc's *half-width* varies, linearly from `min_half_width_deg`
    (zone 0, the narrowest -- most under-covered) to `max_half_width_deg`
    (the last zone -- `180` spans the full azimuth range, equivalent to a
    complete ring). Keeping `theta_center_deg` fixed across zones (rather
    than each zone's own local convention) means a single fixed query
    direction -- the azimuth diametrically opposite `theta_center_deg` --
    is a genuinely consistent "how well is the view I haven't taken
    covered" probe across every zone: for the narrowest zone that
    direction is deep outside the tiny observed arc; for the widest
    (effectively full-ring) zone, that same direction is already covered.
    A real, monotonic angular-coverage gradient, by construction -- what
    the directional BQ variance is asked to recover, not assumed to.
    """
    rng = np.random.default_rng(2)
    zone_centers = [np.array([i * zone_spacing, 0.0, 0.0]) for i in range(n_zones)]
    half_widths = np.linspace(min_half_width_deg, max_half_width_deg, n_zones)

    objects = []
    cameras = []
    zone_camera_ranges = []
    for center, half_width in zip(zone_centers, half_widths):
        objects += thin_rod_cluster(rng, center, n_rods=14, spread=0.8)
        zone_cams = turntable_arc(
            radius=radius, n_views=n_views_per_zone, theta_center_deg=theta_center_deg,
            half_width_deg=float(half_width), phi_deg=35.0,
        )
        zone_cams = translate_cameras(zone_cams, center)
        start = len(cameras)
        cameras += zone_cams
        zone_camera_ranges.append((start, len(cameras)))

    query_theta_deg = (theta_center_deg + 180.0) % 360.0
    info = dict(
        zone_centers=np.array(zone_centers),
        half_widths_deg=half_widths,
        zone_camera_ranges=zone_camera_ranges,
        theta_center_deg=theta_center_deg,
        query_theta_deg=query_theta_deg,
    )
    spec = RenderSceneSpec(objects=objects, cameras=cameras, resolution=400, fov_deg=fov_deg)
    return spec, info


def nbv_test_scene(
    n_train_views: int = 10,
    n_candidate_views: int = 16,
    n_eval_views: int = 16,
    radius: float = 6.5,
    eval_radius: float = 7.5,
    train_theta_center_deg: float = 200.0,
    train_half_width_deg: float = 12.0,
    fov_deg: float = 40.0,
):
    """ROADMAP.md milestone 4 (active-view / NBV combination experiment)
    test scene: a single thin-rod cluster at the origin (same geometry as
    differentiation_scene's zones, via thin_rod_cluster -- deliberately
    the same construction, since this scene *is* differentiation_scene's
    narrow zone, standalone, extended with a next-view candidate pool
    and a held-out evaluation set). Three camera roles, distinguished by
    index range in the returned info dict rather than by anything in
    RenderSceneSpec itself (this renderer doesn't need to know the roles,
    only nbv_experiment.py does):

    - `train_idx`: a narrow arc of `n_train_views` cameras (identical
      construction to differentiation_scene's narrow zone) -- the
      "already observed, under-covered" starting point an NBV policy
      would begin from.
    - `candidate_idx`: a full turntable ring of up to `n_candidate_views`
      poses, at the *training* radius, excluding angles within
      `train_half_width_deg + 5` of the train arc's own center -- the
      discrete next-view pose set a policy scores and picks from.
    - `eval_idx`: a dense ring at a different radius, never a training
      candidate, used only to measure reconstruction quality after
      adding a chosen next-view -- kept disjoint from `candidate_idx` so
      "which view got picked" and "how do we measure whether it helped"
      never share a camera.
    """
    rng = np.random.default_rng(1)
    center = np.array([0.0, 0.0, 0.0])
    objects = thin_rod_cluster(rng, center, n_rods=14, spread=0.8)

    train_cameras = turntable_arc(
        radius=radius, n_views=n_train_views, theta_center_deg=train_theta_center_deg,
        half_width_deg=train_half_width_deg, phi_deg=35.0,
    )

    def angular_dist(a, b):
        return abs((a - b + 180) % 360 - 180)

    margin = train_half_width_deg + 5.0
    candidate_thetas = [
        t for t in np.linspace(0, 360, n_candidate_views, endpoint=False)
        if angular_dist(t, train_theta_center_deg) > margin
    ]
    candidate_cameras = [turntable_camera(radius, 35.0, t) for t in candidate_thetas]

    eval_cameras = turntable_ring(radius=eval_radius, n_views=n_eval_views, phi_deg=35.0)

    cameras = train_cameras + candidate_cameras + eval_cameras
    info = dict(
        train_idx=np.arange(0, len(train_cameras)),
        candidate_idx=np.arange(len(train_cameras), len(train_cameras) + len(candidate_cameras)),
        eval_idx=np.arange(len(train_cameras) + len(candidate_cameras), len(cameras)),
        candidate_thetas=np.array(candidate_thetas),
        train_theta_center_deg=train_theta_center_deg,
    )
    spec = RenderSceneSpec(objects=objects, cameras=cameras, resolution=400, fov_deg=fov_deg)
    return spec, info
