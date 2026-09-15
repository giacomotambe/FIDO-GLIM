# Module 2 — Bounding-box pipeline

*Supervised — consumes an external detector/tracker's boxes.*

`DynamicBBoxRejection` is deliberately simple: it holds a list of externally-supplied `BoundingBox`es, ages them, and removes any point that falls inside one.

```
External detector (ROS 2 topic)
   │  oriented 3D bounding boxes + optional velocity
   ▼
Age filter  →  OBB containment test  →  velocity-inflated ellipsoid test
   │
   ├──▶ Static frame  →  GLIM
   └──▶ Dynamic frame →  diagnostics
```

`reject()` ages every stored box each call (`bbox_ages_[i] < max_bbox_frames_`), then tests every point against both `bbox.contains(point)` and `bbox.contains_inflated(point, params)` — a hit against either drops the point into the dynamic bucket and breaks out of the box loop. `insert_bounding_boxes()` applies a flat `inflate_margin` to every incoming box before storing it.

## Velocity-inflated footprint

When a box carries a velocity, `contains_inflated()` doesn't grow a static margin — it shifts a 2D ellipse footprint along the XY velocity direction so the ellipse extends further *ahead* of the object than behind it, covering the volume it's expected to sweep before the next detection arrives. `VelocityInflationParams` is shared verbatim between [Module 1](module1_voxel.md) and Module 2 to avoid duplicating the same physics twice.

| Struct field | Config key | Shipped value |
|---|---|---|
| `v_fwd_k` | `velocity_inflation_k` | 0.85 |
| `v_rear_k` | `rear_inflation_k`, then overridden by `reverse_velocity_inflation_k` | 0.35 (effective) |
| `v_lat_k` | `lateral_inflation_k` | 0.30 |
| `v_vert_k` | `vertical_inflation_k` | <span class="unused">0.20 — legacy, unused by the 2D footprint test</span> |
| `v_min` | `velocity_inflation_min` | 0.1 m/s activation floor |
| `v_max_speed` | `velocity_inflation_max_speed` | 3.0 m/s saturation |
| `ellipse_box_cover_scale` | `ellipse_box_cover_scale` | √2 — sized so the ellipse still covers the inflated box's corners |

!!! warning "Config field ordering matters here"
    `VelocityInflationParams::from_config()` reads `rear_inflation_k` into `v_rear_k` first, then immediately re-reads `reverse_velocity_inflation_k` into the *same field* — the second read wins. `config_bbox_rejection.json` ships both keys (0.15 and 0.35 respectively); the effective rear multiplier is **0.35**, not 0.15.

!!! note "Config keys present but not read"
    `config_bbox_rejection.json` also ships `bbox_potentially_dynamic_min_track_age`, `bbox_static_min_track_age`, `bbox_min_static_velocity` and `bbox_rejection_obstacle_min_removed_points`. Only `inflate_margin`, `max_bbox_frames`, and the `VelocityInflationParams` fields above are actually consumed by `dynamic_bounding_box_rejection.cpp` / `bounding_box.cpp` — the other four are reserved for track-age gating logic that isn't wired up in this module yet.

## Parameters

See [`config_bbox_rejection.json`](parameters.md#config_bbox_rejectionjson) in the configuration reference for the full parameter table, including the shipped defaults above.
