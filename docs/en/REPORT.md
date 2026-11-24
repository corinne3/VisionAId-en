# VisionAId — Report of the night of 21 November 2026 (phase 1)

> English translation of the team's [`REPORT.md`](../../REPORT.md). Written by the coding assistant for the team ("you" = the team, "I" = the assistant). The phone part came next: see [`REPORT_PHASE2.md`](REPORT_PHASE2.md).

## Summary (10 lines)

1. **The pipeline runs end to end on your video** (`make run-offline`, 485 frames in 69 s on the M4) and produces `annotated.mp4` (+ a version with sound), `session.rrd`, `guidance.wav` and `report.json`.
2. **The grey and orange bottle is found and told apart**: YOLOE-26s-seg + DINOv2-small + multiplicative colour score. 98% recall on augmentations, 0 false positives on 607 negatives from the video (black thermos, beer bottles, orange bag…). A bottle tinted blue is rejected.
3. In the video, the bottle is confirmed on 60% of frames (from frame 104); no confirmed track has an incompatible colour.
4. **Depth**: Depth Anything 3 not cleanly available, fallback to DAv2-Metric-Indoor-Small. A scale factor of 0.76, estimated from the bottle height, gives a camera at 1.39 m and the bottle at 0.87 m above the floor (on the table).
5. **Pose**: RGB-D odometry (ORB + PnP AP3P + symmetric refinement): 0% frames lost; round trip over 10 s: error of 1.8% of the distance travelled.
6. **Simulator** (100 seeds per scenario, real modules): success 95 / 94 / 96%, collision-free 96 / 96 / 99% (room / corridor / sudden obstacle). The plan's criteria are met.
7. On the video, guidance ends in `ARRIVED` 0.23 m from the bottle; the guidance azimuth differs from the direction actually taken by 7° (median).
8. **Real time**: graph of 7 ZeroMQ processes stable for 30 s, clean stop in 0.65 s, instructions at 20 Hz. End-to-end latency p50 382 ms / p95 480 ms: **the 300 ms target is not met** (bottlenecks identified below).
9. Tests: **107 fast + 23 slow, all green**, no `xfail`. 14 commits (one per phase).
10. Points of attention: the monocular costmap is too cluttered to be used as a simulator map, the obstacle alert is very present on the video, and an **expired Hugging Face token** is configured on your machine (see "To do on your side").

## To do on your side

- **Expired "hackathon" Hugging Face token**: it made even public downloads fail. I worked around it in the code (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` in `visionaid/__init__.py`), but it must be renewed or removed (`hf auth logout`).
- The files in `assets/` (MOV + HEIC) are not committed (28 MB); I converted them into `data/` (ignored by git).
- Installing the `clip` package from GitHub was refused by the environment policy: a small shim (`perception/clip_shim.py`) provides the same tokenizer through `transformers`.

## Phase table

| Phase | Status | Tests (fast + slow) | Note |
|---|---|---|---|
| 0 Foundation | ✅ | 6 | open3d pinned to 0.19 |
| 1 Sources & intrinsics | ✅ | 12 | rotation checked against ffmpeg |
| 2 Real-time bus | ✅ | 14 | ZeroMQ conflate, sequential mode |
| 3 Detection & ReID | ✅ | 11 + 8 | track jump bug found and fixed |
| 4 Depth | ✅ | 6 + 3 | DAv2 (DA3 unavailable) |
| 5 Pose (VO) | ✅ | 7 + 3 | "per pair" criterion on keyframes |
| 6 3D fusion | ✅ | 11 + 3 | scale factor 0.76 |
| 7 Planning | ✅ | 9 | A*/D* Lite in numba |
| 8 Guidance | ✅ | 14 | |
| 9 Binaural audio | ✅ | 8 | parametric model |
| 10 Simulator | ✅ | 8 + 1 (100 seeds) | real map not navigable |
| 11 Integration | ✅ / latency ⚠️ | 1 + 5 | p95 480 ms > 300 ms |
| 12 Optimisations | ⚠️ partial | — | CoreML tested, not kept |
| 13 Report | ✅ | — | `scripts/night_run.sh` green |

No test removed, no `xfail`. Modified or reinterpreted thresholds are justified in [`DECISIONS.md`](DECISIONS.md): costmap height bands, forgetting, "per pair" VO criterion, body radius for the real map.

## Key metrics

| Area | Measure | Value |
|---|---|---|
| ReID | calibrated threshold (max F1) | 0.575 |
| ReID | recall on 2 × 30 augmentations | 98.3% |
| ReID | false positives on video negatives | 0% (607 crops, max 0.53) |
| Perception | bottle confirmed on the video | 60% of frames, 0 colour incompatibility |
| Throughput (M4, MPS) | YOLOE-26s-seg alone | ~54 fps |
| | detection + ReID + tracking | ~24 fps (39 ms) |
| | DAv2-Metric-Indoor-Small (518 px) | ~7.4 fps (125-135 ms) |
| | visual odometry | ~38 ms/frame |
| Depth | bottle height (after scaling) | 0.251 m for 0.25 m real |
| VO | `lost` frames | 0% |
| VO | camera height | 1.39 m (σ 8 cm) |
| VO | 10 s round trip | 1.8% of the distance |
| Map | 3D position of the bottle | σ 7.5 cm, 0.87 m above the floor |
| Simulator | success / collision-free | room 95 / 96%, corridor 94 / 96%, sudden 96 / 99% |
| Simulator | replanning (obstacle actually appeared) | 95.5% |
| Simulator | travelled / optimal length (median) | 0.93-0.96 (see note) |
| Offline | final state / distance | ARRIVED, 0.23 m |
| Offline | guidance consistency | 7° median, 23° p75 (71 samples) |
| Real time | end-to-end latency | p50 382 ms, p95 480 ms |
| Real time | effective rates | source 30 Hz, perception 13 Hz, geometry 5 Hz, plan 4 Hz, instructions 20 Hz |

Note on the ratio < 1: the "optimal" length is computed with the planner's inflation radius (0.35 m) up to 0.45 m from the object, whereas the simulated agent can brush past at 0.20 m and stops at 0.5 m. The ≤ 1.6 criterion is therefore met with margin.

## Captures

| Searching (4 s) | Chair alert (6 s) | Approach (12 s) | Arrival (15 s) |
|---|---|---|---|
| ![](../report/annotated_4s.jpg) | ![](../report/annotated_6s.jpg) | ![](../report/annotated_12s.jpg) | ![](../report/annotated_15s.jpg) |

Final costmap (white free, black occupied, pink inflated zone, blue trajectory, orange bottle): ![](../report/costmap_final.png)

Simulator, corridor scenario (ground truth on the left, built map on the right): ![](../report/sim_corridor_2.jpg)

Full videos: `outputs/annotated_audio.mp4` (to listen to with headphones), `outputs/sim/*.mp4`, `outputs/session.rrd` (opens with `rerun outputs/session.rrd`).

## Important decisions (details in `DECISIONS.md`)

- **Multiplicative ReID score** (`emb^0.6 · colour^0.4`): with a weighted sum, a blue bottle passed, because DINOv2 is not very sensitive to hue.
- **Tracker appearance gating**: the confirmed track jumped onto the cereal box next to it when the bottle was not detected.
- **Scale factor 0.76** estimated on the bottle, confirmed by the camera height and the table height.
- **Costmap**: floor noisy at ±7 cm, hence obstacle band [0.10; 1.90] m with a 4 cm/m tolerance; real-time forgetting (half-life of 20 s).
- **numba** for A*/D* Lite (2.9 ms / 1.9 ms incremental on 200×200) and for per-sample audio rendering.
- **Guidance, fixes from the simulator**: look-ahead point in line of sight, alert limited to the body corridor, end of approach along the path. The simulator went from ~45% to ~95% success.

## Known limits

- **Real-time latency** > 300 ms. Bottlenecks: geometry node 170-190 ms (depth competing for the GPU with YOLOE, plus VO), mapping ~90 ms (back-projection, rays, distance transform). Leads: DAv2 at 364 px, incremental costmap, cost computed only by the planner, or better, ARKit/LiDAR depth.
- **Cluttered monocular costmap** (73% of known cells occupied): usable for local guidance, not as a simulation map (start and target disconnected).
- **Frequent obstacle alert on the video** (7 s out of 16): partly legitimate (chair brushed, table walked along). To be validated with a user; no `NAVIGATING` state appears on this short video.
- YOLOE visual prompt, SOFA HRTF, Depth Anything 3 and pose graph optimisation: not done.
- A single 16 s video and two photos: ReID calibration would deserve more negatives (other rooms, other bottles).

## Next steps

1. **Plug in ARKit (pose) + WebRTC (images)**: the `PoseSource` / `FrameSource` interfaces are ready (`ArkitPoseSource`, `WebRTCSource` are documented stubs). The ARKit pose replaces the VO, and LiDAR depth (if available) replaces DAv2, which removes the main bottleneck and the scale factor.
2. **Audio rendering on the phone** from the target in world coordinates: the server sends `ObjectEstimate` + path (a few bytes) and the phone spatialises locally with its own pose at 60 Hz. The audio loop latency no longer depends on the network.
3. **Head tracking** (AirPods / IMU): `T_cam_head` already exists in the config; it only needs to be fed dynamically.
4. TSDF or voxel fusion for a less noisy map, then replay the "real map" scenario.
5. User tests: tuning alert thresholds and sounds.

## How to resume

```bash
make test-fast      # 107 tests, ~1 min
make test-all       # + 23 slow tests, ~9 min
make run-offline    # outputs in outputs/
make run-realtime   # multi-process graph, 30 s
make night          # everything, logged in outputs/night_run.log
```
