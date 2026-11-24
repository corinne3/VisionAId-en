# VisionAId — Implementation plan for Claude Code (autonomous execution)

> English translation of the team's [`PLAN_VISIONAID.md`](../../PLAN_VISIONAID.md): the plan the team gave to the coding assistant for the first night of the hackathon ("you" = the assistant, "I" = the team). The phone part is in [`PLAN_PHASE2.md`](PLAN_PHASE2.md). `gourde` = water bottle.

> **Execution mode: autonomous, end to end, without intermediate validation.**
> You implement all the phases in order, without stopping to ask for my opinion.
> Each phase ends with green automatic tests (or tests marked `xfail` with a justification), a commit, then you move on to the next one.
> I read `PROGRESS.md`, `DECISIONS.md` and `REPORT.md` in the morning.

---

## 0. Context

### Product goal
Guide a visually impaired person to an object detected in real time in a room, from a monocular video stream. Guidance is by sound and binaural. It avoids obstacles and recomputes the path continuously.

### What we build tonight
The whole pipeline on a Mac M4 (Apple Silicon, PyTorch MPS), fed by **a recorded video** read as a real-time stream (*replay*). Streaming from the phone (WebRTC + ARKit pose) will come later. The architecture must allow it without a rewrite, through the `FrameSource` and `PoseSource` interfaces.

### Input data (provided by me)
- `data/video/appart.mp4`: video of my apartment filmed with the phone (monocular, no LiDAR, no IMU in the file).
- `data/ref/gourde_*.jpg`: one or more photos of the target object, **a grey and orange water bottle**.
- The target object appears in the video.

If a file is missing: the tests depending on it are `skip` with a clear message, and you continue with the synthetic tests. You never block.

### Expected outputs in the morning
- `outputs/annotated.mp4`: annotated video (detection, anchored object, costmap inset, path, guidance arrow, state).
- `outputs/session.rrd`: full Rerun recording (images, depth, point cloud, trajectory, costmap, path, target).
- `outputs/guidance.wav`: binaural audio track synchronised with the video.
- `outputs/report.json`: numerical metrics.
- `REPORT.md`: readable summary (what works, figures, failures, next steps).

---

## 1. Autonomy rules (to follow strictly)

1. **Never wait for human input.** No interactive command, no question. In case of ambiguity, choose the simplest option that works and record it in `DECISIONS.md` (date, context, choice, rejected alternative).
2. **Follow the order of the phases.** A phase is finished when its tests pass: `pytest -m "not slow"` green, then the phase's `slow` tests.
3. **Git commit after each phase**, message `phase N: <summary>`. The `data/` folder and the model weights are in `.gitignore`.
4. **Update `PROGRESS.md`** at the end of each phase: status (✅ / ⚠️ partial / ❌), tests, duration, problems.
5. **Time-box of ~90 min per phase.** Beyond that, mark the phase ⚠️, put the blocking tests in `pytest.mark.xfail(reason=...)` with a precise reason, and move on. Later phases must stay runnable thanks to the fallbacks.
6. **Never delete a test or silently weaken a threshold.** Any threshold change is justified in `DECISIONS.md`.
7. **Dependencies: 2 installation attempts at most**, then switch to the listed fallback and record it.
8. **Mandatory timeouts**: `pytest-timeout`, 120 s by default, 1800 s for tests marked `slow`. No process may hang all night. Bus subprocesses have a timeout and are killed cleanly at the end of the test.
9. **Network**: only `pip`/`uv`, Hugging Face and Ultralytics weight downloads.
10. **Do not run big models in every unit test.** Unit tests use synthetic data or stubs. Real models go into tests marked `slow`, with their outputs cached (`.cache/`) for re-runs.

---

## 2. Code conventions

- Python 3.11, environment managed with `uv` (fallback `venv` + `pip`).
- **Never use `super()`.** Prefer composition, `Protocol` (typing) and `dataclass`. If inheritance is unavoidable (e.g. `torch.nn.Module`), call `ParentClass.__init__(self, ...)` explicitly.
- Type hints everywhere, `ruff` for lint and format, short docstrings in French.
- All configuration is in `config/default.yaml`, loaded into dataclasses. No magic numbers in the code.
- Geometric conventions, to document in `docs/frames.md`:
  - OpenCV camera frame: x right, y down, z forward;
  - world frame: origin at the first pose, **z up** after alignment on the floor (see phase 6);
  - poses as 4×4 matrices `T_world_cam` (float64);
  - units in metres, seconds and radians.
- Each message carries a `t` (timestamp in seconds, source clock) and a `frame_idx`.
- Torch device: `mps` if available, otherwise `cpu`. Detection centralised in `visionaid/utils/device.py`.

### Target tree
```
visionaid/
  pyproject.toml
  Makefile                      # setup, test-fast, test-all, run-offline, run-realtime, night
  config/default.yaml
  data/                         # gitignored: video/, ref/
  outputs/                      # gitignored
  docs/frames.md
  src/visionaid/
    utils/      device.py, config.py, timing.py, logging.py
    io/         sources.py (FrameSource, VideoFileSource, ReplaySource, WebRTCSource stub)
                intrinsics.py
    bus/        messages.py, transport.py (ZeroMQ), nodes.py, runner.py
    perception/ detector.py, reid.py, tracker.py, depth.py
    pose/       source.py (PoseSource, VisualOdometry, ArkitPoseSource stub), vo.py
    mapping/    geometry.py, floor.py, costmap.py, object_anchor.py
    planning/   astar.py, dstar_lite.py, lookahead.py
    guidance/   state_machine.py, cues.py
    audio/      binaural.py, renderer.py
    sim/        world.py, agent.py, scenarios.py
    viz/        rerun_logger.py, overlay.py
    apps/       run_offline.py, run_realtime.py, calibrate_reid.py, inspect_data.py
  tests/
    conftest.py                 # synthetic fixtures, data paths, markers
    unit/  data/  integration/
  scripts/night_run.sh
  PROGRESS.md  DECISIONS.md  REPORT.md
```

### pytest markers
- `unit`: fast, synthetic, no model or real data.
- `data`: need the video or the photos (`skip` if missing).
- `slow`: load a real model (detection, depth, re-identification).
- `integration`: full pipeline.

---

## 3. Phases

Each phase gives the goal, the deliverables, the tests and the definition of "done".

### Phase 0 — Project foundation
**Deliverables**
- `pyproject.toml` with the dependencies: numpy, opencv-python, torch, torchvision, transformers, ultralytics, open3d, pyzmq, msgpack, msgpack-numpy, rerun-sdk, soundfile, scipy, pyyaml, pytest, pytest-timeout, ruff.
- Config, logging, `device.py`, `Makefile`, `.gitignore`, `PROGRESS.md`, `DECISIONS.md`.
- `apps/inspect_data.py`: shows the metadata of the video (resolution, fps, duration, **rotation**) and of each reference photo, and saves 10 sampled frames in `outputs/inspect/`.

**Tests**
- `unit`: config loading, device selection.
- `data`: the video opens, fps > 0, at least 1 readable reference photo.

**Done when** `make test-fast` is green.

---

### Phase 1 — Image sources and intrinsics
**Deliverables**
- `FrameSource` (Protocol): `read() -> Frame | None`, `intrinsics`, `fps`.
- `VideoFileSource`:
  - handles rotation (metadata);
  - resizes to a configurable width (default 640 px);
  - provides timestamps from the video.
- `ReplaySource`: wraps a file source and delivers it **at real-time pace**. If the consumer is late, frames are skipped, never queued.
- `WebRTCSource`: stub raising `NotImplementedError`, with a docstring describing the expected contract (frames + timestamped pose).
- `intrinsics.py`: default K from a configurable horizontal field of view (default 69°, typical of a smartphone wide-angle in video). Option `estimate_from_model` if the chosen depth model exposes intrinsics. Option to load an OpenCV (checkerboard) calibration if provided.

**Tests**
- `unit`: on a synthetic video generated in the test (cv2.VideoWriter, 3 s, known pattern):
  - number of frames;
  - strictly increasing timestamps;
  - rotation correctly applied;
  - `ReplaySource` at 1× takes about 3 s (±10%);
  - with a slow consumer, `ReplaySource` skips frames instead of building up delay.
- `unit`: consistent K (cx, cy at the centre, fx > 0).
- `data`: full read of `appart.mp4` without error.

---

### Phase 2 — Real-time bus
Multi-process architecture, one node per block, **ZeroMQ pub/sub** communication serialised with msgpack (+ msgpack-numpy).

**Deliverables**
- `messages.py`: dataclasses `Frame`, `Pose`, `Depth`, `Detections`, `ObjectEstimate`, `Costmap`, `Path`, `GuidanceCue`, with (de)serialisation.
- `transport.py`:
  - `Publisher` and `Subscriber`;
  - **"latest value only"** option (`ZMQ_CONFLATE` or non-blocking drain) for heavy streams (images, depth);
  - normal queue for light streams.
- `nodes.py`: generic node (loop, target rate, latency measurement, clean stop on signal).
- `runner.py`: starts and stops a graph of nodes described in the config.
- Equivalent **single-process** mode, calling the blocks in sequence on each frame. It is used for the deterministic offline mode and for tests. Both modes share exactly the same business code.

**Tests**
- `unit`:
  - serialisation round trip for each message (with numpy arrays);
  - local pub/sub: 1,000 light messages received in order;
  - in conflate mode, a slow subscriber only receives recent messages (age < 2 periods);
  - local median latency < 5 ms for a 640×480 image;
  - runner stop < 2 s without zombie processes.

---

### Phase 3 — Detection and re-identification of the bottle
The key point: **find *this* grey and orange bottle, not just any bottle.**

**Deliverables**
- `detector.py`: open-vocabulary **YOLOE** detector (latest generation available in Ultralytics, ideally YOLOE-26, segmentation variant, small size).
  - Configurable text prompt, default `["water bottle", "bottle", "flask"]`.
  - **Visual prompt** from the reference photos, if the API allows it.
  - Outputs: boxes, masks, scores.
  - Fallbacks in order: YOLOE-11 → YOLO-World v2 → YOLO26/YOLO11 COCO class `bottle`. Record the choice.
- `reid.py`: embedding of masked crops with **DINOv2-small** (transformers). Fallback: CLIP/MobileCLIP via open_clip.
  - Gallery built from the reference photos and their augmentations.
  - Score = maximum cosine similarity with the gallery, combined with a **colour score**: HSV histogram, proportions of grey and orange in the mask.
  - Configurable weighting.
- `apps/calibrate_reid.py`: computes the decision threshold.
  - Positives: augmentations of the reference photos (rotation, scale, blur, brightness, partial crop, 30% occlusion).
  - Negatives: crops of other objects detected in the video (broad prompt `["cup","bottle","chair","bag","box"]`), excluding crops too similar to the reference.
  - Threshold chosen to maximise F1. Saved in `config/reid_threshold.yaml`.
- `tracker.py`: ByteTrack (built into Ultralytics, or a simple IoU + Kalman implementation). The "it is the bottle" decision is taken **per track**, with hysteresis: confirmed after N frames above the threshold (default 3), lost after M frames without detection (default 10).

**Tests**
- `unit`: with a stub detector and encoder, check the track logic and hysteresis (confirmation, loss, reacquisition).
- `slow`, `data`:
  - on each reference photo, the bottle is detected and re-identified (score ≥ threshold);
  - on 30 augmentations per photo, recall ≥ 90%;
  - on the negatives, false positive rate ≤ 5%;
  - **colour robustness**: on the photo with shifted hue (orange → blue), the score falls below the threshold (proof that colour matters).
- `slow`, `data` on the video:
  - at least one confirmed "bottle" track;
  - fraction of frames where the bottle is confirmed, recorded in `report.json` (informative);
  - no confirmed track on an object whose dominant hue is incompatible (colour check);
  - detection throughput measured on MPS, recorded.

---

### Phase 4 — Metric depth
**Deliverables**
- `depth.py` with interchangeable backends (config `depth.backend`):
  1. **Depth Anything 3 Metric** (`depth-anything/DA3METRIC-LARGE`) for quality;
  2. **DA3-Small** (relative) + scale re-anchoring for speed;
  3. Fallback: **Depth Anything V2 Metric Indoor Small** via `transformers.pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")`.

  Record the chosen backend and its throughput on the M4.
- Output: depth map in metres at image resolution, plus a confidence if available.
- Optional temporal smoothing (exponential average guided by the pose, disabled by default).
- Disk cache of depths per frame (`.cache/depth/<video_hash>/<idx>.npz`), so that later phases and re-runs are fast.

**Tests**
- `unit`: with a stub backend, shape, dtype, values > 0, cache path.
- `slow`, `data` on 50 sampled frames:
  - median depth in [0.3; 8] m (indoor);
  - median relative variation between consecutive frames < 15%;
  - **scale check with a known object**: the estimated 3D height of the bottle (mask + depth + K) is within ±40% of `target.height_m` (config, default 0.25 m). If the test fails, apply a global scale factor estimated on the bottle, record it, and test again.

---

### Phase 5 — 6DoF pose by visual odometry
The video has no IMU. The pose is therefore estimated by **RGB-D odometry**, using metric depth as the D channel. The interface must make it possible to plug in the ARKit pose later.

**Deliverables**
- `PoseSource` (Protocol):
  - `VisualOdometry` (implementation);
  - `ArkitPoseSource` (documented stub).
- `vo.py`, main method:
  - ORB features (OpenCV) + matching + 3D back-projection of the reference frame's points thanks to depth;
  - `solvePnPRansac` to obtain the transformation;
  - **keyframe** strategy (new keyframe if translation > 0.15 m, rotation > 10° or inliers < threshold);
  - **quality score** per pose (number of inliers, reprojection error).
- Alternative method: `open3d.pipelines.odometry.compute_rgbd_odometry`. Enabled by config, and used automatically if PnP fails on a frame.
- Tracking loss handling: pose marked `quality=lost`, reset on the next keyframe. The loss is reported downstream (guidance must know it).
- Optional, if time allows: small pose graph optimisation on the keyframes. Do not start before finishing phases 6 to 11.

**Tests**
- `unit`, synthetic:
  - random 3D point cloud, known camera trajectory (circle + translation), projection with K, 0.5 px noise → relative translation error < 3% and rotation error < 1° per pair;
  - cumulative drift over 100 frames < 5% of the distance travelled;
  - injected outliers (30%): the result stays within tolerances;
  - textureless frame: the pose is `lost`, without exception.
- `slow`, `data` on the video:
  - no jump > 0.5 m or > 30° between two consecutive non-`lost` frames;
  - fraction of `lost` frames < 10% (otherwise record and analyse);
  - **camera height above the floor** (after phase 6) between 0.8 and 1.9 m, standard deviation < 0.15 m;
  - **round-trip consistency**: on a 10 s segment read forwards then backwards, the final pose difference is < 10% of the distance travelled.

---

### Phase 6 — 3D fusion: floor, costmap, object anchoring
**Deliverables**
- `geometry.py`: depth → point cloud back-projection, frame changes, subsampling (configurable pixel step).
- `floor.py`:
  - floor plane estimation by RANSAC (Open3D or numpy) on the points of the lower half of the image;
  - definition of the world "up" axis (z) and of the camera height;
  - plane tracked and updated slowly.
- `costmap.py`: 2D top-down grid (5 cm resolution, dynamic size):
  - occupancy in **log-odds**;
  - a point is an obstacle if it is in the height band [0.05; 1.90] m above the floor, otherwise it marks the cell as free floor;
  - simplified ray casting to free the cells between the camera and the obstacle;
  - progressive **forgetting** (decay towards unknown) to handle moving obstacles;
  - **inflation** by the body radius (config, 0.35 m) and decreasing cost around obstacles, to prefer clear paths;
  - three cell states: free, occupied, unknown.
- `object_anchor.py`:
  - 3D position of the bottle = median of the 3D points under the mask (or under the shrunk box), transformed into the world frame;
  - 3D Kalman filter (or exponential average weighted by confidence);
  - **memory**: the position is kept when the object leaves the field of view, with growing uncertainty;
  - states `unknown` / `seen` / `remembered` / `stale`.

**Tests**
- `unit`, synthetic:
  - noisy flat floor → plane recovered (angle < 2°, height < 3 cm);
  - 0.5 m box on the floor → occupied cells at the right place (IoU > 0.8);
  - points on the ceiling (2.3 m) ignored;
  - inflation: distance obstacle → first free cell ≈ radius ± 1 cell;
  - decay: a removed obstacle becomes unoccupied again within N updates;
  - anchoring: object seen from 5 noisy poses → position within 5 cm; out of view for 3 s → `remembered` state, position kept.
- `slow`, `data` on the video:
  - the final costmap contains non-trivial free and occupied areas (free > 20%, occupied > 2%);
  - standard deviation of the bottle's 3D position over the frames where it is confirmed < 0.30 m;
  - export of a PNG image of the final costmap in `outputs/`.

---

### Phase 7 — Path planning
**Deliverables**
- `astar.py`: 8-connected A* on the costmap (cost = distance + proximity cost).
  - Option: unknown cells have a moderate configurable cost ("cautious optimist" approach).
- `dstar_lite.py`: incremental D* Lite, reused between cycles. It updates when cells change or the start moves.
- `lookahead.py`:
  - path smoothing (line-of-sight shortcuts);
  - **look-ahead point** at a configurable distance (default 1.0 m) on the path. Guidance points to it, not directly to the object.
- Planning goal = the free cell closest to the object (the object is often on a piece of furniture, hence in an "occupied" cell).

**Tests**
- `unit`:
  - known grids, length equal to Dijkstra's (brute-force reference) within 1e-6;
  - unreachable goal → `None`, without infinite loop;
  - D* Lite gives the same length as A* recomputed after random insertion or removal of obstacles (50 random scenarios);
  - performance: 200×200 grid, A* < 50 ms and incremental D* Lite < 10 ms on average;
  - paths stay out of inflated cells;
  - the look-ahead point is on the path, at the right distance, or is the goal if the path is shorter.

---

### Phase 8 — Guidance logic
**Deliverables**
- `state_machine.py`, states:
  - `SEARCHING`: object never seen → slow sweep instruction;
  - `NAVIGATING`: valid path → heading to the look-ahead point;
  - `APPROACHING`: less than 1.2 m → direct guidance to the object, with elevation (furniture height);
  - `ARRIVED`: less than 0.5 m;
  - `OBSTACLE_ALERT`: priority, if an obstacle is less than 0.6 m away in the ±30° frontal cone;
  - `TRACKING_LOST`: `lost` pose → stop instruction.
- Hysteresis on all transitions (no state flickering).
- `cues.py`: produces a `GuidanceCue` containing:
  - azimuth and elevation relative to the head (for now head = camera; provide a `T_cam_head` field in the config, identity by default);
  - distance, state, alert level;
  - possible short voice message, rate-limited (no more than one every 3 s).

**Tests**
- `unit`, table-driven: input sequences (distances, lost pose, obstacle) → expected state sequences.
  - No more than one state change per 0.3 s with noisy input.
  - Correct azimuth sign: target on the left → positive azimuth according to the documented convention.
  - Obstacle alert has priority over navigation.
  - Rate limiting of voice messages.

---

### Phase 9 — Binaural audio rendering
**Deliverables**
- `binaural.py`:
  - **deterministic parametric model**, without download:
    - Woodworth ITD (head radius 8.75 cm);
    - ILD through an azimuth-dependent head shadow filter;
    - front/back cue through spectral filtering (treble attenuation for the rear);
  - option: SOFA HRTF (KEMAR) if `sofar` installs and the file downloads. Otherwise, stay on the parametric model and record it.
- `renderer.py`:
  - continuous beacon spatialised on the look-ahead point;
  - beep rate according to distance (faster when getting closer);
  - pitch according to elevation in `APPROACHING`;
  - distinct, priority sound for `OBSTACLE_ALERT`;
  - silence + stop beep for `TRACKING_LOST`;
  - short jingle for `ARRIVED`;
  - azimuth interpolation between two cues (no audible jump);
  - two modes: synchronised WAV writing (offline) and optional real-time playback (`sounddevice`, disabled by default and in the tests).

**Tests**
- `unit`:
  - source at +60° (left) → left channel **ahead** (cross-correlation lag consistent with the theoretical ITD ± 0.1 ms) and **louder** (ILD > 3 dB); symmetric on the right;
  - source at 0° → almost identical channels;
  - front vs back at the same lateral azimuth → lower high-frequency energy at the back;
  - no clipping (|x| < 1);
  - no click: maximum sample jump bounded during an azimuth change;
  - WAV duration equal to the session duration ± 1 frame.

---

### Phase 10 — Closed-loop simulator
Essential to test guidance without a real user.

**Deliverables**
- `sim/world.py`: 2D top-down world (ground-truth grid). Scenario generator:
  - rectangular rooms with random furniture;
  - corridor with a door;
  - obstacle appearing along the way;
  - **real map** = costmap from the video (phase 6), if available.
- Simulated sensor: partial costmap built only with what the agent "sees" (70° cone, 4 m range, noise).
- `sim/agent.py`: virtual user following the `GuidanceCue`:
  - turns towards the azimuth with a reaction delay (300 ms) and angular noise (σ = 10°);
  - walks at 0.6 m/s, stops on alert or tracking loss;
  - in `SEARCHING`, sweeps on the spot.
- The simulator reuses **exactly** the real `mapping`, `planning` and `guidance` modules.

**Tests**
- `unit`, with 100 random seeds per scenario type:
  - success rate (arrival < 0.5 m from the object in < 120 simulated seconds) ≥ 90%;
  - **0 collisions** with the ground truth in ≥ 95% of runs;
  - travelled length / optimal path ratio ≤ 1.6 (median);
  - appearing obstacle scenario: successful replanning in ≥ 90% of cases.
- Output: GIF or MP4 of 3 representative runs in `outputs/sim/`, and statistics in `report.json`.

---

### Phase 11 — Integration
**Deliverables**
- `viz/rerun_logger.py`: logs all streams, with timeline = video timestamp. To plug in progressively from phase 3 if possible.
- `viz/overlay.py`: rendering of `annotated.mp4`:
  - box and mask of the bottle;
  - other detections in grey;
  - state and distance;
  - heading arrow;
  - mini-costmap with path and position.
- `apps/run_offline.py`:
  - deterministic single-process mode on the whole video;
  - produces `annotated.mp4`, `session.rrd`, `guidance.wav` (muxed into a copy `annotated_audio.mp4` with ffmpeg if available) and `report.json`.
- `apps/run_realtime.py`:
  - multi-process graph on the bus, `ReplaySource` at 1×;
  - measures latencies per stage (p50/p95), the rate of skipped frames and the effective frequencies;
  - multi-rate strategy: detection on every available frame, depth and pose as fast as possible, planning at 5 Hz, cues at 20 Hz.
- **Guidance consistency evaluation on the video** (informative, not blocking): during the segments where the camera gets closer to the bottle, angle between the guidance azimuth and the real direction of movement in the next second (median, recorded).

**Tests**
- `integration`, `data`, `slow`:
  - `run_offline` runs to the end without exception;
  - the 4 output files exist and are not empty;
  - the bottle is confirmed at least once;
  - the `ARRIVED` or `APPROACHING` state is reached if the video ends near the bottle (estimated final distance < 1.2 m). Otherwise, informative test.
- `integration`, `data`:
  - `run_realtime` runs for 30 s without crash and stops cleanly;
  - end-to-end p95 latency measured and recorded. Target < 300 ms: if exceeded, record the faulty stages and the optimisation leads (CoreML export of YOLO, DA3-Small, lower resolution) without blocking.
- `integration`, `unit`: the full pipeline runs on a synthetic video with stubs (no models), to check the wiring in < 30 s.

---

### Phase 12 — Optimisations (only if everything else is ✅ or ⚠️)
In order of priority:
1. CoreML export of the detector (`model.export(format="coreml")`) and switch by config. Tests: same detections within ±5% score, throughput gain recorded.
2. DA3-Small depth + scale re-anchoring via the bottle or the floor. Phase 4 tests replayed.
3. Pose graph optimisation on the keyframes.

---

### Phase 13 — Final report
- `scripts/night_run.sh`:
  - runs all the tests (`pytest -m "not slow"` then `pytest -m slow`);
  - runs `run_offline` then `run_realtime` (30 s);
  - aggregates the results;
  - everything is logged in `outputs/night_run.log`;
  - the script runs under `caffeinate -i` if available.
- `REPORT.md`, in French:
  - 10-line summary;
  - phase table (status, tests passed/failed/xfail);
  - key metrics: re-identification recall and false positives, throughput per model, % of `lost` frames, standard deviation of the bottle position, simulator success, p50/p95 latencies;
  - captures (costmap PNG, a few annotated frames);
  - important decisions (pointer to `DECISIONS.md`);
  - known limits and **next steps**, in particular:
    - plugging in ARKit (pose) + WebRTC;
    - audio rendered on the phone from the target in world coordinates;
    - head tracking.

---

## 4. Overall success criteria (in order of importance)
1. The offline pipeline runs end to end on my video and produces the outputs.
2. The grey and orange bottle is found and told apart from other objects, with green tests on my photos.
3. The simulator shows guidance that reaches the object without collision in ≥ 90% of cases.
4. The real-time mode runs in replay with measured latencies.
5. Everything is tested, committed and documented so that I can pick up in the morning.
