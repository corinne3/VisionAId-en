# VisionAId — Phase 2 plan: from the phone to the path, in real time

> English translation of the team's [`PLAN_PHASE2.md`](../../PLAN_PHASE2.md): the plan given to the coding assistant for the phone part ("you" = the assistant). The result is in [`REPORT_PHASE2.md`](REPORT_PHASE2.md). Session names and voice commands are kept as in the code (`gourde` = water bottle, `clés` = keys, `couloir` = corridor, `piece` = room).

> **Execution mode: autonomous, with a single human stop (recordings on the phone).**
> This plan extends [`PLAN_VISIONAID.md`](PLAN_VISIONAID.md) (phases 0 to 13, finished). The autonomy rules (§1) and code conventions (§2) of that plan remain valid as they are, unless stated otherwise below.
> Phase numbering continues from 14, so that `phase N: …` commit messages stay unique.

---

## 0. Context

### Goal of this step
Replace the replayed video with **the live stream of a phone worn on the chest**, and run on the computer all the processing **up to the path**:
- acquisition;
- pose;
- detection;
- depth;
- 3D fusion;
- planning.

**Out of scope**:
- the guidance format, HRTF audio and voice (next step);
- the memory, face and support options (post-its at the bottom of the diagram).

### Chosen block diagram
```
Phone (camera + gyro + accel) ─► phone → computer link ─┬─► Object detection (fixed prompt) ─┐
                                                         ├─► Depth ────────────────────────────┼─► 3D fusion ─► Path
                                                         └─► Pose (VO + gyro + gravity) ───────┘
```
Real time is a cross-cutting constraint: latency measured at each stage, always in "latest value only" mode.

### Decisions already taken with the team
- **Reference phone**: iPhone 12, without LiDAR. The code also targets Android (Chrome), without guaranteeing it for lack of a test device.
- **Acquisition through a web page** served by the computer over HTTPS and opened in the phone's browser. No native app: it stays in reserve if the pose is too unstable, in which case we would switch to ARKit, which works without LiDAR.
- **Pose**: only from the video and the basic sensors. The gyroscope gives rotation, the accelerometer gravity, and visual odometry (VO) gives the displacement.
- **Phone attached to the chest, in portrait.** We consider that **head = torso**, with a fixed camera → body transformation.
- **Fixed prompt** chosen at launch, for example "où est ma gourde" (where is my bottle) or "où sont mes clés" (where are my keys).
- **Two loops eventually**: a slow loop on the computer up to the path, and a fast loop on the phone for audio. This step only builds the slow loop. The messages must however already allow the second one: the target and the path are expressed in world coordinates.

### Transport choice (to validate on review)
**Main transport: secure WebSocket, JPEG images and sensors timestamped on the same phone clock.** WebRTC stays optional (phase 21).

Why WebSocket rather than WebRTC:
- **Exact timestamping.** Each image is grabbed by the page with its `performance.now()` timestamp, the same clock as the sensor events. Image ↔ IMU synchronisation is therefore direct. With WebRTC, it would have to be rebuilt from the RTP timestamps.
- **No jitter buffer, no ICE/mDNS negotiation.** ICE negotiation and mDNS addresses are a classic source of failures between Safari and `aiortc` on a local network.
- **Simple back-pressure.** The phone only sends a new image when the previous one has been acknowledged, so no delay can build up.
- **Acceptable cost.** At 640×480, a JPEG image at quality 0.7 weighs about 40 KB, i.e. about 0.6 MB/s at 15 fps, which passes easily on local Wi-Fi.

---

## 1. Rules specific to this step (in addition to §1 of the first plan)

1. **Single human stop (§3, "Checkpoint").** As long as real recordings are missing, the corresponding `data` tests are `skip` with a clear message, and you move on with synthetic data and the fake phone. You never block. In `PROGRESS.md`, write `⏸ waiting for data` for the part concerned.
2. **The fake phone is the reference for automation.** A Python client (`tools/fake_phone.py`) speaks exactly the same protocol as the web page. All end-to-end tests go through it. The web page itself is validated by hand on the iPhone, through the checkpoint checklist.
3. **Network.** As before (pip/uv, Hugging Face, Ultralytics weights). **No external service** for the certificate, the QR code or a relay: everything is generated locally.
4. **Privacy.** Recorded sessions (images of the apartment) go into `data/sessions/`, ignored by git. They are never pushed.
5. **Do not break what exists.** The current 130 tests stay green at each phase. `run_offline` on `appart.mp4` must keep working.

---

## 2. Added formats and conventions

### Frames
To add to `docs/frames.md`:
- **Device frame** (W3C DeviceMotion specification): x to the right, y to the top of the screen, z out of the screen, towards the user.
- **Rear camera in portrait**: the rotation `R_cam_imu` is a priori `diag(1, −1, −1)`. It is **checked by calibration** (phase 17), because axis and unit conventions vary from one browser to another (deg/s or rad/s).
- **Body frame** = head frame (head = torso assumption): x forward horizontal, y left, z up. `T_cam_body` is fixed and given by the config. The old `T_cam_head` becomes an alias of `T_cam_body`.

### Messages
New messages in `bus/messages.py`, serialisable like the others:
- `ImuSample`:
  - `t`: phone time, in s;
  - `gyro`: rad/s, device frame;
  - `accel`: m/s², gravity included;
  - `accel_lin`: m/s², optional;
  - `orientation`: quaternion, optional.
- `ImuBatch`: list of `ImuSample`, sent every ~50 ms.
- `Frame` gains the fields `t_phone` (capture timestamp on the phone) and `t_recv` (reception timestamp on the computer). Both are optional, to stay compatible with the video.
- `ClockSync`: estimated offset between the phone clock and the computer clock, with its uncertainty.

### WebSocket protocol
Endpoint `wss://<pc>:8443/ws`.
- **Binary messages**: a msgpack header with a fixed-length prefix, followed by the JPEG. The header contains `{type: "frame", seq, t_phone, w, h}`.
- **JSON text messages**:
  - `{type: "imu", samples: [...]}`;
  - `{type: "hello", ua, screen, camera: {w, h, facing}}`;
  - `{type: "ping", t_phone}` and the reply `{type: "pong", t_phone, t_pc}`;
  - `{type: "ack", seq}`, sent by the computer to the phone;
  - `{type: "stats", ...}`, sent by the computer to the phone for display.
- **Protocol version** in `hello`. The server cleanly refuses an unknown version.

### Recorded session format
Folder `data/sessions/<YYYYMMDD-HHMMSS>_<name>/`:
- `frames/`: the received JPEGs, as they are, without recompression;
- `frames.jsonl`: one line per image (`seq`, `t_phone`, `t_recv`, `w`, `h`, `file`);
- `imu.jsonl`: one line per sample;
- `meta.json`: device (user agent), protocol version, clock synchronisation, declared camera height, target, notes;
- `video.mp4`: optional, generated afterwards for viewing (ffmpeg).

> Note from the fork: the final implementation uses a JSON header instead of msgpack, and the fake phone lives in `src/visionaid/phone/fake_phone.py`; see [`protocol.md`](protocol.md) and [`DECISIONS.md`](DECISIONS.md).

---

## 3. Phases

### Phase 14 — Phase 2 foundation
**Deliverables**
- Dependencies:
  - `aiohttp`, for HTTPS, WebSocket and static files;
  - `cryptography`, for the local certificate authority and certificate;
  - `qrcode`, to show the URL in the terminal.
- New messages (§2) and their serialisation.
- `docs/frames.md`: frames added (§2).
- `docs/protocol.md`: WebSocket protocol and session format.
- New config sections:
  - `phone`, with the port, the requested resolution, the JPEG quality, the max frames/s and the IMU rate;
  - `body`, with `T_cam_body` and the declared camera height `camera_height_m` (default 1.30 m);
  - `session`, with the folder and the recording option.

**Tests**
- `unit`: serialisation round trip of the new messages.
- `unit`: a `Frame` without `t_phone` stays valid, for compatibility.

---

### Phase 15 — Acquisition server and web page
**Deliverables**
- `phone/certs.py`: generates a local certificate authority and a server certificate covering `localhost`, the local IP and `<hostname>.local`, in `.certs/` (ignored by git).
  - The server also exposes `http://<pc>:8080/ca.crt`, over plain HTTP, to install the profile on the iPhone.
  - Installation procedure in `docs/phone_setup.md`: Settings → General → VPN & Device Management, then About → Certificate Trust Settings.
- `phone/server.py` (aiohttp):
  - serves the web page and the WebSocket;
  - publishes on the bus the `Frame`s (decoded JPEG, resized to `video.width`) and the `ImuBatch`es;
  - answers `ping`s and acknowledges each image;
  - records the session if `--record` is passed;
  - accepts only one phone at a time and cleanly handles disconnection and reconnection.
- `phone/web/index.html` + `app.js`, in plain JavaScript, without framework or CDN:
  - **Start**: a big "Start" button triggers, in the same gesture, `DeviceMotionEvent.requestPermission()` (iOS) and `getUserMedia` on the rear camera. The Wake Lock keeps the screen from going to sleep.
  - **Sending images**: the page grabs images at N fps from the `<video>` through a canvas, as JPEG, and only sends an image after the previous one has been acknowledged.
  - **Sending sensors**: `devicemotion` events are grouped every 50 ms. The page sends a `ping` every second.
  - **Display**: frames/s sent, IMU rate, round-trip latency, connection state, and the **level** (phone tilt) to help place it correctly on the chest.
  - **"Calibration" mode**: sends full-resolution images, on request ("Capture" button).
  - **Compatibility**: detection of available features, with fallback if the permission does not exist (Android) and clear error messages (HTTPS missing, permission refused, camera not found).
- `apps/run_phone.py`:
  - starts the server;
  - shows the URL and a QR code in the terminal;
  - option `--record <name>`.
- `tools/fake_phone.py`: Python client that follows the protocol and replays a video (or synthetic images) with a consistent synthetic IMU. It accepts a configurable clock offset, jitter and losses.

**Tests**
- `unit`: the certificates are generated and valid for the expected names; a TLS client trusting the local authority connects.
- `integration`: fake phone → server for 10 s.
  - The images received on the bus are in order and none is duplicated.
  - The received rate is ≥ 90% of the sent rate.
  - With a slow consumer, the image age (`t_recv` − sending) stays < 2 periods: no build-up.
- `integration`: disconnection then reconnection of the fake phone; the server resumes without restart.
- `integration`: complete and consistent recorded session (number of images, IMU covering the whole duration, valid `meta.json`).
- `unit`: an unknown protocol version is refused with a message.

---

### Phase 16 — Session replay
**Deliverables**
- `io/session.py`:
  - `SessionSource` (FrameSource) replays a recorded session **at real-time pace**, with frame skipping like `ReplaySource`;
  - `SessionImu` provides the IMU samples between two images.
- `run_offline` and the real-time graph accept `--session <folder>` instead of the video.
- `apps/session_info.py`: summary of a session (duration, real frames/s, gaps, IMU rate, clock offset) and generation of `video.mp4` for viewing.

**Tests**
- `unit`: a synthetic session recorded by the fake phone replays identically (same images, same timestamps).
- `unit`: at each image, the returned IMU window exactly covers the interval between two images.
- `data` (skip without a real session): each session in `data/sessions/` reads from start to end.

---

### ⏸ Human checkpoint — Recordings with the iPhone 12
At this point, you write `docs/checkpoint_enregistrements.md`, the procedure for the team ([English translation](checkpoint_recordings.md)), then **you continue** with phases 17 to 20 on synthetic data. The `data` tests stay `skip` until the data arrives.

**Procedure requested from the team**
1. **Setup**: install the certificate, open the URL, check the page checklist (frames/s, IMU, latency shown). If the Wi-Fi isolates devices, connect the computer to the iPhone's Personal Hotspot.
2. **Camera calibration**:
   - show the checkerboard on the computer screen (`/calib/board`) and measure the size of one square with a ruler;
   - in calibration mode, capture 15 to 20 views from varied angles.
3. **IMU calibration**: phone on the chest,
   - 5 s still;
   - then 3 slow rotations on the spot (left then right);
   - then 3 fast rotations;
   - then 5 s still.
4. **Navigation sessions**, phone on the chest, declaring the camera height measured from the floor:
   - `gourde_piece`: start 3-4 m from the bottle, look for it by turning, then walk to it;
   - `cles_piece`: same with the keys placed on a piece of furniture;
   - `couloir`: walk through a passage with obstacles;
   - `rotations_rapides`: turn fast on the spot, to test pose robustness.
5. **Reference photos** of the keys (5 to 10, varied backgrounds) in `data/ref/cles_*.jpg`.
6. **Feedback** to note in the session (`meta.json` → `notes`): heating, dropouts, discomfort of the holder.

**Web page checklist to validate by hand**
- [ ] HTTPS accepted without warning after installing the certificate.
- [ ] Sensor and camera permission obtained in a single gesture.
- [ ] The screen does not go to sleep for 5 min.
- [ ] At least 15 frames/s received and at least 50 Hz of IMU, for 5 min, without latency drift.
- [ ] Automatic reconnection after a Wi-Fi dropout.

---

### Phase 17 — Camera and sensor calibration
**Deliverables**
- `apps/calibrate_camera.py`: OpenCV checkerboard calibration from a calibration-mode session.
  - Rejection of blurred or badly detected views.
  - Output: `config/calib/<device>_<w>x<h>.yaml`, with K and the distortion.
  - Consistency check against the expected field of view of the iPhone 12, recorded.
- `calib/imu_cam.py`: automatically estimates three quantities from a session.
  - **Gyroscope bias**: on the still segments.
  - **Image ↔ IMU time offset**: it covers the camera capture latency. It is estimated by cross-correlation between the norm of the gyroscope angular velocity and the visual rotation speed (homography or VO between successive images).
  - **Rotation `R_cam_imu`**: Kabsch alignment of gyroscope ↔ visual angular velocities. The result is compared with the prior value, which also detects a unit error (deg/s instead of rad/s).
- Results are written to `config/calib/<device>_imu.yaml` and loaded automatically from the session's user agent, with fallback to default values.

**Tests**
- `unit`: synthetic checkerboard rendered with a known K → K recovered within 2%.
- `unit`: synthetic IMU with a known bias, a 40 ms offset and a known rotation → bias recovered within 10%, offset within 5 ms, rotation within 2°.
- `unit`: gyroscope in deg/s injected → unit error detected and reported.
- `data` (skip without session): calibration on the real sessions, with a reprojection error below 0.5 px, recorded.

---

### Phase 18 — Fused pose (VO + gyroscope + gravity)
**Deliverables**
- `pose/imu.py`:
  - gyroscope integration between two images, corrected for bias and offset;
  - gravity estimation through low-pass filtering of the accelerometer, robust to walking.
- `pose/vio.py` (`VisualInertialOdometry`), wrapping the existing VO:
  - **rotation prior**: the integrated gyroscope rotation is used as the initial estimate for PnP and for predicting matches, which narrows the search and reduces losses in fast rotation;
  - **gravity**: the world z axis is fixed by the accelerometer from the first second. The floor estimated by RANSAC (phase 6) then serves as a check and no longer dictates the vertical axis;
  - **fallback when VO is lost**: orientation continues with the gyroscope alone, the position is frozen and the quality becomes `degraded`. The `lost` state is only declared after N seconds;
  - **scale**: the declared camera height (`body.camera_height_m`) is the reference. The depth scale factor is re-estimated continuously from the distance to the floor, which replaces the fixed 0.76 factor calibrated on the old video. It stays bounded and smoothed.
- `PoseSource` selectable by config: `vo` (current), `vio` (new, default for phone sessions), `arkit` (stub kept).

**Tests**
- `unit`: synthetic sequences from phase 5 enriched with a simulated IMU (noise, bias, offset):
  - vertical error < 2°;
  - in fast rotation (150°/s), the VIO `lost` frame rate ≤ half that of VO alone;
  - translation accuracy at least equal to that of VO alone.
- `unit`: VO artificially cut for 1 s → orientation stays within 3° of the truth, quality goes to `degraded` then back to `ok`.
- `unit`: depth scale biased by 30% → the correction factor converges within 5% of the truth thanks to the camera height.
- `data` (skip without session): camera height estimated within 5 cm of the declared value; `lost` rate on `rotations_rapides`, VIO vs VO, recorded.

---

### Phase 19 — Fixed prompt and multiple targets
**Deliverables**
- `config/targets/<name>.yaml`: one card per target object.
  - Trigger phrases ("où est ma gourde", "où sont mes clés").
  - Text prompts for YOLOE (in English).
  - Reference photos, optional.
  - Real height, optional, for the scale check.
  - Optional colour model. The current colour score becomes specific to each card; keys have none, or a metallic one, depending on measurements.
- `perception/target.py`:
  - resolution of the fixed prompt to a card, through a normalised lookup table (lower case, accents, articles), without a language model;
  - error message listing the known targets if nothing matches.
- Re-identification becomes generic.
  - **With reference photos**: DINOv2 gallery and threshold calibrated per card (`config/reid_threshold_<name>.yaml`, same procedure as in phase 3).
  - **Without photos**: detection from text alone, with a stricter confidence threshold, and the limit is recorded.
- Option `--target "où sont mes clés"` on `run_offline`, `run_phone` and the real-time graph.

**Tests**
- `unit`: phrasing variants ("Où est ma gourde ?", "ma gourde", "gourde") lead to the right card, and an unknown phrase returns a clear error.
- `unit`: cards without colour and without photos load and the pipeline runs with stubs.
- `data`, `slow`: the bottle stays at ≥ 95% recall and 0% false positives on the phase 3 data (non-regression).
- `data`, `slow` (skip without key photos): recall and false positives for the keys on `cles_piece`, **informative**, recorded without a blocking threshold, because it is a difficult object.

---

### Phase 20 — Real time from the phone to the path
**Deliverables**
- `apps/run_live.py`: multi-process graph.
  ```
  phone_server ─► perception ─┐
               ─► geometry (depth + VIO, receives Frame + ImuBatch) ─► world ─► planner
  ```
  - The `guidance` and audio `sink` nodes are removed from this graph. A `path_sink` node logs and measures.
- **Live visualisation**: Rerun in viewer mode, started by the app, with the annotated image, the map, the trajectory, the path, the target and the latencies. A `.rrd` recording is made in parallel.
- **Measurements**:
  - data age at each stage, from `t_phone` converted to computer time;
  - end-to-end latency `capture → path published` (p50/p95);
  - effective rates and skipped frame rate;
  - written to `outputs/live_report.json`.
- The same graph accepts `--session` (real-time replay) and `--fake-phone`, for tests and tuning without a phone.
- **Degraded behaviours**:
  - phone disconnected → the nodes stay alive, the path is marked invalid, then everything restarts on reconnection;
  - pose lost → invalid path until recovery.

**Tests**
- `integration`: fake phone → path published in less than 60 s after start (model warm-up included), on a synthetic scene with stubs.
- `integration`, `slow`: fake phone replaying `appart.mp4` with a synthetic IMU for 30 s.
  - No crash, clean stop in less than 2 s.
  - At least 2 Hz of valid paths once the bottle is confirmed.
  - Latencies recorded.
- `integration`: disconnection and reconnection of the fake phone along the way → recovery without restart.
- `data`, `slow` (skip without session): replay of `gourde_piece` at 1×.
  - The bottle is confirmed.
  - The path is valid at least once.
  - **Capture → path latency target p95 < 700 ms**, recorded. It is wider than in phase 11 because the network is added. If the target is missed, record the faulty stages without blocking.

---

### Phase 21 — Optimisations (only if 14 to 20 are ✅ or ⚠️)
In order of priority:
1. **Depth** at reduced resolution (364 px) and on one image out of N, the IMU pose covering the intermediate images. Latency gain and effect on map accuracy recorded.
2. **Incremental costmap**: only the cells touched by the new rays are updated.
3. **WebRTC transport** as an option (`aiortc`), only if the JPEG WebSocket turns out to be limiting. Phase 15 tests are replayed on this transport.

---

### Phase 22 — Report
- `scripts/night_run.sh` extended:
  - fast and slow tests;
  - `run_live --fake-phone` for 30 s;
  - replay of each available real session.
- `REPORT_PHASE2.md`, in French, on the model of `REPORT.md`:
  - summary;
  - phase table;
  - metrics: received rates, latency per stage, calibration results, VIO vs VO, key detection;
  - Rerun captures;
  - checkpoint status (what was validated on the iPhone, what is waiting for data);
  - next steps towards guidance and audio on the phone.

---

## 4. Success criteria (in order of importance)
1. **Reliable acquisition**: the iPhone 12 sends the computer at least 15 frames/s and 50 Hz of IMU for 5 min, without delay build-up (image age p95 < 200 ms on local Wi-Fi).
2. **Replayable sessions**: a recorded session replays deterministically and serves as a test.
3. **Robust pose**: vertical within 2° thanks to gravity, fewer losses in fast rotation than with VO alone, scale held by the camera height.
4. **Live path**: a path to the target is computed and visible live in Rerun, updated at at least 2 Hz, with a measured latency (target p95 < 700 ms).
5. **Fixed prompt**: the bottle works as before; the keys are measured and documented.
6. **Tracking**: everything is tested, committed (`phase N: …`), documented (`PROGRESS.md`, `DECISIONS.md`, `docs/`).
