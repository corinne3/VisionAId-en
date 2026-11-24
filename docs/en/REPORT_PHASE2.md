# VisionAId — Phase 2 report: from the phone to the path (21 November 2026)

> English translation of the team's [`REPORT_PHASE2.md`](../../REPORT_PHASE2.md). The French original is the reference. File names, session names and voice commands are kept as in the code (`gourde` = water bottle, `clés` = keys, `couloir` = corridor, `piece` = room).

Plan executed: `PLAN_PHASE2.md`, phases 14 to 22. Details are in `PROGRESS.md`, the choices in `DECISIONS.md` ([English translation](DECISIONS.md)).

## Summary

1. **Acquisition from the phone, ready to test on the iPhone 12.** The computer serves a web page over HTTPS, with a certificate from a local authority to be installed once. The page sends the rear camera as JPEG and the gyroscope and accelerometer to the computer over WebSocket, timestamped on the same clock. It has **not yet run on a real phone**: that is the checkpoint.
2. **Back-pressure without delay build-up.** The computer acknowledges each frame after taking it in, and the phone only sends the next one at that moment. With a computer 2.5 times too slow, frame age stays below 2 periods (test).
3. **Synchronised clock** through paired ping/pong: offset recovered within ±20 ms in test.
4. **Sessions recorded and replayable** identically: offline, in real time and in tests, like a video.
5. **Automatic calibration**, validated on synthetic data:
   - camera with a checkerboard shown on the computer screen: K recovered within 2%;
   - gyroscope ↔ camera: bias within 10%, time offset within 5 ms, orientation within 2°;
   - a deg/s ↔ rad/s unit error is detected and corrected.
6. **Fused pose (VIO).**
   - The gyroscope predicts rotation, the accelerometer fixes the vertical (error below 2° with a simulated walk).
   - If the video drops out, orientation continues on the gyroscope for 2 s (`degraded` state). In fast rotation with simulated blur: **0% losses**, against 55% for VO alone.
   - The depth scale is re-anchored on the declared camera height.
7. **Fixed prompt.** "Où est ma gourde" (where is my bottle) or "où sont mes clés" (where are my keys) selects a target card: detection prompts, reference photos, colour model, threshold. Without photos, keys are recognised from text alone. The bottle is unchanged: 8 slow non-regression tests green.
8. **Real-time graph from the phone to the path.** With the fake phone replaying the video and the real models:
   - valid paths at **4.1 Hz**;
   - capture → path latency **p50 336 ms, p95 415 ms**, target 700 ms;
   - path invalidated during a dropout, then automatic recovery.
9. **Optimisation tested and rejected.** Depth at 364 px reduces p95 latency to 275 ms, but clearly degrades guidance (7° → 46°). 518 px is kept.
10. **Tests**: `night_run.sh` green. 177 fast tests pass; among the 30 slow or `data` tests, 24 pass and 6 are `skip` (phone data missing). Offline, real time and fake phone OK. Tests on real phone data are `skip` until recordings exist. No test removed, no threshold weakened.

## What awaits the team (checkpoint)

Everything is described step by step in [`docs/checkpoint_enregistrements.md`](../checkpoint_enregistrements.md) (French), with the setup in [`docs/phone_setup.md`](phone_setup.md). About 30 minutes:

1. install the certificate on the iPhone and go through the web page checklist;
2. record the sessions `calib_camera` (checkerboard), `calib_imu` (rotations), `gourde_piece`, `cles_piece`, `couloir`, `rotations_rapides`, with the measured camera height;
3. copy 5 to 10 photos of the keys into `data/ref/cles_*.jpg`.

Then these commands run all the rest:
```bash
uv run python -m visionaid.apps.calibrate_phone --camera data/sessions/<…>_calib_camera --imu data/sessions/<…>_calib_imu
uv run pytest -m "data" -q --timeout=1800
make night
```
For live use:
```bash
uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 1.32
```
This command opens Rerun, which shows live the annotated image, the map, the trajectory, the path and the target.

## Phase table

| Phase | Status | Tests | Note |
|---|---|---|---|
| 14 Foundation | ✅ | 6 | IMU/clock messages, documented protocol |
| 15 Server and web page | ✅, page to validate on iPhone | 8 + 5 | HTTPS with local authority, WebSocket, fake phone |
| 16 Session replay | ✅ | 7 + 2 + 1 data | video or session everywhere (offline, real time, caches) |
| ⏸ Checkpoint | pending | — | procedure written |
| 17 Calibration | ✅ synthetic, ⏸ real | 9 + 2 data | checkerboard, gyroscope ↔ camera, units |
| 18 VIO | ✅ synthetic, ⏸ real | 10 + 2 data | gyroscope, gravity, `degraded`, scale |
| 19 Fixed prompt | ✅ bottle, ⏸ keys | 19 + 1 + 1 data | target cards, generic ReID |
| 20 Real time up to the path | ✅ fake phone, ⏸ iPhone | 1 + 2 slow | 4.1 Hz, p95 415 ms |
| 21 Optimisations | ⚠️ partial | — | 364 px measured, not kept |
| 22 Report | ✅ | — | `night_run.sh` extended |

## Key metrics

| Area | Measure | Value | Source |
|---|---|---|---|
| Transport | frame age p95 (local, 15 fps) | < 133 ms (2 periods) | fake phone test |
| Transport | consumer 2.5 times too slow | frames always fresh, no queue | test |
| Clock | recovered offset | ± 20 ms (synthetic: ± half RTT) | test |
| Camera calibration | K, synthetic checkerboard | < 2%, reprojection < 0.5 px | test |
| IMU calibration | offset / bias / rotation | < 5 ms / < 10% / < 2° | test |
| VIO | vertical (simulated walk) | < 2° | test |
| VIO | fast rotation with blur | 0% `lost` (VO alone: 55%) | test |
| VIO | 1 s video dropout | orientation < 3°, back to `ok` in < 4 frames | test |
| Scale | depth biased by 30% | corrected to < 5% | test |
| Real time, stubs | first valid path | 5.6 s after connection | test |
| Real time, video + real models | valid paths | 4.1 Hz | fake phone, 30 s |
| Real time, video + real models | capture → path p50 / p95 | 336 / 415 ms (target 700) | fake phone, 30 s |
| Real time, video + real models | geometry (depth + VIO) | ~165 ms per frame, 5.4 Hz | fake phone, 30 s |
| Depth 364 px | capture → path p95 | 275 ms, but guidance 46° (vs 7°) | not kept |
| Offline (non-regression) | final state / distance | ARRIVED at 0.18 m | original video |

## Important decisions (details in `DECISIONS.md`)

- **WebSocket + JPEG rather than WebRTC**: exact timestamping on the sensor clock, no fragile network negotiation, simple back-pressure. WebRTC would bring nothing measurable on a local network.
- **Acknowledgement after ingestion**: the phone never goes faster than the computer.
- **VIO by composition, without a Kalman filter.** Position stays visual: double integration of a phone accelerometer drifts in less than a second. The gyroscope is used for prediction, checking and fallback. The accelerometer gives the vertical.
- **Floor constrained by gravity** when it is known, and **scale from camera height**: replaces the fixed 0.76 factor calibrated on a single video.
- **Target cards = config overrides.** Without `--target`, nothing changes: it is the bottle, as before.
- **Planner with a watchdog**: never a stale path after a dropout.

## Known limits

- **Nothing has run on a real phone yet.** Several points will only be checked at the checkpoint:
  - Safari permissions and keeping the screen on;
  - the real signs and units of the gyroscope (calibration checks them);
  - the real IMU rate and the Wi-Fi latency.
- **Gravity and scale have only been validated on synthetic data.** The test on the real video has no real gravity: the accelerometer is zero there for lack of data.
- **Keys**: no real figures. In text-only mode, two objects of the same category are not distinguished.
- **`degraded` pose**: position is frozen. If the user walks during a visual loss of more than one second, the position drifts until relocalisation.
- **Android**: code compatible (permissions, certificate), but untested for lack of a device.
- **Camera distortion**: estimated but not corrected in the pipeline (pinhole model).

## Next steps

1. **Checkpoint**: recordings with the iPhone, calibration, `data` tests. This is the prerequisite for everything else.
2. **Guidance** (next step of the project): the computer already publishes the target and the path in world coordinates. They still need to be sent back to the phone (return channel on the same WebSocket) and the sound rendered on the phone with its own orientation, with the Web Audio API and its HRTF `PannerNode`: this is the fast loop. The web page is the right starting point.
3. If the real pose is too unstable: go through ARKit in a small native app. Without LiDAR, the ARKit pose is still better than our VIO.
4. Options from the diagram: memory, face, support.

## How to resume

```bash
make test-fast                       # 176 fast tests, ~2 min
make run-phone                       # acquisition only (record: --record <name>)
make run-live                        # phone → path, live Rerun
make run-live-fake                   # same with the fake phone (apartment video)
uv run python -m visionaid.apps.session_info --all
make night                           # everything, logged in outputs/night_run.log
```

> Note from the fork: after this report, the team continued with phases 23 to 29 (sound guidance on the phone, known-cell paths, voice commands and announcements, demo mode). See [`DECISIONS.md`](DECISIONS.md) and [`technical_overview.md`](technical_overview.md).
