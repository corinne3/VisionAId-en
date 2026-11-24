# Progress

> English translation of the team's [`PROGRESS.md`](../../PROGRESS.md). `gourde` = water bottle, `clés`/`cles` = keys.

| Phase | Status | Tests | Duration | Notes |
|---|---|---|---|---|
| 0 — Foundation | ✅ | 6 passed (unit + data) | ~25 min | open3d pinned to 0.19 (libusb missing for 0.20) |
| 1 — Sources & intrinsics | ✅ | 12 new (unit + data) | ~15 min | rotation checked against ffmpeg autorotation |
| 2 — Real-time bus | ✅ | 14 new (unit) | ~20 min | HWM raised to 10,000 (1,000 messages in a burst saturated HWM=1000) |
| 3 — Detection & ReID | ✅ | 11 unit + 8 slow/data | ~1 h | YOLOE-26s-seg + DINOv2-small; threshold 0.575; augmented recall 98%, FPR 0%; target confirmed on 60% of frames; track jump bug fixed |
| 4 — Metric depth | ✅ | 6 unit + 3 slow/data | ~20 min | DAv2-Metric-Indoor-Small (DA3 unavailable); 7.4 fps; bottle height 0.33 m vs 0.25 m |
| 5 — Pose (RGB-D odometry) | ✅ | 7 unit + 3 slow/data | ~45 min | 0% lost, 35 keyframes, round trip 0.6%; "per pair" criterion on keyframes (see DECISIONS); camera height tested in phase 6 |
| 6 — 3D fusion | ✅ | 11 unit + 3 slow/data | ~50 min | scale factor 0.76; camera 1.39 m (σ 8 cm); bottle σ 7.5 cm at 0.87 m above the floor; costmap 27% free / 73% occupied (furnished room) |
| 7 — Planning | ✅ | 9 unit | ~30 min | A*/D* Lite in numba (2.9 ms / 1.9 ms incremental on 200×200) |
| 8 — Guidance logic | ✅ | 14 unit | ~20 min | priorities, 0.15 m hysteresis, 0.3 s hold, messages ≥ 3 s |
| 9 — Binaural audio | ✅ | 8 unit | ~25 min | Woodworth ITD, Brown-Duda ILD, rear cue, no clicks or clipping |
| 10 — Simulator | ✅ | 8 unit + 1 slow (100 seeds × 3) | ~2 h | success 95/94/96%, collision-free 96/96/99%, median ratio ≈ 0.94; real map not navigable (informative) |
| 11 — Integration | ✅ (latency ⚠️) | 1 unit/integration (stubs) + 5 slow | ~1 h 15 | offline 485 frames in 72 s, ARRIVED at 0.23 m, guidance consistency 7° (median); real time 30 s OK, p95 ≈ 450-650 ms > 300 ms |
| 12 — Optimisations | ⚠️ partial | — | ~15 min | CoreML export tested, not kept (score difference 5.8%, slower); DA3-Small and pose graph not done |
| 13 — Final report | ✅ | night_run: 107 fast + 23 slow green, offline and real time OK | ~20 min | REPORT.md, captures in docs/report/ |

## Project phase 2: from the phone to the path (`PLAN_PHASE2.md`)

| Phase | Status | Tests | Duration | Notes |
|---|---|---|---|---|
| 14 — Phase 2 foundation | ✅ | 6 unit | ~20 min | messages `ImuSample`, `ImuBatch`, `ClockSync`, `Frame.t_phone/t_recv`; config `phone`, `body`, `session`; `docs/protocol.md` |
| 15 — Server, web page, fake phone | ✅ (page to validate on iPhone) | 8 unit + 5 integration | ~1 h 15 | HTTPS with local authority, WebSocket JPEG + IMU, acknowledgement after ingestion, clock by paired ping/pong, recorded sessions; web page syntax-checked and served, not yet tested on a real phone |
| 16 — Session replay | ✅ | 7 unit + 2 integration + 1 data (skip) | ~40 min | `SessionSource`, `SessionImu`, `open_media` (video or session) wired into offline, the perception/depth caches and real time (`SessionReplayBlock`); `apps/session_info.py` |
| ⏸ Checkpoint | waiting for data | — | — | procedure in `docs/checkpoint_enregistrements.md`; phases 17 to 20 continue on synthetic data |
| 17 — Camera and sensor calibration | ✅ (synthetic) / ⏸ real data | 9 unit + 2 data (skip) | ~1 h | synthetic checkerboard: K within 2%; synthetic IMU: offset within 5 ms, bias within 10%, rotation within 2°, unit error detected; automatic per-device loading (`config/calib/`) |
| 18 — Fused pose (VIO) | ✅ (synthetic) / ⏸ real data | 9 unit + 2 data/slow (skip) | ~1 h 15 | vertical < 2° (simulated walk), fast rotations with blur: 0% `lost` vs 55% with VO alone, 1 s dropout: orientation < 3°, depth scale re-anchored within 5% by camera height; VIO chosen automatically when the IMU is present (offline, real time) |
| 19 — Fixed prompt and multiple targets | ✅ (bottle) / ⏸ real keys | 19 unit + 1 integration + 1 data/slow (skip) | ~50 min | cards `config/targets/{gourde,cles}.yaml`, phrase → card by key words (no language model), ReID without colour or text-only if no photo; bottle unchanged (8 slow phase 3 tests green); `--target` on run_offline, run_realtime, run_phone, calibrate_reid |
| 20 — Real time from the phone to the path | ✅ (fake phone) / ⏸ real iPhone | 1 integration + 2 slow (1 skip) | ~1 h 15 | 6-node graph (+ live or file Rerun); stubs: path in 5.6 s, dropout → invalid path then recovery; real video + real models: valid paths at 4.1 Hz, capture → path latency p50 336 ms / p95 415 ms (target 700 ms) |
| 21 — Optimisations | ⚠️ partial (measured, not kept) | — | ~25 min | depth at 364 px: p95 latency 429 → 275 ms, geometry 5.4 → 10.6 Hz, but guidance consistency 7° → 46° (median) and arrival 0.18 → 0.35 m: **518 px kept**, 364 available through `--depth-width`; incremental costmap and WebRTC not done (not limiting) |
| 22 — Report | ✅ | night_run: 176 fast + 24 slow green (6 skip: phone data), offline / real time / fake phone OK | ~30 min | `REPORT_PHASE2.md`; fix after the first real trial: certificate re-issued on the fly when the IP changes, fake phone on a free port |
| Real iPhone 12 data | ✅ | 13 data/slow tests green | ~1 h 30 | camera calibration 0.06 px (field of view 66.9°), gyroscope correlated at 0.996; fixed: units in cm, Safari iOS axes, accelerometer sign, table taken for the floor; VIO: losses 10.4% → 0.4% in fast rotations; real-time bottle replay: 4.3 paths/s, p95 457 ms; keys: confirmed on 9% of frames |
| 29 — Voice: request (STT) and announcements (TTS) | ✅ (fake phone, real models without video) / ⏸ real iPhone | 12 unit + 1 integration | one night | `--command stt` (Talk button, Whisper on the computer or Safari) or `--command prompt`; announcements: request understood, object found (where + route, path suspended), approach ("Tu arrives vers la table…" — you are getting close to the table…), correction; furniture detected in the same YOLOE pass (target detections unchanged) and mapped; live target change (ReID prepared at startup, ~10 ms); Whisper-small: 0.6 s per request |

> Note from the fork: on a Windows PC without GPU, the fork uses Whisper-base and lower frame rates; see [`setup/windows-android`](../../setup/windows-android/README.md) and [`results/windows_android_runs.md`](../../results/windows_android_runs.md).
