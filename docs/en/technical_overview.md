# VisionAId — Technical overview

*Corinne Bergès, November 2026. Complete French version with all the maths: [dossier_technique_fr.md](../dossier_technique_fr.md).*
*Original code: [SonDePoisson/VisionAid](https://github.com/SonDePoisson/VisionAid) (version 01af1e1, 22/11/2026).*

## 1. Pipeline

| Block | Tool | Why this choice | Main limit |
|---|---|---|---|
| Phone | Web page (getUserMedia, DeviceMotion, Wake Lock, Web Audio, speechSynthesis) | nothing to install; iPhone and Android | HTTPS mandatory → local certificate authority |
| Transport | aiohttp WebSocket, JPEG + JSON headers, acknowledgement | image and IMU stamped on the **same phone clock**; back-pressure → no accumulated delay | public Wi-Fi often isolates devices |
| Detection | YOLOE-26s-seg (open vocabulary) | new object = new text prompt, no retraining | small or metallic objects are hard |
| Re-identification | DINOv2-small embeddings × HSV colour score | "my" bottle, not any bottle; 98 % recall, 0 false positives on 607 negatives | needs reference photos |
| Depth | Depth Anything V2 Metric Indoor Small | metric depth from one image, no LiDAR needed | noisy beyond 2-3 m, scale bias |
| Pose | RGB-D visual odometry (ORB + AP3P/RANSAC) + gyroscope rotation prior + accelerometer gravity | fast to validate; robust to fast rotations (0 % lost vs 55 % vision only) | position drifts without loop closure |
| 3D fusion | gravity-aligned floor plane, scale from camera height, log-odds occupancy grid (5 cm), object anchor | Bayesian update as a simple addition; map that forgets slowly | cluttered map with monocular depth |
| Planning | A* / D* Lite (Numba), body margin 0.35 m, goal within arm's reach, known free cells only | incremental replanning in ~2 ms | — |
| Guidance | prioritised state machine with hysteresis; direction smoothed in the world frame | stable cues, validated by a simulator (100 runs per scenario) | — |
| Sound | constant-power stereo pan, layered sounds (path, obstacles, arrival, search) by Romain Benoit | clear left/right, same rendering live and offline | front/back coded by the shortest turn |
| Voice | Whisper (speech → text), rule-based target matching, phone text-to-speech at 4 moments | fast, predictable, no LLM | CPU-only PCs: transcription blocks the connection |
| Real time | ZeroMQ publish/subscribe, one process per node, "latest value only" (conflation) | slow nodes skip frames instead of lagging | — |

## 2. Key maths (short)

- **Pinhole camera**: `u = fx·X/Z + cx`; back-projection `X = (u − cx)·Z/fx`; `fx = (W/2)/tan(HFOV/2)`.
- **Re-identification score**: `score = s^0.6 · c^0.4`, `s` = cosine similarity of DINOv2 embeddings, `c` = colour share. A product (not a sum) so that a wrong colour cancels the score. Threshold chosen by maximising F1.
- **PnP + RANSAC**: minimise the reprojection error `Σ ρ(‖u_i − π(K(R·X_i + t))‖²)` (ρ = Huber); RANSAC needs `N = log(1−p)/log(1−w³)` draws (≈ 35 for p = 0.99, w = 0.5).
- **Gyroscope integration**: `R_{k+1} = R_k · exp([ω·Δt]×)` (Rodrigues formula).
- **Gravity**: complementary filter `û ← normalise((1−k)·û_pred + k·â)`, k = 0.05; the accelerometer is not used for position (a 0.05 m/s² bias gives 2.5 m of error after 10 s).
- **Camera-IMU calibration**: time offset by cross-correlation of rotation-rate norms; rotation by Kabsch (`H = Σ a_i b_iᵀ = UΣVᵀ`, `R = V·diag(1,1,det(VUᵀ))·Uᵀ`). Finding: Safari iOS permutes the gyroscope axes and flips the accelerometer sign.
- **Scale**: `s = h_declared / h_measured`, smoothed in log space, bounded to [0.5, 2].
- **Occupancy grid**: log-odds `L = log(p/(1−p))`; hit `+0.85`, pass-through `−0.4`; half-life 20 s; obstacles marked up to 3 m only.
- **Planning**: A* with `f = g + h` (admissible octile heuristic); D* Lite keeps `g` and `rhs` and only repairs inconsistent cells.
- **Direction smoothing**: first-order low-pass in the world frame (τ = 0.3 s, ≤ 90°/s), then `az = θ_world − body_yaw`, so path changes glide but body turns are immediate.
- **Stereo pan**: `pan = −az/90°`, `θ = (pan+1)·π/4`, gains `cos θ` and `sin θ` → `L² + R² = 1` (constant perceived loudness).
- **Clock sync**: `offset = t_pc − (t_phone + RTT/2)`, using the exchange with the smallest round trip.

## 3. Design decisions

| Decision | Rejected alternative |
|---|---|
| Web page served by the PC | native app / Flutter (ARKit kept as a fallback) |
| WebSocket + JPEG + acknowledgement | WebRTC |
| Compute on the PC | everything on the phone |
| Open-vocabulary detector + re-identification | closed-set YOLO (no "water bottle" class of *mine*) |
| Composed visual-inertial odometry | full EKF / MSCKF |
| Paths through known free cells only | unknown cells with a penalty |
| Stereo pan | generic HRTF (front/back confusions without head tracking) |
| No LLM in the loop | LLM agent |
| Open source, free, local | paid cloud APIs |

## 4. Measured performance

| | Mac M4 + iPhone | Windows CPU + Android |
|---|---|---|
| Valid paths | 4.1 Hz | 0.5 Hz |
| Capture → path latency p50 / p95 | 336 / 415 ms | 2.2 / 2.9 s |
| Image age p50 | ~40 ms | 35-46 ms |
| Left/right sound | ✅ | ✅ |

## 5. Limits and safety

- Not a safety device: the cane or guide dog remains essential.
- Sound follows the chest, not the head.
- On CPU-only machines, use stop-and-go.
- Tested with sighted people only so far.
