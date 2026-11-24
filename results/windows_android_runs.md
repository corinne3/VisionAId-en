# Live runs — Windows PC (CPU only) + Samsung Android

Hardware: Windows 11 laptop, Intel CPU, 8 GB RAM, no GPU · Samsung, Android 10, Chrome · phone hotspot.

## 1. Acquisition only (21/11, 14:40)

| Measure | Value | Target (plan) |
|---|---|---|
| Images received by the PC | 10 /s | ≥ 15 /s |
| IMU | 60 Hz | ≥ 50 Hz |
| Image age p50 | 35 ms | < 200 ms |
| Network round trip | 3 ms | — |

→ Web page, local certificate, WebSocket and sensors validated on Android/Chrome.

## 2. Full chain, first attempt without settings (21/11, 16:15-17:10)

- Object detected but never anchored on the map (`object_states: unknown`), 0 valid paths.
- Causes found in the code: detection ↔ depth tolerance of ±3 frames, pose considered stale after 1.5 s. Both assume a fast machine.

## 3. Full chain with local settings (21/11, 19:20)

| Measure | Value |
|---|---|
| Valid paths | 75 / 91 |
| Path rate | 0.43 Hz |
| Capture → path latency p50 / p95 | 2.6 s / 3.6 s |
| Object states | seen 17, remembered 68 |

## 4. With guidance and sound on the phone (22/11, 01:30)

| Measure | Value |
|---|---|
| Valid paths | 190 / 240 (79 %) |
| Path rate | 0.52 Hz |
| Capture → path latency p50 / p95 | 2.2 s / 2.9 s |
| Object states | seen 74, remembered 196 |
| Left/right sound in the earphones | ✅ correct |

## 5. Voice mode (22/11, 11:35)

| Measure | Value |
|---|---|
| Valid paths | 49 / 145 |
| Capture → path latency p50 / p95 | 2.6 s / 4.5 s |
| Guidance to the bottle | ✅ reached |
| Voice request (Whisper-base on CPU) | ❌ not matched; request typed instead |
| Phone disconnections | 2 × `No PONG received` during transcription |

→ On a CPU-only PC, Whisper occupies all cores in the same process as the phone server, which then misses its heartbeat. Proposed fix: run speech recognition in a separate process, or use the phone's recognition (`--stt phone`).

## Portability findings

| Issue | Cause | Local fix | Durable fix proposed |
|---|---|---|---|
| "mes clés" not recognised | text files read as cp1252 | `PYTHONUTF8=1` | `encoding="utf-8"` in `open()` |
| Object never anchored | ±3-frame tolerance hard-coded | `phone.max_fps: 3` | tolerance in seconds, in config |
| Paths always invalid | pose stale after 1.5 s | `stale_s = 6` | `stale_s` in config |
| Gravity inverted on Android | iPhone calibration (`accel_sign = -1`, Safari-specific) applied to every phone | `phone.device: samsung` | calibration chosen per device |
| Model download stuck | Hugging Face Xet protocol | `HF_HUB_DISABLE_XET=1` | — |
| Whisper fails | ffmpeg missing on Windows | install ffmpeg | document it |
| TLS unit test fails | Windows-specific race in the test | none | make the test robust |
