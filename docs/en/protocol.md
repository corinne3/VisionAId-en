# Phone → computer protocol (version 1)

> English translation of the team's [`docs/protocol.md`](../protocol.md). Message types and field names are identical to the code.

The page `https://<pc>:8443/` opened in the phone's browser sends the camera and the sensors to the server (`apps/run_phone.py`) over a secure WebSocket `wss://<pc>:8443/ws`. The certificate is issued by a local authority generated on the computer. Its root certificate can be downloaded over plain HTTP at `http://<pc>:8080/ca.crt` (see [`phone_setup.md`](phone_setup.md)).

Only one phone is connected at a time. A new connection replaces the old one.

## Phone → computer

### Text messages (JSON)

| `type` | Fields | Role |
|---|---|---|
| `hello` | `protocol` (int), `ua` (user agent), `camera: {w, h}`, `imu` (bool), `gyro_units` (`"deg/s"`), `t_phone` | Mandatory first message. Unknown version: `error` then close. |
| `imu` | `samples: [[t, gx, gy, gz, ax, ay, az], …]` | Batch every ~50 ms. `t` in s (phone clock), gyro in the announced unit, acceleration including gravity in m/s², W3C device frame. |
| `ping` | `id`, `t_phone`, `prev_id`, `rtt` | Every second. `rtt` (s) is the round trip measured for ping `prev_id` (the previous exchange): the computer thus matches delay and RTT of the same exchange. |
| `notes` | `text` | Free notes, saved in the session's `meta.json`. |
| `command` | `text`, `alternatives` (list, optional), `source` | User request already as text (Safari recognition, or typed): "je cherche ma gourde" (I'm looking for my bottle), "stop". `stt` mode only. |
| `command_audio` | `mime` (`audio/mp4`…), `data` (base64) | Recorded request (a few seconds), transcribed by Whisper on the computer. |
| `bye` | — | Voluntary stop. |

### Binary messages (images)
Structure: `[4 bytes: header length L, big-endian][L bytes: UTF-8 JSON header][JPEG]`.

Header: `{"type": "frame", "seq": int, "t_phone": float, "w": int, "h": int, "calib": bool}`.
- `seq` is strictly increasing.
- `calib = true` marks a calibration image, sent at full resolution and on request. The server records it but does not publish it on the processing bus.

**Back-pressure**: the computer acknowledges an image **after decoding it and passing it** to processing. The phone only sends image `seq` after receiving `ack` for the previous image, or after `ack_timeout_s` (1 s) in case of loss. It always grabs the most recent image at sending time: nothing piles up.

## Computer → phone (JSON)

| `type` | Fields | Role |
|---|---|---|
| `welcome` | `protocol`, `session` (name or `null`), `max_fps`, `jpeg_quality`, `imu_batch_ms`, `capture_width`, `capture_height`, `ack_timeout_s`, `ping_interval_s`, `audio` | Reply to `hello`, with the settings imposed by the computer. `audio`: sound rendering settings (see below), or `null` if the sounds are missing. |
| `ack` | `seq` | Image received and decoded. |
| `pong` | `id`, `t_phone`, `t_pc` | Reply to the ping. |
| `stats` | `fps_in`, `imu_hz`, `offset_ms`, `rtt_ms`, `frame_age_ms` | Every second, for display on the phone. |
| `error` | `message` | Fatal error (version, format), followed by close. |
| `sound` | `t`, `state`, `mode`, `az`, `pan`, `dist`, `interval`, `gain_db`, `obstacle_db`, `event` | Sound instruction, ~20 per second (`realtime.cue_hz`). See "Sound". |
| `speech` | `id`, `text`, `duck` | Sentence to speak (browser speech synthesis), once per `id`. `duck`: the path layer goes silent until the end of the sentence (object found announcement); obstacles stay audible. |
| `heard` | `text`, `action` (`search`, `stop`, `unknown`), `target` | What the computer understood of the request (display). |

`welcome.voice`: `mode` (`prompt` or `stt`), `stt` (`whisper` or `phone`), `max_record_s`, `tts` (the phone speaks the sentences), `lang`, `rate`, `target` (object name in `prompt` mode). In `stt` mode, the page shows the big **Talk** button; the microphone is only open during the request (a Bluetooth headset then switches back to stereo).

## Sound (computer → phone → headset)

The computer decides what to play (`audio/director.py`), the phone executes it with Web Audio (`web/audio.js`). The sounds are served by the computer at `/sounds/<file>` (folder `audio.sounds_dir`). Two overlaid layers: the **path** (repeated note, stereo pan) and the **obstacles** (looping pad, volume according to distance).

`welcome.audio`: `sounds` (`guide`, `search`, `arrived`, `obstacle` → URL), `gains_db` (per sound), `master_gain_db`, `smoothing_s` (pan), `obstacle_smoothing_s`, `lost_obstacle_db`, `watchdog_s`.

`sound` message:

| Field | Meaning |
|---|---|
| `mode` | path layer: `search` (repeated search sound), `guide` (repeated path note), `lost` (tracking lost: no more path), `arrived` (silence after the arrival sound), `silent` |
| `az` | path direction (rad, positive to the left) |
| `pan` | constant-power stereo pan: −1 = fully left (nothing on the right), 0 = centre, +1 = fully right. Path behind: ±1 on the side of the shortest U-turn |
| `interval` | spacing between two notes (s), reduced within `audio.guide_accel_m` of the target |
| `gain_db` | note gain |
| `obstacle_db` | gain of the obstacle pad (≤ −120: muted), smoothed over `obstacle_smoothing_s` |
| `event` | `arrived`: play the arrival sound once |

Without a `sound` message for `watchdog_s` (1.5 s) during guidance, the phone switches to "tracking lost" (no more path, pad at `lost_obstacle_db`): a frozen guide would be dangerous.

## Publishing on the bus (computer side)
- `frame` → `Frame(t=t_phone, frame_idx=seq, image=BGR resized to video.width, K, t_phone, t_recv)`.
- `imu` → `ImuBatch` (gyro converted to rad/s), one message per batch received.
- `clock` → `ClockSync` at each offset update.

## Recorded session
Folder `data/sessions/<YYYYMMDD-HHMMSS>_<name>/` (ignored by git):

| File | Content |
|---|---|
| `frames/<seq:06d>.jpg` | received JPEGs, without recompression |
| `calib/<seq:06d>.jpg` | calibration images (`calib = true`) |
| `frames.jsonl` | `{"seq", "t_phone", "t_recv", "w", "h", "file"}` per image |
| `imu.jsonl` | `{"t", "gyro": [rad/s ×3], "accel": [m/s² ×3], "t_recv"}` per sample |
| `clock.jsonl` | `{"t_recv", "offset", "uncertainty", "rtt"}` |
| `meta.json` | `protocol`, `ua`, `camera`, `gyro_units`, `camera_height_m`, `target`, `notes`, start/end, number of images and samples |
| `video.mp4` | optional, generated by `apps/session_info.py` for viewing |
