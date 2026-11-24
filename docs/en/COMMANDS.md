# VisionAId — main commands (English translation of `COMMANDES.md`)

Run every command from the project folder. `Ctrl+C` stops the current command.
On Windows, see also [setup/windows-android](../../setup/windows-android/README.md) (local settings for a CPU-only PC).

> **Rules of use**
> - Only one "live" or "real-time replay" command at a time: they use the same ports. Stop one before starting the other.
> - At start-up, allow about 25 s for the models to load before the image arrives. An empty Rerun window at first is normal.
> - Do not close the Rerun window during a live run or a replay: the computation goes on, but nothing is displayed any more.
> - Camera height (`--height`): in metres (`1.5`) or centimetres (`150`), both are understood.
> - The system understands and speaks **French**: target phrases stay in French (`"où est ma gourde"` = "where is my water bottle", `"où sont mes clés"` = "where are my keys").

---

## 1. Live with the phone

### Live + Rerun visualisation
```bash
uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 150 --rerun live
```
Then, on the phone browser: `https://<computer-name>.local:8443/` (or the IP address printed by the command). Headphones connected to the phone, phone on the chest, then **Start**. The guidance sound plays in the headphones; **Test the sound** checks left, right and front. Turn off "Mono audio" in the phone's accessibility settings.

For the keys: `--target "où sont mes clés"`.

### Live + visualisation + session recording
```bash
uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 150 --rerun both --record bottle_test2
```
- `--record <name>` records the session in `data/sessions/<date>_<name>/`, replayable later (section 2).
- `--rerun both` shows Rerun and also saves `outputs/live_session.rrd`. This file is overwritten by the next live run.

### Recording only, without computation
Lighter: useful for calibration or data collection.
```bash
uv run python -m visionaid.apps.run_phone --record keys_test2 --target "où sont mes clés" --height 150
```

### Useful live options
| Option | Effect |
|---|---|
| `--rerun live` | visualisation only |
| `--rerun both` | visualisation + `outputs/live_session.rrd` file |
| `--rerun file` | file only, no window |
| `--rerun off` | nothing |
| `--duration 60` | automatic stop after 60 s |
| `--demo` | demo: a single bottle in the room, recognised as soon as it is detected (no reference photos, no colour) |
| `--depth-width 364` | faster depth (−35 % latency), but less precise guidance |

### Voice request ("je cherche ma gourde", "I'm looking for my water bottle")
```bash
uv run python -m visionaid.apps.run_live --command stt --height 150 --rerun live
```
Without `--target`, voice mode is chosen automatically. On the phone, after **Start**, the page invites you (in French) to press the big **🎤 Talk / Parler** button (beep), then say "je cherche ma gourde" or "où sont mes clés": recording stops by itself after the sentence (or press again). The PC transcribes (Whisper, ~0.6 s on a Mac), then:
1. "Je cherche la gourde. Tourne lentement sur toi-même." ("I'm looking for the bottle. Turn slowly on the spot."), then the search sound;
2. object found: the path sound pauses for one sentence saying where the object is and how to get there ("The bottle is at the back of the room, in front of you, about 4 metres away, behind the sofa, on the table. The path goes around the sofa on the left, follow the sound.");
3. guidance with the path sound only;
4. near the furniture holding the object (or within 1.6 m): "You're reaching the table, the bottle will be slightly on your left.", the path sound continues until the arrival sound.

Other requests: "stop" / "arrête" (silence), a new request at any time ("je cherche mes clés"). Sentence not understood: "Je n'ai pas compris… Tu peux chercher tes clés ou ta gourde." ("I didn't understand… You can look for your keys or your bottle."), the current search continues. The "ou écrire" (or type) field of the page sends the request from the keyboard (tests).

| Option | Effect |
|---|---|
| `--command prompt --target "…"` | object fixed at start-up; spoken announcements are still made |
| `--command stt` | the user says what they are looking for (Speak button) |
| `--stt whisper` | (default) the phone records, the PC transcribes. First run: model download (~1 GB) |
| `--stt phone` | the phone's own speech recognition; falls back to Whisper on failure |
| `--tts auto` | (default) voice in the phone headphones when live, on the Mac speaker when replaying |
| `--tts mac` / `--tts off` | voice on the Mac / no voice |

In voice mode, recognition of every object (`config/targets/` sheets) is prepared at start-up (~10 s more). Furniture (table, sofa, chair, bed, shelf…) is detected together with the object and used in the sentences; it appears in purple in the 3D world view (`world/meubles`). Spoken sentences are shown in Rerun (`status`) and in the log.

---

## 2. Replay a recorded session

Available sessions:
```bash
ls data/sessions
```

### Replay "like live", with the 3D world view
```bash
uv run python -m visionaid.apps.run_live --session data/sessions/20261121-140325_gourde_piece --rerun live
```
The phone is not used. The session loops at the end. The target object is taken from the session. Spoken announcements play on the computer speaker (`--tts off` to mute them).

### Offline replay → annotated video with the guidance sound
```bash
uv run python -m visionaid.apps.run_offline --session data/sessions/20261121-140325_gourde_piece
```
Then open `outputs/annotated_audio.mp4` (macOS: `open`, Windows: `start`).
Listen with headphones: spoken announcements are mixed in (macOS `Thomas` voice, `tts.mac_voice`; not available on Windows). The same replay also produces `outputs/session.rrd` and `outputs/guidance.wav`.
⚠️ On an 8 GB CPU-only PC, the offline replay of a long session may run out of memory: use `--max-frames`.

### Guidance sound only (tuning the sounds)
```bash
uv run python -m visionaid.apps.render_sound data/sessions/<session> --video
```
Writes `outputs/son/<session>/`: `guidage.wav` (headphones) and `camera_son.mp4` (camera image + sound). The first run replays the vision and stores the cues in `consignes.json`; later runs only redo the sound, in a few seconds: change a setting in the `audio` section of `config/default.yaml` and listen again. `--recompute` replays the vision again.

To use a session recorded on another computer: copy its folder `data/sessions/<date>_<name>/` (with `frames.jsonl`, `meta.json`, `imu.jsonl` and the images) into `data/sessions/`.

### Session summary
Duration, frames per second, IMU, gaps:
```bash
uv run python -m visionaid.apps.session_info --all
```

---

## 3. Open a Rerun recording (`.rrd`)

```bash
uv run rerun --port 9877 outputs/live_session.rrd
```
`--port 9877` opens a separate window. Without it, the file is added to the window already open (left panel, *Recordings*).

**3D view navigation:** left click + drag to rotate, wheel to zoom, right click + drag to pan, double click to reframe. Click the eye of an item in the left panel to hide it (e.g. `world/nuage`, the point cloud).

---

## 3 bis. Videos for a presentation

Export a session as three videos: annotated camera, top-down map and 3D world.
```bash
uv run python -m visionaid.apps.export_videos data/sessions/20261121-182903_gourde_direct2
```
Videos are written to `outputs/videos/<session>/`: `camera.mp4`, `carte.mp4` (map) and `monde3d.mp4` (3D world). Option `--only monde3d` to redo only one.

---

## 4. Calibration (once per phone)

Detailed procedure (French): `docs/checkpoint_enregistrements.md`.
```bash
uv run python -m visionaid.apps.run_phone --record calib_camera
```
Show the checkerboard `src/visionaid/phone/web/board.html` full screen, then:
```bash
uv run python -m visionaid.apps.run_phone --record calib_imu --height 150
uv run python -m visionaid.apps.calibrate_phone --camera data/sessions/<…>_calib_camera --imu data/sessions/<…>_calib_imu
```
Results go to `config/calib/<device>_*.yaml` and are used automatically for the device named in `phone.device` (default `iphone12`). **For another phone, change `phone.device`**: the iPhone calibration includes a Safari-specific accelerometer sign flip.

### Sound demo on simulated scenarios (no models needed)
```bash
uv run python -m visionaid.apps.demo_sound
```
Writes `outputs/demo_sons/{room_0,corridor_1,appearing_0}.mp4`: simulated scenario with the guidance sound (headphones).

### Guidance sounds
Original sounds by Romain Benoit in `assets/sounds/` (`VisionAid_Sounds 1` to `4`), prepared versions (mono, single pattern, equalised loudness, loopable pad) in `assets/sounds/processed/`. After changing an original sound:
```bash
uv run python -m visionaid.audio.prepare
```
Two superimposed layers:
- **path** (sound 2; sound 4 while the object is not found): repeated note, stereo-panned towards the path (fully left = 100 % left ear; path behind = fully on the side of the shortest turn). The tempo speeds up within 2 m of the target;
- **obstacles** (sound 1): looping pad, louder and louder between 1.5 m and 0.4 m, always below the path;
- **arrival** (sound 3): once, then silence. Tracking lost: no path sound, pad at medium volume.

Settings (tempo, distances, obstacle volumes, hysteresis behind…): `audio` section of `config/default.yaml`.

---

## 5. Without a phone: fake phone

Replays a video as if it came from the phone, to test the whole live chain.
```bash
uv run python -m visionaid.apps.run_live --fake-phone data/video/appart.mp4 --duration 60 --rerun live
```

---

## 6. Path diagnostics (guidance quality on sessions)

For each navigation session: when the first valid path appears, time spent searching, share of valid paths, blocked starts and direction jumps; also the share of paths through never-seen areas (`unknown_path_frac`, must stay at 0) and jumps of the sound direction between frames (`sound_jumps_over_20`, `sound_jumps_over_45`).
```bash
uv run python -m visionaid.apps.diag_path --all --out outputs/diag
uv run python -m visionaid.apps.diag_path --compare outputs/diag_before outputs/diag --png outputs/diag/comparison.png
```

---

## 7. Tests

```bash
make test-fast                              # ~190 fast tests, ~2 min (Windows: uv run pytest -m "not slow" -q)
uv run pytest -m data -q --timeout=1800     # tests on real data, ~10 min
make night                                  # everything, log in outputs/night_run.log (macOS/Linux)
```

---

## 8. Quick troubleshooting

| Problem | Solution |
|---|---|
| `address already in use` / port 8443 | a live run or replay is already running: `Ctrl+C` in its terminal |
| Empty Rerun window | wait ~25 s (models loading); check it was not closed |
| "Connection not private" on the phone | install the certificate and trust it: `docs/phone_setup.md`, step 3 |
| The phone cannot reach the computer | same network (or the phone's hotspot); public Wi-Fi often isolates devices |
| More details | [`phone_setup.md`](phone_setup.md) (connection), [`REPORT_PHASE2.md`](REPORT_PHASE2.md) (report), [`DECISIONS.md`](DECISIONS.md) (design decisions) |
