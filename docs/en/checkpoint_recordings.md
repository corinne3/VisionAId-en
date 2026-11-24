# Checkpoint: recordings with the iPhone 12

> English translation of the team's [`docs/checkpoint_enregistrements.md`](../checkpoint_enregistrements.md). Written by the coding assistant for the team during the hackathon ("I" = the assistant). Session names and voice commands are kept as in the code (`gourde` = water bottle, `clés` = keys, `couloir` = corridor, `piece` = room).

These recordings are the only data I cannot produce myself. They are used to calibrate the camera and the sensors, to measure the fused pose and to test key detection. The rest of the plan moves on without them: the tests that depend on them are `skip` as long as they are missing.

Total time: about 30 minutes.

## Before starting
- Follow [`phone_setup.md`](phone_setup.md): network, certificate installed, page that opens without warning.
- Measure with a tape the **camera height** of the phone attached to the chest, from the floor (for example 1.32 m).
- Attach the phone **in portrait, screen facing you, rear camera facing forward**. The level bubble must be green.
- Photograph the keys: **5 to 10 photos** on varied backgrounds, to copy into `data/ref/cles_1.jpg`, `cles_2.jpg`, etc. The normal camera app is enough.

Each recording is started by a command on the computer. Then tap **Start** on the phone, and **Stop** at the end. `Ctrl+C` stops the server.

## 1. Camera calibration (`calib_camera`)
```bash
uv run python -m visionaid.apps.run_phone --record calib_camera
```
1. On the computer, open `https://localhost:8443/calib/board` full screen, brightness at maximum. Measure with a ruler the **side of one square** (in mm) and write it down.
2. On the phone, tick **Calibration mode**. The phone is held in the hand, not on the chest.
3. Take **15 to 20 views** with "Capture a view":
   - from the front, at an angle (left, right, top, bottom, about 30-45°);
   - close and far;
   - the checkerboard fully visible and sharp on each view.
4. In the Notes field, write `case = <mm> mm`, then "Send notes".

## 2. Sensor calibration (`calib_imu`)
```bash
uv run python -m visionaid.apps.run_phone --record calib_imu --height <height>
```
Phone on the chest, facing a textured area (shelf, wall with objects), in normal mode:
1. **5 s still**.
2. **3 slow rotations** on the spot, about 90° to the left and back, then 90° to the right and back.
3. **3 fast rotations**, same amplitudes.
4. Lean the upper body forward then backward, twice.
5. **5 s still**.

## 3. Navigation sessions
For each, start the command, stand at the starting point, tap Start, walk the route, then Stop.

| Name | Command | Route |
|---|---|---|
| `gourde_piece` | `--record gourde_piece --target "où est ma gourde" --height <h>` | Start 3-4 m from the bottle placed on a piece of furniture, without looking at it. Turn slowly on the spot until you see it, then walk to it avoiding the furniture. |
| `cles_piece` | `--record cles_piece --target "où sont mes clés" --height <h>` | Same, with the keys placed on a piece of furniture (table, chest of drawers). |
| `couloir` | `--record couloir --target "où est ma gourde" --height <h>` | Walk through a corridor or passage with 1 or 2 obstacles (chair, bag), the bottle at the end. |
| `rotations_rapides` | `--record rotations_rapides --height <h>` | On the spot: turn fast to the left and to the right, about 180° each time, for 30 s, then walk 3 m. |

The full command is `uv run python -m visionaid.apps.run_phone …`.

## 4. Check
```bash
uv run python -m visionaid.apps.session_info --all
```
For each session, check:
- `fps_median` ≥ 15;
- `imu_hz` ≥ 50;
- `gaps_over_3_periods` close to 0;
- `frame_age_ms_p95` < 200 ms.

## Web page checklist
- [ ] HTTPS accepted without warning after installing the certificate.
- [ ] Sensor and camera permissions obtained in a single gesture.
- [ ] The screen does not go to sleep for 5 minutes.
- [ ] At least 15 frames/s received and at least 50 Hz of IMU for 5 minutes, without frame age drift.
- [ ] Automatic reconnection after a Wi-Fi dropout: turn off the phone's Wi-Fi for 5 s, then turn it back on.

Write down the problems encountered (heating, dropouts, discomfort of the holder) in the page's Notes field, or tell me directly.

## Next
Tell me that the data is in `data/sessions/` and `data/ref/`. I then run:
- the camera and sensor calibration (phase 17);
- the `data` tests of phases 16 to 20;
- the replay of the sessions in the full pipeline.
