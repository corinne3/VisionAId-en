# Configuration B — Windows PC (CPU only) + Android phone

Tested on: Windows 11 laptop, Intel CPU, **8 GB RAM, no GPU**, Samsung Android 10, Chrome.

## 1. Install (once, with internet)

Open **PowerShell** and run:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e      # needed by Whisper (voice requests)
```

Close and reopen PowerShell, then get this repository and run the setup script:

```powershell
cd C:\dev
git clone https://github.com/corinne3/VisionAId-en.git visionaid
cd visionaid
powershell -ExecutionPolicy Bypass -File setup\windows-android\setup_windows.ps1
```

The script:
1. installs the dependencies (`uv sync`, CPU PyTorch);
2. sets two environment variables (`PYTHONUTF8=1`, `HF_HUB_DISABLE_XET=1`);
3. applies the **local settings for a slow machine** (table below).

Then allow the two ports in the Windows firewall (PowerShell **as administrator**, once):

```powershell
New-NetFirewallRule -DisplayName "VisionAId" -Direction Inbound -Protocol TCP -LocalPort 8080,8443 -Action Allow -Profile Any
```

## 2. Local settings for a slow machine

| Setting | Default | Here | Why |
|---|---|---|---|
| `phone.device` | `iphone12` | `samsung` | the iPhone calibration includes a Safari-specific sign flip of the accelerometer; applied to Android it inverts gravity |
| `phone.max_fps` | 15 | 3 | detection and depth must fall on frames at most ±3 frames apart; on CPU they never did |
| `stale_s` (planner, guidance) | 1.5 s | 6 s | on CPU a new pose arrives every ~2 s; otherwise every path is marked invalid |
| Whisper model | `small` | `base` | lighter for 8 GB RAM |

These are **local** changes (not committed). To undo them: `git checkout -- config/default.yaml src/visionaid/apps/live_blocks.py`.

## 3. Network

Public Wi-Fi networks (eduroam, hotels) usually **isolate devices from each other**: the phone cannot reach the PC.
→ Turn on the **phone's hotspot** and connect the PC to it. Traffic between phone and PC stays local.

Quick test: on the PC `python -m http.server 8000`, on the phone `http://<PC IPv4>:8000`. If the page loads, the network is fine.

## 4. Phone certificate (once)

1. Start the server: `cd C:\dev\visionaid` then `uv run python -m visionaid.apps.run_phone`
2. On the phone (Chrome): `http://<PC IPv4>:8080/` → download the certificate.
3. **Settings** → search "certificate" → **Install from device storage** → **CA certificate** → install anyway.
4. Close Chrome completely, reopen, then open `https://<PC IPv4>:8443/`.

## 5. Run

Start the PC side first, then press **Démarrer** on the phone page.

```powershell
cd C:\dev\visionaid
# fixed target (most reliable on CPU)
uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 130 --depth-width 364 --record my_run
# demo mode: a single bottle in the room
uv run python -m visionaid.apps.run_live --demo --target "où est ma gourde" --height 130 --depth-width 364
# voice request recognised by the phone (keeps the PC free)
uv run python -m visionaid.apps.run_live --command stt --stt phone --height 130 --depth-width 364
```

- `--height 130`: camera height above the floor, in cm.
- `--depth-width 364`: faster depth on CPU (less precise).
- Earphones on the phone; turn off "mono audio" in the accessibility settings.
- Move **slowly**, pause facing the object, keep Chrome in the foreground.
- `Ctrl+C` stops and prints the measurements.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Page does not load on the phone | check the IPv4 shown by the server; firewall rule; hotspot instead of public Wi-Fi |
| `mes clés` not recognised | `PYTHONUTF8=1` (set by the script) |
| Model download stuck at 0 % | `HF_HUB_DISABLE_XET=1`, or `uv run hf download <model>` |
| Phone disconnects with `No PONG received` | Whisper is using all the CPU: use `--stt phone` or `--target` |
| `n_valid_paths: 0`, object `unknown` | check the local settings above; only one bottle in view, or add reference photos in `data/ref/` |
