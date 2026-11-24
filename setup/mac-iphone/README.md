# Configuration A — Mac (Apple Silicon) + iPhone (reference)

This is the configuration used by the team during the hackathon (Mac M4, iPhone 12). All commands: [docs/en/COMMANDS.md](../../docs/en/COMMANDS.md).

## 1. Install

```bash
git clone https://github.com/corinne3/VisionAId-en.git visionaid
cd visionaid
uv sync           # Python 3.12, PyTorch with MPS (Apple GPU)
```

## 2. Network and certificate

- Use the **iPhone hotspot** if the Wi-Fi isolates devices.
- Start the server: `uv run python -m visionaid.apps.run_phone`
- On the iPhone (Safari): `http://<Mac IP>:8080/` → download the profile.
- **Settings → General → VPN & Device Management** → install "VisionAId CA locale".
- **Settings → General → About → Certificate Trust Settings** → enable full trust.
- Open `https://<Mac IP>:8443/` → **Démarrer**.

## 3. Run

```bash
# fixed target
uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 130
# voice request (Whisper-small on the Mac, ~0.6 s per sentence)
uv run python -m visionaid.apps.run_live --command stt --height 130
# demo mode (a single bottle in the room)
uv run python -m visionaid.apps.run_live --demo --target "où est ma gourde" --height 130
```

The iPhone 12 calibration (camera + gyroscope/accelerometer) is loaded by default (`phone.device: iphone12`).
For another iPhone, record a calibration session (see section 4 of [COMMANDS.md](../../docs/en/COMMANDS.md)).

## Measured (team reports)

| Measure | Value |
|---|---|
| Valid paths | 4.1 Hz |
| Capture → path latency p50 / p95 | 336 / 415 ms |
| Simulator success (100 runs per scenario) | room 96 %, appearing obstacle 95 %, corridor 89 % |
