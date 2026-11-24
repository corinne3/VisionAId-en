# Phone setup (iPhone 12, Safari)

> English translation of the team's [`docs/phone_setup.md`](../phone_setup.md). The French original is the reference. For the Windows PC + Android setup, see [`setup/windows-android/README.md`](../../setup/windows-android/README.md).

The phone opens a web page served by the computer. Safari only allows the camera and motion sensors on a trusted **HTTPS** page, so the computer's local certificate authority must be installed once. This certificate is generated in `.certs/`, which is never committed.

## 1. Network
The phone and the computer must be on the **same network**. If the Wi-Fi isolates devices from each other (common at hackathons: the page never loads), turn on the iPhone's **Personal Hotspot** and connect the computer to it.

## 2. Start the server
```bash
uv run python -m visionaid.apps.run_phone
```
The terminal shows the `https://<ip>:8443/` addresses and a QR code.

## 3. Install the certificate (once per computer)
1. On the iPhone, open `http://<computer-ip>:8080/` in **Safari**, then tap "Download the certificate". Safari offers to install a profile: accept.
2. **Settings → General → VPN & Device Management** → profile "VisionAId CA locale" → **Install**.
3. **Settings → General → About → Certificate Trust Settings** → turn on **full trust** for "VisionAId CA locale".

If the computer's IP address changes, the server automatically re-issues its certificate with the same authority: nothing to reinstall.
On Android, the procedure is similar: Settings → Security → Encryption & credentials → Install a certificate → CA certificate.

## 4. Start the capture
1. Open `https://<computer-ip>:8443/` in Safari, or scan the QR code.
2. Attach the phone **to the chest, in portrait, screen facing you** (rear camera facing forward).
3. Tap **Start**, then allow access to **motion** and **camera**.
4. Check that:
   - the computer receives at least 15 frames/s;
   - the IMU is above 50 Hz;
   - the frame age stays stable;
   - the level bubble is green (phone upright).

The screen stays on thanks to the Screen Wake Lock (Safari ≥ 16.4). With an older version, turn off Auto-Lock.

## Troubleshooting
| Symptom | Likely cause | Fix |
|---|---|---|
| "Page not secure" | page opened over `http://` | use `https://…:8443/` |
| Certificate warning | full trust not enabled | step 3.3 |
| The page does not load | isolating Wi-Fi, macOS firewall | Personal Hotspot; allow Python in System Settings → Network → Firewall |
| IMU "missing" | motion permission denied | Settings → Safari → Motion & Orientation Access, then reload |
| Frozen frames | page hidden or screen locked | keep Safari in the foreground |
| "This connection is not private" although the certificate is installed | the Mac changed network (Wi-Fi ↔ hotspot) and the certificate does not cover the new address | the server re-issues it by itself within a few seconds ("Network address changed" message in the terminal): reload the page with the **new** address shown |
| "Cannot connect to the server" on Personal Hotspot | the Wi-Fi address is used while the Mac is on the hotspot (172.20.10.x address), or Chrome has no local network access | use the `172.20.10.x` address shown in the terminal; for Chrome: Settings → Chrome → Local Network on |
| "Your iPhone is not connected to the Internet" on a 5G hotspot | **IPv6-only** network (Orange 5G…): the Mac only has a translation IPv4 `192.0.0.2`, which is unreachable | up-to-date server (listens on IPv6): open `https://<mac-name>.local:8443/` (first address shown) or an IPv6 address in brackets |
| The phone never reaches the Mac | the phone showing the page is not on the **same network** as the Mac (it is on direct 5G) | simplest: connect the Mac to the hotspot **of the iPhone that films**; otherwise put both devices on the same hotspot or Wi-Fi |
| Chrome on iPhone | Chrome uses the iOS certificates | install the certificate **with Safari** (step 3), then Chrome works too |
