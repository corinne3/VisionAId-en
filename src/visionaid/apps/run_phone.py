"""Acquisition only: starts the phone server, shows the URL (QR code) and records if requested.

Examples:
    uv run python -m visionaid.apps.run_phone                      # live stream, without recording
    uv run python -m visionaid.apps.run_phone --record gourde_piece --target "où est ma gourde"
"""

from __future__ import annotations

import argparse
import json
import time

from visionaid.phone.server import PhoneServer
from visionaid.utils.config import load_config
from visionaid.utils.logging import get_logger

log = get_logger("run_phone")


def print_qr(url: str) -> None:
    """Shows a QR code of the URL in the terminal (generated locally)."""
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as e:  # terminal without unicode, etc.
        log.debug("QR code unavailable: %s", e)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--record", default=None, help="name of the session to record")
    ap.add_argument("--target", default=None, help="fixed prompt (recorded in the session)")
    ap.add_argument("--height", type=float, default=None, help="measured camera height (m or cm)")
    ap.add_argument("--notes", default="", help="free notes for the session")
    ap.add_argument("--duration", type=float, default=None, help="automatic stop (s)")
    args = ap.parse_args(argv)
    overrides = {}
    if args.height:
        from visionaid.utils.config import length_to_m

        overrides["body.camera_height_m"] = length_to_m(args.height, "camera height")
    if args.target:  # check right now that the target is known
        overrides["target.query"] = args.target
    cfg = load_config(overrides=overrides)
    meta = {"target": args.target, "notes": args.notes}
    srv = PhoneServer(cfg, record=args.record, meta=meta).start()
    urls = srv.urls()
    print("\n=== VisionAId: open on the phone (same Wi-Fi network) ===")
    for u in urls:
        print("  ", u)
    print_qr(urls[0])
    if srv.http_port > 0:
        host = urls[0].split("//")[1].rsplit(":", 1)[0]
        print(
            f"Certificate to install once: http://{host}:{srv.http_port}/  (docs/phone_setup.md)"
        )
    print("Ctrl+C to stop.\n")
    t0 = time.time()
    try:
        while args.duration is None or time.time() - t0 < args.duration:
            time.sleep(2.0)
            s = srv.stats
            log.info(
                "images %d (%.1f/s) | IMU %d (%.0f Hz) | age p50 %s ms",
                s.n_frames,
                s.rate(s.frame_times),
                s.n_imu,
                s.rate(s.imu_times),
                (f"{s.summary()['frame_age_ms_p50']:.0f}" if s.frame_ages else "–"),
            )
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    summ = srv.summary()
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    return summ


if __name__ == "__main__":
    main()
