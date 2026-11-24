"""Local certificate authority and server certificate (HTTPS required by Safari for camera and sensors).

Everything is generated on the computer, without an external service. The root certificate (`ca.crt`) is installed once
on the phone (see `docs/phone_setup.md`); the server certificate is regenerated automatically
if the local IP address changes.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CA_NAME = "VisionAId CA locale"
SERVER_DAYS = 397  # iOS refuses server certificates valid for more than 398 days
CA_DAYS = 3650


@dataclass
class CertPaths:
    """Paths of the generated files."""

    ca_cert: Path
    ca_key: Path
    server_cert: Path
    server_key: Path


def _ifconfig_ipv6() -> set[str]:
    """Global IPv6 addresses of the interfaces (macOS/BSD `ifconfig`, Linux `ip`), without link-local."""
    import subprocess

    out = ""
    for cmd in (["ifconfig"], ["ip", "-6", "addr"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
            break
        except (OSError, subprocess.SubprocessError):
            continue
    ips = set()
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "inet6" and len(parts) > 1:
            a = parts[1].split("%")[0].split("/")[0]
            try:
                ip = ipaddress.ip_address(a)
            except ValueError:
                continue
            if ip.is_global or (ip.is_private and not ip.is_link_local and not ip.is_loopback):
                ips.add(str(ip))
    return ips


def local_ips() -> list[str]:
    """Local addresses reachable by the phone: IPv4 and IPv6 (IPv6-only networks,
    e.g. 5G hotspot where the Mac only has a translation IPv4 192.0.0.x)."""
    ips: set[str] = {"127.0.0.1"}
    for fam, probe in (
        (socket.AF_INET, "10.255.255.255"),
        (socket.AF_INET6, "2001:4860:4860::8888"),
    ):
        try:  # default outgoing interface, without sending a packet
            s = socket.socket(fam, socket.SOCK_DGRAM)
            try:
                s.connect((probe, 1))
                ips.add(s.getsockname()[0].split("%")[0])
            finally:
                s.close()
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips |= _ifconfig_ipv6()
    return sorted(ips)


def reachable_ips() -> list[str]:
    """Addresses to offer to the phone: without loopback or translation IPv4 (192.0.0.0/29)."""
    out = []
    for a in local_ips():
        ip = ipaddress.ip_address(a)
        if ip.is_loopback or (ip.version == 4 and ip in ipaddress.ip_network("192.0.0.0/29")):
            continue
        out.append(a)
    return out


def url_host(a: str) -> str:
    """URL host: brackets for an IPv6."""
    return f"[{a}]" if ":" in a else a


def host_names() -> list[str]:
    """DNS names covered by the server certificate."""
    h = socket.gethostname().split(".")[0]
    names = {"localhost", h, f"{h}.local"}
    return sorted(n for n in names if n)


def _key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _write_key(key, path: Path) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def make_ca(paths: CertPaths) -> None:
    """Creates the local certificate authority."""
    key = _key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CA_NAME)])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    _write_key(key, paths.ca_key)
    paths.ca_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def make_server_cert(paths: CertPaths, names: list[str], ips: list[str]) -> None:
    """Issues the server certificate signed by the local authority."""
    ca_key = serialization.load_pem_private_key(paths.ca_key.read_bytes(), None)
    ca_cert = x509.load_pem_x509_certificate(paths.ca_cert.read_bytes())
    key = _key()
    now = _now()
    san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=SERVER_DAYS))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(key, paths.server_key)
    paths.server_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def server_cert_covers(paths: CertPaths, names: list[str], ips: list[str]) -> bool:
    """True if the server certificate exists, does not expire within 7 days and covers names and IPs."""
    if not paths.server_cert.exists() or not paths.server_key.exists():
        return False
    cert = x509.load_pem_x509_certificate(paths.server_cert.read_bytes())
    if cert.not_valid_after_utc < _now() + dt.timedelta(days=7):
        return False
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    have_dns = set(san.get_values_for_type(x509.DNSName))
    have_ip = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    return set(names) <= have_dns and set(ips) <= have_ip


def ensure_certs(
    directory: str | Path, names: list[str] | None = None, ips: list[str] | None = None
) -> CertPaths:
    """Creates the authority if needed and (re)issues the server certificate if it no longer covers the host."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = CertPaths(d / "ca.crt", d / "ca.key", d / "server.crt", d / "server.key")
    names = names or host_names()
    ips = ips or local_ips()
    if not paths.ca_cert.exists() or not paths.ca_key.exists():
        make_ca(paths)
        for p in (paths.server_cert, paths.server_key):
            p.unlink(missing_ok=True)
    if not server_cert_covers(paths, names, ips):
        make_server_cert(paths, names, ips)
    return paths


def server_ssl_context(paths: CertPaths) -> ssl.SSLContext:
    """Server-side TLS context."""
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(str(paths.server_cert), str(paths.server_key))
    return ctx


def client_ssl_context(paths: CertPaths) -> ssl.SSLContext:
    """TLS context of a client that trusts the local authority (fake phone, tests)."""
    ctx = ssl.create_default_context(cafile=str(paths.ca_cert))
    return ctx
