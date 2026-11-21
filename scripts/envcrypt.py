"""Encripta/desencripta o arquivo .env com GPG simétrico (gpg --symmetric, AES256).

O arquivo encriptado é .env-$USER.gpg (usuário atual) e pode ser aberto
diretamente com `gpg -d .env-$USER.gpg`.

Uso:
    uv run scripts/envcrypt.py encrypt [--in .env] [--out .env-$USER.gpg]
    uv run scripts/envcrypt.py decrypt [--in .env-$USER.gpg] [--out .env] [--force]

A senha é pedida pelo próprio gpg (pinentry) ou lida da variável ENVCRYPT_PASSWORD.
"""

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ENCRYPTED = Path(f".env-{getpass.getuser()}.gpg")


def run_gpg(args: list[str]) -> bytes:
    gpg = shutil.which("gpg")
    if gpg is None:
        sys.exit("Erro: gpg não encontrado no PATH.")
    cmd = [gpg, "--quiet", "--no-symkey-cache"]
    stdin = None
    if pw := os.environ.get("ENVCRYPT_PASSWORD"):
        cmd += ["--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0"]
        stdin = pw.encode()
    result = subprocess.run(cmd + args, input=stdin, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"Erro do gpg:\n{result.stderr.decode().strip()}")
    return result.stdout


def write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def encrypt(src: Path, dst: Path) -> None:
    run_gpg(["--yes", "--symmetric", "--cipher-algo", "AES256", "--output", str(dst), str(src)])
    print(f"{src} -> {dst} (encriptado)")


def decrypt(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        sys.exit(f"Erro: {dst} já existe. Use --force para sobrescrever.")
    plaintext = run_gpg(["--decrypt", str(src)])
    write_private(dst, plaintext)
    print(f"{src} -> {dst} (desencriptado)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encrypt", help=f"encripta .env -> {DEFAULT_ENCRYPTED}")
    enc.add_argument("--in", dest="src", type=Path, default=Path(".env"))
    enc.add_argument("--out", dest="dst", type=Path, default=DEFAULT_ENCRYPTED)

    dec = sub.add_parser("decrypt", help=f"desencripta {DEFAULT_ENCRYPTED} -> .env")
    dec.add_argument("--in", dest="src", type=Path, default=DEFAULT_ENCRYPTED)
    dec.add_argument("--out", dest="dst", type=Path, default=Path(".env"))
    dec.add_argument("--force", action="store_true", help="sobrescreve o destino")

    args = parser.parse_args()
    if not args.src.exists():
        sys.exit(f"Erro: {args.src} não encontrado.")
    if args.cmd == "encrypt":
        encrypt(args.src, args.dst)
    else:
        decrypt(args.src, args.dst, args.force)


if __name__ == "__main__":
    main()
