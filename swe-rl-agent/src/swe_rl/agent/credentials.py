"""OS credential store. No plaintext or alternate file-backend fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _backend() -> Any:
    try:
        import keyring

        backend = keyring.get_keyring()
        module = type(backend).__module__
        backends = list(backend.backends) if module == "keyring.backends.chainer" else [backend]
        allowed = (
            "keyring.backends.macOS",
            "keyring.backends.Windows",
            "keyring.backends.SecretService",
            "keyring.backends.libsecret",
        )
        if not backends or not all(type(item).__module__.startswith(allowed) for item in backends):
            raise RuntimeError(
                "A supported OS credential store is required; use environment keys on headless hosts"
            )
        return backend
    except ImportError:
        raise RuntimeError("Install keyring to save keys in the OS credential store") from None


def secret_reference(profile: Path, provider: str, endpoint: str) -> str:
    identity = f"{profile.resolve()}:{provider}:{endpoint}"
    return "keyring://swe-rl/" + hashlib.sha256(identity.encode()).hexdigest()


def store_secret(reference: str, value: str) -> None:
    try:
        _backend().set_password("swe-rl", reference, value)
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("OS credential store could not save the key") from None


def load_secret(reference: str) -> str:
    if not reference.startswith("keyring://swe-rl/"):
        raise ValueError("Invalid credential reference")
    try:
        return _backend().get_password("swe-rl", reference) or ""
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("OS credential store could not retrieve the key") from None


def delete_secret(reference: str) -> None:
    try:
        backend = _backend()
        if backend.get_password("swe-rl", reference) is not None:
            backend.delete_password("swe-rl", reference)
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("OS credential store could not remove the key") from None
