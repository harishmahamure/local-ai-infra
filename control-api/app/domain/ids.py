from __future__ import annotations

import secrets
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def ulid() -> str:
    ms = int(time.time() * 1000)
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    return _encode(ms, 10) + _encode(rand, 16)


def new_job_id() -> str:
    return f"job_{ulid()}"


def new_asset_id() -> str:
    return f"ast_{ulid()}"


def new_trace_id() -> str:
    return f"tr_{ulid()}"
