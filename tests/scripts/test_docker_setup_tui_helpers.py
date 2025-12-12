from __future__ import annotations

import string

from docker_setup_tui import FLOWER_PASSWORD_SYMBOLS, MIN_FLOWER_PASSWORD_LENGTH, generate_flower_credentials


def test_generate_flower_credentials_strength() -> None:
    username, password = generate_flower_credentials()
    username2, password2 = generate_flower_credentials()

    assert username.startswith("flower_")
    assert len(username) > len("flower_")
    assert username != username2  # High entropy expected

    assert len(password) >= MIN_FLOWER_PASSWORD_LENGTH
    assert password != password2

    allowed = set(string.ascii_letters + string.digits + FLOWER_PASSWORD_SYMBOLS)
    assert set(password).issubset(allowed)

    assert any(c.islower() for c in password)
    assert any(c.isupper() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in FLOWER_PASSWORD_SYMBOLS for c in password)
