from webui.services.api_key_service import ApiKeyService


def test_generate_key_format_includes_uuid_prefix() -> None:
    key_id = "550e8400-e29b-41d4-a716-446655440000"
    raw_key = ApiKeyService._generate_key(key_id)

    assert raw_key.startswith("smtk_550e8400_")
    parts = raw_key.split("_", 2)
    assert parts[0] == "smtk"
    assert parts[1] == "550e8400"
    assert len(parts[2]) >= 20


def test_hash_key_is_sha256_hex() -> None:
    raw_key = "smtk_550e8400_secret"
    hashed1 = ApiKeyService._hash_key(raw_key)
    hashed2 = ApiKeyService._hash_key(raw_key)

    assert hashed1 == hashed2
    assert len(hashed1) == 64
    assert all(c in "0123456789abcdef" for c in hashed1)

