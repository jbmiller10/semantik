"""Unit tests for SecretEncryption and Fernet key utilities."""

import pytest
from cryptography.fernet import Fernet

from shared.utils.encryption import DecryptionError, EncryptionNotConfiguredError, SecretEncryption, generate_fernet_key


@pytest.fixture(autouse=True)
def _reset_encryption():
    """Reset encryption state before and after each test."""
    SecretEncryption.reset()
    yield
    SecretEncryption.reset()


@pytest.fixture()
def valid_key() -> str:
    """Generate a valid Fernet key for testing."""
    return Fernet.generate_key().decode()


class TestSecretEncryptionInitialize:
    """Tests for SecretEncryption.initialize()."""

    def test_initialize_with_valid_key_returns_true(self, valid_key):
        """Valid Fernet key should enable encryption."""
        result = SecretEncryption.initialize(valid_key)

        assert result is True
        assert SecretEncryption.is_configured() is True
        assert SecretEncryption.is_initialized() is True

    def test_initialize_with_empty_key_returns_false(self):
        """Empty string key should disable encryption."""
        result = SecretEncryption.initialize("")

        assert result is False
        assert SecretEncryption.is_configured() is False
        assert SecretEncryption.is_initialized() is True

    def test_initialize_with_whitespace_key_returns_false(self):
        """Whitespace-only key should disable encryption."""
        result = SecretEncryption.initialize("   ")

        assert result is False
        assert SecretEncryption.is_configured() is False

    def test_initialize_with_none_returns_false(self):
        """None key should disable encryption."""
        result = SecretEncryption.initialize(None)

        assert result is False
        assert SecretEncryption.is_configured() is False
        assert SecretEncryption.is_initialized() is True

    def test_initialize_with_invalid_key_raises_valueerror(self):
        """Invalid key format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Fernet key"):
            SecretEncryption.initialize("not-a-valid-fernet-key")

    def test_initialize_strips_whitespace_from_key(self, valid_key):
        """Whitespace around valid key should be stripped."""
        padded_key = f"  {valid_key}  "

        result = SecretEncryption.initialize(padded_key)

        assert result is True
        assert SecretEncryption.is_configured() is True


class TestSecretEncryptionState:
    """Tests for state check methods."""

    def test_is_configured_true_after_valid_init(self, valid_key):
        """is_configured() returns True when encryption is enabled."""
        SecretEncryption.initialize(valid_key)

        assert SecretEncryption.is_configured() is True

    def test_is_configured_false_after_empty_key(self):
        """is_configured() returns False when encryption is disabled."""
        SecretEncryption.initialize("")

        assert SecretEncryption.is_configured() is False

    def test_is_configured_false_before_initialization(self):
        """is_configured() returns False before initialize() is called."""
        assert SecretEncryption.is_configured() is False

    def test_is_initialized_true_even_when_disabled(self):
        """is_initialized() returns True even when encryption is disabled."""
        SecretEncryption.initialize("")

        assert SecretEncryption.is_initialized() is True
        assert SecretEncryption.is_configured() is False

    def test_is_initialized_false_before_init(self):
        """is_initialized() returns False before initialize() is called."""
        assert SecretEncryption.is_initialized() is False

    def test_get_key_id_returns_hex_string(self, valid_key):
        """get_key_id() returns 16-char hex fingerprint of key."""
        SecretEncryption.initialize(valid_key)

        key_id = SecretEncryption.get_key_id()

        assert len(key_id) == 16
        assert all(c in "0123456789abcdef" for c in key_id)

    def test_get_key_id_returns_none_when_disabled(self):
        """get_key_id() returns 'none' when encryption is disabled."""
        SecretEncryption.initialize("")

        assert SecretEncryption.get_key_id() == "none"

    def test_get_key_id_deterministic(self, valid_key):
        """Same key should produce same key_id."""
        SecretEncryption.initialize(valid_key)
        key_id_1 = SecretEncryption.get_key_id()

        SecretEncryption.reset()
        SecretEncryption.initialize(valid_key)
        key_id_2 = SecretEncryption.get_key_id()

        assert key_id_1 == key_id_2


class TestSecretEncryptionEncrypt:
    """Tests for SecretEncryption.encrypt()."""

    def test_encrypt_returns_bytes(self, valid_key):
        """encrypt() returns bytes ciphertext."""
        SecretEncryption.initialize(valid_key)

        result = SecretEncryption.encrypt("my-secret")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encrypt_raises_when_not_configured(self):
        """encrypt() raises EncryptionNotConfiguredError when disabled."""
        SecretEncryption.initialize("")

        with pytest.raises(EncryptionNotConfiguredError):
            SecretEncryption.encrypt("my-secret")

    def test_encrypt_raises_when_not_initialized(self):
        """encrypt() raises EncryptionNotConfiguredError before init."""
        with pytest.raises(EncryptionNotConfiguredError):
            SecretEncryption.encrypt("my-secret")

    def test_encrypt_different_outputs_for_same_input(self, valid_key):
        """Fernet uses random IV, so same input produces different ciphertext."""
        SecretEncryption.initialize(valid_key)

        ciphertext_1 = SecretEncryption.encrypt("same-secret")
        ciphertext_2 = SecretEncryption.encrypt("same-secret")

        # Same input should decrypt to same output
        assert SecretEncryption.decrypt(ciphertext_1) == SecretEncryption.decrypt(ciphertext_2)
        # But ciphertexts should differ due to random IV
        assert ciphertext_1 != ciphertext_2

    def test_encrypt_handles_unicode(self, valid_key):
        """encrypt() handles unicode strings correctly."""
        SecretEncryption.initialize(valid_key)

        secret = "password with émojis! 🔐"
        ciphertext = SecretEncryption.encrypt(secret)
        decrypted = SecretEncryption.decrypt(ciphertext)

        assert decrypted == secret

    def test_encrypt_handles_empty_string(self, valid_key):
        """encrypt() handles empty string."""
        SecretEncryption.initialize(valid_key)

        ciphertext = SecretEncryption.encrypt("")
        decrypted = SecretEncryption.decrypt(ciphertext)

        assert decrypted == ""


class TestSecretEncryptionDecrypt:
    """Tests for SecretEncryption.decrypt()."""

    def test_decrypt_returns_original_plaintext(self, valid_key):
        """decrypt() returns original plaintext after encryption."""
        SecretEncryption.initialize(valid_key)
        original = "my-secret-password"

        ciphertext = SecretEncryption.encrypt(original)
        decrypted = SecretEncryption.decrypt(ciphertext)

        assert decrypted == original

    def test_decrypt_raises_when_not_configured(self):
        """decrypt() raises EncryptionNotConfiguredError when disabled."""
        SecretEncryption.initialize("")

        with pytest.raises(EncryptionNotConfiguredError):
            SecretEncryption.decrypt(b"some-ciphertext")

    def test_decrypt_raises_when_not_initialized(self):
        """decrypt() raises EncryptionNotConfiguredError before init."""
        with pytest.raises(EncryptionNotConfiguredError):
            SecretEncryption.decrypt(b"some-ciphertext")

    def test_decrypt_raises_on_invalid_token(self, valid_key):
        """decrypt() raises DecryptionError on corrupted ciphertext."""
        SecretEncryption.initialize(valid_key)

        with pytest.raises(DecryptionError, match="encryption key may have changed"):
            SecretEncryption.decrypt(b"not-valid-fernet-token")

    def test_decrypt_raises_on_wrong_key(self, valid_key):
        """decrypt() raises DecryptionError when using different key."""
        SecretEncryption.initialize(valid_key)
        ciphertext = SecretEncryption.encrypt("my-secret")

        # Reinitialize with different key
        different_key = Fernet.generate_key().decode()
        SecretEncryption.reset()
        SecretEncryption.initialize(different_key)

        with pytest.raises(DecryptionError):
            SecretEncryption.decrypt(ciphertext)

    def test_decrypt_handles_long_secrets(self, valid_key):
        """decrypt() handles long secrets (e.g., SSH keys)."""
        SecretEncryption.initialize(valid_key)
        long_secret = "A" * 10000  # 10KB secret

        ciphertext = SecretEncryption.encrypt(long_secret)
        decrypted = SecretEncryption.decrypt(ciphertext)

        assert decrypted == long_secret


class TestSecretEncryptionReset:
    """Tests for SecretEncryption.reset()."""

    def test_reset_clears_state(self, valid_key):
        """reset() clears all state."""
        SecretEncryption.initialize(valid_key)
        assert SecretEncryption.is_configured() is True
        assert SecretEncryption.is_initialized() is True

        SecretEncryption.reset()

        assert SecretEncryption.is_configured() is False
        assert SecretEncryption.is_initialized() is False
        assert SecretEncryption.get_key_id() == "none"

    def test_reset_allows_reinitialization(self, valid_key):
        """After reset(), can initialize with new key."""
        SecretEncryption.initialize(valid_key)
        key_id_1 = SecretEncryption.get_key_id()

        SecretEncryption.reset()

        new_key = Fernet.generate_key().decode()
        SecretEncryption.initialize(new_key)
        key_id_2 = SecretEncryption.get_key_id()

        assert key_id_1 != key_id_2
        assert SecretEncryption.is_configured() is True


class TestGenerateFernetKey:
    """Tests for generate_fernet_key()."""

    def test_generate_fernet_key_returns_valid_key(self):
        """generate_fernet_key() returns a valid Fernet key."""
        key = generate_fernet_key()

        # Should be a string
        assert isinstance(key, str)

        # Should be usable with SecretEncryption
        result = SecretEncryption.initialize(key)
        assert result is True

        # Should work for encrypt/decrypt
        ciphertext = SecretEncryption.encrypt("test")
        assert SecretEncryption.decrypt(ciphertext) == "test"

    def test_generate_fernet_key_unique_each_call(self):
        """generate_fernet_key() produces unique keys each call."""
        keys = [generate_fernet_key() for _ in range(10)]

        assert len(set(keys)) == 10  # All unique

    def test_generate_fernet_key_correct_length(self):
        """generate_fernet_key() returns 44-character base64 key."""
        key = generate_fernet_key()

        assert len(key) == 44
        assert key.endswith("=")  # Base64 padding
