"""
Decryption utilities for BrowseComp encrypted dataset.

Uses SHA256 key derivation + XOR cipher to decrypt questions and answers.
Each task has a unique 'canary' password for decryption.
"""

import hashlib
import base64


def derive_key(password: str, length: int) -> bytes:
    """
    Derive a key from password using SHA256, repeated to match length.

    Args:
        password: The canary/password string from the CSV
        length: Required key length (matches ciphertext length)

    Returns:
        Key bytes repeated/truncated to match length
    """
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    # Repeat key to match length: key * (full_repeats) + key[remaining]
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """
    XOR decrypt base64-encoded ciphertext.

    Args:
        ciphertext_b64: Base64-encoded encrypted data
        password: Password for decryption

    Returns:
        Decrypted plaintext string

    Raises:
        ValueError: If base64 decoding fails
    """
    try:
        encrypted = base64.b64decode(ciphertext_b64)
    except Exception as e:
        raise ValueError(f"Failed to decode base64: {e}")

    # Derive key of same length as ciphertext
    key = derive_key(password, len(encrypted))

    # XOR decrypt
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))

    # Decode with error replacement for robustness
    return decrypted.decode('utf-8', errors='replace')


def decrypt_task(problem_b64: str, answer_b64: str, canary: str) -> tuple[str, str]:
    """
    Decrypt a complete task (problem + answer).

    Args:
        problem_b64: Base64-encoded encrypted problem
        answer_b64: Base64-encoded encrypted answer
        canary: Password for decryption

    Returns:
        Tuple of (decrypted_problem, decrypted_answer)

    Example:
        >>> problem, answer = decrypt_task(
        ...     "YUZcsui4ccp0...",
        ...     "ERFEy6PzLg==",
        ...     "BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT ONLINE. canary GUID browsecomp:26b5c67b..."
        ... )
        >>> print(problem)
        "What was the name of..."
    """
    problem = decrypt(problem_b64, canary)
    answer = decrypt(answer_b64, canary)
    return problem, answer
