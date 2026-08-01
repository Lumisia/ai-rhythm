"""내용 해시.

자산 식별과 프론트 무결성 검증이 전부 sha256 을 쓴다. 계산 방식이
한 곳에만 있어야 값이 갈리지 않는다.
"""

import hashlib
from pathlib import Path

_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """파일 내용의 sha256 을 소문자 16진수로 돌려준다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
