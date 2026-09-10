# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MultiModalResourceConnector file:// allowlist enforcement.

These tests exercise the ``file://`` local-media allowlist at the connector
level (``resource_connector._load_file_url``), independent of the HTTP layer.
The behavior is platform-agnostic: it only depends on ``pathlib``/``urllib``.

The security invariant under test is that a rejected ``file://`` reference is
**never passed to MediaIO** — rejection must happen before any I/O, not after.
Every reject test therefore asserts both the raised error and that the
recording MediaIO saw zero load_file calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    resolve_allowed_local_media_path,
    resolve_local_media_path,
)


class _RecordingMediaIO(MediaIO[Path]):
    """Fake MediaIO that records the file paths it is asked to load."""

    def __init__(self) -> None:
        self.loaded_paths: list[Path] = []

    def load_bytes(self, data: bytes) -> Path:
        return Path("bytes")

    def load_base64(self, media_type: str, data: str) -> Path:
        return Path("base64")

    def load_file(self, filepath: Path) -> Path:
        self.loaded_paths.append(filepath)
        return filepath


def _as_localhost_file_url(path: Path) -> str:
    """Rewrite ``file:///C:/...`` into the ``file://localhost/C:/...`` form."""
    assert path.as_uri().startswith("file://")
    return "file://localhost" + path.as_uri()[len("file://") :]


# ---------------------------------------------------------------------------
# resolve_allowed_local_media_path
# ---------------------------------------------------------------------------


def test_allowed_local_media_path_must_be_existing_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="must be a directory"):
        resolve_allowed_local_media_path(missing)

    plain_file = tmp_path / "not_a_dir"
    plain_file.write_text("x")
    with pytest.raises(ValueError, match="must be a directory"):
        resolve_allowed_local_media_path(plain_file)


def test_resolve_allowed_local_media_path_normalizes_redundant_components(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    doubled = tmp_path / "refs" / ".." / "refs"
    assert resolve_allowed_local_media_path(str(doubled)) == media_dir.resolve()


# ---------------------------------------------------------------------------
# _load_file_url allowlist enforcement
# ---------------------------------------------------------------------------


def test_load_file_disabled_without_allowlist(tmp_path: Path) -> None:
    connector = MultiModalResourceConnector()
    media_io = _RecordingMediaIO()
    audio = (
        tmp_path / "ref.wav"
    )  # need not exist: rejection happens before any path handling

    with pytest.raises(RuntimeError, match="Local file loading is disabled"):
        connector.load_resource(audio.as_uri(), media_io)

    assert media_io.loaded_paths == []


def test_load_file_inside_allowlist(tmp_path: Path) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    audio = media_dir / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=media_dir)
    media_io = _RecordingMediaIO()

    result = connector.load_resource(audio.as_uri(), media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_file_accepts_localhost_file_url(tmp_path: Path) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    audio = media_dir / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=media_dir)
    media_io = _RecordingMediaIO()

    result = connector.load_resource(_as_localhost_file_url(audio), media_io)

    assert result == audio.resolve()


def test_load_file_rejects_remote_file_netloc(tmp_path: Path) -> None:
    connector = MultiModalResourceConnector(allowed_local_media_path=tmp_path)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="netloc is not supported"):
        connector.load_resource("file://remotehost/reference.wav", media_io)

    assert media_io.loaded_paths == []


def test_load_file_rejects_path_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_resource(outside.as_uri(), media_io)

    assert media_io.loaded_paths == []


def test_load_file_rejects_traversal_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    # URL containing ".." that would resolve outside the allowlist.
    traversal_url = (allowed / ".." / "outside.wav").as_uri()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_resource(traversal_url, media_io)

    assert media_io.loaded_paths == []


def test_load_file_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    link = allowed / "escape.wav"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # Windows without developer mode cannot create symlinks.
        pytest.skip(f"symlink creation unsupported: {exc}")

    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_resource(link.as_uri(), media_io)

    assert media_io.loaded_paths == []


def test_load_file_passes_resolved_path_to_media_io(tmp_path: Path) -> None:
    """The connector resolves/normalizes the path; existence is media_io's job."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()
    missing = allowed / "ghost.wav"

    result = connector.load_resource(missing.as_uri(), media_io)

    assert media_io.loaded_paths == [missing.resolve()]
    assert result == missing.resolve()


# ---------------------------------------------------------------------------
# load_resource dispatch (data / unsupported schemes)
# ---------------------------------------------------------------------------


def test_load_resource_data_url_passes_to_base64() -> None:
    connector = MultiModalResourceConnector()
    media_io = _RecordingMediaIO()

    result = connector.load_resource("data:audio/wav;base64,QUJDRA==", media_io)

    assert result == Path("base64")


def test_load_resource_unsupported_scheme() -> None:
    connector = MultiModalResourceConnector()

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        connector.load_resource("ftp://example.com/reference.wav", _RecordingMediaIO())


# ---------------------------------------------------------------------------
# Bare local paths (load_local_path / resolve_local_media_path)
# ---------------------------------------------------------------------------
#
# Bare paths are a trusted-local convenience: they stay allowed when no
# allowlist is configured (backwards compatible), but once an allowlist IS
# configured they must resolve inside it -- the same policy that ``file://``
# enforces. Rejection must happen before MediaIO sees the path.


def test_resolve_local_media_path_allowed_without_allowlist(tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"  # existence is media_io's concern, not policy's
    resolved = resolve_local_media_path(str(audio), allowed_local_media_path=None)

    assert resolved == audio.resolve()


def test_load_local_path_allowed_without_allowlist(tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector()
    media_io = _RecordingMediaIO()

    result = connector.load_local_path(audio, media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_local_path_inside_allowlist(tmp_path: Path) -> None:
    media_dir = tmp_path / "refs"
    media_dir.mkdir()
    audio = media_dir / "ref.wav"
    audio.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=media_dir)
    media_io = _RecordingMediaIO()

    result = connector.load_local_path(audio, media_io)

    assert result == audio.resolve()
    assert media_io.loaded_paths == [audio.resolve()]


def test_load_local_path_rejects_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(outside, media_io)

    assert media_io.loaded_paths == []


def test_load_local_path_rejects_traversal_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(allowed / ".." / "outside.wav", media_io)

    assert media_io.loaded_paths == []


def test_load_local_path_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    link = allowed / "escape.wav"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # Windows without developer mode cannot create symlinks.
        pytest.skip(f"symlink creation unsupported: {exc}")

    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    media_io = _RecordingMediaIO()

    with pytest.raises(ValueError, match="not within allowed directory"):
        connector.load_local_path(link, media_io)

    assert media_io.loaded_paths == []
