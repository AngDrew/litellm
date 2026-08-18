from pathlib import Path


def test_runtime_image_includes_configured_callback_source() -> None:
    assert "COPY --from=builder /app/callbacks /app/callbacks" in (
        Path(__file__).resolve().parents[2] / "Dockerfile"
    ).read_text(encoding="utf-8")
