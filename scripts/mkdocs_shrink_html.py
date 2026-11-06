# Copyright 2026 Marvin van Aalst, Anna Matuszyńska

"""mkdocs hook that keeps the notebook pages small (see `hooks:` in mkdocs.yml).

mkdocs-jupyter renders every notebook as a standalone page: it inlines the same
~580 kB of JupyterLab CSS into each one and embeds plots as base64 PNGs. Across 28
notebooks that was 16 MB of duplicated CSS and 4 MB of PNGs per docs version, and
mike keeps a full copy of every version on gh-pages, which has to stay under
GitHub Pages' 1 GB limit.

- Large inline <style> blocks are written once to `assets/jupyter-<hash>.css` and
  replaced by a <link> to it, in the same place so the cascade is unchanged.
- Embedded PNGs are re-encoded as lossless WebP (same pixels, ~60% smaller) when
  that is smaller.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# Smaller <style> blocks (page-specific tweaks) stay inline.
_MIN_SHARED_CSS = 50_000
_STYLE = re.compile(r"<style([^>]*)>(.*?)</style>", re.DOTALL)
_PNG = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")

_css: dict[str, str] = {}


def _to_webp(match: re.Match[str]) -> str:
    png = base64.b64decode(match.group(1))
    try:
        image = Image.open(io.BytesIO(png))
        buf = io.BytesIO()
        image.save(buf, format="WEBP", lossless=True, method=6)
    except OSError:  # not a PNG Pillow can read: leave it as it is
        return match.group(0)
    webp = buf.getvalue()
    if len(webp) >= len(png):
        return match.group(0)
    return "data:image/webp;base64," + base64.b64encode(webp).decode("ascii")


def on_pre_build(config: MkDocsConfig) -> None:  # noqa: ARG001
    """Forget the stylesheets of a previous build (mkdocs serve rebuilds)."""
    _css.clear()


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:  # noqa: ARG001
    """Move large inline CSS to shared files and shrink embedded PNGs."""
    root = "../" * page.url.count("/")

    def share(match: re.Match[str]) -> str:
        attrs, css = match.group(1), match.group(2)
        if len(css) < _MIN_SHARED_CSS or "media=" in attrs:
            return match.group(0)
        name = f"assets/jupyter-{hashlib.sha256(css.encode()).hexdigest()[:12]}.css"
        _css[name] = css
        return f'<link rel="stylesheet" href="{root}{name}">'

    return _PNG.sub(_to_webp, _STYLE.sub(share, output))


def on_post_build(config: MkDocsConfig, **_: Any) -> None:
    """Write the shared stylesheets."""
    for name, css in _css.items():
        path = Path(config.site_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(css, encoding="utf-8")
