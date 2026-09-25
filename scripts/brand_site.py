"""Remove the marimo branding that `marimo export html-wasm` writes and marimo cannot configure.

marimo offers three hooks for an exported app: `app_title`, `css_file`, and `html_head_file`.
`app.py` uses all three. The export still writes a default description, two web manifests, and
the marimo icons. marimo 0.24.2 ignores `[tool.marimo.opengraph]` in a WASM export. This script
replaces what is left.

Run it from the repository root after each export:

    uv run python scripts/brand_site.py dist

The script stops with an error when the export does not look as expected. A marimo upgrade that
changes the template then fails the build and does not publish a site that still says marimo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

APP_NAME = "Simple Topic Modeling"
"""The product name."""

SHORT_NAME = "Topic Modeling"
"""The name under an icon on a home screen."""

DESCRIPTION = (
    "Simple Topic Modeling finds the themes in a collection of documents. "
    "It runs fully in your browser, so your texts never leave your computer."
)
"""The one-line description. `web/head.html` holds the same text."""

SOURCE_URL = "https://github.com/maehr/simple-topic-modeling"
"""The source repository."""

BRAND_COLOR = "#0f766e"
"""The icon background and the theme colour of the manifests."""

DEFAULT_DESCRIPTION = '<meta name="description" content="a marimo app" />'
"""The description that marimo writes into every export."""

ICONS: dict[str, int] = {
    "favicon-16x16.png": 16,
    "favicon-32x32.png": 32,
    "apple-touch-icon.png": 180,
    "android-chrome-192x192.png": 192,
    "android-chrome-512x512.png": 512,
    "logo.png": 512,
}
"""Each marimo icon file in the export, and its size in pixels."""

MANIFESTS = ("manifest.json", "site.webmanifest")
"""The two web manifests in the export."""


def strip_default_description(html: str) -> str:
    """Remove the marimo default description, so the one from `web/head.html` is the only one.

    Example:
        >>> strip_default_description(
        ...     '<head><meta name="description" content="a marimo app" /></head>'
        ... )
        '<head></head>'
        >>> strip_default_description("<head></head>")
        Traceback (most recent call last):
        ValueError: The export has no marimo default description. Check the marimo template.
    """
    if DEFAULT_DESCRIPTION not in html:
        raise ValueError("The export has no marimo default description. Check the marimo template.")
    return html.replace(DEFAULT_DESCRIPTION, "")


def set_theme_color(html: str) -> str:
    """Replace the marimo theme colour with the brand colour.

    Example:
        >>> set_theme_color('<meta name="theme-color" content="#000000" />')
        '<meta name="theme-color" content="#0f766e" />'
    """
    return html.replace(
        '<meta name="theme-color" content="#000000" />',
        f'<meta name="theme-color" content="{BRAND_COLOR}" />',
    )


def add_noscript(html: str) -> str:
    """Insert a text summary after `<body>` for crawlers and readers without JavaScript.

    Example:
        >>> page = add_noscript("<body><div id='root'></div></body>")
        >>> page.index("<body>") < page.index("<noscript>") < page.index("<div")
        True
        >>> "WebAssembly" in page
        True
        >>> add_noscript("<html></html>")
        Traceback (most recent call last):
        ValueError: The export has no <body> tag.
    """
    if "<body>" not in html:
        raise ValueError("The export has no <body> tag.")
    block = (
        "<noscript>"
        f"<h1>{APP_NAME}</h1>"
        f"<p>{DESCRIPTION}</p>"
        "<p>This app needs JavaScript and WebAssembly. Turn on JavaScript and reload the page.</p>"
        f'<p><a href="{SOURCE_URL}">Read the source code on GitHub.</a></p>'
        "</noscript>"
    )
    return html.replace("<body>", f"<body>\n{block}", 1)


def rewrite_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a web manifest with the app name, description, and colour.

    Example:
        >>> rewrite_manifest({"name": "marimo", "short_name": "marimo", "icons": []})
        {'name': 'Simple Topic Modeling', 'short_name': 'Topic Modeling', 'icons': [], \
'description': 'Simple Topic Modeling finds the themes in a collection of documents. \
It runs fully in your browser, so your texts never leave your computer.', \
'theme_color': '#0f766e'}
    """
    return {
        **data,
        "name": APP_NAME,
        "short_name": SHORT_NAME,
        "description": DESCRIPTION,
        "theme_color": BRAND_COLOR,
    }


def draw_icon(size: int) -> Image.Image:
    """Draw the app icon: three bars of different lengths, like the weights of three topics.

    The icon is drawn at 512 pixels and then scaled, so each size has the same shape.

    Example:
        >>> icon = draw_icon(32)
        >>> icon.size, icon.mode
        ((32, 32), 'RGBA')
    """
    base = 512
    image = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, base - 1, base - 1), radius=112, fill=BRAND_COLOR)
    left, height, gap = 104, 64, 44
    top = (base - 3 * height - 2 * gap) // 2
    for index, length in enumerate((304, 224, 144)):
        y = top + index * (height + gap)
        draw.rounded_rectangle((left, y, left + length, y + height), radius=32, fill="white")
    if size == base:
        return image
    return image.resize((size, size), Image.Resampling.LANCZOS)


def brand(site: Path) -> None:
    """Rewrite the page, the manifests, and the icons in an exported site."""
    index = site / "index.html"
    html = index.read_text(encoding="utf-8")
    index.write_text(
        add_noscript(set_theme_color(strip_default_description(html))), encoding="utf-8"
    )

    for name in MANIFESTS:
        path = site / name
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(rewrite_manifest(data), indent=2) + "\n", encoding="utf-8")

    for name, size in ICONS.items():
        if not (site / name).exists():
            raise FileNotFoundError(f"The export has no {name}. Check the marimo template.")
        draw_icon(size).save(site / name)
    draw_icon(512).save(site / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])

    # The export copies marimo's own prompt for coding assistants. It does not belong on the site.
    (site / "CLAUDE.md").unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    """Brand the site in the directory that the first argument names."""
    if len(argv) != 2:
        print("Usage: python scripts/brand_site.py SITE_DIR", file=sys.stderr)
        return 2
    brand(Path(argv[1]))
    print(f"Branded the site in {argv[1]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
