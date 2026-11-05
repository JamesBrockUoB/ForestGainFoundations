"""Drop-in replacement for streamlit-drawable-canvas with a real paint-bucket tool.

Frontend: paint_canvas_frontend/index.html (plain JS, no build step).
Keep this file next to the Streamlit app.
"""

import base64
import io
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

PAINT_RES = 768  # must match RES in frontend

_component = components.declare_component(
    "paint_canvas",
    path=str(Path(__file__).parent / "paint_canvas_frontend"),
)


def _to_data_url(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def paint_canvas(
    background,
    display_px,
    tool,
    color,
    brush,
    token,
    initial_paint=None,
    key=None,
    layer="paint",
):
    """Render the canvas. Returns {paint: PNG data URL, n: int} after each edit, else None.

    layer: "paint" | "dw"
    color should be green for paint and yellow for DW, but the frontend decides from layer.
    """
    return _component(
        background=_to_data_url(background),
        display_px=int(display_px),
        tool=tool,
        color=color,
        brush=int(brush),
        token=str(token),
        initial_paint=initial_paint,
        key=key,
        layer=layer,
        default=None,
    )


def decode_paint(data_url):
    """PNG data URL -> (PAINT_RES, PAINT_RES, 4) uint8 RGBA array, or None."""
    if not data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    rgba = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
    if rgba.shape[:2] != (PAINT_RES, PAINT_RES):
        raise ValueError(
            f"Canvas image is {rgba.shape[1]}×{rgba.shape[0]}; "
            f"expected {PAINT_RES}x{PAINT_RES}."
        )
    return rgba


def paint_masks(rgba):
    """Split green and yellow canvas pixels by nearest configured color."""
    r, g, b, a = (rgba[..., i].astype(np.int32) for i in range(4))
    painted = a > 0

    green_distance = r**2 + (g - 200) ** 2 + b**2
    yellow_distance = (r - 255) ** 2 + (g - 208) ** 2 + b**2

    green = painted & (green_distance <= yellow_distance)
    yellow = painted & (yellow_distance < green_distance)
    return green, yellow
