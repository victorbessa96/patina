#!/usr/bin/env python3
"""Neutral-studio equirect for umber's IBL default (wave-4 item 6).

Generates assets/env/studio-neutral.png: a 1024x512 LDR equirect with a
soft neutral overcast gradient (bright zenith, mid horizon, dark ground),
a slight warm key light and a cool fill — no licensed content, fully
procedural.

PyPNG-style dependency-free: the PNG encoder below is ~25 lines of stdlib
(struct + zlib). Fully deterministic (pure math, no RNG): re-running
produces byte-identical output.

Layout convention (see umber-gpu/src/ibl.rs): the PNG is stored
viewer-conventional — displayed TOP row = +Y zenith. umber's PNG loader
flips rows into GPU order on read, so the top row here lands on the last
data row (v=1 = +Y). Do NOT "fix" the orientation: it is load-bearing.
"""

import math
import os
import struct
import zlib

WIDTH = 1024
HEIGHT = 512

# Gradient stops in sin(elevation): (sin_e, (r, g, b)) linear-light 0..1.
ZENITH = (0.82, 0.85, 0.92)  # soft top light, neutral-cool
HORIZON = (0.55, 0.55, 0.56)  # neutral mid
NADIR = (0.16, 0.14, 0.13)  # dark warm-gray ground

KEY_AZIMUTH = 0.30  # fraction of full turn
KEY_ELEVATION = math.radians(35.0)
KEY_COLOR = (0.55, 0.38, 0.22)
KEY_STRENGTH = 0.55
KEY_WIDTH = 0.45  # radians (gaussian sigma)

FILL_AZIMUTH = 0.80
FILL_ELEVATION = math.radians(10.0)
FILL_COLOR = (0.30, 0.38, 0.52)
FILL_STRENGTH = 0.30
FILL_WIDTH = 0.70


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def angular_distance(az1, el1, az2, el2):
    # Great-circle distance via the spherical law of cosines, clamped
    # against rounding at coincident points.
    c = (
        math.sin(el1) * math.sin(el2)
        + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    )
    return math.acos(max(-1.0, min(1.0, c)))


def pixel(u, v):
    """u, v in [0, 1): u = longitude fraction, v = 0 at displayed top."""
    az = u * 2.0 * math.pi
    el = (0.5 - v) * math.pi  # displayed top = +pi/2 zenith
    s = math.sin(el)
    if s >= 0.0:
        base = lerp(HORIZON, ZENITH, s)
    else:
        base = lerp(HORIZON, NADIR, -s)
    key = KEY_STRENGTH * math.exp(
        -((angular_distance(az, el, KEY_AZIMUTH * 2 * math.pi, KEY_ELEVATION)) / KEY_WIDTH) ** 2
    )
    fill = FILL_STRENGTH * math.exp(
        -((angular_distance(az, el, FILL_AZIMUTH * 2 * math.pi, FILL_ELEVATION)) / FILL_WIDTH) ** 2
    )
    rgb = tuple(
        min(1.0, max(0.0, b + KEY_COLOR[i] * key + FILL_COLOR[i] * fill))
        for i, b in enumerate(base)
    )
    return rgb


def write_png(path, width, height, rows):
    def chunk(ctype, data):
        out = struct.pack(">I", len(data)) + ctype + data
        out += struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        return out

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def main():
    rows = []
    for y in range(HEIGHT):
        v = (y + 0.5) / HEIGHT
        row = []
        for x in range(WIDTH):
            u = (x + 0.5) / WIDTH
            r, g, b = pixel(u, v)
            row += [round(r * 255), round(g * 255), round(b * 255), 255]
        rows.append(row)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "env", "studio-neutral.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_png(out, WIDTH, HEIGHT, rows)
    print(f"wrote {out} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
