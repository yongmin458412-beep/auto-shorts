#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
        return
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def _ensure_dependencies() -> None:
    _ensure_package("PIL", "pillow")
    _ensure_package("requests", "requests")


_ensure_dependencies()

import requests
from PIL import Image, ImageDraw, ImageFont, ImageStat


# Canvas/Layout constants
WIDTH = 864
HEIGHT = 1536
CENTER_X = 432

TITLE_POS = (432, 255)
IMAGE_CENTER = (432, 800)
SUBTITLE_POS = (432, 1250)

MAX_TEXT_WIDTH = 760
MAX_IMAGE_HEIGHT = 850

TITLE_FONT_SIZE = 90
SUBTITLE_FONT_SIZE = 60

STROKE_WIDTH = 4
STROKE_FILL = "black"

UI_SAFE_BOTTOM = HEIGHT - 150
MIN_GAP_IMAGE_SUBTITLE = 40

# Font paths / URLs
SCRIPT_DIR = Path(__file__).resolve().parent
FONT_DIR = SCRIPT_DIR / "fonts"
TITLE_FONT_PATH = FONT_DIR / "Pretendard-Black.otf"
JP_TITLE_FONT_PATH = FONT_DIR / "NotoSansJP-Black.otf"

PRETENDARD_BLACK_URL = (
    "https://github.com/orioncactus/pretendard/releases/latest/download/Pretendard-Black.otf"
)
NOTO_SANS_JP_BLACK_URL = (
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansJP-Black.otf"
)
PRETENDARD_BLACK_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/orioncactus/pretendard/main/packages/pretendard/dist/public/static/Pretendard-Black.otf",
]
NOTO_SANS_JP_BLACK_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Black.otf",
]


def ensure_fonts() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    _download_if_missing(PRETENDARD_BLACK_URL, TITLE_FONT_PATH, PRETENDARD_BLACK_FALLBACK_URLS)
    _download_if_missing(NOTO_SANS_JP_BLACK_URL, JP_TITLE_FONT_PATH, NOTO_SANS_JP_BLACK_FALLBACK_URLS)


def _download_if_missing(url: str, target_path: Path, fallbacks: List[str] | None = None) -> None:
    if target_path.exists() and target_path.stat().st_size > 10_000:
        return
    urls = [url] + list(fallbacks or [])
    last_error = ""
    for candidate in urls:
        try:
            response = requests.get(candidate, timeout=90, stream=True)
            response.raise_for_status()
            with target_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        file.write(chunk)
            if target_path.exists() and target_path.stat().st_size > 10_000:
                return
        except Exception as exc:
            last_error = str(exc)
            continue
    raise RuntimeError(f"폰트 다운로드 실패: {target_path.name} ({last_error or 'unknown error'})")


def _contains_japanese(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def _pick_text_color(image: Image.Image, center: Tuple[int, int], sample_w: int = 320, sample_h: int = 180) -> str:
    x, y = center
    left = max(0, x - sample_w // 2)
    top = max(0, y - sample_h // 2)
    right = min(image.width, left + sample_w)
    bottom = min(image.height, top + sample_h)
    patch = image.crop((left, top, right, bottom)).convert("RGB")
    stat = ImageStat.Stat(patch)
    r, g, b = stat.mean
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 160 else "white"


def _wrap_text_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return [""]

    lines: List[str] = []
    for paragraph in text.splitlines() or [text]:
        p = paragraph.strip()
        if not p:
            continue

        if " " in p:
            words = p.split(" ")
            cur = ""
            for word in words:
                test = (cur + " " + word).strip()
                w = draw.textlength(test, font=font)
                if w <= max_width or not cur:
                    cur = test
                else:
                    lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
        else:
            cur = ""
            for ch in p:
                test = cur + ch
                w = draw.textlength(test, font=font)
                if w <= max_width or not cur:
                    cur = test
                else:
                    lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)

    return lines or [""]


def _fit_text_layout(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    initial_size: int,
    max_width: int,
    line_limit: int = 2,
) -> Tuple[ImageFont.FreeTypeFont, str]:
    size = max(10, int(initial_size))
    font = ImageFont.truetype(str(font_path), size=size)
    lines = _wrap_text_by_width(draw, text, font, max_width)

    if len(lines) > line_limit:
        size = max(10, size - 10)
        font = ImageFont.truetype(str(font_path), size=size)
        lines = _wrap_text_by_width(draw, text, font, max_width)

    if len(lines) > line_limit:
        lines = lines[:line_limit]
        # 마지막 줄 말줄임
        last = lines[-1]
        while last:
            probe = last + "…"
            if draw.textlength(probe, font=font) <= max_width:
                lines[-1] = probe
                break
            last = last[:-1].rstrip()
        if not last:
            lines[-1] = "…"

    return font, "\n".join(lines)


def _paste_center_image(canvas: Image.Image, image_path: Path) -> Tuple[int, int, int, int]:
    if (not image_path) or (not image_path.exists()) or (not image_path.is_file()):
        # Placeholder square
        side = 520
        x1 = IMAGE_CENTER[0] - side // 2
        y1 = IMAGE_CENTER[1] - side // 2
        x2 = x1 + side
        y2 = y1 + side
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=32, fill=(220, 220, 220), outline=(140, 140, 140), width=4)
        draw.text((IMAGE_CENTER[0], IMAGE_CENTER[1]), "NO IMAGE", fill="#333333", anchor="mm")
        return (x1, y1, x2, y2)

    with Image.open(image_path) as src:
        image = src.convert("RGBA")

    # Keep aspect ratio, limit height
    scale = min(1.0, float(MAX_IMAGE_HEIGHT) / float(max(1, image.height)))
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))

    # Fit horizontally as well (so it stays visually centered and safe)
    if new_w > MAX_TEXT_WIDTH:
        scale2 = float(MAX_TEXT_WIDTH) / float(new_w)
        new_w = max(1, int(round(new_w * scale2)))
        new_h = max(1, int(round(new_h * scale2)))

    image = image.resize((new_w, new_h), Image.LANCZOS)

    x1 = IMAGE_CENTER[0] - new_w // 2
    y1 = IMAGE_CENTER[1] - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Keep inside canvas
    x1 = max(0, min(WIDTH - new_w, x1))
    y1 = max(0, min(HEIGHT - new_h, y1))
    x2 = x1 + new_w
    y2 = y1 + new_h

    canvas.alpha_composite(image, (x1, y1))
    return (x1, y1, x2, y2)


def render_layout(
    title_text: str,
    subtitle_text: str,
    image_path: str,
    output_path: str,
    background_color: str = "#F3F1EA",
) -> Path:
    ensure_fonts()

    canvas = Image.new("RGBA", (WIDTH, HEIGHT), background_color)
    draw = ImageDraw.Draw(canvas)

    # Title font: Korean default, Japanese fallback by content
    title_font_path = JP_TITLE_FONT_PATH if _contains_japanese(title_text) else TITLE_FONT_PATH
    subtitle_font_path = JP_TITLE_FONT_PATH if _contains_japanese(subtitle_text) else TITLE_FONT_PATH

    title_font, title_wrapped = _fit_text_layout(
        draw,
        text=title_text,
        font_path=title_font_path,
        initial_size=TITLE_FONT_SIZE,
        max_width=MAX_TEXT_WIDTH,
        line_limit=2,
    )

    # Render centered title (anchor="mm")
    title_color = _pick_text_color(canvas, TITLE_POS)
    draw.multiline_text(
        TITLE_POS,
        title_wrapped,
        font=title_font,
        fill=title_color,
        anchor="mm",
        align="center",
        spacing=max(8, int(title_font.size * 0.18)),
        stroke_width=STROKE_WIDTH,
        stroke_fill=STROKE_FILL,
    )

    # Center image
    image_bbox = _paste_center_image(canvas, Path(image_path) if image_path else Path(""))

    # Subtitle layout and spacing rule (>= 40px below image)
    subtitle_font, subtitle_wrapped = _fit_text_layout(
        draw,
        text=subtitle_text,
        font_path=subtitle_font_path,
        initial_size=SUBTITLE_FONT_SIZE,
        max_width=MAX_TEXT_WIDTH,
        line_limit=2,
    )

    spacing = max(8, int(subtitle_font.size * 0.18))
    sub_bbox = draw.multiline_textbbox(
        SUBTITLE_POS,
        subtitle_wrapped,
        font=subtitle_font,
        anchor="mm",
        align="center",
        spacing=spacing,
        stroke_width=STROKE_WIDTH,
    )
    sub_h = max(1, sub_bbox[3] - sub_bbox[1])

    subtitle_y = SUBTITLE_POS[1]
    min_sub_top = image_bbox[3] + MIN_GAP_IMAGE_SUBTITLE
    if (subtitle_y - sub_h // 2) < min_sub_top:
        subtitle_y = min_sub_top + sub_h // 2

    # UI safe area
    if (subtitle_y + sub_h // 2) > UI_SAFE_BOTTOM:
        subtitle_y = UI_SAFE_BOTTOM - sub_h // 2

    subtitle_color = _pick_text_color(canvas, (CENTER_X, subtitle_y))
    draw.multiline_text(
        (CENTER_X, subtitle_y),
        subtitle_wrapped,
        font=subtitle_font,
        fill=subtitle_color,
        anchor="mm",
        align="center",
        spacing=spacing,
        stroke_width=STROKE_WIDTH,
        stroke_fill=STROKE_FILL,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, quality=95)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 864x1536 Shorts layout with auto font download.")
    parser.add_argument("--title", required=True, help="Title text")
    parser.add_argument("--subtitle", required=True, help="Subtitle text")
    parser.add_argument("--image", default="", help="Path to center image")
    parser.add_argument("--output", default="./output_layout.png", help="Output image path")
    parser.add_argument("--bg", default="#F3F1EA", help="Background color (hex)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = render_layout(
        title_text=args.title,
        subtitle_text=args.subtitle,
        image_path=args.image,
        output_path=args.output,
        background_color=args.bg,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
