from __future__ import annotations

import json
import base64
import os
import random
import re
import textwrap
import time
from html import unescape
from urllib.parse import urljoin, urlencode
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup

import gspread
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

MOVIEPY_AVAILABLE = True
MOVIEPY_ERROR = ""
try:
    import numpy as np
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        concatenate_videoclips,
        vfx,
    )
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except Exception as exc:
    MOVIEPY_AVAILABLE = False
    MOVIEPY_ERROR = str(exc)
    np = None
    AudioFileClip = CompositeAudioClip = ImageClip = concatenate_videoclips = vfx = None
    Image = ImageDraw = ImageFilter = ImageFont = None


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    value = _get_secret(key)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _get_list(key: str) -> List[str]:
    value = _get_secret(key, "")
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_json(key: str) -> Optional[Dict[str, Any]]:
    value = _get_secret(key)
    if not value:
        b64_value = _get_secret(f"{key}_B64", "")
        if b64_value:
            try:
                decoded = base64.b64decode(b64_value).decode("utf-8")
                return json.loads(decoded)
            except Exception:
                return None
        return None
    try:
        return json.loads(value)
    except Exception:
        fixed = _fix_private_key_json(value)
        if fixed != value:
            try:
                return json.loads(fixed)
            except Exception:
                return None
        return None


def _fix_private_key_json(value: str) -> str:
    if "private_key" not in value:
        return value
    pattern = r'("private_key"\s*:\s*")(?P<key>-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----\s*)(?P<suffix>")'

    def repl(match: re.Match) -> str:
        key = match.group("key")
        if "\\n" in key:
            return match.group(0)
        key_fixed = key.replace("\r\n", "\n").replace("\n", "\\n")
        return f'{match.group(1)}{key_fixed}{match.group("suffix")}'

    return re.sub(pattern, repl, value, flags=re.S)


def _get_query_param(key: str) -> str:
    try:
        params = st.query_params
        value = params.get(key)
        if isinstance(value, list):
            return value[0] if value else ""
        return value or ""
    except Exception:
        try:
            params = st.experimental_get_query_params()
            value = params.get(key, [""])[0]
            return value or ""
        except Exception:
            return ""


@dataclass
class AppConfig:
    openai_api_key: str
    openai_model: str
    elevenlabs_api_key: str
    elevenlabs_voice_ids: List[str]
    elevenlabs_model: str
    elevenlabs_stability: float
    elevenlabs_similarity: float
    sheet_id: str
    google_service_account_json: Optional[Dict[str, Any]]
    assets_dir: str
    manifest_path: str
    output_dir: str
    font_path: str
    width: int
    height: int
    fps: int
    enable_youtube_upload: bool
    youtube_client_id: str
    youtube_client_secret: str
    youtube_refresh_token: str
    youtube_privacy_status: str
    serpapi_api_key: str
    pexels_api_key: str
    ja_dialect_style: str
    telegram_bot_token: str
    telegram_admin_chat_id: str
    telegram_timeout_sec: int
    telegram_offset_path: str
    bboom_list_url: str
    bboom_max_fetch: int
    used_links_path: str
    trend_query: str
    trend_max_results: int
    approve_keywords: List[str]
    swap_keywords: List[str]


def load_config() -> AppConfig:
    assets_dir = _get_secret("ASSETS_DIR", "data/assets")
    manifest_path = _get_secret("MANIFEST_PATH", "data/manifests/assets.json")
    output_dir = _get_secret("OUTPUT_DIR", "data/output")
    return AppConfig(
        openai_api_key=_get_secret("OPENAI_API_KEY", "") or "",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
        elevenlabs_api_key=_get_secret("ELEVENLABS_API_KEY", "") or "",
        elevenlabs_voice_ids=_get_list("ELEVENLABS_VOICE_IDS")
        or ([voice for voice in [_get_secret("ELEVENLABS_VOICE_ID", "")] if voice]),
        elevenlabs_model=_get_secret("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        or "eleven_multilingual_v2",
        elevenlabs_stability=float(_get_secret("ELEVENLABS_STABILITY", "0.5") or 0.5),
        elevenlabs_similarity=float(_get_secret("ELEVENLABS_SIMILARITY", "0.8") or 0.8),
        sheet_id=_get_secret("SHEET_ID", "") or "",
        google_service_account_json=_get_json("GOOGLE_SERVICE_ACCOUNT_JSON"),
        assets_dir=assets_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        font_path=_get_secret("FONT_PATH", "") or "",
        width=int(_get_secret("VIDEO_WIDTH", "1080") or 1080),
        height=int(_get_secret("VIDEO_HEIGHT", "1920") or 1920),
        fps=int(_get_secret("VIDEO_FPS", "30") or 30),
        enable_youtube_upload=_get_bool("YOUTUBE_UPLOAD_ENABLED", False),
        youtube_client_id=_get_secret("YOUTUBE_CLIENT_ID", "") or "",
        youtube_client_secret=_get_secret("YOUTUBE_CLIENT_SECRET", "") or "",
        youtube_refresh_token=_get_secret("YOUTUBE_REFRESH_TOKEN", "") or "",
        youtube_privacy_status=_get_secret("YOUTUBE_PRIVACY_STATUS", "public") or "public",
        serpapi_api_key=_get_secret("SERPAPI_API_KEY", "") or "",
        pexels_api_key=_get_secret("PEXELS_API_KEY", "") or "",
        ja_dialect_style=_get_secret("JA_DIALECT_STYLE", "") or "",
        telegram_bot_token=_get_secret("TELEGRAM_BOT_TOKEN", "") or "",
        telegram_admin_chat_id=_get_secret("TELEGRAM_ADMIN_CHAT_ID", "") or "",
        telegram_timeout_sec=int(_get_secret("TELEGRAM_TIMEOUT_SEC", "600") or 600),
        telegram_offset_path=_get_secret("TELEGRAM_OFFSET_PATH", "data/state/telegram_offset.json")
        or "data/state/telegram_offset.json",
        bboom_list_url=_get_secret("BBOOM_LIST_URL", "https://m.bboom.naver.com/best")
        or "https://m.bboom.naver.com/best",
        bboom_max_fetch=int(_get_secret("BBOOM_MAX_FETCH", "30") or 30),
        used_links_path=_get_secret("USED_LINKS_PATH", "data/state/used_links.json")
        or "data/state/used_links.json",
        trend_query=_get_secret("TREND_QUERY", "日本 トレンド ハッシュタグ") or "日本 トレンド ハッシュタグ",
        trend_max_results=int(_get_secret("TREND_MAX_RESULTS", "8") or 8),
        approve_keywords=_get_list("APPROVE_KEYWORDS") or ["승인", "approve", "ok", "yes"],
        swap_keywords=_get_list("SWAP_KEYWORDS") or ["교환", "swap", "change", "next"],
    )


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def normalize_hashtags(tags: List[str]) -> List[str]:
    cleaned: List[str] = []
    for tag in tags:
        if not tag:
            continue
        tag = tag.strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        cleaned.append(tag)
    return cleaned


def generate_script(
    config: AppConfig,
    seed_text: str,
    language: str = "ja",
    beats_count: int = 7,
    allowed_tags: List[str] | None = None,
    trend_context: str = "",
    dialect_style: str = "",
) -> Dict[str, Any]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")
    allowed_tags = allowed_tags or [
        "shock",
        "laugh",
        "awkward",
        "angry",
        "cute",
        "facepalm",
        "wow",
        "plot",
        "ending",
    ]
    tag_list = ", ".join(allowed_tags)
    system_text = (
        "You are a short-form video scriptwriter. "
        "Return ONLY valid JSON with keys: "
        "title_ja, description_ja, hashtags_ja (array), beats (array). "
        "Each beat: {text, tag}. "
        "Keep beats punchy, 1 line each, no emojis in text."
    )
    style_line = "Style: Japanese, fast, comedic, meme-like. "
    if dialect_style:
        style_line += (
            f"Use {dialect_style} dialect for ALL Japanese text. "
            "Keep it friendly and natural, avoid offensive stereotypes. "
        )
    user_text = (
        f"Seed: {seed_text}\n"
        f"Language: {language}\n"
        f"Beats: {beats_count}\n"
        f"Allowed tags: {tag_list}\n"
        f"Trend context: {trend_context}\n"
        f"{style_line}"
        "Hook hard in the first 3 seconds and aim for a loop ending. "
        "Hashtags: 3-6 items, reflect recent JP trends if context provided. "
        "Keep title short and punchy. "
        "Length: 20-55 seconds total. "
        "Output JSON only."
    )
    client = OpenAI(api_key=config.openai_api_key)
    response = client.responses.create(
        model=config.openai_model,
        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    )
    output_text = getattr(response, "output_text", "") or ""
    result = extract_json(output_text)
    if not result:
        raise RuntimeError("Failed to parse JSON from model response")
    result["hashtags_ja"] = normalize_hashtags(result.get("hashtags_ja", []))
    return result


def pick_voice_id(voice_ids: List[str]) -> str:
    if not voice_ids:
        return ""
    return random.choice(voice_ids)


def tts_elevenlabs(
    config: AppConfig,
    text: str,
    output_path: str,
    voice_id: str,
) -> str:
    if not config.elevenlabs_api_key:
        raise RuntimeError("ELEVENLABS_API_KEY가 없습니다.")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID가 없습니다.")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": config.elevenlabs_api_key,
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": config.elevenlabs_model,
        "voice_settings": {
            "stability": config.elevenlabs_stability,
            "similarity_boost": config.elevenlabs_similarity,
        },
    }
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(response.content)
    return output_path


@dataclass
class AssetItem:
    asset_id: str
    path: str
    tags: List[str]
    kind: str


def ensure_dirs(paths: List[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_manifest(manifest_path: str) -> List[AssetItem]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    return [
        AssetItem(
            asset_id=item["asset_id"],
            path=item["path"],
            tags=item.get("tags", []),
            kind=item.get("kind", "image"),
        )
        for item in raw
    ]


def save_manifest(manifest_path: str, items: List[AssetItem]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump([item.__dict__ for item in items], file, ensure_ascii=False, indent=2)


def add_asset(
    manifest_path: str,
    asset_path: str,
    tags: List[str],
    kind: str = "image",
) -> AssetItem:
    items = load_manifest(manifest_path)
    asset_id = f"asset_{len(items)+1:04d}"
    new_item = AssetItem(asset_id=asset_id, path=asset_path, tags=tags, kind=kind)
    items.append(new_item)
    save_manifest(manifest_path, items)
    return new_item


def list_tags(items: List[AssetItem]) -> List[str]:
    tags: List[str] = []
    for item in items:
        for tag in item.tags:
            if tag not in tags:
                tags.append(tag)
    return tags


def filter_assets_by_tags(items: List[AssetItem], tags: List[str]) -> List[AssetItem]:
    if not tags:
        return items
    tag_set = set(tags)
    return [item for item in items if tag_set.intersection(set(item.tags))]


def pick_asset(items: List[AssetItem], tags: List[str]) -> Optional[AssetItem]:
    candidates = filter_assets_by_tags(items, tags)
    if not candidates:
        return random.choice(items) if items else None
    return random.choice(candidates)


def tags_from_text(text: str) -> List[str]:
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _fit_image_to_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    target_width, target_height = size
    original_width, original_height = image.size
    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def _make_background(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    background = image.resize(size, Image.LANCZOS)
    return background.filter(ImageFilter.GaussianBlur(radius=18))


def _draw_text_block(
    image: Image.Image,
    text: str,
    font_path: str,
    font_size: int,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
) -> Image.Image:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, font_size)
    max_width = box[2] - box[0]
    wrapped = textwrap.fill(text, width=18)
    lines = wrapped.split("\n")
    line_height = font.getbbox("Ag")[3] + 6
    total_height = line_height * len(lines)
    start_y = box[1] + max((box[3] - box[1] - total_height) // 2, 0)
    for line in lines:
        line_width = font.getbbox(line)[2]
        line_x = box[0] + max((max_width - line_width) // 2, 0)
        draw.text((line_x, start_y), line, font=font, fill=fill, stroke_width=4, stroke_fill=(0, 0, 0))
        start_y += line_height
    return image


def _compose_frame(
    asset_path: str,
    text: str,
    size: Tuple[int, int],
    font_path: str,
) -> Image.Image:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    base = Image.open(asset_path).convert("RGB")
    background = _make_background(base, size)
    foreground = _fit_image_to_canvas(base, size)
    composed = Image.alpha_composite(background.convert("RGBA"), foreground.convert("RGBA")).convert("RGB")
    width, height = size
    text_box = (80, int(height * 0.68), width - 80, height - 120)
    return _draw_text_block(composed, text, font_path, font_size=64, box=text_box, fill=(255, 255, 255))


def _estimate_durations(texts: List[str], total_duration: float) -> List[float]:
    lengths = [max(len(text), 1) for text in texts]
    total = sum(lengths)
    raw = [(length / total) * total_duration for length in lengths]
    minimum = 1.2
    adjusted = [max(duration, minimum) for duration in raw]
    scale = total_duration / sum(adjusted)
    return [duration * scale for duration in adjusted]


def render_video(
    config: AppConfig,
    asset_paths: List[str],
    texts: List[str],
    tts_audio_path: str,
    output_path: str,
    bgm_path: str | None = None,
    bgm_volume: float = 0.08,
) -> str:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    audio_clip = AudioFileClip(tts_audio_path)
    durations = _estimate_durations(texts, audio_clip.duration)
    clips: List[ImageClip] = []
    for index, text in enumerate(texts):
        asset_path = asset_paths[min(index, len(asset_paths) - 1)]
        frame_image = _compose_frame(asset_path, text, (config.width, config.height), config.font_path)
        frame_array = np.array(frame_image)
        clip = ImageClip(frame_array).set_duration(durations[index])
        clip = clip.fx(vfx.resize, lambda t: 1 + 0.03 * (t / max(durations[index], 0.1)))
        clips.append(clip)
    video = concatenate_videoclips(clips, method="compose").set_fps(config.fps)
    if bgm_path and os.path.exists(bgm_path):
        bgm_clip = AudioFileClip(bgm_path).volumex(bgm_volume)
        audio = CompositeAudioClip([audio_clip, bgm_clip])
    else:
        audio = audio_clip
    video = video.set_audio(audio)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=config.fps, threads=4)
    audio_clip.close()
    video.close()
    return output_path


def append_publish_log(config: AppConfig, row: Dict[str, str]) -> None:
    if not config.sheet_id:
        raise RuntimeError("SHEET_ID가 없습니다.")
    if not config.google_service_account_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON이 없습니다.")
    client = gspread.service_account_from_dict(config.google_service_account_json)
    sheet = client.open_by_key(config.sheet_id)
    try:
        worksheet = sheet.worksheet("publish_log")
    except Exception:
        worksheet = sheet.add_worksheet(title="publish_log", rows=1000, cols=20)
    headers = [
        "date_jst",
        "title_ja",
        "hashtags_ja",
        "template_id",
        "asset_ids",
        "voice_id",
        "video_path",
        "youtube_video_id",
        "youtube_url",
        "status",
        "error",
    ]
    existing = worksheet.row_values(1)
    if existing != headers:
        worksheet.update("A1", [headers])
    values: List[str] = [row.get(header, "") for header in headers]
    worksheet.append_row(values, value_input_option="USER_ENTERED")


def upload_video(
    config: AppConfig,
    file_path: str,
    title: str,
    description: str,
    tags: Optional[list] = None,
) -> Dict[str, str]:
    if not config.youtube_client_id or not config.youtube_client_secret or not config.youtube_refresh_token:
        raise RuntimeError("YouTube OAuth 설정이 누락되었습니다.")
    credentials = Credentials(
        token=None,
        refresh_token=config.youtube_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=config.youtube_client_id,
        client_secret=config.youtube_client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    youtube = build("youtube", "v3", credentials=credentials)
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": "24",
        },
        "status": {"privacyStatus": config.youtube_privacy_status},
    }
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    video_id = response.get("id", "")
    return {"video_id": video_id, "video_url": f"https://www.youtube.com/watch?v={video_id}"}


def build_google_oauth_url(
    client_id: str,
    redirect_uri: str,
    scope: str,
    prompt_consent: bool = True,
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "include_granted_scopes": "true",
    }
    if prompt_consent:
        params["prompt"] = "consent"
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def exchange_oauth_code_for_token(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> Dict[str, Any]:
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    response = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def collect_images_serpapi(
    query: str,
    api_key: str,
    output_dir: str,
    limit: int = 12,
) -> List[str]:
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY가 없습니다.")
    os.makedirs(output_dir, exist_ok=True)
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "ijn": 0,
        "num": min(limit, 20),
    }
    response = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
    response.raise_for_status()
    results = response.json().get("images_results", [])
    downloaded: List[str] = []
    for item in results[:limit]:
        image_url = item.get("original") or item.get("thumbnail")
        if not image_url:
            continue
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            filename = f"{random.randint(100000, 999999)}.jpg"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as file:
                file.write(image_response.content)
            downloaded.append(file_path)
        except Exception:
            continue
    return downloaded


def get_trend_context(config: AppConfig) -> str:
    if not config.serpapi_api_key:
        return ""
    try:
        params = {
            "engine": "google_news",
            "q": config.trend_query,
            "api_key": config.serpapi_api_key,
            "hl": "ja",
            "gl": "jp",
        }
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        response.raise_for_status()
        news = response.json().get("news_results", [])[: config.trend_max_results]
        titles = [item.get("title", "") for item in news if item.get("title")]
        if not titles:
            return ""
        return " / ".join(titles)
    except Exception:
        return ""


def generate_trend_queries(config: AppConfig, trend_context: str, count: int = 4) -> List[str]:
    if not trend_context or not config.openai_api_key:
        return []
    system_text = (
        "You generate short Japanese search queries for image search. "
        "Return JSON only. Format: {\"queries\": [\"...\"]}."
    )
    user_text = (
        f"Trend context (Japan): {trend_context}\n"
        f"Create {count} short Japanese image-search queries suitable for short-form content visuals. "
        "Avoid brand names if possible. Keep each query under 6 words."
    )
    try:
        client = OpenAI(api_key=config.openai_api_key)
        response = client.responses.create(
            model=config.openai_model,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        output_text = getattr(response, "output_text", "") or ""
        result = extract_json(output_text)
        queries = result.get("queries", []) if isinstance(result, dict) else []
        cleaned: List[str] = []
        for query in queries:
            if isinstance(query, str):
                value = query.strip()
                if value:
                    cleaned.append(value)
        if cleaned:
            return cleaned[:count]
    except Exception:
        return []
    return []


def collect_images_pexels(
    query: str,
    api_key: str,
    output_dir: str,
    limit: int = 12,
    locale: str = "ja-JP",
) -> List[str]:
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY가 없습니다.")
    os.makedirs(output_dir, exist_ok=True)
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": min(limit, 80),
        "orientation": "portrait",
        "locale": locale,
    }
    response = requests.get("https://api.pexels.com/v1/search", params=params, headers=headers, timeout=60)
    response.raise_for_status()
    photos = response.json().get("photos", [])
    downloaded: List[str] = []
    for photo in photos:
        src = photo.get("src", {}) if isinstance(photo, dict) else {}
        image_url = src.get("large") or src.get("original")
        if not image_url:
            continue
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            filename = f"{random.randint(100000, 999999)}.jpg"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as file:
                file.write(image_response.content)
            downloaded.append(file_path)
            if len(downloaded) >= limit:
                break
        except Exception:
            continue
    return downloaded


def collect_images_auto_trend(
    config: AppConfig,
    output_dir: str,
    total_count: int = 12,
    max_queries: int = 4,
) -> Tuple[List[str], List[str]]:
    trend_context = get_trend_context(config)
    queries = generate_trend_queries(config, trend_context, count=max_queries)
    if not queries:
        raise RuntimeError("트렌드 검색어 생성에 실패했습니다. SERPAPI_API_KEY를 확인하세요.")
    if not config.pexels_api_key:
        raise RuntimeError("PEXELS_API_KEY가 없습니다.")
    os.makedirs(output_dir, exist_ok=True)
    collected: List[str] = []
    remaining = total_count
    for query in queries:
        if remaining <= 0:
            break
        per_query = max(1, min(remaining, total_count // max(1, len(queries))))
        items = collect_images_pexels(
            query=query,
            api_key=config.pexels_api_key,
            output_dir=output_dir,
            limit=per_query,
            locale="ja-JP",
        )
        collected.extend(items)
        remaining = total_count - len(collected)
    return collected[:total_count], queries


def fetch_bboom_list(config: AppConfig) -> List[Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(config.bboom_list_url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if "postNo=" not in href and "postno=" not in href:
            continue
        url = urljoin(config.bboom_list_url, href)
        if url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        title = unescape(title)
        if not title:
            continue
        seen.add(url)
        items.append({"url": url, "title": title})
        if len(items) >= config.bboom_max_fetch:
            break
    if items:
        return items
    # Fallback: regex extraction
    for match in re.findall(r'href=["\']([^"\']*postNo=\d+[^"\']*)', html):
        url = urljoin(config.bboom_list_url, match)
        if url in seen:
            continue
        seen.add(url)
        items.append({"url": url, "title": url})
        if len(items) >= config.bboom_max_fetch:
            break
    return items


def fetch_bboom_post_text(url: str) -> Dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    meta_title = soup.find("meta", property="og:title")
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    meta_desc = soup.find("meta", property="og:description")
    desc = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""
    text_blocks = []
    for tag in soup.find_all(["p", "span"]):
        text = tag.get_text(" ", strip=True)
        if text and len(text) > 5:
            text_blocks.append(text)
        if len(text_blocks) >= 6:
            break
    content = " ".join(text_blocks)
    if not content:
        content = desc
    return {"title": title, "content": content}


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload, timeout=30)


def get_telegram_updates(token: str, offset: int) -> List[Dict[str, Any]]:
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset, "timeout": 10}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("result", [])


def wait_for_approval(config: AppConfig, progress, status_box) -> str:
    start_time = time.time()
    offset = _load_offset(config.telegram_offset_path)
    approve_set = {kw.lower() for kw in config.approve_keywords}
    swap_set = {kw.lower() for kw in config.swap_keywords}
    while time.time() - start_time < config.telegram_timeout_sec:
        _status_update(progress, status_box, 0.25, "텔레그램 승인 대기")
        try:
            updates = get_telegram_updates(config.telegram_bot_token, offset)
        except Exception:
            updates = []
        for update in updates:
            update_id = update.get("update_id", 0)
            offset = max(offset, update_id + 1)
            message = update.get("message", {})
            chat = message.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if config.telegram_admin_chat_id and chat_id != str(config.telegram_admin_chat_id):
                continue
            text = (message.get("text") or "").strip().lower()
            if any(word in text for word in approve_set):
                _save_offset(config.telegram_offset_path, offset)
                return "approve"
            if any(word in text for word in swap_set):
                _save_offset(config.telegram_offset_path, offset)
                return "swap"
        _save_offset(config.telegram_offset_path, offset)
        time.sleep(5)
    return "approve"


def _write_local_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_hashtags(tags: List[str]) -> str:
    return " ".join(tags)


def _read_json_file(path: str, default: Any) -> Any:
    if not path:
        return default
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def _write_json_file(path: str, data: Any) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_used_links(path: str) -> Dict[str, Any]:
    return _read_json_file(path, {"used": []})


def _is_used_link(used_data: Dict[str, Any], url: str) -> bool:
    used_list = used_data.get("used", [])
    return any(item.get("url") == url for item in used_list)


def _mark_used_link(path: str, url: str, status: str, title: str = "") -> None:
    used_data = _load_used_links(path)
    used_list = used_data.get("used", [])
    used_list.append(
        {
            "url": url,
            "status": status,
            "title": title,
            "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    used_data["used"] = used_list
    _write_json_file(path, used_data)


def _load_offset(path: str) -> int:
    data = _read_json_file(path, {"offset": 0})
    return int(data.get("offset", 0))


def _save_offset(path: str, offset: int) -> None:
    _write_json_file(path, {"offset": offset})

def _missing_required(config: AppConfig) -> List[str]:
    missing = []
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not config.elevenlabs_api_key:
        missing.append("ELEVENLABS_API_KEY")
    if not config.elevenlabs_voice_ids:
        missing.append("ELEVENLABS_VOICE_ID 또는 ELEVENLABS_VOICE_IDS")
    if not config.sheet_id:
        missing.append("SHEET_ID")
    if not config.google_service_account_json:
        missing.append("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not config.font_path:
        missing.append("FONT_PATH")
    if config.enable_youtube_upload:
        if not config.youtube_client_id:
            missing.append("YOUTUBE_CLIENT_ID")
        if not config.youtube_client_secret:
            missing.append("YOUTUBE_CLIENT_SECRET")
        if not config.youtube_refresh_token:
            missing.append("YOUTUBE_REFRESH_TOKEN")
    return missing


def _status_update(progress, status_box, pct: float, message: str) -> None:
    if progress:
        progress.progress(min(max(pct, 0.0), 1.0))
    if status_box:
        status_box.info(f"진행 상태: {message}")


def _script_plan_text(script: Dict[str, Any]) -> str:
    beats = script.get("beats", [])
    hook = beats[0].get("text", "") if beats else ""
    ending = beats[-1].get("text", "") if beats else ""
    middle = beats[1].get("text", "") if len(beats) > 2 else ""
    return (
        f"제목(안): {script.get('title_ja','')}\n"
        f"훅: {hook}\n"
        f"전개: {middle}\n"
        f"오치: {ending}\n"
        f"해시태그: {' '.join(script.get('hashtags_ja', []))}"
    )


def _auto_bboom_flow(config: AppConfig, progress, status_box) -> None:
    if not config.telegram_bot_token or not config.telegram_admin_chat_id:
        st.error("텔레그램 봇 설정이 필요합니다. TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID를 확인하세요.")
        return
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        st.error("에셋이 없습니다. 먼저 이미지를 추가하세요.")
        return

    _status_update(progress, status_box, 0.05, "네이버 뿜 인기글 수집")
    try:
        items = fetch_bboom_list(config)
    except Exception as exc:
        st.error(f"뿜 수집 실패: {exc}")
        return
    if not items:
        st.error("가져올 인기글이 없습니다.")
        return

    used_data = _load_used_links(config.used_links_path)
    trend_context = get_trend_context(config)

    for item in items:
        url = item.get("url", "")
        if not url or _is_used_link(used_data, url):
            continue
        _status_update(progress, status_box, 0.12, "글 내용 분석")
        try:
            post = fetch_bboom_post_text(url)
        except Exception:
            post = {"title": item.get("title", ""), "content": ""}
        seed = f"{post.get('title','')}\n{post.get('content','')}"

        _status_update(progress, status_box, 0.18, "스크립트 초안 생성")
        script = generate_script(
            config=config,
            seed_text=seed,
            beats_count=7,
            trend_context=trend_context,
            dialect_style=config.ja_dialect_style,
        )
        plan_text = _script_plan_text(script)
        request_text = (
            f"[승인 요청]\n링크: {url}\n"
            f"제목: {post.get('title','')}\n"
            f"{plan_text}\n\n"
            "응답: 승인 / 교환"
        )
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, request_text)
        decision = wait_for_approval(config, progress, status_box)
        if decision == "swap":
            _mark_used_link(config.used_links_path, url, "swap", post.get("title", ""))
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, "교환 처리됨. 다음 인기글로 진행합니다.")
            continue

        _status_update(progress, status_box, 0.3, "TTS 생성")
        texts = [beat.get("text", "") for beat in script.get("beats", [])]
        tags = [beat.get("tag", "") for beat in script.get("beats", [])]
        assets = []
        for tag in tags:
            asset = pick_asset(manifest_items, [tag])
            if asset:
                assets.append(asset.path)
        if not assets:
            st.error("태그에 맞는 에셋이 없습니다.")
            return
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
        voice_id = pick_voice_id(config.elevenlabs_voice_ids)
        tts_elevenlabs(config, "。".join(texts), audio_path, voice_id=voice_id)

        _status_update(progress, status_box, 0.6, "영상 렌더링")
        output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
        )

        video_id = ""
        video_url = ""
        if config.enable_youtube_upload:
            _status_update(progress, status_box, 0.85, "유튜브 업로드")
            result = upload_video(
                config=config,
                file_path=output_path,
                title=script.get("title_ja", ""),
                description=script.get("description_ja", "") + "\n\n" + " ".join(script.get("hashtags_ja", [])),
                tags=script.get("hashtags_ja", []),
            )
            video_id = result.get("video_id", "")
            video_url = result.get("video_url", "")
        else:
            _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")

        log_row = {
            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "title_ja": script.get("title_ja", ""),
            "hashtags_ja": " ".join(script.get("hashtags_ja", [])),
            "template_id": "default",
            "asset_ids": ",".join([a for a in assets]),
            "voice_id": voice_id,
            "video_path": output_path,
            "youtube_video_id": video_id,
            "youtube_url": video_url,
            "status": "ok",
            "error": "",
        }
        try:
            append_publish_log(config, log_row)
        except Exception:
            pass
        _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)
        _mark_used_link(config.used_links_path, url, "approved", post.get("title", ""))
        _status_update(progress, status_box, 1.0, "완료")
        st.video(output_path)

        summary_text = f"[완료]\n제목: {script.get('title_ja','')}\n요약: {script.get('description_ja','')}\n"
        if video_url:
            summary_text += f"유튜브 링크: {video_url}"
        else:
            summary_text += f"로컬 파일: {output_path}"
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)
        return

    st.warning("사용 가능한 인기글이 더 이상 없습니다.")


def run_streamlit_app() -> None:
    st.set_page_config(page_title="숏츠 자동화 스튜디오", layout="wide")
    config = load_config()
    ensure_dirs(
        [
            config.assets_dir,
            os.path.join(config.assets_dir, "images"),
            os.path.join(config.assets_dir, "inbox"),
            os.path.join(config.assets_dir, "bgm"),
            os.path.join(config.assets_dir, "sfx"),
            os.path.dirname(config.manifest_path),
            config.output_dir,
        ]
    )

    st.sidebar.title("숏츠 자동화 스튜디오")
    st.sidebar.subheader("상태")
    st.sidebar.write(f"자동 업로드: {'켜짐' if config.enable_youtube_upload else '꺼짐'}")
    st.sidebar.write(f"MoviePy 사용 가능: {'예' if MOVIEPY_AVAILABLE else '아니오'}")
    if not MOVIEPY_AVAILABLE:
        st.sidebar.error(f"MoviePy 오류: {MOVIEPY_ERROR}")

    st.sidebar.subheader("필수 API/설정")
    st.sidebar.markdown(
        "- `OPENAI_API_KEY`\n"
        "- `ELEVENLABS_API_KEY`\n"
        "- `ELEVENLABS_VOICE_ID` 또는 `ELEVENLABS_VOICE_IDS`\n"
        "- `SHEET_ID`\n"
        "- `GOOGLE_SERVICE_ACCOUNT_JSON`\n"
        "- `FONT_PATH`"
    )
    st.sidebar.subheader("자동 승인(텔레그램)")
    st.sidebar.markdown(
        "- `TELEGRAM_BOT_TOKEN`\n"
        "- `TELEGRAM_ADMIN_CHAT_ID`\n"
        "- `TELEGRAM_TIMEOUT_SEC`"
    )
    st.sidebar.subheader("선택")
    st.sidebar.markdown(
        "- `YOUTUBE_*` (자동 업로드)\n"
        "- `SERPAPI_API_KEY` (트렌드 수집)\n"
        "- `PEXELS_API_KEY` (트렌드 이미지 자동 수집)\n"
        "- `JA_DIALECT_STYLE` (일본어 사투리 스타일)"
    )
    missing = _missing_required(config)
    if missing:
        st.sidebar.warning("누락된 설정: " + ", ".join(missing))

    page = st.sidebar.radio("메뉴", ["생성", "토큰", "에셋", "로그"])

    manifest_items = load_manifest(config.manifest_path)
    all_tags = list_tags(manifest_items)

    if page == "생성":
        st.header("생성")
        if missing:
            st.error("필수 API/설정이 누락되어 있습니다. 좌측 사이드바를 확인하세요.")
        progress = st.progress(0.0)
        status_box = st.empty()

        st.subheader("네이버 뿜 자동 생성(승인 포함)")
        auto_button = st.button("뿜 인기글로 자동 생성 시작")
        if auto_button:
            _auto_bboom_flow(config, progress, status_box)

        st.divider()
        seed_text = st.text_area("아이디어/요약 입력", height=120)
        beats_count = st.slider("문장(비트) 수", 5, 9, 7)
        tag_filter = st.multiselect("허용 태그", options=all_tags, default=all_tags[:5])
        generate_button = st.button("스크립트 생성")

        if generate_button and seed_text:
            _status_update(progress, status_box, 0.05, "스크립트 생성 중")
            script = generate_script(
                config=config,
                seed_text=seed_text,
                beats_count=beats_count,
                allowed_tags=tag_filter or all_tags,
                trend_context=get_trend_context(config),
                dialect_style=config.ja_dialect_style,
            )
            st.session_state["script"] = script
            _status_update(progress, status_box, 0.2, "스크립트 생성 완료")

        script = st.session_state.get("script")
        if script:
            st.subheader("스크립트")
            title = st.text_input("제목(일본어)", value=script.get("title_ja", ""))
            description = st.text_area("설명(일본어)", value=script.get("description_ja", ""), height=80)
            hashtags = st.text_input(
                "해시태그(공백 구분)",
                value=_format_hashtags(script.get("hashtags_ja", [])),
            )
            beats_df = pd.DataFrame(script.get("beats", []))
            edited_beats = st.data_editor(beats_df, num_rows="fixed")

            render_button = st.button("영상 만들기")
            if render_button:
                if missing:
                    st.error("필수 API/설정이 누락되어 있어 진행할 수 없습니다.")
                    return
                if not MOVIEPY_AVAILABLE:
                    st.error(f"MoviePy가 설치되지 않았습니다: {MOVIEPY_ERROR}")
                    return
                beats = edited_beats.to_dict(orient="records")
                if not beats:
                    st.error("렌더링할 문장이 없습니다.")
                elif not manifest_items:
                    st.error("에셋이 없습니다. 먼저 이미지를 추가하세요.")
                else:
                    _status_update(progress, status_box, 0.1, "문장/태그 정리")
                    texts = [beat.get("text", "") for beat in beats]
                    tags = [beat.get("tag", "") for beat in beats]
                    assets = []
                    for tag in tags:
                        asset = pick_asset(manifest_items, [tag])
                        if asset:
                            assets.append(asset.path)
                    if not assets:
                        st.error("태그에 맞는 에셋이 없습니다.")
                    else:
                        _status_update(progress, status_box, 0.3, "TTS 생성")
                        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
                        voice_id = pick_voice_id(config.elevenlabs_voice_ids)
                        tts_elevenlabs(config, "。".join(texts), audio_path, voice_id=voice_id)
                        _status_update(progress, status_box, 0.6, "영상 렌더링")
                        output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
                        render_video(
                            config=config,
                            asset_paths=assets,
                            texts=texts,
                            tts_audio_path=audio_path,
                            output_path=output_path,
                        )
                        video_id = ""
                        video_url = ""
                        if config.enable_youtube_upload:
                            _status_update(progress, status_box, 0.85, "유튜브 업로드")
                            result = upload_video(
                                config=config,
                                file_path=output_path,
                                title=title,
                                description=description + "\n\n" + hashtags,
                                tags=hashtags.split(),
                            )
                            video_id = result.get("video_id", "")
                            video_url = result.get("video_url", "")
                        else:
                            _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")
                        log_row = {
                            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "title_ja": title,
                            "hashtags_ja": hashtags,
                            "template_id": "default",
                            "asset_ids": ",".join([a for a in assets]),
                            "voice_id": voice_id,
                            "video_path": output_path,
                            "youtube_video_id": video_id,
                            "youtube_url": video_url,
                            "status": "ok",
                            "error": "",
                        }
                        try:
                            append_publish_log(config, log_row)
                        except Exception:
                            pass
                        _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)
                        _status_update(progress, status_box, 1.0, "완료")
                        st.video(output_path)
                        if video_url:
                            st.success(video_url)

    if page == "토큰":
        st.header("유튜브 리프레시 토큰 발급")
        st.markdown(
            "1) OAuth 클라이언트 ID/Secret 입력\n"
            "2) Redirect URI 설정\n"
            "3) 승인 URL 열기 → 로그인/동의\n"
            "4) code를 붙여넣고 토큰 교환"
        )
        st.caption(
            "Redirect URI는 Google Cloud Credentials에 등록되어 있어야 합니다. "
            "로컬이라면 보통 http://localhost:8501 를 사용합니다."
        )

        client_id = st.text_input("OAuth 클라이언트 ID", value=config.youtube_client_id or "")
        client_secret = st.text_input(
            "OAuth 클라이언트 Secret", value=config.youtube_client_secret or "", type="password"
        )
        redirect_uri = st.text_input(
            "Redirect URI",
            value=_get_secret("OAUTH_REDIRECT_URI", "") or "",
            placeholder="http://localhost:8501",
        )
        scope = st.text_input(
            "Scope",
            value="https://www.googleapis.com/auth/youtube.upload",
        )
        prompt_consent = st.checkbox("재동의 강제(prompt=consent)", value=True)

        if st.button("승인 URL 생성"):
            if not client_id or not redirect_uri:
                st.error("클라이언트 ID와 Redirect URI를 입력하세요.")
            else:
                st.session_state["auth_url"] = build_google_oauth_url(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    scope=scope.strip(),
                    prompt_consent=prompt_consent,
                )

        auth_url = st.session_state.get("auth_url", "")
        if auth_url:
            st.markdown(f"[승인 페이지 열기]({auth_url})")
            st.code(auth_url)

        default_code = _get_query_param("code")
        auth_code = st.text_input("승인 코드(code)", value=default_code or "")

        if st.button("토큰 교환"):
            if not client_id or not client_secret or not redirect_uri or not auth_code:
                st.error("모든 값을 입력한 후 진행하세요.")
            else:
                try:
                    result = exchange_oauth_code_for_token(
                        client_id=client_id,
                        client_secret=client_secret,
                        code=auth_code.strip(),
                        redirect_uri=redirect_uri,
                    )
                    refresh_token = result.get("refresh_token", "")
                    if refresh_token:
                        st.success("리프레시 토큰 발급 성공")
                        st.code(refresh_token)
                        st.info("이 값을 `.streamlit/secrets.toml`의 `YOUTUBE_REFRESH_TOKEN`에 넣으세요.")
                    else:
                        st.warning(
                            "refresh_token이 응답에 없습니다. "
                            "처음 승인일 때만 내려오는 경우가 많습니다. "
                            "prompt=consent를 켜고 다시 승인하세요."
                        )
                        st.json(result)
                except Exception as exc:
                    st.error(f"토큰 교환 실패: {exc}")

    if page == "에셋":
        st.header("에셋")
        upload_files = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        tag_input = st.text_input("태그(쉼표 구분)")
        if st.button("업로드 저장") and upload_files:
            for file in upload_files:
                save_path = os.path.join(config.assets_dir, "images", file.name)
                with open(save_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
                add_asset(config.manifest_path, save_path, tags_from_text(tag_input))
            st.success("에셋이 저장되었습니다.")

        st.subheader("AI 이미지 수집(SerpAPI)")
        collect_query = st.text_input("검색어")
        collect_count = st.slider("수집 개수", 4, 20, 8)
        if st.button("인박스로 수집"):
            try:
                inbox_dir = os.path.join(config.assets_dir, "inbox")
                collected = collect_images_serpapi(
                    query=collect_query,
                    api_key=config.serpapi_api_key,
                    output_dir=inbox_dir,
                    limit=collect_count,
                )
                st.success(f"{len(collected)}개 이미지를 인박스에 저장했습니다.")
            except Exception as exc:
                st.error(f"수집 실패: {exc}")

        st.subheader("일본 트렌드 자동 수집(Pexels)")
        st.caption("SerpAPI로 일본 트렌드 키워드를 만들고, Pexels에서 이미지를 자동 수집합니다.")
        auto_count = st.slider("자동 수집 개수", 4, 30, 12, key="auto_collect_count")
        auto_queries = st.slider("검색어 개수", 2, 6, 4, key="auto_collect_queries")
        if st.button("일본 트렌드 자동 수집"):
            try:
                inbox_dir = os.path.join(config.assets_dir, "inbox")
                collected, queries = collect_images_auto_trend(
                    config=config,
                    output_dir=inbox_dir,
                    total_count=auto_count,
                    max_queries=auto_queries,
                )
                st.write("사용 검색어:", ", ".join(queries))
                st.success(f"{len(collected)}개 이미지를 인박스에 저장했습니다.")
            except Exception as exc:
                st.error(f"자동 수집 실패: {exc}")

        inbox_dir = os.path.join(config.assets_dir, "inbox")
        inbox_files = [
            os.path.join(inbox_dir, f)
            for f in os.listdir(inbox_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if inbox_files:
            st.subheader("인박스")
            inbox_tags = st.text_input("인박스 태그(쉼표 구분)")
            selected_files: List[str] = []
            for file_path in inbox_files:
                st.image(file_path, width=200)
                if st.checkbox(f"선택: {os.path.basename(file_path)}", key=f"select_{file_path}"):
                    selected_files.append(file_path)
            if st.button("선택한 짤 저장"):
                for file_path in selected_files:
                    add_asset(config.manifest_path, file_path, tags_from_text(inbox_tags))
                st.success("선택한 짤이 라이브러리에 추가되었습니다.")

        st.subheader("라이브러리")
        if manifest_items:
            selected_tag = st.selectbox("태그 필터", options=["(전체)"] + all_tags)
            filtered = manifest_items if selected_tag == "(전체)" else filter_assets_by_tags(manifest_items, [selected_tag])
            for item in filtered:
                st.image(item.path, width=200, caption=",".join(item.tags))
        else:
            st.info("아직 에셋이 없습니다.")

    if page == "로그":
        st.header("로그")
        local_log_path = os.path.join(config.output_dir, "runs.jsonl")
        if os.path.exists(local_log_path):
            with open(local_log_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[-50:]
            records = [json.loads(line) for line in lines]
            st.dataframe(pd.DataFrame(records))
        else:
            st.info("아직 로그가 없습니다.")


def run_batch(count: int, seed: str, beats: int) -> None:
    config = load_config()
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        raise RuntimeError("에셋이 없습니다. 먼저 이미지를 추가하세요.")
    for index in range(count):
        script = generate_script(
            config,
            seed,
            beats_count=beats,
            trend_context=get_trend_context(config),
            dialect_style=config.ja_dialect_style,
        )
        beats_list = script.get("beats", [])
        texts = [beat.get("text", "") for beat in beats_list]
        tags = [beat.get("tag", "") for beat in beats_list]
        assets = []
        for tag in tags:
            asset = pick_asset(manifest_items, [tag])
            if asset:
                assets.append(asset.path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}_{index}.mp3")
        voice_id = pick_voice_id(config.elevenlabs_voice_ids)
        tts_elevenlabs(config, "。".join(texts), audio_path, voice_id=voice_id)
        output_path = os.path.join(config.output_dir, f"shorts_{now}_{index}.mp4")
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
        )
        video_id = ""
        video_url = ""
        if config.enable_youtube_upload:
            result = upload_video(
                config=config,
                file_path=output_path,
                title=script.get("title_ja", ""),
                description=script.get("description_ja", ""),
                tags=script.get("hashtags_ja", []),
            )
            video_id = result.get("video_id", "")
            video_url = result.get("video_url", "")
        log_row = {
            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "title_ja": script.get("title_ja", ""),
            "hashtags_ja": " ".join(script.get("hashtags_ja", [])),
            "template_id": "default",
            "asset_ids": ",".join([a for a in assets]),
            "voice_id": voice_id,
            "video_path": output_path,
            "youtube_video_id": video_id,
            "youtube_url": video_url,
            "status": "ok",
            "error": "",
        }
        try:
            append_publish_log(config, log_row)
        except Exception:
            pass
        _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)


if os.getenv("RUN_BATCH") == "1":
    run_batch(
        count=int(os.getenv("BATCH_COUNT", "2")),
        seed=os.getenv("BATCH_SEED", "일본어 밈 숏츠"),
        beats=int(os.getenv("BATCH_BEATS", "7")),
    )
else:
    run_streamlit_app()
