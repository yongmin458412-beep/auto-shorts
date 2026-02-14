from __future__ import annotations

import json
import ast
import base64
import mimetypes
import os
import random
import re
import shutil
import subprocess
import textwrap
import time
from html import unescape
from urllib.parse import urljoin, urlencode, urlparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup

PYSUBS2_AVAILABLE = True
PYSUBS2_ERROR = ""
try:
    import pysubs2
except Exception as exc:
    PYSUBS2_AVAILABLE = False
    PYSUBS2_ERROR = str(exc)
    pysubs2 = None

import gspread
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError, ResumableUploadError

MOVIEPY_AVAILABLE = True
MOVIEPY_ERROR = ""
try:
    import numpy as np
    CV2_AVAILABLE = True
    CV2_ERROR = ""
    try:
        import cv2
    except Exception as exc:
        CV2_AVAILABLE = False
        CV2_ERROR = str(exc)
        cv2 = None
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    # Pillow 10.0+에서 ANTIALIAS 제거됨 → MoviePy 내부 호환성 패치
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
        vfx,
    )
except Exception as exc:
    MOVIEPY_AVAILABLE = False
    MOVIEPY_ERROR = str(exc)
    np = None
    CV2_AVAILABLE = False
    CV2_ERROR = "MoviePy unavailable"
    cv2 = None
    AudioFileClip = CompositeAudioClip = ImageClip = VideoFileClip = concatenate_videoclips = vfx = None
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


def _ensure_writable_dir(path: str, fallback: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".write_test")
        with open(test_path, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_path)
        return path
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _ensure_writable_file(path: str, fallback: str) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        test_path = os.path.join(os.path.dirname(path), ".write_test")
        with open(test_path, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_path)
        return path
    except Exception:
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        return fallback


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


def _guess_ext_from_type(content_type: str) -> str:
    if not content_type:
        return ""
    content_type = content_type.lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    return ""


def download_images_from_urls(urls: List[str], output_dir: str, limit: int = 30) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    downloaded: List[str] = []
    for raw_url in urls:
        if len(downloaded) >= limit:
            break
        url = raw_url.strip()
        if not url:
            continue
        try:
            # 네이버 CDN은 Referer 필수
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://blog.naver.com/",
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                continue
            # 너무 작은 파일(아이콘 등) 제외 - 5KB 미만
            if len(response.content) < 5120:
                continue
            ext = os.path.splitext(urlparse(url).path)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                ext = _guess_ext_from_type(content_type) or ".jpg"
            filename = f"{random.randint(100000, 999999)}{ext}"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as file:
                file.write(response.content)
            downloaded.append(file_path)
        except Exception:
            continue
    return downloaded


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip().strip('"').strip("'")
    if url.startswith("//"):
        url = "https:" + url
    if "pstatic.net" in url and "?" in url:
        url = url.split("?", 1)[0]
    return url


_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://blog.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


def _naver_blog_to_postview_url(url: str) -> Optional[str]:
    """
    blog.naver.com/{blogId}/{logNo} 형식을
    PostView.naver?blogId=...&logNo=... 형식으로 변환.
    이미 PostView 형식이면 그대로 반환.
    """
    # 이미 PostView URL인 경우
    if "PostView" in url or "postview" in url.lower():
        return url
    # 표준 형식: blog.naver.com/{blogId}/{logNo}
    match = re.search(
        r"blog\.naver\.com/([^/?#]+)/(\d+)",
        url,
        flags=re.I,
    )
    if match:
        blog_id, log_no = match.group(1), match.group(2)
        return (
            f"https://blog.naver.com/PostView.naver"
            f"?blogId={blog_id}&logNo={log_no}&isInf=true"
        )
    # m.blog.naver.com/{blogId}/{logNo}
    match = re.search(
        r"m\.blog\.naver\.com/([^/?#]+)/(\d+)",
        url,
        flags=re.I,
    )
    if match:
        blog_id, log_no = match.group(1), match.group(2)
        return (
            f"https://blog.naver.com/PostView.naver"
            f"?blogId={blog_id}&logNo={log_no}&isInf=true"
        )
    return None


def _is_naver_post_image(url: str) -> bool:
    """
    네이버 블로그 본문 실제 이미지인지 판별.
    postfiles / blogfiles 도메인만 허용.
    UI 아이콘(ssl.pstatic.net), 썸네일(mblogthumb-phinf),
    프로필(phinf.pstatic.net) 등은 모두 제외.
    """
    if not url:
        return False
    # 허용 도메인: 실제 본문 첨부 이미지
    allowed = (
        "postfiles.pstatic.net",
        "blogfiles.pstatic.net",
    )
    if any(d in url for d in allowed):
        # 경로에 이미지 확장자 또는 네이버 업로드 경로 포함 확인
        path_lower = url.lower()
        if any(ext in path_lower for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", "/mjax", "mjpeg")):
            return True
        # 확장자 없어도 postfiles 경로면 허용 (네이버는 확장자 생략 많음)
        return True
    return False


def extract_image_urls_from_html(html: str, naver_mode: bool = False) -> List[str]:
    """
    naver_mode=True 이면 postfiles/blogfiles 도메인만 추출 (본문 이미지 전용).
    naver_mode=False 이면 기존 방식 전체 추출.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()

    # 1) <img> 태그 모든 속성
    for img in soup.find_all("img"):
        for attr in (
            "src", "data-src", "data-lazy-src", "data-original",
            "data-actualsrc", "data-lazy", "data-url",
        ):
            value = img.get(attr)
            if value:
                urls.add(_normalize_url(value))

    # 2) style 속성 background-image
    for tag in soup.find_all(style=True):
        style = tag.get("style", "") or ""
        for match in re.findall(r"url\(([^)]+)\)", style, flags=re.I):
            value = match.strip().strip("'").strip('"')
            if value:
                urls.add(_normalize_url(value))

    # 3) 정규식: 일반 이미지 URL
    for match in re.findall(
        r"https?://[^\"'\s<>]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\"'\s<>]*)?",
        html,
        flags=re.I,
    ):
        urls.add(_normalize_url(match))

    # 4) 정규식: 네이버 본문 이미지 CDN (postfiles / blogfiles 만)
    for match in re.findall(
        r"https?://(?:postfiles|blogfiles)\.pstatic\.net/[^\"'\s<>]+",
        html,
        flags=re.I,
    ):
        urls.add(_normalize_url(match))

    # 5) JSON 데이터 안의 이미지 URL
    for match in re.findall(
        r'"(?:url|src|imageUrl|photoUrl)"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
        html,
        flags=re.I,
    ):
        urls.add(_normalize_url(match))

    cleaned = []
    for url in urls:
        if not url or url.startswith("data:"):
            continue
        if naver_mode:
            # 네이버 모드: 본문 실제 이미지만
            if not _is_naver_post_image(url):
                continue
        else:
            # 일반 모드: 명백한 UI 요소만 제외
            if any(x in url for x in [
                "favicon", "icon_", "blank.", "loading.",
                "ssl.pstatic.net",        # 네이버 UI 정적 리소스
                "mblogthumb-phinf",       # 썸네일
                "phinf.pstatic.net",      # 프로필 이미지
                "dthumb.pstatic.net",     # 동적 썸네일
            ]):
                continue
        cleaned.append(url)
    return cleaned


def fetch_post_image_urls(url: str) -> List[str]:
    if not url:
        return []

    is_naver = "naver.com" in url
    collected: List[str] = []

    # PostView URL 변환 (네이버 iframe 우회)
    postview_url = _naver_blog_to_postview_url(url) if is_naver else None
    urls_to_try: List[str] = []
    if postview_url and postview_url != url:
        urls_to_try.append(postview_url)
    urls_to_try.append(url)

    # 모바일 버전 추가 (네이버 PC URL인 경우)
    if "blog.naver.com" in url and "m.blog.naver.com" not in url:
        urls_to_try.append(url.replace("blog.naver.com", "m.blog.naver.com"))

    for try_url in urls_to_try:
        try:
            response = requests.get(try_url, headers=_NAVER_HEADERS, timeout=30)
            response.raise_for_status()
            found = extract_image_urls_from_html(response.text, naver_mode=is_naver)
            if found:
                return found
            collected.extend(found)
        except Exception:
            continue

    return list(dict.fromkeys(collected))


def collect_images_from_post_urls(
    post_urls: List[str],
    output_dir: str,
    limit: int = 50,
) -> Tuple[int, int]:
    all_urls: List[str] = []
    for post_url in post_urls:
        try:
            urls = fetch_post_image_urls(post_url)
            all_urls.extend(urls)
        except Exception:
            continue
    if not all_urls:
        return 0, 0
    downloaded = download_images_from_urls(all_urls, output_dir, limit=limit)
    return len(all_urls), len(downloaded)


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
    openai_vision_model: str
    openai_tts_voices: List[str]
    openai_tts_voice_preference: List[str]
    openai_tts_model: str
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
    bgm_mode: str
    bgm_volume: float
    bgm_fallback_enabled: bool
    asset_overlay_mode: str
    max_video_duration_sec: float
    require_approval: bool
    auto_run_daily: bool
    auto_run_hour: int
    auto_run_times: List[str]
    auto_run_tz: str
    auto_run_state_path: str
    auto_run_lock_path: str
    used_topics_path: str
    generated_bg_dir: str
    use_generated_bg_priority: bool
    primary_photo_path: str
    telegram_bot_token: str
    telegram_admin_chat_id: str
    telegram_timeout_sec: int
    telegram_offset_path: str
    telegram_debug_logs: bool
    telegram_timeline_only: bool
    approve_keywords: List[str]
    swap_keywords: List[str]
    pixabay_api_key: str
    pixabay_bgm_enabled: bool
    use_bg_videos: bool
    render_threads: int
    use_korean_template: bool
    caption_max_chars: int
    caption_hold_ratio: float
    caption_trim: bool
    use_ass_subtitles: bool
    thumbnail_enabled: bool
    thumbnail_use_hook: bool
    thumbnail_max_chars: int
    use_minecraft_parkour_bg: bool
    tts_provider: str
    tts_force_cute_voice: bool
    tts_baby_voice: bool
    tts_baby_pitch: float
    force_fresh_media_on_start: bool
    force_fresh_minecraft_download: bool
    enable_majisho_tag: bool
    majisho_asset_path: str
    use_japanese_caption_style: bool
    linktree_url: str
    enable_pinned_comment: bool
    highlight_clip_enabled: bool
    highlight_clip_duration_sec: float
    highlight_clip_sample_fps: float
    highlight_clip_max_scan_sec: float
    ab_test_enabled: bool
    background_ab_epsilon: float
    background_ab_lookback: int
    background_ab_max_sample: int
    retry_pending_uploads: bool
    pending_uploads_path: str
    youtube_api_key: str
    ab_report_enabled: bool
    ab_report_hour: int
    ab_report_days: int
    ab_report_max_items: int
    ab_report_state_path: str
    video_metrics_state_path: str
    jp_youtube_only: bool
    # 플랫폼: instagram(기본), youtube, tiktok(준비)
    upload_platform: str
    enable_instagram_upload: bool
    instagram_access_token: str
    instagram_user_id: str
    instagram_use_popular_audio: bool


def load_config() -> AppConfig:
    assets_dir = _get_secret("ASSETS_DIR", "data/assets")
    manifest_path = _get_secret("MANIFEST_PATH", "data/manifests/assets.json")
    output_dir = _ensure_writable_dir(
        _get_secret("OUTPUT_DIR", "data/output") or "data/output",
        "/tmp/auto_shorts_output",
    )
    telegram_offset_path = _ensure_writable_file(
        _get_secret("TELEGRAM_OFFSET_PATH", "data/state/telegram_offset.json")
        or "data/state/telegram_offset.json",
        "/tmp/auto_shorts_state/telegram_offset.json",
    )
    auto_run_state_path = _ensure_writable_file(
        _get_secret("AUTO_RUN_STATE_PATH", "data/state/auto_run_state.json")
        or "data/state/auto_run_state.json",
        "/tmp/auto_shorts_state/auto_run_state.json",
    )
    auto_run_lock_path = _ensure_writable_file(
        _get_secret("AUTO_RUN_LOCK_PATH", "data/state/auto_run.lock")
        or "data/state/auto_run.lock",
        "/tmp/auto_shorts_state/auto_run.lock",
    )
    used_topics_path = _ensure_writable_file(
        _get_secret("USED_TOPICS_PATH", "data/state/used_topics.json")
        or "data/state/used_topics.json",
        "/tmp/auto_shorts_state/used_topics.json",
    )
    pending_uploads_path = _ensure_writable_file(
        _get_secret("PENDING_UPLOADS_PATH", "data/state/pending_uploads.json")
        or "data/state/pending_uploads.json",
        "/tmp/auto_shorts_state/pending_uploads.json",
    )
    ab_report_state_path = _ensure_writable_file(
        _get_secret("AB_REPORT_STATE_PATH", "data/state/ab_report_state.json")
        or "data/state/ab_report_state.json",
        "/tmp/auto_shorts_state/ab_report_state.json",
    )
    video_metrics_state_path = _ensure_writable_file(
        _get_secret("VIDEO_METRICS_STATE_PATH", "data/state/video_metrics_state.json")
        or "data/state/video_metrics_state.json",
        "/tmp/auto_shorts_state/video_metrics_state.json",
    )
    pixabay_api_key = (_get_secret("PIXABAY_API_KEY", "") or "").strip()
    pexels_api_key = (_get_secret("PEXELS_API_KEY", "") or "").strip()
    auto_run_hour = int(_get_secret("AUTO_RUN_HOUR", "18") or 18)
    auto_run_times = _get_list("AUTO_RUN_TIMES") or ["12:30", "20:30"]
    return AppConfig(
        openai_api_key=_get_secret("OPENAI_API_KEY", "") or "",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
        openai_vision_model=_get_secret("OPENAI_VISION_MODEL", "") or "",
        openai_tts_voices=_get_list("OPENAI_TTS_VOICES")
        or ([v for v in [_get_secret("OPENAI_TTS_VOICE", "shimmer")] if v]),
        openai_tts_voice_preference=_get_list("OPENAI_TTS_VOICE_PREFERENCE"),
        openai_tts_model=_get_secret("OPENAI_TTS_MODEL", "tts-1") or "tts-1",
        sheet_id=_get_secret("SHEET_ID", "") or "",
        google_service_account_json=_get_json("GOOGLE_SERVICE_ACCOUNT_JSON"),
        assets_dir=assets_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        font_path=(_get_secret("FONT_PATH", "") or "").strip()
        or _ensure_japanese_font(_get_secret("ASSETS_DIR", "data/assets") or "data/assets"),
        width=int(_get_secret("VIDEO_WIDTH", "1080") or 1080),
        height=int(_get_secret("VIDEO_HEIGHT", "1920") or 1920),
        fps=int(_get_secret("VIDEO_FPS", "30") or 30),
        enable_youtube_upload=_get_bool("YOUTUBE_UPLOAD_ENABLED", False),
        youtube_client_id=_get_secret("YOUTUBE_CLIENT_ID", "") or "",
        youtube_client_secret=_get_secret("YOUTUBE_CLIENT_SECRET", "") or "",
        youtube_refresh_token=_get_secret("YOUTUBE_REFRESH_TOKEN", "") or "",
        youtube_privacy_status=_get_secret("YOUTUBE_PRIVACY_STATUS", "public") or "public",
        serpapi_api_key=_get_secret("SERPAPI_API_KEY", "") or "",
        pexels_api_key=pexels_api_key,
        ja_dialect_style=(
            _get_secret(
                "JA_DIALECT_STYLE",
                "関西寄りの親しみあるタメ口（全国視聴者に通じる軽い方言）",
            )
            or "関西寄りの親しみあるタメ口（全国視聴者に通じる軽い方言）"
        ),
        bgm_mode=_get_secret("BGM_MODE", "auto") or "auto",  # 기본값 auto: BGM 자동 선택
        bgm_volume=float(_get_secret("BGM_VOLUME", "0.2") or 0.2),  # 일본 쇼츠: TTS 가독성 위해 20%
        bgm_fallback_enabled=_get_bool("BGM_FALLBACK_ENABLED", True),
        asset_overlay_mode=_get_secret("ASSET_OVERLAY_MODE", "off") or "off",
        max_video_duration_sec=float(_get_secret("MAX_VIDEO_DURATION_SEC", "59") or 59),
        require_approval=_get_bool("REQUIRE_APPROVAL", True),
        auto_run_daily=_get_bool("AUTO_RUN_DAILY", True),
        auto_run_hour=auto_run_hour,
        auto_run_times=auto_run_times,
        auto_run_tz=_get_secret("AUTO_RUN_TZ", "Asia/Tokyo") or "Asia/Tokyo",
        auto_run_state_path=auto_run_state_path,
        auto_run_lock_path=auto_run_lock_path,
        used_topics_path=used_topics_path,
        generated_bg_dir=_get_secret("GENERATED_BG_DIR", "data/assets/generated_bg") or "data/assets/generated_bg",
        use_generated_bg_priority=_get_bool("USE_GENERATED_BG_PRIORITY", False),
        primary_photo_path=(_get_secret("PRIMARY_PHOTO_PATH", "") or "").strip(),
        telegram_bot_token=_get_secret("TELEGRAM_BOT_TOKEN", "") or "",
        telegram_admin_chat_id=_get_secret("TELEGRAM_ADMIN_CHAT_ID", "") or "",
        telegram_timeout_sec=int(_get_secret("TELEGRAM_TIMEOUT_SEC", "600") or 600),
        telegram_offset_path=telegram_offset_path,
        telegram_debug_logs=_get_bool("TELEGRAM_DEBUG_LOGS", False),
        telegram_timeline_only=_get_bool("TELEGRAM_TIMELINE_ONLY", True),
        approve_keywords=_get_list("APPROVE_KEYWORDS") or ["승인", "approve", "ok", "yes"],
        swap_keywords=_get_list("SWAP_KEYWORDS") or ["교환", "swap", "change", "next"],
        pixabay_api_key=pixabay_api_key,
        pixabay_bgm_enabled=_get_bool("PIXABAY_BGM_ENABLED", False),
        use_bg_videos=_get_bool("USE_BG_VIDEOS", True),  # 기본값 True: 배경영상 항상 활성화
        render_threads=int(_get_secret("RENDER_THREADS", "1") or 1),
        use_korean_template=_get_bool("USE_KOREAN_TEMPLATE", True),
        caption_max_chars=int(_get_secret("CAPTION_MAX_CHARS", "14") or 14),
        caption_hold_ratio=float(_get_secret("CAPTION_HOLD_RATIO", "0.9") or 0.9),
        caption_trim=_get_bool("CAPTION_TRIM", True),
        use_ass_subtitles=_get_bool("USE_ASS_SUBTITLES", True),
        thumbnail_enabled=_get_bool("THUMBNAIL_ENABLED", True),
        thumbnail_use_hook=_get_bool("THUMBNAIL_USE_HOOK", True),
        thumbnail_max_chars=int(_get_secret("THUMBNAIL_MAX_CHARS", "22") or 22),
        use_minecraft_parkour_bg=_get_bool("USE_MINECRAFT_PARKOUR_BG", True),
        tts_provider=(_get_secret("TTS_PROVIDER", "") or "openai").strip().lower() or "openai",
        tts_force_cute_voice=_get_bool("TTS_FORCE_CUTE_VOICE", True),
        tts_baby_voice=_get_bool("TTS_BABY_VOICE", True),
        tts_baby_pitch=float(_get_secret("TTS_BABY_PITCH", "1.20") or 1.20),
        force_fresh_media_on_start=_get_bool("FORCE_FRESH_MEDIA_ON_START", True),
        force_fresh_minecraft_download=_get_bool("FORCE_FRESH_MINECRAFT_DOWNLOAD", True),
        enable_majisho_tag=_get_bool("ENABLE_MAJISHO_TAG", True),
        majisho_asset_path=(
            _get_secret("MAJISHO_ASSET_PATH", "data/assets/branding/majisho.png")
            or "data/assets/branding/majisho.png"
        ),
        use_japanese_caption_style=_get_bool("USE_JAPANESE_CAPTION_STYLE", True),
        linktree_url=(_get_secret("LINKTREE_URL", "") or "").strip(),
        enable_pinned_comment=_get_bool("ENABLE_PINNED_COMMENT", False),
        highlight_clip_enabled=_get_bool("HIGHLIGHT_CLIP_ENABLED", True),
        highlight_clip_duration_sec=float(_get_secret("HIGHLIGHT_CLIP_DURATION_SEC", "55") or 55),
        highlight_clip_sample_fps=float(_get_secret("HIGHLIGHT_CLIP_SAMPLE_FPS", "2") or 2),
        highlight_clip_max_scan_sec=float(_get_secret("HIGHLIGHT_CLIP_MAX_SCAN_SEC", "900") or 900),
        ab_test_enabled=_get_bool("AB_TEST_ENABLED", True),
        background_ab_epsilon=float(_get_secret("BACKGROUND_AB_EPSILON", "0.2") or 0.2),
        background_ab_lookback=int(_get_secret("BACKGROUND_AB_LOOKBACK", "30") or 30),
        background_ab_max_sample=int(_get_secret("BACKGROUND_AB_MAX_SAMPLE", "12") or 12),
        retry_pending_uploads=_get_bool("RETRY_PENDING_UPLOADS", True),
        pending_uploads_path=pending_uploads_path,
        youtube_api_key=(_get_secret("YOUTUBE_API_KEY", "") or "").strip(),
        ab_report_enabled=_get_bool("AB_REPORT_ENABLED", True),
        ab_report_hour=int(_get_secret("AB_REPORT_HOUR", "20") or 20),
        ab_report_days=int(_get_secret("AB_REPORT_DAYS", "7") or 7),
        ab_report_max_items=int(_get_secret("AB_REPORT_MAX_ITEMS", "20") or 20),
        ab_report_state_path=ab_report_state_path,
        video_metrics_state_path=video_metrics_state_path,
        upload_platform=(_get_secret("UPLOAD_PLATFORM", "youtube") or "youtube").strip().lower(),
        enable_instagram_upload=_get_bool("ENABLE_INSTAGRAM_UPLOAD", False),
        instagram_access_token=(_get_secret("INSTAGRAM_ACCESS_TOKEN", "") or "").strip(),
        instagram_user_id=(_get_secret("INSTAGRAM_USER_ID", "") or "").strip(),
        instagram_use_popular_audio=_get_bool("INSTAGRAM_USE_POPULAR_AUDIO", True),
        jp_youtube_only=_get_bool("JP_YOUTUBE_ONLY", True),
    )


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    blocks = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return text


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "\\" and in_string:
            escape = not escape
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if in_string:
            escape = False
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
        escape = False
    return ""


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    raw = text.strip()
    raw = _strip_code_fences(raw)
    candidates: List[str] = []
    if raw:
        candidates.append(raw)
    extracted = _extract_first_json_object(raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    for candidate in candidates:
        cand = candidate.strip()
        if not cand:
            continue
        # 흔한 JSON 오류 보정: trailing comma 제거
        cand = re.sub(r",\\s*([}\\]])", r"\\1", cand)
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"story_timeline": parsed}
        except Exception:
            pass
        # single-quote dict 등 파이썬 literal fallback
        try:
            parsed = ast.literal_eval(cand)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"story_timeline": parsed}
        except Exception:
            pass
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


def _compress_to_story_parts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = ["hook", "problem", "failure", "success", "point", "reaction"]
    picked: Dict[str, Dict[str, Any]] = {}
    for item in items:
        role = item.get("role")
        if role in order and role not in picked:
            picked[role] = item
    if len(picked) == len(order):
        compressed = [picked[r] for r in order]
        for idx, item in enumerate(compressed, start=1):
            item["order"] = idx
        return compressed
    return items


_FORBIDDEN_MYSTERY_PATTERNS = re.compile(
    r"(幽霊|心霊|怪談|呪い|悪魔|オカルト|降霊|UFO|宇宙人|ミイラ|古代文明|中世|王朝|海賊船)",
    re.IGNORECASE,
)
_TOO_OLD_YEAR_PATTERNS = re.compile(r"(17\d{2}|18\d{2})")
_MODERN_DAILY_HINT_PATTERNS = re.compile(
    r"(スマホ|アプリ|SNS|インスタ|YouTube|TikTok|Netflix|Amazon|Uber|コンビニ|マクドナルド|スタバ|コカ.?コーラ|ペプシ|ナイキ|iPhone|AirPods|サブスク|デリバリー|QR決済|ポイント|クーポン|カップ麺|ラーメン)",
    re.IGNORECASE,
)


def _check_story_quality(items: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> List[str]:
    issues: List[str] = []
    required = ["hook", "problem", "failure", "success", "point", "reaction"]
    roles = {item.get("role") for item in items}
    missing = [r for r in required if r not in roles]
    if missing:
        issues.append(f"역할 누락: {', '.join(missing)}")
        return issues

    def _first_text(role: str) -> str:
        for item in items:
            if item.get("role") == role:
                return str(item.get("script_ja", "")).strip()
        return ""

    hook = _first_text("hook")
    problem = _first_text("problem")
    failure = _first_text("failure")
    success = _first_text("success")
    point = _first_text("point")
    reaction = _first_text("reaction")

    def _len_bad(text: str) -> bool:
        return len(text) < 6 or len(text) > 80

    if _len_bad(hook):
        issues.append("Hook 길이 부적절")
    if _len_bad(problem):
        issues.append("Problem 길이 부적절")
    if _len_bad(failure):
        issues.append("Failure 길이 부적절")
    if _len_bad(success):
        issues.append("Success 길이 부적절")
    if _len_bad(point):
        issues.append("Point 길이 부적절")
    if _len_bad(reaction):
        issues.append("Reaction 길이 부적절")

    # 일본어 여부(히라가나/カタカナ/一部漢字) 간단 체크
    def _has_japanese(text: str) -> bool:
        return bool(re.search(r"[\\u3040-\\u30ff]", text))

    for label, txt in [("hook", hook), ("problem", problem), ("failure", failure), ("success", success), ("point", point), ("reaction", reaction)]:
        if txt and not _has_japanese(txt):
            issues.append(f"{label} 일본어 느낌 부족")

    # Hook 임팩트
    if not re.search(r"(衝撃|ヤバ|マジ|嘘|閲覧注意|知らない|やべ|危険|99%|ガチ|まじか|ほんま)", hook):
        issues.append("Hook 임팩트 부족")
    if not re.search(r"(知ってた|まだ|おい|消えかけ|マジ|嘘|[?？])", hook):
        issues.append("Hook 어그로/질문 부족")
    # Problem: 裏側/問題系
    if not re.search(r"(問題|裏|秘密|闇|危険|違和感|炎上|黒歴史|やばい|罠|トリック)", problem):
        issues.append("Problem 내용 약함")
    if problem and not problem.startswith(("実は", "そもそも", "実際")):
        issues.append("Problem 전개 연결 약함")
    # Failure: 실패/비판
    if not re.search(r"(失敗|炎上|批判|最悪|叩かれ|伸びなかった|滑った|ハマらなかった|空振り)", failure):
        issues.append("Failure 내용 약함")
    if failure and not failure.startswith(("でも", "だけど", "ところが")):
        issues.append("Failure 전환 연결 약함")
    # Success: 반전/성공
    if not re.search(r"(でも|ところが|実は|まさか|逆転|成功|大ヒット|一気に伸びた|バズった)", success):
        issues.append("Success 반전 부족")
    if success and not success.startswith(("ところが", "しかし", "でも")):
        issues.append("Success 전환 연결 약함")
    # Point: 핵심
    if not re.search(r"(ポイント|理由|核心|決め手|ミソ|キモ)", point):
        issues.append("Point 문장 약함")
    if point and not point.startswith(("ガチでヤバいポイントは", "結局", "要するに")):
        issues.append("Point 정리문 약함")
    # Reaction: 1인칭
    if not re.search(r"(俺|私|やば|マジ|草|w|笑|ほんま|なんやねん|うわ)", reaction):
        issues.append("Reaction 1인칭/감정 부족")

    meta = meta or {}
    title_ja = str(meta.get("title_ja", "") or "")
    topic_en = str(meta.get("topic_en", "") or "")
    whole_text = " ".join(
        [title_ja, topic_en]
        + [str(item.get("script_ja", "") or "") for item in items]
    )
    if _FORBIDDEN_MYSTERY_PATTERNS.search(whole_text):
        issues.append("초자연/유령 계열 소재 금지")
    if _TOO_OLD_YEAR_PATTERNS.search(whole_text):
        issues.append("1800년대 이전/유사 고전 연도 소재 금지")
    if not _MODERN_DAILY_HINT_PATTERNS.search(whole_text):
        issues.append("현대 일상 공감 키워드 부족")

    return issues


_HOOK_TOPIC_LABEL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"(mcdonald|マクドナルド|맥도날드)", re.IGNORECASE), "マクドナルド"),
    (re.compile(r"(coca[- ]?cola|コカ.?コーラ|코카콜라)", re.IGNORECASE), "コカ・コーラ"),
    (re.compile(r"(pepsi|ペプシ|펩시)", re.IGNORECASE), "ペプシ"),
    (re.compile(r"(nike|ナイキ)", re.IGNORECASE), "ナイキ"),
    (re.compile(r"(instagram|インスタ|인스타)", re.IGNORECASE), "インスタ"),
    (re.compile(r"(ramen|ラーメン|라면)", re.IGNORECASE), "ラーメン"),
]


def _topic_label_for_hook(meta: Dict[str, Any], hook_text: str = "") -> str:
    joined = " ".join(
        [
            str(meta.get("topic_en", "") or ""),
            str(meta.get("title_ja", "") or ""),
            str(hook_text or ""),
        ]
    )
    for pattern, label in _HOOK_TOPIC_LABEL_PATTERNS:
        if pattern.search(joined):
            return label
    title = str(meta.get("title_ja", "") or "").strip()
    if title:
        # 일본어 제목의 앞쪽 명사구를 최대한 짧게 사용
        title = re.sub(r"[【】\[\]()（）].*$", "", title).strip()
        return title[:10] if title else "この話"
    return "この話"


def _is_strong_hook(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"(知ってた|まだ|おい|消えかけ|嘘|ヤバ|マジ|[?？])", text)
    )


def _enforce_aggressive_hook(meta: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    for item in items:
        if item.get("role") != "hook":
            continue
        original = str(item.get("script_ja", "") or "").strip()
        if _is_strong_hook(original):
            return items
        label = _topic_label_for_hook(meta, original)
        forced = f"おい、{label}が一時期ガチで終わりかけたって知ってた？"
        item["script_ja"] = forced
        if not str(item.get("script_ko", "") or "").strip():
            item["script_ko"] = f"야, {label}가 한때 진짜 망할 뻔한 거 알고 있었어?"
        return items
    return items


def _normalize_story_line(text: str, max_len: int = 90) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return value
    return value


def _ensure_prefix(text: str, prefix: str) -> str:
    value = str(text or "").strip()
    if not value:
        return prefix.rstrip("、")
    if value.startswith(prefix) or value.startswith(prefix.rstrip("、")):
        return value
    return f"{prefix}{value}"


def _enforce_story_arc(meta: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hook→Problem→Failure→Success→Point→Reaction 흐름이 명확히 보이도록
    역할별 연결 접두어와 최소 임팩트를 보정.
    """
    if not items:
        return items
    label = _topic_label_for_hook(meta, "")
    fixed: List[Dict[str, Any]] = []
    for item in items:
        cur = dict(item)
        role = str(cur.get("role", "") or "").strip().lower()
        ja = _normalize_story_line(cur.get("script_ja", ""))
        if role == "problem":
            if len(ja) < 8:
                ja = f"実は、{label}にはほとんど知られてない裏の仕組みがある。"
            elif not ja.startswith(("実は", "そもそも", "実際")):
                ja = _ensure_prefix(ja, "実は、")
        elif role == "failure":
            if not re.search(r"(失敗|炎上|批判|最悪|叩かれ|伸びなかった|滑った)", ja):
                ja = f"でも最初は、{ja}って空気で普通に叩かれて失敗扱いされた。"
            elif not ja.startswith(("でも", "だけど", "ところが")):
                ja = _ensure_prefix(ja, "でも最初は、")
        elif role == "success":
            if not re.search(r"(ところが|でも|逆転|成功|一気に伸びた|バズった)", ja):
                ja = f"ところが、見せ方を変えた瞬間に評価が逆転して一気に伸びた。"
            elif not ja.startswith(("ところが", "しかし", "でも")):
                ja = _ensure_prefix(ja, "ところが、")
        elif role == "point":
            if not re.search(r"(ポイント|理由|決め手|ミソ|キモ)", ja):
                ja = f"ガチでヤバいポイントは、{ja}"
            elif not ja.startswith(("ガチでヤバいポイントは", "結局", "要するに")):
                ja = _ensure_prefix(ja, "ガチでヤバいポイントは、")
        elif role == "reaction":
            if not re.search(r"(俺|私|やば|マジ|草|w|笑|うわ)", ja):
                ja = f"俺は正直、{ja}"
        cur["script_ja"] = _normalize_story_line(ja, max_len=92)
        if not str(cur.get("script_ko", "") or "").strip():
            cur["script_ko"] = _to_ko_literal_tone(cur["script_ja"])
        fixed.append(cur)
    return fixed


def _inject_majisho_asmr_beat(script: Dict[str, Any], enabled: bool = True) -> Dict[str, Any]:
    if not enabled:
        return script
    timeline = _get_story_timeline(script)
    if not timeline:
        return script
    # 중복 삽입 방지
    for row in timeline:
        role = str(row.get("role", "") or "").strip().lower()
        txt = str(row.get("script_ja", "") or "")
        if role == "asmr_tag" or "マジショ" in txt:
            return script

    hook_idx = 0
    for idx, row in enumerate(timeline):
        if str(row.get("role", "") or "").strip().lower() == "hook":
            hook_idx = idx
            break

    insert_item = {
        "order": hook_idx + 2,
        "role": "asmr_tag",
        "script_ja": "マジショ",
        "script_ko": "마지쇼",
        "visual_search_keyword": "cute anime mascot マジショ sign",
        "duration": "1s",
    }
    new_timeline = timeline[: hook_idx + 1] + [insert_item] + timeline[hook_idx + 1 :]
    for idx, row in enumerate(new_timeline, start=1):
        if isinstance(row, dict):
            row["order"] = idx
    script["story_timeline"] = new_timeline
    script["content"] = new_timeline
    return script


def _apply_majisho_interlude_assets(
    config: AppConfig,
    roles: List[str],
    bg_video_paths: List[Optional[str]],
    bg_image_paths: List[Optional[str]],
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    if not getattr(config, "enable_majisho_tag", True):
        return bg_video_paths, bg_image_paths
    if not roles:
        return bg_video_paths, bg_image_paths
    target_idx = -1
    for idx, role in enumerate(roles):
        if role == "asmr_tag":
            target_idx = idx
            break
    if target_idx < 0:
        return bg_video_paths, bg_image_paths
    asset_path = str(getattr(config, "majisho_asset_path", "") or "").strip()
    if not asset_path or not os.path.exists(asset_path):
        return bg_video_paths, bg_image_paths

    if len(bg_video_paths) < len(roles):
        bg_video_paths = (bg_video_paths + [None] * len(roles))[: len(roles)]
    if len(bg_image_paths) < len(roles):
        bg_image_paths = (bg_image_paths + [_ensure_placeholder_image(config)] * len(roles))[: len(roles)]

    ext = os.path.splitext(asset_path)[1].lower()
    if ext in {".mp4", ".mov", ".webm", ".mkv"}:
        bg_video_paths[target_idx] = asset_path
        bg_image_paths[target_idx] = _ensure_placeholder_image(config)
    else:
        bg_video_paths[target_idx] = None
        bg_image_paths[target_idx] = asset_path
    return bg_video_paths, bg_image_paths


# ─────────────────────────────────────────────
# NEW: 뿜 글 분위기 카테고리 분류
# ─────────────────────────────────────────────

# 지원 카테고리 (BGM 검색어 + 에셋 태그에 공통 사용)
CONTENT_CATEGORIES = [
    "humor",       # 유머/웃김
    "touching",    # 감동/눈물
    "shocking",    # 충격/반전
    "heartwarming",# 훈훈/힐링
    "cringe",      # 어이없음/공감
    "exciting",    # 신남/에너지
    "sad",         # 슬픔/공감
    "anger",       # 분노/황당
]

# 카테고리 → Pixabay 검색 키워드 매핑
BGM_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "humor":        ["funny", "comedic", "quirky upbeat"],
    "touching":     ["emotional piano", "heartfelt", "touching cinematic"],
    "shocking":     ["suspense", "dramatic tension", "thriller"],
    "heartwarming": ["warm acoustic", "uplifting gentle", "feel good"],
    "cringe":       ["awkward comedy", "silly funny", "quirky"],
    "exciting":     ["energetic upbeat", "hype electronic", "motivation"],
    "sad":          ["sad piano", "melancholy", "emotional"],
    "anger":        ["intense dramatic", "aggressive rock", "tension"],
    # 폭로/고발 쇼츠용 무드 키워드
    "mystery_suspense": ["suspense", "mystery", "dark ambient", "thriller"],
    "fast_exciting": ["action", "fast beat", "energetic", "epic"],
}

# 카테고리 → 에셋 tags 매핑 (기존 태그 시스템과 연결)
CATEGORY_TO_ASSET_TAGS: Dict[str, List[str]] = {
    "humor":        ["laugh", "awkward"],
    "touching":     ["cute", "ending"],
    "shocking":     ["shock", "wow"],
    "heartwarming": ["cute", "plot"],
    "cringe":       ["facepalm", "awkward"],
    "exciting":     ["wow", "shock"],
    "sad":          ["ending", "plot"],
    "anger":        ["angry", "facepalm"],
}


def analyze_content_category(
    config: AppConfig,
    text: str,
) -> str:
    """뿜 글 텍스트를 분석해서 CONTENT_CATEGORIES 중 하나를 반환."""
    if not config.openai_api_key or not text:
        return "humor"
    system_text = (
        "You are a content mood classifier. "
        "Given Korean text, classify its mood into exactly one category. "
        f"Categories: {', '.join(CONTENT_CATEGORIES)}. "
        "Return JSON only: {\"category\": \"...\"}"
    )
    user_text = f"Text: {text[:800]}"
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
        category = result.get("category", "humor")
        if category in CONTENT_CATEGORIES:
            return category
    except Exception:
        pass
    return "humor"


# ─────────────────────────────────────────────
# NEW: Pixabay Audio BGM 자동 다운로드
# ─────────────────────────────────────────────

def _pick_pixabay_audio_url(hit: Dict[str, Any]) -> str:
    candidates: List[str] = []
    audio = hit.get("audio")
    if isinstance(audio, str):
        candidates.append(audio)
    elif isinstance(audio, dict):
        for key in (
            "url",
            "previewURL",
            "previewUrl",
            "preview",
            "large",
            "medium",
            "small",
            "download",
            "downloadURL",
            "downloadUrl",
            "file",
        ):
            value = audio.get(key)
            if isinstance(value, str):
                candidates.append(value)
    for key in ("audio_url", "url", "previewURL", "previewUrl", "preview", "downloadURL", "downloadUrl"):
        value = hit.get(key)
        if isinstance(value, str):
            candidates.append(value)
    for value in candidates:
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def fetch_bgm_from_pixabay(
    api_key: str,
    category: str,
    output_dir: str,
    custom_query: str = "",  # AI가 생성한 BGM 검색 쿼리 (우선 사용)
    config: Optional["AppConfig"] = None,
) -> Optional[str]:
    """
    카테고리에 맞는 BGM을 Pixabay에서 검색 후 다운로드.
    custom_query가 있으면 우선 사용, 없으면 카테고리 키워드 사용.
    성공하면 저장된 파일 경로 반환, 실패하면 None.
    """
    if not api_key:
        return None
    if isinstance(custom_query, list):
        custom_query = " ".join(custom_query)
    custom_query = custom_query or ""  # None/list 방어
    if custom_query.strip():
        query = custom_query.strip()
    else:
        keywords = BGM_CATEGORY_KEYWORDS.get(category, ["upbeat"])
        query = random.choice(keywords)
    os.makedirs(output_dir, exist_ok=True)
    def _bgm_log(message: str) -> None:
        if config is not None:
            try:
                _telemetry_log(message, config)
            except Exception:
                pass
        _append_bgm_debug(message)

    try:
        # 저장 경로가 쓰기 불가면 /tmp 폴백
        output_dir = _ensure_writable_dir(output_dir, "/tmp/auto_shorts_bgm")
        _bgm_log(f"Pixabay BGM 저장 경로: {output_dir}")
        # Pixabay audio API endpoint
        music_params = {
            "key": api_key,
            "q": query,
            "per_page": 10,
            "safesearch": "true",
        }
        music_response = requests.get(
            "https://pixabay.com/api/audio/",
            params=music_params,
            timeout=30,
        )
        if music_response.status_code != 200:
            body_head = (music_response.text or "")[:160].replace("\n", " ")
            _bgm_log(
                f"Pixabay BGM API 실패: status={music_response.status_code} body={body_head}"
            )
            return None
        payload = music_response.json()
        hits = payload.get("hits", []) or []
        _bgm_log(f"Pixabay BGM 결과 수: {len(hits)} (query='{query}')")
        if not hits:
            return None
        # 랜덤으로 하나 선택
        hit = random.choice(hits[:5])
        audio_url = _pick_pixabay_audio_url(hit)
        if not audio_url:
            _bgm_log(f"Pixabay BGM URL 추출 실패: keys={list(hit.keys())}")
            return None
        audio_response = requests.get(audio_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        audio_response.raise_for_status()
        filename = f"pixabay_{category}_{random.randint(10000, 99999)}.mp3"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(audio_response.content)
        _bgm_log(f"Pixabay BGM 다운로드 완료: {os.path.basename(file_path)}")
        return file_path
    except Exception as exc:
        _bgm_log(f"Pixabay BGM 예외: {exc}")
        return None


def get_or_download_bgm(
    config: AppConfig,
    category: str,
    custom_query: str = "",  # AI 생성 BGM 키워드 (우선 사용)
) -> Optional[str]:
    """
    1) assets/bgm/{category}/ 에 기존 파일이 있으면 랜덤 선택
    2) 없으면 Pixabay에서 다운로드 후 반환 (custom_query 우선)
    """
    custom_query = custom_query or ""  # None 방어
    bgm_category_dir = os.path.join(config.assets_dir, "bgm", category)
    os.makedirs(bgm_category_dir, exist_ok=True)
    existing = _list_audio_files(bgm_category_dir)
    if existing:
        return random.choice(existing)
    if config.pixabay_api_key:
        path = fetch_bgm_from_pixabay(
            api_key=config.pixabay_api_key,
            category=category,
            output_dir=bgm_category_dir,
            custom_query=custom_query,
            config=config,
        )
        if path:
            return path
    # fallback: 기존 bgm 디렉토리
    return pick_bgm_path(config)


# ─────────────────────────────────────────────
# 일본인 타겟 숏츠 대본 생성 시스템
# ─────────────────────────────────────────────

# BGM 무드 카테고리 (mystery_suspense / fast_exciting)
BGM_MOOD_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "mystery_suspense": {
        "description": "폭로/고발/긴장감 — 미스터리, 조작, 충격 계열",
        "folder": "mystery_suspense",
    },
    "fast_exciting": {
        "description": "반전/속도감 — 에너지 높은 비트",
        "folder": "fast_exciting",
    },
}

BGM_MOOD_ALIASES: Dict[str, str] = {
    "mystery": "mystery_suspense",
    "suspense": "mystery_suspense",
    "exciting": "fast_exciting",
    "emotional": "mystery_suspense",
    "informative": "mystery_suspense",
}

# 롤렛 주제 풀 — LLM이 이 리스트를 참고해 매번 새 주제 생성
JP_CONTENT_THEMES: List[str] = [
    "マクドナルドのポテトが店舗で味ブレしにくい運用の裏側",
    "コカ・コーラとペプシの味覚テストで起きる錯覚トリック",
    "Instagramのおすすめ欄が伸びる投稿と沈む投稿の差",
    "TikTokの冒頭3秒で離脱率が決まる理由",
    "NetflixのサムネABテストが視聴行動を変える仕組み",
    "Amazonレビュー表示順のロジックと購買への影響",
    "Uberのサージ価格とユーザー心理の駆け引き",
    "コンビニ新商品が短期間で消える選定ロジック",
    "スタバ期間限定メニューの希少性マーケ戦略",
    "Nikeスニーカー抽選販売が熱狂を生む設計",
    "AirPodsが外れにくく感じる設計と錯覚ポイント",
    "カップ麺ヒット商品が生まれる試作サイクルの裏話",
]

# 반응형 고정댓글 후보 (랜덤 선택)
PINNED_COMMENT_VARIANTS: List[str] = [
    "これ信じる？コメントで教えて。",
    "あなたなら騙される？正直に教えて。",
    "実はこれ知ってた？体験談ちょうだい。",
    "どっち派？意見で揉めよう。",
    "これガチだと思う？反論歓迎。",
]

# 시스템 프롬프트 (LLM에 직접 전달)
JP_SHORTS_SYSTEM_PROMPT: str = """あなたは日本のYouTube Shortsトレンドを熟知した『現代ミステリー／裏話ストーリーテラー』。視聴者が1秒も離脱しないテンポで書け。

[スタイル]
1) 口調: 日本語のタメ口。です/ます禁止。
2) テンポ最優先。短文・辛口・シニカルに。
3) 最後は必ず1人称の独り言で締める。
4) 事実は「意外性・裏側・衝撃」に寄せる。

[テーマ制約]
1) 幽霊・怪談・超自然・オカルトは禁止。
2) 1800年代以前、古代史/中世史メインの話は禁止。
3) いまの生活で使うブランド/アプリ/食べ物/サービスだけ扱う。
4) 視聴者が「今日の自分に関係ある」と感じる具体例を必ず入れる。

[ストーリー構成: 6段階]
1) Hook: 目を止める挑発。例「まだこれ使ってるの？」
2) Problem: 裏側の問題／違和感を提示。
3) Failure: 最初は失敗・炎上・批判。
4) Success: でも/ところが/実は…で逆転。
5) Point: 成功の核心を一言で言い切る。
6) Reaction: 1人称の独り言で締める。

[構成強化ルール]
- 各パートは必ず前のパートの結果として続くこと（因果を切らない）。
- 接続語を明示すること（実は / でも最初は / ところが / つまり）。
- Hook→Problem→Failureで緊張を上げ、Success→Pointで回収する。

[注意]
- 句読点少なめ・テンポ重視
- 長文禁止、1行を短く
- script_jaは必ず日本語。script_koは確認用の韓国語訳だが、意訳せず「直訳寄り」で口調の強さ/温度感を維持すること

[JSON出力フォーマット]
{
  "meta": {
    "topic_en": "テーマ英語",
    "title_ja": "日本語タイトル(釣り/衝撃系)",
    "hashtags": ["#雑学", "#裏話", "#都市伝説", "#衝撃", "#ミステリー"],
    "pinned_comment": "視聴者参加を促す日本語質問",
    "pinned_comment_ko": "上の日本語コメントの韓国語直訳(口調を維持)",
    "bgm_mood": "mystery_suspense" または "fast_exciting"
  },
  "story_timeline": [
    {"order": 1, "role": "hook", "script_ja": "日本語タメ口", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"},
    {"order": 2, "role": "problem", "script_ja": "日本語タメ口", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"},
    {"order": 3, "role": "failure", "script_ja": "日本語タメ口", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"},
    {"order": 4, "role": "success", "script_ja": "日本語タメ口", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"},
    {"order": 5, "role": "point", "script_ja": "日本語タメ口", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"},
    {"order": 6, "role": "reaction", "script_ja": "日本語タメ口の独り言", "script_ko": "韓国語直訳(거친 말투 유지)", "visual_search_keyword": "英語検索キーワード"}
  ]
}
"""


_KO_LITERAL_ENDING_RULES: List[Tuple[str, str]] = [
    (r"입니다\.$", "임."),
    (r"입니다$", "임"),
    (r"였습니다\.$", "였음."),
    (r"였습니다$", "였음"),
    (r"습니다\.$", "음."),
    (r"습니다$", "음"),
    (r"하세요\.$", "해."),
    (r"하세요$", "해"),
    (r"해요\.$", "함."),
    (r"해요$", "함"),
    (r"해요\?$", "함?"),
]


def _to_ko_literal_tone(text: str) -> str:
    """
    일본어 원문의 거친 템포를 살리기 위해 한국어 번역을 직역/반말 톤으로 정규화.
    """
    value = str(text or "").strip()
    if not value:
        return value
    for pattern, repl in _KO_LITERAL_ENDING_RULES:
        value = re.sub(pattern, repl, value)
    # 존댓말 축소
    value = value.replace("합니다", "함").replace("했어요", "했음")
    value = value.replace("그래요", "그럼").replace("하지만", "근데")
    return value.strip()


def _compose_bilingual_text(
    ja_text: str,
    ko_text: str,
    ko_prefix: str = "(한글 직역)",
) -> str:
    ja = (ja_text or "").strip()
    ko = _to_ko_literal_tone(ko_text)
    if not ja and not ko:
        return ""
    if not ko or ko == ja:
        return ja or ko
    prefix = f"{ko_prefix} " if ko_prefix else ""
    return f"{ja}\n{prefix}{ko}"


def generate_viral_script() -> Dict[str, Any]:
    client = OpenAI()

    system_message = """당신은 현재 일본 유튜브 쇼츠 트렌드를 섭렵한 **'바이럴 마케팅 천재'**이자 **'현대 미스터리 스토리텔러'**입니다.
당신의 임무는 1) **클릭을 안 할 수 없는 썸네일/제목**을 설계하고, 2) **현대 문명 속 충격적인 비하인드 스토리**로 시청자를 붙잡아두며, 3) **댓글 창을 폭발시키는 질문**을 던지는 것입니다.

### 1. 주제 선정 (Modern & Mystery)
- **소재:** 100년 전 역사가 아닌, **지금 당장** 우리가 쓰는 앱, 먹는 음식, 입는 브랜드(Nike, McDonald's, Instagram, Ramen 등).
- **각도:** "이게 원래 마약이었다고?", "이게 원래 군사용이었다고?" 식의 **일상 속 배신감**을 주는 소재.

### 2. 메타데이터 전략 (일본 트렌드 반영)
- **썸네일 텍스트:** 영상 첫 화면에 크게 박힐 텍스트. 5~8글자 이내의 일본어. (예: 閲覧注意, 99%が誤解, 衝撃の正体)
- **해시태그:** 대형 키워드(#雑学)와 니치 키워드(#裏話)를 섞어서 알고리즘 타겟팅.
- **고정 댓글:** 시청자가 자기 경험을 말하고 싶어서 안달 나게 만드는 질문. (논쟁 유발 or 공감 유도)

### 3. 대본 작성 (몰랐숏 스타일)
- **말투:** 일본어 반말(Tameguchi). 빠르고 시니컬하게. (~Desu/Masu 절대 금지)
- **흐름:**
    1. **Hook:** "너 이거 쓰면서도 몰랐지?" (무시/도발)
    2. **Secret:** "사실 이거 원래 ㅇㅇㅇ였어." (충격)
    3. **Twist:** "근데 어떤 천재가 이걸 바꿔서 대박 난 거야." (전환)
    4. **Outro:** "아, 나도 하나 사러 가야겠다." (1인칭 혼잣말 엔딩)

### 4. JSON 출력 형식 (파이썬 연동용 - 엄격 준수)
반드시 아래 JSON 포맷으로만 응답하세요.

{
  "meta": {
    "topic_en": "주제 키워드 (영어)",
    "title_ja": "일본어 제목 (클릭을 부르는 어그로성 제목, 예: 【衝撃】マックのポテトが美味すぎる恐ろしい理由)",
    "thumbnail_text_ja": "썸네일용 텍스트 (짧고 강렬하게, 예: 中毒の正体)",
    "hashtags": ["#雑学", "#都市伝説", "#衝撃", "#ライフハック", "#裏話"],
    "pinned_comment_ja": "고정 댓글 (시청자 참여 유도, 예: あなたはマック派？モス派？コメントで教えて！)",
    "bgm_mood": "mystery_suspense"
  },
  "shorts_script": [
    {
      "order": 1,
      "role": "hook",
      "duration": "4s",
      "txt_ja": "일본어 대본 (시청자가 지금 쓰고 있는 물건 지적)",
      "txt_ko": "한국어 번역 (확인용)",
      "visual_keyword_en": "close up of modern product (specific brand or item)"
    },
    {
      "order": 2,
      "role": "secret",
      "duration": "10s",
      "txt_ja": "일본어 대본 (충격적인 과거/비밀 폭로)",
      "txt_ko": "한국어 번역",
      "visual_keyword_en": "black and white or shocking contrast image (e.g., war, chemical, trash)"
    },
    {
      "order": 3,
      "role": "twist",
      "duration": "15s",
      "txt_ja": "일본어 대본 (성공의 반전 포인트)",
      "txt_ko": "한국어 번역",
      "visual_keyword_en": "money falling or people cheering or factory production"
    },
    {
      "order": 4,
      "role": "outro",
      "duration": "5s",
      "txt_ja": "일본어 대본 (친구 같은 1인칭 혼잣말)",
      "txt_ko": "한국어 번역 (예: 아.. 소름 돋네..)",
      "visual_keyword_en": "person shrugging or looking surprised or eating product"
    }
  ]
}
"""

    user_message = "현대인들이 흥미로워할 랜덤 주제 하나를 선정해서 작성해줘."

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    output_text = getattr(response, "output_text", "") or ""
    return json.loads(output_text)


def generate_script_jp(
    config: AppConfig,
    extra_hint: str = "",
) -> Dict[str, Any]:
    """
    LLM이 주제를 스스로 선정해 일본인 타겟 숏츠 대본을 JSON으로 생성.
    extra_hint: 추가로 힌트를 줄 때 사용 (예: 특정 지역, 키워드)
    """
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")

    dialect_style = (getattr(config, "ja_dialect_style", "") or "").strip()
    theme_pool = "\n".join(f"- {t}" for t in JP_CONTENT_THEMES)
    user_text = (
        "以下のテーマ例を参考に、日本人向けの現代ミステリー/裏話ショート台本を作成してください。\n"
        "Hook → Problem → Failure → Success → Point → Reaction の6段構成を守り、タメ口でテンポよく書いてください。\n"
        "各行が前行の結果として自然につながるように書いてください（因果関係を必ず維持）。\n"
        "幽霊/怪談/オカルト/古代史/中世史/1800年代以前のネタは使わないでください。\n"
        "視聴者の日常に直結するテーマ(ブランド・アプリ・食べ物・サービス)のみ選んでください。\n"
        + (f"日本語の口調は必ず「{dialect_style}」のニュアンスで統一してください。\n" if dialect_style else "")
        + "必ずJSONのみを出力してください。\n\n"
        f"[テーマ例]\n{theme_pool}\n\n"
        + (f"[追加ヒント]\n{extra_hint}\n\n" if extra_hint else "")
        + "上記のシステムプロンプトとJSON形式を厳守し、純粋なJSONのみを出力してください。"
    )

    client = OpenAI(api_key=config.openai_api_key)
    last_error = ""
    feedback = ""
    for attempt in range(3):
        prompt = user_text
        if feedback:
            prompt += "\n\n[품질 피드백]\n" + feedback + "\n위 문제를 반드시 해결하고 JSON만 출력하세요."
        response = client.responses.create(
            model=config.openai_model,
            input=[
                {"role": "system", "content": JP_SHORTS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        output_text = getattr(response, "output_text", "") or ""
        result = extract_json(output_text)
        if not result:
            last_error = "LLM JSON 파싱 실패"
            try:
                _telemetry_log(f"LLM JSON 파싱 실패. 원문 일부:\n{output_text[:1600]}", config)
            except Exception:
                pass
            try:
                with open("/tmp/auto_shorts_llm_output.log", "w", encoding="utf-8") as file:
                    file.write(output_text)
            except Exception:
                pass
            feedback = "JSON 형식 위반 또는 불완전한 출력"
            continue

        # ── 새 스키마 (meta + story_timeline[]) 정규화 ───────────
        meta = result.get("meta", {}) if isinstance(result.get("meta", {}), dict) else {}
        story_list = result.get("story_timeline", None)
        if not isinstance(story_list, list):
            # 구 스키마 fallback
            story_list = result.get("content", []) if isinstance(result.get("content"), list) else []

        # meta 필드 검증
        title_ja = meta.get("title_ja") or meta.get("title") or meta.get("video_title") or "ミステリーショーツ"
        meta["title_ja"] = title_ja
        meta["title"] = title_ja
        if not meta.get("topic_en"):
            meta["topic_en"] = ""
        hashtags = meta.get("hashtags", [])
        if isinstance(hashtags, str):
            hashtags = [t for t in re.split(r"[,\s]+", hashtags) if t]
        meta["hashtags"] = normalize_hashtags(hashtags if isinstance(hashtags, list) else [])
        mood_raw = meta.get("bgm_mood", "")
        if mood_raw in BGM_MOOD_ALIASES:
            meta["bgm_mood"] = BGM_MOOD_ALIASES[mood_raw]
        if meta.get("bgm_mood") not in BGM_MOOD_CATEGORIES:
            meta["bgm_mood"] = "mystery_suspense"
        # 고정댓글은 여러 버전 중 랜덤 선택 (LLM 생성 문장도 옵션에 포함)
        comment_pool = list(PINNED_COMMENT_VARIANTS)
        if meta.get("pinned_comment"):
            comment_pool.append(str(meta["pinned_comment"]))
        meta["pinned_comment"] = random.choice(comment_pool)
        pinned_ko_raw = meta.get("pinned_comment_ko") or meta["pinned_comment"]
        meta["pinned_comment_ko"] = _to_ko_literal_tone(str(pinned_ko_raw or ""))
        # 해시태그 최소 4개 유지
        if isinstance(meta.get("hashtags"), list) and len(meta["hashtags"]) < 4:
            defaults = ["#雑学", "#裏話", "#都市伝説", "#衝撃", "#ミステリー"]
            for tag in defaults:
                if len(meta["hashtags"]) >= 4:
                    break
                if tag not in meta["hashtags"]:
                    meta["hashtags"].append(tag)

        # story_timeline 정렬 및 검증
        normalized: List[Dict[str, Any]] = []
        allowed_roles = {"hook", "problem", "failure", "success", "point", "reaction"}
        forced_brand_hint = _detect_brand_query_hint(
            str(meta.get("topic_en", "") or meta.get("title_ja", "") or ""),
            str(meta.get("title_ja", "") or ""),
            "",
        )
        order_map = {
            "hook": 1,
            "problem": 2,
            "failure": 3,
            "success": 4,
            "point": 5,
            "reaction": 6,
        }
        for idx, raw in enumerate(story_list or []):
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role", "")).strip().lower()
            if role not in allowed_roles:
                if "reaction" in role or "outro" in role:
                    role = "reaction"
                elif role.startswith("twist") or "success" in role or "반전" in role:
                    role = "success"
                elif "failure" in role or "실패" in role:
                    role = "failure"
                elif "problem" in role or "위기" in role:
                    role = "problem"
                elif "point" in role or "포인트" in role:
                    role = "point"
                elif role == "hook":
                    role = "hook"
                else:
                    role = "problem" if idx > 0 else "hook"
            order_val = raw.get("order")
            if not isinstance(order_val, int):
                order_val = order_map.get(role, idx + 1)
            script_ja = str(raw.get("script_ja") or "").strip()
            script_ko = str(raw.get("script_ko") or "").strip()
            if not script_ko and script_ja:
                script_ko = script_ja
            script_ko = _to_ko_literal_tone(script_ko)
            visual_kw = raw.get("visual_search_keyword") or raw.get("visual_keyword_en") or ""
            visual_kw = str(visual_kw).strip()
            visual_kw = _refine_visual_keyword(
                visual_kw,
                meta.get("topic_en", ""),
                role,
                script_ja,
                forced_brand_hint=forced_brand_hint,
            )
            normalized.append(
                {
                    "order": order_val,
                    "role": role,
                    "script_ja": script_ja,
                    "script_ko": script_ko,
                    "visual_search_keyword": visual_kw,
                }
            )

        normalized = sorted(normalized, key=lambda x: x.get("order", 99))
        normalized = _compress_to_story_parts(normalized)
        normalized = _enforce_aggressive_hook(meta, normalized)
        normalized = _enforce_story_arc(meta, normalized)
        issues = _check_story_quality(normalized, meta=meta)
        if issues:
            feedback = " / ".join(issues)
            last_error = "스토리 구조 품질 미달"
            if attempt < 2:
                continue
            # 마지막 시도면 경고 로그만 남기고 진행 (파이프라인 중단 방지)
            try:
                _telemetry_log(f"스토리 품질 경고(통과 처리): {feedback}", config)
            except Exception:
                pass

        result["meta"] = meta
        result["story_timeline"] = normalized
        # 하위 호환: UI/기존 로직에서 content를 참조하는 경우 대응
        result["content"] = normalized
        return result

    raise RuntimeError(last_error or "LLM 스토리 생성 실패")


def _get_story_timeline(script: Dict[str, Any]) -> List[Dict[str, Any]]:
    timeline = script.get("story_timeline")
    if isinstance(timeline, list):
        return timeline
    content_list = script.get("content")
    if isinstance(content_list, list):
        return content_list
    return []


def _script_to_beats(script: Dict[str, Any]) -> List[str]:
    """generate_script_jp 결과(새 스키마)를 TTS/영상용 텍스트 리스트로 변환."""
    timeline = _get_story_timeline(script)
    if timeline:
        texts: List[str] = []
        for item in timeline:
            ja = item.get("script_ja", "")
            ko = item.get("script_ko", "")
            if ja:
                texts.append(ja)
            elif ko:
                texts.append(ko)
        return texts
    # 구 스키마 fallback (하위 호환)
    texts: List[str] = []
    hook = script.get("hook_3_sec", "")
    if hook:
        texts.append(hook)
    for line in script.get("body_script", []):
        if line:
            texts.append(line)
    outro = script.get("cta_outro", "")
    if outro:
        texts.append(outro)
    return texts


def _script_to_beats_ko(script: Dict[str, Any]) -> List[str]:
    """generate_script_jp 결과에서 한국어 텍스트 리스트를 반환 (참고용)."""
    timeline = _get_story_timeline(script)
    if timeline:
        return [item.get("script_ko", item.get("script_ja", "")) for item in timeline]
    texts_ko: List[str] = []
    hook_ko = script.get("hook_3_sec_ko", script.get("hook_3_sec", ""))
    if hook_ko:
        texts_ko.append(hook_ko)
    for line in script.get("body_script_ko", script.get("body_script", [])):
        if line:
            texts_ko.append(line)
    outro_ko = script.get("cta_outro_ko", script.get("cta_outro", ""))
    if outro_ko:
        texts_ko.append(outro_ko)
    return texts_ko


def _script_to_visual_keywords(script: Dict[str, Any]) -> List[str]:
    """각 세그먼트의 visual_search_keyword 리스트 반환."""
    timeline = _get_story_timeline(script)
    if timeline:
        topic = str((script.get("meta", {}) or {}).get("topic_en", "") or "")
        title = str((script.get("meta", {}) or {}).get("title_ja", "") or "")
        forced_brand_hint = _detect_brand_query_hint(topic, title, "")
        keywords: List[str] = []
        for item in timeline:
            role = str(item.get("role", "") or "")
            script_ja = str(item.get("script_ja", "") or "")
            raw_kw = (
                item.get("visual_search_keyword")
                or item.get("visual_keyword_en")
                or "archival newspaper headline close up"
            )
            keywords.append(
                _refine_visual_keyword(
                    str(raw_kw),
                    topic,
                    role,
                    script_ja,
                    forced_brand_hint=forced_brand_hint,
                )
            )
        return keywords
    # 구 스키마 fallback — 전체 공통 키워드 반복
    default_kw = script.get("bg_search_query", "archival newspaper headline close up")
    texts = _script_to_beats(script)
    return [default_kw] * len(texts)


def _script_to_roles(script: Dict[str, Any]) -> List[str]:
    """각 세그먼트의 role 리스트 반환."""
    timeline = _get_story_timeline(script)
    if timeline:
        roles: List[str] = []
        for item in timeline:
            role = str(item.get("role", "")).strip().lower()
            if role in {"asmr_tag", "asmr", "tag"}:
                role = "asmr_tag"
            if "reaction" in role or "outro" in role:
                role = "reaction"
            elif role.startswith("twist") or "success" in role or "반전" in role:
                role = "success"
            elif "conflict" in role:
                role = "problem"
            roles.append(role or "problem")
        return roles
    # 구 스키마 fallback
    texts = _script_to_beats(script)
    if not texts:
        return []
    if len(texts) == 1:
        return ["hook"]
    return ["hook"] + ["body"] * (len(texts) - 2) + ["outro"]


def _build_tts_segments(texts: List[str], roles: List[str]) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    for idx, text in enumerate(texts):
        line = str(text or "").strip()
        if not line:
            continue
        role = str(roles[idx] if idx < len(roles) else "body")
        segments.append({"text": line, "role": role})
    return segments


def _build_caption_styles(roles: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if not roles:
        return ["default"] * count
    styles: List[str] = []
    for i in range(count):
        role = roles[i] if i < len(roles) else ""
        if role == "reaction":
            styles.append("reaction")
        elif role == "asmr_tag":
            styles.append("asmr_tag")
        else:
            styles.append("default")
    return styles


def _select_caption_variant(config: AppConfig) -> str:
    if not getattr(config, "ab_test_enabled", False):
        return "default"
    return random.choice(["default", "japanese_variety"])


def _apply_caption_variant(styles: List[str], variant: str) -> List[str]:
    if not styles:
        return styles
    if variant == "japanese_variety":
        return [
            "reaction" if s == "reaction" else ("asmr_tag" if s == "asmr_tag" else "japanese_variety")
            for s in styles
        ]
    if variant == "default":
        return [
            "reaction" if s == "reaction" else ("asmr_tag" if s == "asmr_tag" else "default_plain")
            for s in styles
        ]
    return styles


def _load_ab_records(config: AppConfig, lookback: int) -> List[Dict[str, Any]]:
    path = os.path.join(config.output_dir, "ab_tests.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()[-max(lookback, 1):]
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        return records
    except Exception:
        return []


_HORROR_BG_TERMS = [
    "horror", "creepy", "ghost", "haunted", "murder", "serial killer",
    "missing", "kidnap", "crime", "slasher", "bloody", "terror",
    "ホラー", "恐怖", "怪談", "心霊", "呪い", "幽霊", "殺人", "失踪", "誘拐", "事件",
    "공포", "귀신", "괴담", "심령", "저주", "살인", "실종", "납치", "범죄",
]
_HUMOR_DAILY_BG_TERMS = [
    "funny", "humor", "comedy", "meme", "daily", "everyday", "lifehack", "relatable",
    "ギャグ", "コメディ", "おもしろ", "爆笑", "ネタ", "日常", "あるある", "ライフハック",
    "유머", "개그", "웃긴", "밈", "일상", "꿀팁", "공감",
]


def _infer_background_mode_by_content(
    meta: Optional[Dict[str, Any]],
    texts: Optional[List[str]],
    visual_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    meta = meta or {}
    merged: List[str] = []
    for key in ("topic_en", "title_ja", "thumbnail_text_ja", "title"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            merged.append(val.strip())
    if texts:
        merged.extend([t for t in texts if isinstance(t, str) and t.strip()])
    if visual_keywords:
        merged.extend([k for k in visual_keywords if isinstance(k, str) and k.strip()])
    if not merged:
        return None
    joined = " ".join(merged).lower()
    if any(term.lower() in joined for term in _HORROR_BG_TERMS):
        return "image"
    if any(term.lower() in joined for term in _HUMOR_DAILY_BG_TERMS):
        return "minecraft"
    return None


def _select_background_mode(
    config: AppConfig,
    meta: Optional[Dict[str, Any]] = None,
    texts: Optional[List[str]] = None,
    visual_keywords: Optional[List[str]] = None,
) -> str:
    modes = ["minecraft", "image"]
    if not getattr(config, "use_minecraft_parkour_bg", True) or not getattr(config, "use_bg_videos", True):
        return "image"
    forced = _infer_background_mode_by_content(meta, texts, visual_keywords)
    if forced:
        return forced
    epsilon = float(getattr(config, "background_ab_epsilon", 0.2))
    lookback = int(getattr(config, "background_ab_lookback", 30))
    if not getattr(config, "ab_test_enabled", False):
        return random.choice(modes)
    if random.random() < max(0.0, min(epsilon, 1.0)):
        return random.choice(modes)
    records = _load_ab_records(config, lookback)
    if not records:
        return random.choice(modes)
    # YouTube API 키가 있으면 조회수 기반으로 가중치
    if getattr(config, "youtube_api_key", ""):
        stats = {"minecraft": [], "image": []}
        max_sample = int(getattr(config, "background_ab_max_sample", 12))
        for rec in records:
            mode = rec.get("background_mode")
            vid = rec.get("youtube_video_id") or rec.get("youtube_video_id", "")
            if mode in stats and vid and len(stats[mode]) < max_sample:
                views = _fetch_youtube_stats(vid, config.youtube_api_key).get("viewCount", 0)
                if views:
                    stats[mode].append(views)
        avg_m = sum(stats["minecraft"]) / max(1, len(stats["minecraft"]))
        avg_i = sum(stats["image"]) / max(1, len(stats["image"]))
        total = avg_m + avg_i
        if total > 0:
            r = random.random() * total
            return "minecraft" if r < avg_m else "image"
        return random.choice(modes)
    # API 키가 없으면 성공 횟수 기반
    counts = {"minecraft": 0, "image": 0}
    for rec in records:
        mode = rec.get("background_mode")
        if mode in counts and rec.get("status", "ok") == "ok":
            counts[mode] += 1
    total = counts["minecraft"] + counts["image"]
    if total <= 0:
        return random.choice(modes)
    r = random.random() * total
    return "minecraft" if r < counts["minecraft"] else "image"


_GENERIC_VIDEO_TOKENS = {
    "calm", "relax", "relaxing", "nature", "landscape", "scenery",
    "ocean", "sea", "forest", "sky", "clouds", "sunset", "sunrise",
    "ambient", "background", "broll", "b-roll", "slow", "soft", "peaceful",
    "video", "image", "photo", "footage", "clip", "scene", "vertical",
    "dramatic", "cinematic", "aesthetic", "beautiful", "trending",
}
_VISUAL_SEARCH_STOPWORDS = {
    "close", "up", "shot", "shots", "style", "mood", "with", "and", "the",
    "man", "woman", "people", "person", "face", "looking", "holding", "view",
    "photo", "image", "video", "footage", "dramatic", "cinematic", "background",
    "reaction", "scene", "light", "lights", "street", "city", "old", "new",
}
_ROLE_QUERY_HINTS: Dict[str, str] = {
    "hook": "brand logo app icon close up countdown timer",
    "asmr_tag": "cute mascot logo whisper text bubble",
    "problem": "old advertisement archive newspaper",
    "failure": "closed store empty street",
    "success": "crowded store launch success",
    "point": "brand product detail close up",
    "reaction": "person shocked in store",
}
_BRAND_QUERY_HINTS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(mcdonald|マクドナルド|맥도날드|ronald|ランランルー|란란루)", re.IGNORECASE),
        "McDonalds logo golden arches Ronald McDonald mascot storefront",
    ),
    (re.compile(r"(coca[- ]?cola|コカ.?コーラ|코카콜라)", re.IGNORECASE), "Coca Cola logo bottle vending machine"),
    (re.compile(r"(pepsi|ペプシ|펩시)", re.IGNORECASE), "Pepsi logo can soda advertisement"),
    (re.compile(r"(nike|ナイキ)", re.IGNORECASE), "Nike logo sneaker store display"),
    (re.compile(r"(instagram|インスタ|인스타)", re.IGNORECASE), "Instagram logo smartphone app screen"),
    (re.compile(r"(tiktok|틱톡|ティックトック|douyin)", re.IGNORECASE), "TikTok logo app icon smartphone screen"),
    (re.compile(r"(netflix|넷플릭스|ネトフリ|ネットフリックス)", re.IGNORECASE), "Netflix logo red N app icon streaming interface"),
    (re.compile(r"(ramen|ラーメン|라면)", re.IGNORECASE), "ramen noodles bowl close up"),
]


def _extract_query_tokens(text: str) -> List[str]:
    tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text or "") if t]
    return [
        t for t in tokens
        if len(t) >= 3 and t not in _VISUAL_SEARCH_STOPWORDS
    ]


def _detect_brand_query_hint(topic_en: str, script_ja: str, keyword: str) -> str:
    combined = " ".join([topic_en or "", script_ja or "", keyword or ""])
    for pattern, hint in _BRAND_QUERY_HINTS:
        if pattern.search(combined):
            return hint
    return ""


def _extract_countdown_hint(topic_en: str, script_ja: str, keyword: str) -> str:
    src = " ".join([topic_en or "", script_ja or "", keyword or ""])
    if not src.strip():
        return ""
    m = re.search(r"(\d{1,2})\s*(?:秒|sec|seconds?)", src, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d{1,2})\b", src)
    if not m:
        return ""
    try:
        sec = int(m.group(1))
    except Exception:
        return ""
    if sec <= 0 or sec > 30:
        return ""
    return f"{sec} second countdown timer"


def _refine_visual_keyword(
    keyword: str,
    topic_en: str,
    role: str,
    script_ja: str,
    forced_brand_hint: str = "",
) -> str:
    kw = (keyword or "").strip()
    topic = (topic_en or "").strip()
    role_hint = _ROLE_QUERY_HINTS.get(role, "dramatic close up")
    brand_hint = forced_brand_hint.strip() or _detect_brand_query_hint(topic, script_ja, kw)
    countdown_hint = _extract_countdown_hint(topic, script_ja, kw)
    if brand_hint:
        extras: List[str] = []
        if role == "hook":
            extras.append("attention grabbing")
        if countdown_hint:
            extras.append(countdown_hint)
        return " ".join([brand_hint] + extras + [role_hint]).strip()

    if not kw:
        kw = topic
    topic_tokens = _extract_query_tokens(topic)
    kw_tokens = _extract_query_tokens(kw)
    token_set = set(kw_tokens)
    is_generic = (not token_set) or token_set.issubset(_GENERIC_VIDEO_TOKENS) or len(token_set) <= 2

    merged_tokens = kw_tokens[:]
    if topic_tokens:
        if is_generic:
            merged_tokens = topic_tokens[:5]
        else:
            for token in topic_tokens:
                if token not in merged_tokens:
                    merged_tokens.append(token)
    if not merged_tokens:
        merged_tokens = ["mystery", "story"]

    refined = " ".join(merged_tokens[:6]).strip()
    if role_hint and role_hint not in refined:
        refined = f"{refined} {role_hint}".strip()

    if role == "hook" and not any(
        w in refined.lower()
        for w in ("brand", "restaurant", "product", "app", "logo", "bottle", "sneaker")
    ):
        refined = f"{refined} headline close up".strip()
    if countdown_hint and countdown_hint not in refined:
        refined = f"{refined} {countdown_hint}".strip()
    return refined or "mystery story headline close up"


def _fallback_mood_key(mood: str) -> str:
    if mood in BGM_MOOD_ALIASES:
        mood = BGM_MOOD_ALIASES[mood]
    if mood in {"mystery_suspense", "mystery", "suspense"}:
        return "mystery"
    if mood in {"fast_exciting", "exciting"}:
        return "exciting"
    if mood in {"emotional"}:
        return "emotional"
    return "informative"


def match_bgm_by_mood(config: AppConfig, mood: str) -> Optional[str]:
    """
    mood(mystery_suspense/fast_exciting)에 맞는 BGM 파일 반환.
    1) assets/bgm/{mood}/ 폴더에서 랜덤 선택
    2) 없으면 Pixabay에서 자동 다운로드 (API 키 있을 때만)
    3) 그래도 없으면 assets/bgm 폴더에서 폴백
    """
    if mood in BGM_MOOD_ALIASES:
        mood = BGM_MOOD_ALIASES[mood]
    mood_info = BGM_MOOD_CATEGORIES.get(mood, BGM_MOOD_CATEGORIES["mystery_suspense"])
    folder_name = mood_info["folder"]
    bgm_dir = os.path.join(config.assets_dir, "bgm", folder_name)
    os.makedirs(bgm_dir, exist_ok=True)

    # 기존 파일 있으면 랜덤 선택
    existing = [
        os.path.join(bgm_dir, f)
        for f in os.listdir(bgm_dir)
        if f.lower().endswith((".mp3", ".wav", ".ogg", ".m4a"))
    ]
    if existing:
        return random.choice(existing)

    # Minecraft BGM 슬롯 우선 (일본 쇼츠 + Minecraft Parkour 배경 시)
    if getattr(config, "use_minecraft_parkour_bg", False):
        minecraft_dir = os.path.join(config.assets_dir, "bgm", "minecraft")
        minecraft_files = _list_audio_files(minecraft_dir)
        if not minecraft_files:
            # YouTube에서 Minecraft BGM 자동 수집
            _telemetry_log("Minecraft BGM YouTube 자동 수집 시도", config)
            downloaded = fetch_youtube_minecraft_bgm(minecraft_dir)
            if downloaded:
                minecraft_files = [downloaded]
        if minecraft_files:
            return random.choice(minecraft_files)

    # Pixabay 자동 다운로드 (명시적으로 켠 경우에만)
    if config.pixabay_api_key and getattr(config, "pixabay_bgm_enabled", False):
        _append_bgm_debug(f"Pixabay BGM 시도 (mood={mood})")
        _telemetry_log(f"Pixabay BGM 다운로드 시도 (mood={mood})", config)
        query_pool = BGM_CATEGORY_KEYWORDS.get(mood, [])
        custom_query = random.choice(query_pool) if query_pool else ""
        downloaded = fetch_bgm_from_pixabay(
            api_key=config.pixabay_api_key,
            category=mood,
            output_dir=bgm_dir,
            custom_query=custom_query,
            config=config,
        )
        if downloaded and os.path.exists(downloaded):
            return downloaded
        _append_bgm_debug("Pixabay BGM 결과 없음")
        _telemetry_log("Pixabay BGM 다운로드 실패/결과 없음", config)
    else:
        if not config.pixabay_api_key:
            _append_bgm_debug("PIXABAY_API_KEY 미설정")
            _telemetry_log("PIXABAY_API_KEY 미설정: BGM 자동 다운로드 생략", config)
        else:
            _append_bgm_debug("PIXABAY_BGM_ENABLED=false: Pixabay BGM 생략")
            _telemetry_log("PIXABAY_BGM_ENABLED=false: Pixabay BGM 생략", config)

    # Minecraft BGM 슬롯 (일본 쇼츠 + Minecraft Parkour 배경 시 우선)
    if getattr(config, "use_minecraft_parkour_bg", False):
        minecraft_dir = os.path.join(config.assets_dir, "bgm", "minecraft")
        minecraft_files = _list_audio_files(minecraft_dir)
        if not minecraft_files:
            downloaded = fetch_youtube_minecraft_bgm(minecraft_dir)
            if downloaded:
                minecraft_files = [downloaded]
        if minecraft_files:
            return random.choice(minecraft_files)

    # 로컬 일반 폴더 폴백
    fallback_dir = os.path.join(config.assets_dir, "bgm")
    fallback_files = _list_audio_files(fallback_dir)
    if fallback_files:
        return random.choice(fallback_files)
    # 최후: 간단한 합성 BGM 생성
    if getattr(config, "bgm_fallback_enabled", True):
        try:
            gen_dir = os.path.join(config.assets_dir, "bgm", "generated")
            os.makedirs(gen_dir, exist_ok=True)
            filename = f"generated_{mood}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
            out_path = os.path.join(gen_dir, filename)
            _generate_bgm_fallback(out_path, float(getattr(config, "max_video_duration_sec", 59.0)), _fallback_mood_key(mood))
            _append_bgm_debug(f"합성 BGM 생성 완료: {os.path.basename(out_path)}")
            _telemetry_log(f"합성 BGM 생성 완료: {os.path.basename(out_path)}", config)
            return out_path
        except Exception as exc:
            _append_bgm_debug(f"합성 BGM 생성 실패: {exc}")
            _telemetry_log(f"합성 BGM 생성 실패: {exc}", config)
            try:
                fallback_dir = "/tmp/auto_shorts_bgm"
                os.makedirs(fallback_dir, exist_ok=True)
                filename = f"generated_{mood}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
                out_path = os.path.join(fallback_dir, filename)
                _generate_bgm_fallback(
                    out_path,
                    float(getattr(config, "max_video_duration_sec", 59.0)),
                    _fallback_mood_key(mood),
                )
                _append_bgm_debug(f"합성 BGM 생성 완료(/tmp): {os.path.basename(out_path)}")
                _telemetry_log(f"합성 BGM 생성 완료(/tmp): {os.path.basename(out_path)}", config)
                return out_path
            except Exception as exc2:
                _append_bgm_debug(f"/tmp 합성 BGM 생성 실패: {exc2}")
                _telemetry_log(f"/tmp 합성 BGM 생성 실패: {exc2}", config)
    return None


# 일본 쇼츠용 기본 우선순위: 귀엽고 밝은 톤
ENERGETIC_VOICE_ORDER = ["shimmer", "nova", "alloy", "echo", "fable", "onyx"]
CUTE_VOICE_ORDER = ["shimmer", "nova", "alloy", "echo", "fable", "onyx"]


def pick_voice_id(
    voice_ids: List[str],
    preference: Optional[List[str]] = None,
    force_cute: bool = False,
) -> str:
    clean_voice_ids = [str(v).strip() for v in (voice_ids or []) if str(v).strip()]
    if not clean_voice_ids:
        return "shimmer" if force_cute else ""
    pref_list = [str(v).strip() for v in (preference or []) if str(v).strip() in clean_voice_ids]
    if force_cute:
        for v in CUTE_VOICE_ORDER:
            if v in pref_list:
                return v
        for v in CUTE_VOICE_ORDER:
            if v in clean_voice_ids:
                return v
        if pref_list:
            return pref_list[0]
        return clean_voice_ids[0]
    if pref_list:
        return random.choice(pref_list)
    energetic = [v for v in ENERGETIC_VOICE_ORDER if v in clean_voice_ids]
    return random.choice(energetic or clean_voice_ids)


ALLOWED_REACTION_TAGS = [
    "shock",
    "wow",
    "laugh",
    "awkward",
    "facepalm",
    "angry",
    "cute",
    "plot",
    "ending",
]


def analyze_image_tags(
    config: AppConfig,
    image_path: str,
    allowed_tags: Optional[List[str]] = None,
) -> List[str]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")
    allowed_tags = allowed_tags or ALLOWED_REACTION_TAGS
    model = config.openai_vision_model or config.openai_model
    if not model:
        raise RuntimeError("OPENAI_VISION_MODEL 또는 OPENAI_MODEL이 없습니다.")
    if not os.path.exists(image_path):
        return []
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    system_text = (
        "You are an expert Korean/Japanese internet meme and reaction image classifier. "
        "These images are used as reaction clips in short-form videos (Reels/Shorts/TikTok). "
        "Analyze the image carefully — look at facial expressions, body language, context, text overlays, and overall emotional tone. "
        "Choose 1-3 tags that best describe HOW this image would be used as a reaction in a video. "
        "Tag definitions:\n"
        "  shock: sudden surprise, jaw-drop, eyes wide open, disbelief\n"
        "  wow: amazement, impressed, mind-blown, spectacular\n"
        "  laugh: funny, humorous, comedic, lol expression\n"
        "  awkward: uncomfortable, embarrassed, cringe, nervous\n"
        "  facepalm: disappointment, 'seriously?', exasperation, headshake\n"
        "  angry: frustration, rage, upset, indignant\n"
        "  cute: adorable, heartwarming, sweet, lovely\n"
        "  plot: story setup, suspense, 'what happens next?', tension building\n"
        "  ending: conclusion, resolution, final reveal, outro moment\n"
        "Return JSON only: {\"tags\": [\"...\"], \"reason\": \"brief explanation\"}."
    )
    user_text = (
        f"Allowed tags: {', '.join(allowed_tags)}\n"
        "Look at this image and choose the most fitting reaction tags. "
        "Be precise — if the image shows sadness or crying, do NOT tag it as laugh. "
        "If it shows anger or frustration, use angry or facepalm. "
        "Match the actual emotional content of the image."
    )
    client = OpenAI(api_key=config.openai_api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    output_text = getattr(response, "output_text", "") or ""
    result = extract_json(output_text)
    tags = result.get("tags", []) if isinstance(result, dict) else []
    cleaned = []
    for tag in tags:
        if isinstance(tag, str):
            value = tag.strip().lower()
            if value in allowed_tags:
                cleaned.append(value)
    return list(dict.fromkeys(cleaned))


# ─────────────────────────────────────────────
# NEW: 에셋 이미지 → 콘텐츠 카테고리 분석
# ─────────────────────────────────────────────

def analyze_image_content_category(
    config: AppConfig,
    image_path: str,
) -> str:
    """
    이미지를 분석해서 CONTENT_CATEGORIES 중 가장 잘 어울리는 카테고리 반환.
    기존 analyze_image_tags와 별도로 카테고리 레벨 분류.
    """
    if not config.openai_api_key:
        return "humor"
    model = config.openai_vision_model or config.openai_model
    if not os.path.exists(image_path):
        return "humor"
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    system_text = (
        "You are an expert Korean/Japanese internet meme and reaction image classifier "
        "for short-form video content (Reels/Shorts/TikTok). "
        "Analyze the image carefully — facial expressions, body language, context clues, text overlays, color tone, and overall emotional atmosphere. "
        "Pick the single most fitting category based on the ACTUAL emotional content of the image.\n\n"
        "Category definitions (choose ONE):\n"
        "  humor       — funny, comedic, joke, silly, meme with punchline\n"
        "  touching    — emotional, tearful, moving story, heartfelt, makes you cry\n"
        "  shocking    — jaw-dropping reveal, plot twist, unbelievable fact, WTF moment\n"
        "  heartwarming — wholesome, warm fuzzy feeling, acts of kindness, family/pet love\n"
        "  cringe      — awkward, embarrassing, secondhand embarrassment, 'why would they do that'\n"
        "  exciting    — hype, energetic, celebration, victory, pumped up\n"
        "  sad         — grief, loss, lonely, crying, melancholy, unfortunate situation\n"
        "  anger       — frustration, injustice, rant, outrage, 'this is wrong'\n\n"
        "IMPORTANT: Do not default to 'humor'. Carefully examine what emotion the image actually conveys. "
        "A crying person = sad or touching, NOT humor. An angry face = anger, NOT humor. "
        "Return JSON only: {\"category\": \"one_of_the_above\", \"reason\": \"brief_explanation\"}"
    )
    user_text = (
        "Classify this image into exactly one category based on its actual emotional content. "
        "Look carefully at expressions, context, and atmosphere. "
        "Do NOT assume it is humor just because it's a meme format."
    )
    try:
        client = OpenAI(api_key=config.openai_api_key)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": system_text},
                        {"type": "input_image", "image_url": data_url},
                        {"type": "input_text", "text": user_text},
                    ],
                }
            ],
        )
        output_text = getattr(response, "output_text", "") or ""
        result = extract_json(output_text)
        category = result.get("category", "").strip().lower()
        if category in CONTENT_CATEGORIES:
            return category
        # 유사 단어 매핑 (모델이 가끔 비슷한 단어로 반환할 때)
        fallback_map = {
            "funny": "humor", "comedy": "humor", "comic": "humor",
            "emotional": "touching", "heartfelt": "touching", "cry": "touching",
            "crying": "sad", "grief": "sad", "melancholy": "sad",
            "wholesome": "heartwarming", "warm": "heartwarming", "sweet": "heartwarming",
            "twist": "shocking", "reveal": "shocking", "wtf": "shocking",
            "awkward": "cringe", "embarrass": "cringe",
            "hype": "exciting", "energy": "exciting", "celebration": "exciting",
            "angry": "anger", "rage": "anger", "frustrat": "anger",
        }
        for key, mapped in fallback_map.items():
            if key in category:
                return mapped
    except Exception:
        pass
    return "humor"


def tts_openai(
    config: AppConfig,
    text: str,
    output_path: str,
    voice: str,
) -> str:
    """
    OpenAI TTS API로 음성 생성. 일본어 텍스트 지원.
    """
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")
    if not voice:
        voice = "alloy"
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.openai_tts_model,
        "input": text,
        "voice": voice,
    }
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        json=payload,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(response.content)
    return output_path


def tts_gtts(
    text: str,
    output_path: str,
    lang: str = "ja",
) -> str:
    """
    gTTS로 일본어 TTS 생성. OPENAI_API_KEY 없을 때 폴백용.
    lang='ja'로 일본어 지원.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gTTS가 설치되지 않았습니다. pip install gTTS")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return output_path


def _resolve_ffmpeg_bin() -> str:
    ffmpeg_bin = "ffmpeg"
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return ffmpeg_bin


def _apply_ffmpeg_audio_filter(input_path: str, output_path: str, filter_expr: str) -> bool:
    if not input_path or not os.path.exists(input_path):
        return False
    cmd = [
        _resolve_ffmpeg_bin(),
        "-y",
        "-i",
        input_path,
        "-af",
        filter_expr,
        "-ac",
        "2",
        "-ar",
        "44100",
        output_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return proc.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def _apply_baby_voice_filter(config: AppConfig, audio_path: str) -> str:
    if not getattr(config, "tts_baby_voice", True):
        return audio_path
    pitch = float(getattr(config, "tts_baby_pitch", 1.20) or 1.20)
    pitch = max(1.02, min(pitch, 1.35))
    tempo = max(0.5, min(2.0, 1.0 / pitch))
    filtered = os.path.join(
        os.path.dirname(audio_path) or ".",
        f"baby_{os.path.basename(audio_path)}",
    )
    filt = (
        f"asetrate=44100*{pitch:.4f},"
        f"atempo={tempo:.4f},"
        "aresample=44100,"
        "highpass=f=120,"
        "treble=g=2,"
        "volume=1.08"
    )
    applied = _apply_ffmpeg_audio_filter(audio_path, filtered, filt)
    if (not applied):
        # 환경에 따라 atempo/treble 체인이 실패할 수 있어 단순 체인으로 재시도
        fallback_filt = (
            f"asetrate=44100*{pitch:.4f},"
            "aresample=44100,"
            "highpass=f=120,"
            "volume=1.05"
        )
        applied = _apply_ffmpeg_audio_filter(audio_path, filtered, fallback_filt)
    if applied:
        try:
            os.replace(filtered, audio_path)
        except Exception:
            shutil.copyfile(filtered, audio_path)
            os.remove(filtered)
    return audio_path


def _apply_asmr_voice_filter(audio_path: str) -> str:
    filtered = os.path.join(
        os.path.dirname(audio_path) or ".",
        f"asmr_{os.path.basename(audio_path)}",
    )
    filt = (
        "highpass=f=130,"
        "lowpass=f=3800,"
        "acompressor=threshold=-24dB:ratio=2.2:attack=20:release=220,"
        "volume=0.86"
    )
    if _apply_ffmpeg_audio_filter(audio_path, filtered, filt):
        try:
            os.replace(filtered, audio_path)
        except Exception:
            shutil.copyfile(filtered, audio_path)
            os.remove(filtered)
    return audio_path


def _tts_generate_single(
    config: AppConfig,
    text: str,
    output_path: str,
    voice: str = "",
) -> str:
    provider = (getattr(config, "tts_provider", "") or "openai").strip().lower()
    if provider == "gtts":
        return tts_gtts(text, output_path, lang="ja")
    return tts_openai(config, text, output_path, voice=voice or "shimmer")


def tts_generate(
    config: AppConfig,
    text: str,
    output_path: str,
    voice: str = "",
    segments: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    TTS 생성.
    - 기본: 단일 텍스트
    - segments 제공 시: 세그먼트 단위 합성 (asmr_tag는 ASMR 필터)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not segments:
        result = _tts_generate_single(config, text, output_path, voice=voice or "shimmer")
        return _apply_baby_voice_filter(config, result)

    if not MOVIEPY_AVAILABLE or AudioFileClip is None:
        joined = "。".join(
            [
                str(seg.get("text", "") or "").strip()
                for seg in segments
                if str(seg.get("text", "") or "").strip()
            ]
        )
        result = _tts_generate_single(config, joined, output_path, voice=voice or "shimmer")
        return _apply_baby_voice_filter(config, result)

    try:
        from moviepy.editor import concatenate_audioclips
    except Exception:
        joined = "。".join(
            [
                str(seg.get("text", "") or "").strip()
                for seg in segments
                if str(seg.get("text", "") or "").strip()
            ]
        )
        result = _tts_generate_single(config, joined, output_path, voice=voice or "shimmer")
        return _apply_baby_voice_filter(config, result)

    temp_files: List[str] = []
    clips: List[Any] = []
    try:
        for idx, seg in enumerate(segments):
            line = str(seg.get("text", "") or "").strip()
            if not line:
                continue
            role = str(seg.get("role", "") or "").strip().lower()
            tmp_path = os.path.join(
                os.path.dirname(output_path) or ".",
                f"tts_seg_{idx}_{random.randint(1000, 9999)}.mp3",
            )
            _tts_generate_single(config, line, tmp_path, voice=voice or "shimmer")
            _apply_baby_voice_filter(config, tmp_path)
            if role in {"asmr_tag", "asmr"}:
                _apply_asmr_voice_filter(tmp_path)
            temp_files.append(tmp_path)
            clips.append(AudioFileClip(tmp_path))
        if not clips:
            raise RuntimeError("TTS 세그먼트가 비어 있습니다.")
        merged = concatenate_audioclips(clips)
        merged.write_audiofile(output_path, fps=44100, nbytes=2, codec="mp3", logger=None)
        try:
            merged.close()
        except Exception:
            pass
        return output_path
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


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


def update_asset_tags(
    manifest_path: str,
    tag_map: Dict[str, List[str]],
    keep_existing: bool = True,
) -> int:
    if not tag_map:
        return 0
    items = load_manifest(manifest_path)
    updated = 0
    for item in items:
        if item.asset_id not in tag_map:
            continue
        new_tags = tag_map[item.asset_id]
        if keep_existing:
            merged = list(dict.fromkeys(item.tags + new_tags))
            item.tags = merged
        else:
            item.tags = list(dict.fromkeys(new_tags))
        updated += 1
    if updated:
        save_manifest(manifest_path, items)
    return updated


def remove_assets(
    manifest_path: str,
    asset_ids: List[str],
    delete_files: bool = False,
) -> int:
    if not asset_ids:
        return 0
    items = load_manifest(manifest_path)
    remaining: List[AssetItem] = []
    removed = 0
    remove_set = set(asset_ids)
    for item in items:
        if item.asset_id in remove_set:
            removed += 1
            if delete_files and os.path.exists(item.path):
                try:
                    os.remove(item.path)
                except Exception:
                    pass
        else:
            remaining.append(item)
    if removed:
        save_manifest(manifest_path, remaining)
    return removed


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


def pick_asset_by_category(items: List[AssetItem], category: str) -> Optional[AssetItem]:
    """콘텐츠 카테고리에 맞는 에셋 선택 (카테고리 태그 포함 우선)."""
    # 카테고리 자체가 태그로 저장된 에셋 우선
    category_candidates = [item for item in items if category in item.tags]
    if category_candidates:
        return random.choice(category_candidates)
    # 카테고리→에셋태그 매핑으로 fallback
    mapped_tags = CATEGORY_TO_ASSET_TAGS.get(category, [])
    if mapped_tags:
        return pick_asset(items, mapped_tags)
    return random.choice(items) if items else None


def tags_from_text(text: str) -> List[str]:
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


_SYSTEM_FONT_CANDIDATES = [
    # 트렌디 일본어 폰트가 설치된 경우 우선
    "/usr/share/fonts/truetype/mplus/MPLUSRounded1c-ExtraBold.ttf",
    "/usr/share/fonts/truetype/zenkakugothicnew/ZenKakuGothicNew-Bold.ttf",
    "/usr/share/fonts/truetype/bizudpgothic/BIZUDPGothic-Bold.ttf",
    # Streamlit Cloud / Ubuntu
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unifont/unifont.ttf",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

# 일본어 가독성용 Noto Sans JP Bold (일본 예능 자막 스타일)
_JAPANESE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/mplus/MPLUSRounded1c-ExtraBold.ttf",
    "/usr/share/fonts/truetype/zenkakugothicnew/ZenKakuGothicNew-Bold.ttf",
    "/usr/share/fonts/truetype/bizudpgothic/BIZUDPGothic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Bold.otf",
] + _SYSTEM_FONT_CANDIDATES


_TRENDY_JP_FONT_DOWNLOADS: List[Tuple[str, str]] = [
    (
        "MPLUSRounded1c-ExtraBold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-ExtraBold.ttf",
    ),
    (
        "ZenKakuGothicNew-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/zenkakugothicnew/ZenKakuGothicNew-Bold.ttf",
    ),
    (
        "BIZUDPGothic-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/bizudpgothic/BIZUDPGothic-Bold.ttf",
    ),
    (
        "NotoSansJP[wght].ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
    ),
]


def _ensure_japanese_font(assets_dir: str) -> str:
    """
    ./assets/fonts/ 경로에서 일본어 폰트를 불러오거나, 없으면 Google Fonts에서
    트렌디한 일본어 볼드 폰트를 다운로드합니다. 일본어 자막 깨짐 방지용.
    """
    fonts_dir = os.path.join(assets_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    local_names = [name for name, _ in _TRENDY_JP_FONT_DOWNLOADS] + [
        "NotoSansJP-Bold.otf",
        "NotoSansJP-Bold.ttf",
        "NotoSansCJKjp-Bold.otf",
    ]
    for name in local_names:
        p = os.path.join(fonts_dir, name)
        if os.path.exists(p):
            return p
    for filename, url in _TRENDY_JP_FONT_DOWNLOADS:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 10000:
                path = os.path.join(fonts_dir, filename)
                with open(path, "wb") as f:
                    f.write(resp.content)
                return path
        except Exception:
            continue
    for candidate in _JAPANESE_FONT_CANDIDATES:
        if os.path.exists(candidate) and ImageFont:
            try:
                ImageFont.truetype(candidate, size=24)
                return candidate
            except Exception:
                continue
    return ""


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)
    # 시스템 CJK 폰트 폴백
    for candidate in _SYSTEM_FONT_CANDIDATES:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
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


def _enhance_bg_image_quality(image: Image.Image) -> Image.Image:
    if not MOVIEPY_AVAILABLE:
        return image
    try:
        if CV2_AVAILABLE and cv2 is not None and np is not None:
            arr = np.array(image.convert("RGB"))
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            merged = cv2.merge((l2, a, b))
            rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            kernel = np.array([[0, -1, 0], [-1, 5.3, -1], [0, -1, 0]], dtype=np.float32)
            sharp = cv2.filter2D(rgb, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            return Image.fromarray(sharp)
    except Exception:
        pass
    try:
        return image.filter(ImageFilter.UnsharpMask(radius=1.8, percent=120, threshold=2))
    except Exception:
        return image


def _ease_in_out(x: float) -> float:
    v = max(0.0, min(1.0, float(x)))
    return v * v * (3.0 - 2.0 * v)


def _wrap_cjk_text(text: str, max_width_px: int, font_size: int) -> List[str]:
    """CJK(일본어·한국어) 문자를 픽셀 폭 기준으로 줄바꿈."""
    # 영문은 ~0.55배, CJK는 ~1배 폭 차지
    def char_w(c: str) -> float:
        return 1.0 if ord(c) > 127 else 0.55

    max_chars = max(6, int(max_width_px / (font_size * 0.95)))
    lines: List[str] = []
    paragraphs = str(text or "").splitlines() or [str(text or "")]
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        current_w = 0.0
        for ch in paragraph:
            cw = char_w(ch)
            if current_w + cw > max_chars:
                if current:
                    lines.append(current)
                current = ch
                current_w = cw
            else:
                current += ch
                current_w += cw
        if current:
            lines.append(current)
    return lines or [text]


def _shorten_caption_text(text: str, max_chars: int = 18) -> str:
    if not text:
        return text
    # 문장 분리 — 첫 문장/구문만 사용
    head = re.split(r"[。．.!?？！\\n、,，]", text, maxsplit=1)[0].strip()
    if not head:
        head = text.strip()
    if len(head) > max_chars:
        head = head[:max_chars].rstrip() + "…"
    return head


def _build_caption_texts(texts: List[str], max_chars: int) -> List[str]:
    return [_shorten_caption_text(t, max_chars=max_chars) for t in texts]


def _get_caption_texts(config: AppConfig, texts: List[str]) -> List[str]:
    if getattr(config, "caption_trim", False):
        return _build_caption_texts(texts, config.caption_max_chars)
    return texts


def _pick_thumbnail_text(meta: Optional[Dict[str, Any]], texts: List[str]) -> str:
    meta = meta or {}
    for key in ("thumbnail_text_ja", "title_ja", "title", "video_title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if texts:
        return texts[0]
    return ""


def _extract_video_frame_image(
    video_path: str,
    output_path: str,
    at_ratio: float = 0.2,
) -> Optional[str]:
    """
    렌더링된 영상에서 프레임을 추출해 썸네일 원본으로 사용.
    """
    if not MOVIEPY_AVAILABLE or not VideoFileClip or not Image:
        return None
    if not video_path or not os.path.exists(video_path):
        return None
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0.0)
        if duration <= 0:
            return None
        t = min(max(0.4, duration * at_ratio), max(0.4, duration - 0.4))
        frame = clip.get_frame(t)
        image = Image.fromarray(frame).convert("RGB")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path, "JPEG", quality=92, optimize=True)
        return output_path
    except Exception:
        return None
    finally:
        try:
            if clip:
                clip.close()
        except Exception:
            pass


def _pick_thumbnail_source(
    config: AppConfig,
    bg_image_paths: List[Optional[str]],
    rendered_video_path: str,
    frame_name: str,
) -> str:
    """
    썸네일 소스 우선순위:
    1) 실제 배경 이미지(placeholder 제외)
    2) 렌더링 영상 프레임 추출 이미지
    3) placeholder
    """
    placeholder = _ensure_placeholder_image(config)
    placeholder_abs = os.path.abspath(placeholder)
    for path in (bg_image_paths or []):
        if not path or not os.path.exists(path):
            continue
        if os.path.abspath(path) == placeholder_abs:
            continue
        return path
    frame_path = os.path.join(config.output_dir, frame_name)
    extracted = _extract_video_frame_image(rendered_video_path, frame_path)
    if extracted and os.path.exists(extracted):
        return extracted
    return placeholder


def _apply_korean_template(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
    """트렌디 쇼츠 느낌의 오버레이 (시네마 그라디언트 + 네온 엣지)."""
    overlay = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 배경 전체에 은은한 시네마 컬러 그라디언트
    grad_step = 3
    for y in range(0, canvas_height, grad_step):
        p = y / max(1, canvas_height - 1)
        r = int(10 + 22 * p)
        g = int(14 + 28 * p)
        b = int(18 + 44 * p)
        a = int(55 + 25 * p)
        draw.rectangle([0, y, canvas_width, min(canvas_height, y + grad_step)], fill=(r, g, b, a))

    # 자막 가독성 확보용 상/하단 비네팅
    top_h = int(canvas_height * 0.17)
    for y in range(top_h):
        alpha = int(135 * (1 - y / max(1, top_h)))
        draw.line([(0, y), (canvas_width, y)], fill=(0, 0, 0, alpha))
    bottom_h = int(canvas_height * 0.24)
    for y in range(bottom_h):
        alpha = int(185 * (y / max(1, bottom_h)))
        yy = canvas_height - bottom_h + y
        draw.line([(0, yy), (canvas_width, yy)], fill=(0, 0, 0, alpha))

    # 좌우 네온 엣지 (과하지 않게)
    edge_w = max(5, int(canvas_width * 0.006))
    draw.rectangle([0, 0, edge_w, canvas_height], fill=(79, 208, 255, 72))
    draw.rectangle([canvas_width - edge_w, 0, canvas_width, canvas_height], fill=(255, 108, 92, 66))

    # 상단 칩(짧은 강조 바)
    chip_h = max(8, int(canvas_height * 0.008))
    chip_w = int(canvas_width * 0.22)
    draw.rounded_rectangle(
        [int(canvas_width * 0.06), int(canvas_height * 0.05), int(canvas_width * 0.06) + chip_w, int(canvas_height * 0.05) + chip_h],
        radius=max(4, chip_h // 2),
        fill=(255, 255, 255, 86),
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_korean_thumbnail_style(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """썸네일 전용: 강한 컬러/강조 요소를 추가해 한국 쇼츠 감성 강화."""
    overlay = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    red = (255, 60, 60, 200)
    yellow = (255, 210, 0, 170)

    # 상단 대각 리본
    ribbon_h = int(canvas_height * 0.18)
    ribbon_w = int(canvas_width * 0.78)
    draw.polygon(
        [
            (0, 0),
            (ribbon_w, 0),
            (int(ribbon_w * 0.86), ribbon_h),
            (0, ribbon_h),
        ],
        fill=red,
    )
    # 하단 대각 라이트
    band_h = int(canvas_height * 0.18)
    band_w = int(canvas_width * 0.8)
    draw.polygon(
        [
            (canvas_width - band_w, canvas_height - int(band_h * 0.6)),
            (canvas_width, canvas_height - band_h),
            (canvas_width, canvas_height),
            (canvas_width - int(band_w * 0.2), canvas_height),
        ],
        fill=yellow,
    )

    # 강한 프레임
    frame = max(8, int(canvas_width * 0.01))
    draw.rectangle(
        [frame, frame, canvas_width - frame, canvas_height - frame],
        outline=(255, 255, 255, 120),
        width=max(2, frame // 2),
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def generate_thumbnail_image(
    config: AppConfig,
    bg_image_path: str,
    title_text: str,
    hook_text: str,
    output_path: str,
) -> str:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    W, H = 1280, 720
    base = Image.open(bg_image_path).convert("RGB")
    base = _fit_image_to_canvas(base, (W, H))
    if config.use_korean_template:
        base = _apply_korean_template(base, W, H)
    # 썸네일 전용 강한 스타일
    base = _apply_korean_thumbnail_style(base, W, H)

    main_text = hook_text if config.thumbnail_use_hook else title_text
    main_text = _shorten_caption_text(main_text, max_chars=config.thumbnail_max_chars)
    sub_text = ""
    if main_text != title_text and title_text:
        sub_text = _shorten_caption_text(title_text, max_chars=max(14, config.thumbnail_max_chars))

    # 텍스트 배경 박스 (강한 대비)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = 30
    box_h = int(H * 0.34)
    draw.rectangle([0, H - box_h, W, H], fill=(0, 0, 0, 175))
    accent_h = max(10, int(H * 0.014))
    draw.rectangle([0, H - box_h, W, H - box_h + accent_h], fill=(255, 60, 60, 230))

    # 텍스트 렌더링
    font_main = _load_font(config.font_path, int(W * 0.095))
    font_sub = _load_font(config.font_path, int(W * 0.05))
    lines = _wrap_cjk_text(main_text, W - pad * 2, int(W * 0.095))
    y = H - box_h + pad
    for line in lines:
        try:
            lw = font_main.getbbox(line)[2]
        except Exception:
            lw = len(line) * int(W * 0.08)
        x = max(pad, (W - lw) // 2)
        # 강조 배경
        rect_pad = 14
        rect = [x - rect_pad, y - 6, x + lw + rect_pad, y + int(W * 0.1)]
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(rect, radius=18, fill=(255, 60, 60, 90))
        else:
            draw.rectangle(rect, fill=(255, 60, 60, 90))
        # 그림자 + 두꺼운 스트로크
        draw.text((x + 3, y + 3), line, font=font_main, fill=(0, 0, 0, 180))
        draw.text(
            (x, y),
            line,
            font=font_main,
            fill=(255, 242, 120),
            stroke_width=7,
            stroke_fill=(0, 0, 0),
        )
        y += int(W * 0.1)

    if sub_text:
        y += int(W * 0.02)
        try:
            lw = font_sub.getbbox(sub_text)[2]
        except Exception:
            lw = len(sub_text) * int(W * 0.045)
        x = max(pad, (W - lw) // 2)
        draw.text((x + 2, y + 2), sub_text, font=font_sub, fill=(0, 0, 0, 160))
        draw.text((x, y), sub_text, font=font_sub, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))

    thumb = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    thumb.save(output_path, "JPEG", quality=92, optimize=True)
    return output_path


def _draw_subtitle(
    image: Image.Image,
    text: str,
    font_path: str,
    canvas_width: int,
    canvas_height: int,
    style: str = "default",
    t: Optional[float] = None,
    duration: Optional[float] = None,
    hold_ratio: float = 1.0,
) -> Image.Image:
    """자막을 YouTube Shorts 안전 영역(화면 60% 지점)에 렌더링.
    모바일 Shorts 하단 UI(제목·좋아요·댓글 등)가 화면 하단 ~30%를 덮으므로
    자막을 화면의 55~65% 구간에 고정해 가려지지 않도록 함.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    # 진행률 기반 노출 제어 (자막을 더 짧게 보여줌)
    if duration and t is not None and duration > 0:
        progress = max(0.0, min(t / duration, 1.0))
        if progress > max(0.05, min(hold_ratio, 1.0)):
            return image
    else:
        progress = 0.0

    # 폰트 크기: 요청에 맞춰 아주 소폭 확대
    font_size = max(52, canvas_width // 15)
    pad_x = int(canvas_width * 0.05)
    max_text_w = canvas_width - pad_x * 2
    raw_text = str(text or "")
    ja_text = ""
    ko_text = ""
    if _CAPTION_LANG_SEP in raw_text:
        ja_text, ko_text = raw_text.split(_CAPTION_LANG_SEP, 1)
        ja_text = ja_text.strip()
        ko_text = ko_text.strip()
        is_bilingual = bool(ja_text and ko_text)
        raw_parts: List[str] = []
    else:
        raw_parts = [part.strip() for part in raw_text.splitlines() if part.strip()]
        is_bilingual = len(raw_parts) >= 2

    if is_bilingual:
        if not ja_text and raw_parts:
            ja_text = raw_parts[0]
        if not ko_text and len(raw_parts) > 1:
            ko_text = "\n".join(raw_parts[1:])
        primary_lines = _wrap_cjk_text(ja_text, max_text_w, font_size)[:3]
        secondary_base_size = max(30, int(font_size * 0.64))
        secondary_lines = _wrap_cjk_text(ko_text, max_text_w, secondary_base_size)[:3]
        primary_line_h = font_size + 12
        secondary_line_h = secondary_base_size + 10
        section_gap = 18 if secondary_lines else 0
        total_h = (
            primary_line_h * len(primary_lines)
            + section_gap
            + secondary_line_h * len(secondary_lines)
            + 24
        )
    else:
        primary_lines = _wrap_cjk_text(text, max_text_w, font_size)[:4]
        secondary_lines = []
        secondary_base_size = max(30, int(font_size * 0.64))
        primary_line_h = font_size + 14
        secondary_line_h = secondary_base_size + 10
        section_gap = 0
        total_h = primary_line_h * len(primary_lines) + 20
    # ── Shorts 안전 영역: 화면 55% 지점을 자막 중앙으로 ──
    # 하단 UI 안전선: canvas_height * 0.68 이하
    safe_bottom = int(canvas_height * 0.68)
    box_y = safe_bottom - total_h
    box_y = max(int(canvas_height * 0.45), box_y)  # 최소 45% 아래 유지
    # 스타일 결정 (reaction은 말풍선, japanese_variety는 일본 예능 스타일: 노란색+검정테두리)
    style_key = (style or "").strip().lower()
    is_reaction = style_key in {"reaction", "outro", "outro_loop"}
    is_japanese_variety = style_key == "japanese_variety"
    is_asmr_tag = style_key in {"asmr_tag", "asmr"}
    box_outline = (255, 255, 255, 0)
    accent_fill = (255, 255, 255, 0)
    if is_reaction:
        box_fill = (255, 232, 92, 214)
        box_outline = (255, 255, 255, 120)
        accent_fill = (255, 255, 255, 105)
        text_fill = (25, 20, 0)
        stroke_fill = (0, 0, 0)
        stroke_width = 3
    elif is_japanese_variety:
        # 일본 예능 자막: 순백+노란색 그라데이션 글씨 + 강화된 검정 테두리
        box_fill = (0, 0, 0, 0)   # 배경 박스 없음 (자막만)
        text_fill = (255, 240, 0)  # 선명한 노란색
        stroke_fill = (0, 0, 0)
        stroke_width = 10          # 기존 6 → 10으로 강화 (일본 예능 TV 스타일)
    elif is_asmr_tag:
        box_fill = (255, 238, 246, 215)
        box_outline = (255, 255, 255, 130)
        accent_fill = (255, 255, 255, 130)
        text_fill = (249, 98, 188)
        stroke_fill = (40, 20, 40)
        stroke_width = 4
    else:
        # old-PPT 느낌을 줄이기 위해 글래스 카드 톤으로 변경
        box_fill = (12, 16, 24, 165)
        box_outline = (120, 216, 255, 118)
        accent_fill = (100, 232, 255, 112)
        text_fill = (255, 255, 255)
        stroke_fill = (0, 0, 0)
        stroke_width = 3

    # 애니메이션: 팝업(scale) + 튀어오르기(bounce) + 페이드아웃 + 새벽 빛 효과(glow)
    scale = 1.0
    y_bounce = 0
    alpha_mul = 1.0
    glow_alpha = 0  # 텍스트 주변 발광 효과 강도
    if duration and t is not None and duration > 0:
        pop_t = min(max(progress / 0.12, 0.0), 1.0)  # 팝인 속도 약간 빠르게
        scale = 0.88 + 0.12 * pop_t                   # 스케일 범위 확대 (0.88→1.0)
        y_bounce = int((1.0 - pop_t) * 16)            # 바운스 높이 증가
        glow_alpha = int(max(0, (1.0 - pop_t)) * 80)  # 팝인 초반 발광
        if progress > 0.82:
            alpha_mul = max(0.0, 1.0 - (progress - 0.82) / 0.18)  # 페이드아웃 구간 확대

    # 반투명 배경 박스 (japanese_variety는 박스 없음, reaction은 말풍선)
    box_pad = 18
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(overlay)
    box_rect = [
        pad_x - box_pad,
        box_y - box_pad,
        canvas_width - pad_x + box_pad,
        box_y + total_h + box_pad,
    ]
    if not is_japanese_variety:
        if hasattr(box_draw, "rounded_rectangle"):
            radius = 30 if is_asmr_tag else (26 if is_reaction else 20)
            box_draw.rounded_rectangle(
                box_rect,
                radius=radius,
                fill=box_fill,
                outline=box_outline if box_outline[3] > 0 else None,
                width=2 if box_outline[3] > 0 else 1,
            )
        else:
            box_draw.rectangle(box_rect, fill=box_fill)
        # 카드 상단 하이라이트 바
        if accent_fill[3] > 0:
            accent_h = max(4, int(canvas_height * 0.004))
            box_draw.rectangle(
                [
                    box_rect[0] + 8,
                    box_rect[1] + 8,
                    box_rect[2] - 8,
                    box_rect[1] + 8 + accent_h,
                ],
                fill=accent_fill,
            )

    # 말풍선 꼬리 (reaction 전용)
    if is_reaction:
        tail_w = max(26, int(canvas_width * 0.05))
        tail_h = max(16, int(canvas_width * 0.03))
        tail_x = canvas_width // 2
        tail_y = box_rect[3]
        if tail_y + tail_h < canvas_height * 0.85:
            box_draw.polygon(
                [
                    (tail_x - tail_w // 2, tail_y),
                    (tail_x + tail_w // 2, tail_y),
                    (tail_x, tail_y + tail_h),
                ],
                fill=box_fill,
            )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    text_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    y = box_y
    for line in primary_lines:
        scaled_size = max(20, int(font_size * scale))
        scaled_font = _load_font(font_path, scaled_size)
        try:
            lw = scaled_font.getbbox(line)[2]
        except Exception:
            lw = len(line) * scaled_size
        lx = max(pad_x, (canvas_width - lw) // 2)

        # japanese_variety 스타일: glow(발광) 레이어 먼저 그려서 텍스트 존재감 강화
        if (is_japanese_variety or is_asmr_tag) and glow_alpha > 0:
            glow_size = max(20, scaled_size + 4)
            glow_font = _load_font(font_path, glow_size)
            draw.text(
                (lx - 1, y + y_bounce - 1),
                line,
                font=glow_font,
                fill=((255, 255, 150, glow_alpha) if not is_asmr_tag else (255, 220, 240, glow_alpha)),
                stroke_width=stroke_width + (3 if is_asmr_tag else 4),
                stroke_fill=((255, 220, 0, glow_alpha) if not is_asmr_tag else (255, 160, 220, glow_alpha)),
            )

        draw.text(
            (lx, y + y_bounce),
            line,
            font=scaled_font,
            fill=(text_fill[0], text_fill[1], text_fill[2], int(255 * alpha_mul)),
            stroke_width=stroke_width,
            stroke_fill=(stroke_fill[0], stroke_fill[1], stroke_fill[2], int(255 * alpha_mul)),
        )
        y += primary_line_h

    if secondary_lines:
        # JP/KR 구분선
        div_y = y + max(2, int(canvas_height * 0.002))
        div_x1 = pad_x + 24
        div_x2 = canvas_width - pad_x - 24
        draw.line([(div_x1, div_y), (div_x2, div_y)], fill=(180, 220, 255, int(180 * alpha_mul)), width=2)
        y += section_gap
        secondary_fill = (230, 230, 230)
        secondary_stroke = (0, 0, 0)
        secondary_stroke_width = max(2, stroke_width - 2)
        if is_reaction:
            secondary_fill = (45, 35, 0)
        elif is_japanese_variety:
            secondary_fill = (210, 245, 255)
            secondary_stroke_width = max(2, stroke_width - 3)
        elif is_asmr_tag:
            secondary_fill = (180, 60, 140)
            secondary_stroke = (255, 255, 255)
            secondary_stroke_width = 2
        for line in secondary_lines:
            scaled_size = max(16, int(secondary_base_size * scale))
            scaled_font = _load_font(font_path, scaled_size)
            try:
                lw = scaled_font.getbbox(line)[2]
            except Exception:
                lw = len(line) * scaled_size
            lx = max(pad_x, (canvas_width - lw) // 2)
            draw.text(
                (lx, y + y_bounce),
                line,
                font=scaled_font,
                fill=(secondary_fill[0], secondary_fill[1], secondary_fill[2], int(235 * alpha_mul)),
                stroke_width=secondary_stroke_width,
                stroke_fill=(
                    secondary_stroke[0],
                    secondary_stroke[1],
                    secondary_stroke[2],
                    int(235 * alpha_mul),
                ),
            )
            y += secondary_line_h
    image = Image.alpha_composite(image.convert("RGBA"), text_layer).convert("RGB")
    return image


def _overlay_sticker(
    image: Image.Image,
    asset_path: str,
    canvas_width: int,
    canvas_height: int,
    size: int = 320,
    position: Optional[Tuple[int, int]] = None,
    opacity: float = 1.0,
) -> Image.Image:
    """에셋 이미지를 이모티콘처럼 우하단에 작게 붙입니다."""
    if not os.path.exists(asset_path):
        return image
    try:
        sticker = Image.open(asset_path).convert("RGBA")
        sticker = sticker.resize((size, size), Image.LANCZOS)
        if opacity < 1.0:
            alpha = sticker.split()[-1]
            alpha = alpha.point(lambda p: int(p * max(0.0, min(opacity, 1.0))))
            sticker.putalpha(alpha)
        if position:
            x, y = position
        else:
            margin = 30
            x = canvas_width - size - margin
            y = int(canvas_height * 0.10)   # 화면 상단 10% 위치 — 자막(하단)과 겹치지 않음
        base = image.convert("RGBA")
        base.paste(sticker, (x, y), sticker)
        return base.convert("RGB")
    except Exception:
        return image


def _random_sticker_position(canvas_width: int, canvas_height: int, size: int) -> Tuple[int, int]:
    margin = int(min(canvas_width, canvas_height) * 0.06)
    x_min = margin
    x_max = max(margin, canvas_width - size - margin)
    y_min = int(canvas_height * 0.08)
    y_max = max(y_min, int(canvas_height * 0.38))
    if x_max <= x_min or y_max <= y_min:
        return (canvas_width - size - margin, int(canvas_height * 0.10))
    return (random.randint(x_min, x_max), random.randint(y_min, y_max))


def _maybe_overlay_sticker(
    image: Image.Image,
    asset_path: str,
    canvas_width: int,
    canvas_height: int,
    mode: str = "off",
    size: int = 220,
) -> Image.Image:
    mode_key = (mode or "off").strip().lower()
    if mode_key in {"off", "none", "false", "0"}:
        return image
    position = None
    if mode_key == "random":
        position = _random_sticker_position(canvas_width, canvas_height, size)
    return _overlay_sticker(
        image,
        asset_path,
        canvas_width,
        canvas_height,
        size=size,
        position=position,
        opacity=0.85,
    )


def _compose_frame(
    asset_path: str,
    text: str,
    size: Tuple[int, int],
    font_path: str,
    style: str = "default",
    overlay_mode: str = "off",
    use_template: bool = True,
) -> Image.Image:
    """정적 이미지 배경 프레임 생성 (배경영상 없을 때 fallback)."""
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    base = Image.open(asset_path).convert("RGB")
    background = _fit_image_to_canvas(base, size)
    if use_template:
        background = _apply_korean_template(background, size[0], size[1])
    composed = background.copy()
    width, height = size
    composed = _draw_subtitle(composed, text, font_path, width, height, style=style)
    composed = _maybe_overlay_sticker(composed, asset_path, width, height, mode=overlay_mode, size=200)
    return composed


# 글로벌 미스터리 fallback 쿼리 목록 (Pixabay / Pexels 공용)
_GLOBAL_BG_FALLBACK_QUERIES = [
    "archival newspaper headline",
    "crime scene investigation board",
    "police evidence board close up",
    "old documents desk close up",
    "vintage city night street",
    "factory exterior night",
    "courtroom sketch illustration",
    "mysterious hallway dim light",
]


def _normalize_search_query(query: str) -> str:
    tokens = [
        t for t in re.split(r"[^a-zA-Z0-9]+", (query or "").lower())
        if t and t not in _VISUAL_SEARCH_STOPWORDS
    ]
    if not tokens:
        return (query or "").strip()
    return " ".join(tokens[:7]).strip()


def _build_search_query_candidates(query: str) -> List[str]:
    raw = (query or "").strip()
    reduced = _normalize_search_query(raw)
    candidates = [raw, reduced]
    # 제품/브랜드 키워드만 남긴 짧은 쿼리도 추가
    parts = [t for t in reduced.split() if len(t) >= 4]
    if parts:
        candidates.append(" ".join(parts[:4]))
    # 브랜드 로고 검색어 강제 보강
    brand_hints = _brand_fallback_queries_from_keyword(raw)
    for hint in brand_hints:
        if hint:
            candidates.append(hint)
    # 3초/5초 같은 숫자 문맥이 있으면 카운트다운 이미지 검색어 추가
    countdown_hint = _extract_countdown_hint(raw, raw, raw)
    if countdown_hint:
        candidates.append(countdown_hint)
        candidates.append(f"{countdown_hint} social media")
    uniq: List[str] = []
    for item in candidates:
        if item and item not in uniq:
            uniq.append(item)
    return uniq


def _brand_fallback_queries_from_keyword(keyword: str) -> List[str]:
    hint = _detect_brand_query_hint(keyword, keyword, keyword)
    if not hint:
        return []
    # 브랜드 소재는 로고/앱 아이콘 중심으로 재시도해서 엉뚱한 이미지 방지
    return [
        hint,
        f"{hint} close up",
        f"{hint} official logo",
    ]


def _score_pixabay_hit(hit: Dict[str, Any], query_tokens: List[str]) -> int:
    tags = str(hit.get("tags", "") or "").lower()
    user = str(hit.get("user", "") or "").lower()
    text = f"{tags} {user}"
    if not query_tokens:
        return 0
    score = 0
    for token in query_tokens:
        if token in text:
            score += 5
    # vertical 선호 가중치
    width = int(hit.get("imageWidth") or 0)
    height = int(hit.get("imageHeight") or 0)
    if height > width > 0:
        score += 2
    return score


def fetch_pixabay_video(
    query: str,
    api_key: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
) -> Optional[str]:
    """Pixabay Videos API에서 세로형 한국 배경영상 검색·다운로드.
    세로형 = height > width인 영상만 우선. 없으면 가로형도 허용 (render_video에서 crop).
    저장 위치: /tmp/pixabay_bg/
    """
    if not api_key:
        return None
    save_dir = "/tmp/pixabay_bg"
    os.makedirs(save_dir, exist_ok=True)

    def _try_q(q: str) -> Optional[str]:
        try:
            params = {
                "key": api_key,
                "q": q,
                "video_type": "film",
                "per_page": 15,
                "safesearch": "true",
            }
            resp = requests.get(
                "https://pixabay.com/api/videos/",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            if not hits:
                return None
            random.shuffle(hits)
            # 세로형(portrait) 우선, 없으면 가로형도 허용
            def _pick_url(hit: Dict[str, Any]) -> Optional[str]:
                vids = hit.get("videos", {})
                # large → medium → small 순서로 시도, 해상도 적당한 것 선택
                for key in ("large", "medium", "small"):
                    v = vids.get(key, {})
                    url = v.get("url", "")
                    if url:
                        return url
                return None

            portrait = [h for h in hits if
                        h.get("videos", {}).get("large", {}).get("height", 0) >
                        h.get("videos", {}).get("large", {}).get("width", 1)]
            candidates = portrait or hits
            for hit in candidates[:8]:
                url = _pick_url(hit)
                if not url:
                    continue
                try:
                    vresp = requests.get(url, stream=True, timeout=120)
                    vresp.raise_for_status()
                    fname = f"pxbay_{random.randint(100000, 999999)}.mp4"
                    fpath = os.path.join(save_dir, fname)
                    with open(fpath, "wb") as f_out:
                        for chunk in vresp.iter_content(chunk_size=65536):
                            f_out.write(chunk)
                    return fpath
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # 1차: 요청 쿼리
    result = _try_q(query)
    if result:
        return result
    # 2차~: 글로벌 미스터리 fallback
    for fbq in _GLOBAL_BG_FALLBACK_QUERIES:
        result = _try_q(fbq)
        if result:
            return result
    return None


def fetch_pixabay_image(
    query: str,
    api_key: str,
    output_dir: str = "/tmp/pixabay_images",
) -> Optional[str]:
    if not api_key:
        return None
    save_dir = _ensure_writable_dir(output_dir, "/tmp/pixabay_images")

    def _try_q(q: str) -> Optional[str]:
        try:
            query_tokens = _extract_query_tokens(q)
            params = {
                "key": api_key,
                "q": q,
                "image_type": "photo",
                "orientation": "vertical",
                "per_page": 20,
                "safesearch": "true",
            }
            resp = requests.get("https://pixabay.com/api/", params=params, timeout=30)
            if resp.status_code != 200:
                return None
            hits = resp.json().get("hits", []) or []
            if not hits:
                return None
            ranked = sorted(
                hits,
                key=lambda h: _score_pixabay_hit(h, query_tokens),
                reverse=True,
            )
            candidates = ranked[:8] if ranked else hits[:8]
            # 너무 한 장만 반복되지 않도록 상위권에서 랜덤 선택
            if len(candidates) > 3:
                top = candidates[:3]
                random.shuffle(top)
                candidates = top + candidates[3:]
            for hit in candidates:
                url = hit.get("largeImageURL") or hit.get("webformatURL") or hit.get("previewURL")
                if not url:
                    continue
                try:
                    img_resp = requests.get(url, stream=True, timeout=60)
                    img_resp.raise_for_status()
                    ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                    fname = f"pximg_{random.randint(100000, 999999)}{ext}"
                    fpath = os.path.join(save_dir, fname)
                    with open(fpath, "wb") as out:
                        for chunk in img_resp.iter_content(chunk_size=65536):
                            out.write(chunk)
                    return fpath
                except Exception:
                    continue
        except Exception:
            return None
        return None

    for candidate in _build_search_query_candidates(query):
        result = _try_q(candidate)
        if result:
            return result
    return None


def fetch_serpapi_image(
    query: str,
    api_key: str,
    output_dir: str = "/tmp/serpapi_images",
) -> Optional[str]:
    if not api_key:
        return None
    save_dir = _ensure_writable_dir(output_dir, "/tmp/serpapi_images")
    try:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "num": 12,
            "safe": "active",
        }
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=40)
        if resp.status_code != 200:
            return None
        results = resp.json().get("images_results", []) or []
        if not results:
            return None
        tokens = _extract_query_tokens(query)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for item in results[:12]:
            title = str(item.get("title", "") or "").lower()
            source = str(item.get("source", "") or "").lower()
            text = f"{title} {source}"
            score = 0
            for tok in tokens:
                if tok in text:
                    score += 3
            if "logo" in text:
                score += 2
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [row[1] for row in scored[:6]] if scored else results[:6]
        for item in candidates:
            image_url = item.get("original") or item.get("thumbnail")
            if not image_url:
                continue
            try:
                r = requests.get(image_url, stream=True, timeout=40)
                r.raise_for_status()
                ext = os.path.splitext(urlparse(str(image_url)).path)[1] or ".jpg"
                path = os.path.join(save_dir, f"sp_{random.randint(100000, 999999)}{ext}")
                with open(path, "wb") as out:
                    for chunk in r.iter_content(chunk_size=65536):
                        out.write(chunk)
                return path
            except Exception:
                continue
    except Exception:
        return None
    return None


def fetch_wikimedia_image(
    query: str,
    output_dir: str = "/tmp/wikimedia_images",
) -> Optional[str]:
    save_dir = _ensure_writable_dir(output_dir, "/tmp/wikimedia_images")
    try:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 8,
            "prop": "imageinfo",
            "iiprop": "url",
        }
        resp = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=30)
        if resp.status_code != 200:
            return None
        pages = (resp.json().get("query", {}) or {}).get("pages", {}) or {}
        if not pages:
            return None
        for page in pages.values():
            infos = page.get("imageinfo", []) if isinstance(page, dict) else []
            if not infos:
                continue
            url = str(infos[0].get("url", "") or "")
            if not url:
                continue
            try:
                r = requests.get(url, stream=True, timeout=40)
                r.raise_for_status()
                ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                path = os.path.join(save_dir, f"wm_{random.randint(100000, 999999)}{ext}")
                with open(path, "wb") as out:
                    for chunk in r.iter_content(chunk_size=65536):
                        out.write(chunk)
                return path
            except Exception:
                continue
    except Exception:
        return None
    return None


def fetch_segment_images(
    config: AppConfig,
    keywords: List[str],
) -> List[Optional[str]]:
    if not keywords:
        return []
    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    px_dir = f"/tmp/pixabay_images/{run_stamp}"
    pex_dir = f"/tmp/pexels_bg_images/{run_stamp}"
    sp_dir = f"/tmp/serpapi_images/{run_stamp}"
    wm_dir = f"/tmp/wikimedia_images/{run_stamp}"
    unique_kws = [kw for kw in dict.fromkeys(keywords) if kw]
    kw_to_path: Dict[str, Optional[str]] = {}
    for kw in unique_kws:
        path: Optional[str] = None
        query_candidates = _build_search_query_candidates(kw)
        _telemetry_log(f"이미지 검색 키워드: {kw} -> 후보 {query_candidates[:3]}", config)
        for candidate in query_candidates:
            if config.pixabay_api_key:
                path = fetch_pixabay_image(candidate, config.pixabay_api_key, px_dir)
            if not path and config.pexels_api_key:
                path = fetch_pexels_image(candidate, config.pexels_api_key, pex_dir)
            if not path and config.serpapi_api_key:
                path = fetch_serpapi_image(candidate, config.serpapi_api_key, sp_dir)
            if not path:
                path = fetch_wikimedia_image(candidate, wm_dir)
            if path:
                break
        if not path:
            brand_fallbacks = _brand_fallback_queries_from_keyword(kw)
            fallback_pool = brand_fallbacks if brand_fallbacks else _GLOBAL_BG_FALLBACK_QUERIES
            for fbq in fallback_pool:
                if config.pixabay_api_key:
                    path = fetch_pixabay_image(fbq, config.pixabay_api_key, px_dir)
                if not path and config.pexels_api_key:
                    path = fetch_pexels_image(fbq, config.pexels_api_key, pex_dir)
                if not path and config.serpapi_api_key:
                    path = fetch_serpapi_image(fbq, config.serpapi_api_key, sp_dir)
                if not path:
                    path = fetch_wikimedia_image(fbq, wm_dir)
                if path:
                    break
        kw_to_path[kw] = path
        _telemetry_log(f"이미지 검색 결과: {'성공' if path else '실패'} ({kw})", config)

    placeholder = _ensure_placeholder_image(config)
    return [kw_to_path.get(kw) or placeholder for kw in keywords]


def fetch_pexels_video(
    query: str,
    api_key: str,
    output_dir: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
) -> Optional[str]:
    """Pexels에서 세로형(portrait) royalty-free 한국 배경 영상을 검색·다운로드.
    - 결과가 없으면 _KOREA_BG_FALLBACK_QUERIES 순서로 자동 재시도.
    - 저장 위치: /tmp/ (Streamlit Cloud read-only 레포 우회).
    """
    if not api_key:
        return None
    # 저장은 항상 /tmp/ 에 (레포 내부는 read-only일 수 있음)
    save_dir = "/tmp/pexels_bg"
    os.makedirs(save_dir, exist_ok=True)

    def _try_query(q: str) -> Optional[str]:
        try:
            headers = {"Authorization": api_key}
            params = {"query": q, "per_page": 15, "orientation": "portrait"}
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            videos = resp.json().get("videos", [])
            if not videos:
                return None
            random.shuffle(videos)
            for video in videos[:8]:
                files = sorted(
                    video.get("video_files", []),
                    key=lambda f: abs(f.get("width", 0) - canvas_w),
                )
                for vf in files:
                    link = vf.get("link", "")
                    if not link:
                        continue
                    try:
                        vresp = requests.get(link, stream=True, timeout=120)
                        vresp.raise_for_status()
                        fname = f"bg_{random.randint(100000, 999999)}.mp4"
                        fpath = os.path.join(save_dir, fname)
                        with open(fpath, "wb") as vf_out:
                            for chunk in vresp.iter_content(chunk_size=65536):
                                vf_out.write(chunk)
                        return fpath
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    # 1차: 요청된 쿼리
    result = _try_query(query)
    if result:
        return result
    # 2차~: 글로벌 미스터리 fallback 쿼리 순서대로 재시도
    for fallback_q in _GLOBAL_BG_FALLBACK_QUERIES:
        result = _try_query(fallback_q)
        if result:
            return result
    return None


def fetch_youtube_minecraft_parkour_video(
    output_dir: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    target_count: int = 3,  # 최대 확보할 파일 수 (세그먼트 다양화용)
    config: Optional["AppConfig"] = None,
    force_fresh: bool = False,
) -> Optional[str]:
    """
    YouTube에서 Minecraft Parkour No Copyright 영상을 yt-dlp로 다운로드합니다.
    - 검색어를 여러 개 순환하여 다양한 영상 확보 (세그먼트마다 다른 영상 사용 가능)
    - 1080p 이상 우선 다운로드 (쇼츠 9:16 크롭 시 화질 손실 최소화)
    - output_dir에 target_count개 이상 파일이 있으면 스킵
    - 반환: 다운로드된 파일 경로 또는 None
    """
    os.makedirs(output_dir, exist_ok=True)
    def _mc_log(message: str) -> None:
        if config is not None:
            try:
                _telemetry_log(message, config)
            except Exception:
                pass

    # 기존 파일 목록 확인
    existing_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith((".mp4", ".mkv", ".webm"))
        and os.path.getsize(os.path.join(output_dir, f)) > 500000  # 500KB 이상만 유효
    ]
    if force_fresh and existing_files:
        for path in list(existing_files):
            try:
                os.remove(path)
            except Exception:
                continue
        _mc_log(f"Minecraft 배경 강제 초기화: {len(existing_files)}개 삭제")
        existing_files = []
    if len(existing_files) >= target_count:
        # 이미 충분한 파일이 있으면 랜덤 반환
        _mc_log(f"Minecraft 배경 재사용: 기존 {len(existing_files)}개 파일")
        return random.choice(existing_files)
    if existing_files:
        # 일부 있으면 첫 번째 반환하되 추가 다운로드는 계속 진행
        result_path = existing_files[0]
    else:
        result_path = None

    try:
        import yt_dlp
    except ImportError as exc:
        _mc_log(f"Minecraft 배경 수집 실패(yt-dlp import): {exc}")
        return result_path

    # 다양한 검색어 풀 (매번 다른 영상 확보)
    search_queries = [
        "Minecraft Parkour No Copyright gameplay 1080p",
        "Minecraft Parkour free to use gameplay",
        "Minecraft Parkour no copyright background",
        "Minecraft satisfying parkour gameplay no copyright",
        "Minecraft Parkour gameplay copyright free HD",
    ]

    # 아직 확보 못한 만큼 다운로드 시도
    needed = target_count - len(existing_files)
    downloaded_this_run: List[str] = []

    for query in search_queries:
        if len(downloaded_this_run) >= needed:
            break

        search_url = f"ytsearch15:{query}"
        ydl_opts_info = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # 메타데이터만 빠르게 수집
            "noplaylist": False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(search_url, download=False)
        except Exception as exc:
            _mc_log(f"Minecraft 검색 실패(query={query}): {exc}")
            continue

        entries = info.get("entries") or []
        candidates = []
        for e in entries:
            if not e:
                continue
            dur = e.get("duration") or 0
            views = e.get("view_count") or e.get("play_count") or 0
            vid_id = e.get("id") or ""
            # 이미 다운로드된 파일 ID는 스킵
            already = any(vid_id in f for f in existing_files + downloaded_this_run)
            if vid_id and dur >= 60 and not already:
                candidates.append((views, dur, e))

        if not candidates:
            _mc_log(f"Minecraft 후보 없음(query={query}, entries={len(entries)})")
            continue

        # 조회수 높은 순 정렬 후 상위 3개 중 랜덤 선택 (다양성 확보)
        candidates.sort(key=lambda x: x[0], reverse=True)
        pick = random.choice(candidates[:3])
        best = pick[2]
        video_url = best.get("webpage_url") or best.get("url")
        if not video_url and best.get("id"):
            video_url = f"https://www.youtube.com/watch?v={best['id']}"
        if not video_url:
            continue

        out_tmpl = os.path.join(output_dir, "minecraft_parkour_%(id)s.%(ext)s")
        ydl_opts_dl = {
            # 1080p 이상 우선, 없으면 720p, 최후엔 최고화질
            "format": (
                "bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height>=1080]+bestaudio/"
                "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height>=720]+bestaudio/"
                "best[ext=mp4]/best"
            ),
            "outtmpl": out_tmpl,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 60,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            _mc_log(f"Minecraft 다운로드 실패(video={video_url}): {exc}")
            continue

        # 새로 다운로드된 파일 찾기
        for f in os.listdir(output_dir):
            full = os.path.join(output_dir, f)
            if (
                f.startswith("minecraft_parkour_")
                and f.lower().endswith((".mp4", ".mkv", ".webm"))
                and full not in existing_files
                and full not in downloaded_this_run
                and os.path.getsize(full) > 500000
            ):
                downloaded_this_run.append(full)
                if result_path is None:
                    result_path = full
                break

    if result_path:
        _mc_log(f"Minecraft 배경 확보 완료: {os.path.basename(result_path)}")
    else:
        _mc_log("Minecraft 배경 확보 실패: 모든 쿼리에서 다운로드 결과 없음")

    return result_path


def _is_video_readable(path: Optional[str]) -> bool:
    if not path or not os.path.exists(path):
        return False
    if not MOVIEPY_AVAILABLE or VideoFileClip is None:
        return True
    try:
        clip = VideoFileClip(path)
        _ = float(clip.duration or 0)
        clip.close()
        return True
    except Exception:
        return False


def _fetch_context_video_background(
    config: AppConfig,
    keywords: List[str],
) -> Optional[str]:
    queries = [q for q in dict.fromkeys(keywords or []) if q]
    queries.extend(_GLOBAL_BG_FALLBACK_QUERIES[:3])
    for query in queries[:6]:
        path: Optional[str] = None
        if config.pixabay_api_key:
            path = fetch_pixabay_video(query, config.pixabay_api_key, config.width, config.height)
        if not path and config.pexels_api_key:
            path = fetch_pexels_video(
                query=query,
                api_key=config.pexels_api_key,
                output_dir="/tmp/pexels_bg_videos",
                canvas_w=config.width,
                canvas_h=config.height,
            )
        if path and _is_video_readable(path):
            _telemetry_log(f"컨텍스트 배경 영상 확보: {os.path.basename(path)} (q={query})", config)
            return path
    return None


def fetch_youtube_minecraft_bgm(output_dir: str) -> Optional[str]:
    """
    YouTube에서 'Minecraft BGM No Copyright' 등으로 검색하여
    저작권 프리 Minecraft BGM을 yt-dlp로 오디오만 다운로드합니다.
    - output_dir에 이미 파일이 있으면 스킵
    - 반환: 다운로드된 MP3 경로 또는 None
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = _list_audio_files(output_dir)
    if existing:
        return random.choice(existing)

    try:
        import yt_dlp
    except ImportError:
        return None

    search_queries = [
        "Minecraft BGM No Copyright",
        "Minecraft music royalty free",
        "Minecraft background music calm",
    ]
    for search_query in search_queries:
        search_url = f"ytsearch15:{search_query}"
        ydl_opts_info = {"quiet": True, "no_warnings": True, "extract_flat": False}
        try:
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(search_url, download=False)
        except Exception:
            continue

        entries = info.get("entries") or []
        candidates = []
        for e in entries:
            if not e:
                continue
            dur = e.get("duration") or 0
            if dur and 60 <= dur <= 600:  # 1분~10분
                candidates.append(e)

        if not candidates:
            continue

        best = random.choice(candidates[:5])  # 상위 5개 중 랜덤
        video_url = best.get("webpage_url") or best.get("url")
        if not video_url and best.get("id"):
            video_url = f"https://www.youtube.com/watch?v={best['id']}"
        if not video_url:
            continue

        out_tmpl = os.path.join(output_dir, "minecraft_bgm_%(id)s.%(ext)s")
        ydl_opts_dl = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
                ydl.download([video_url])
        except Exception:
            continue

        for f in os.listdir(output_dir):
            if f.startswith("minecraft_bgm_") and f.lower().endswith((".mp3", ".m4a")):
                return os.path.join(output_dir, f)
    return None


def _generate_bgm_fallback(output_path: str, duration: float, mood: str) -> str:
    """
    Pixabay API 키·로컬 파일 없을 때 numpy로 간단한 ambient 배경음 생성.
    단순 사인파 화음 — 저작권 없음.
    """
    import wave
    import struct
    import math

    sample_rate = 44100
    n = int(duration * sample_rate)
    mood_chords: Dict[str, List[float]] = {
        "mystery":     [110.0, 146.8, 164.8],   # Am chord drone
        "exciting":    [220.0, 277.2, 329.6],   # C major bright
        "informative": [196.0, 246.9, 293.7],   # G major mellow
        "emotional":   [174.6, 220.0, 261.6],   # F major warm
    }
    freqs = mood_chords.get(mood, mood_chords.get("exciting", [220.0, 277.2, 329.6]))
    samples: List[int] = []
    for i in range(n):
        t = i / sample_rate
        fade = min(t / 1.5, 1.0, (duration - t) / 1.5)
        # LFO로 살짝 변동감
        lfo = 1 + 0.008 * math.sin(2 * math.pi * 0.3 * t)
        val = sum(math.sin(2 * math.pi * f * lfo * t) for f in freqs)
        val = val / len(freqs) * 0.18 * fade
        samples.append(max(-32767, min(32767, int(val * 32767))))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output_path


def _estimate_durations(texts: List[str], total_duration: float) -> List[float]:
    lengths = [max(len(text), 1) for text in texts]
    total = sum(lengths)
    raw = [(length / total) * total_duration for length in lengths]
    minimum = 0.8
    adjusted = [max(duration, minimum) for duration in raw]
    scale = total_duration / sum(adjusted)
    return [duration * scale for duration in adjusted]


_HIGHLIGHT_CACHE: Dict[Tuple[str, int], float] = {}
_RUN_NOTIFIER: Optional["RunTimelineNotifier"] = None
_LAST_GENERATED_BG_TOPIC_KEY: str = ""


class RunTimelineNotifier:
    def __init__(self, config: AppConfig, enabled: bool = True):
        self.config = config
        self.enabled = bool(enabled and config.telegram_bot_token and config.telegram_admin_chat_id)
        self.start_ts = time.time()

    def send(self, icon: str, title: str, detail: str | None = None) -> None:
        if not self.enabled:
            return
        ts = _get_local_now(self.config).strftime("%H:%M:%S")
        msg = f"{icon} {ts} - {title}"
        if detail:
            msg += f"\n   \"{detail}\""
        send_telegram_message(
            self.config.telegram_bot_token,
            self.config.telegram_admin_chat_id,
            msg,
            silent=True,
        )

    def finish(self, success_platforms: List[str], next_run: datetime | None = None) -> None:
        elapsed = max(1, int(time.time() - self.start_ts))
        mins, secs = divmod(elapsed, 60)
        duration_text = f"총 소요시간: {mins}분 {secs}초" if mins else f"총 소요시간: {secs}초"
        platforms_text = "성공한 업로드: " + (", ".join(success_platforms) if success_platforms else "없음")
        next_text = ""
        if next_run:
            next_text = f"다음 실행: {next_run.strftime('%Y-%m-%d %H:%M')}"
        detail = "\n".join([duration_text, platforms_text, next_text]).strip()
        self.send("✅", "모든 작업 완료!", detail)
        if next_run:
            delta = next_run - _get_local_now(self.config)
            rest_sec = max(0, int(delta.total_seconds()))
            rest_h, rem = divmod(rest_sec, 3600)
            rest_m, rest_s = divmod(rem, 60)
            rest_text = f"{rest_h}시간 {rest_m}분 {rest_s}초 후 다시 만나요!"
            self.send("😴", "시스템 휴식", rest_text)


def _set_run_notifier(notifier: Optional["RunTimelineNotifier"]) -> None:
    global _RUN_NOTIFIER
    _RUN_NOTIFIER = notifier


def _notify(icon: str, title: str, detail: str | None = None) -> None:
    if _RUN_NOTIFIER:
        _RUN_NOTIFIER.send(icon, title, detail)


def _find_dynamic_segment_start(
    video_path: str,
    target_duration: float,
    sample_fps: float = 2.0,
    max_scan_sec: float = 900.0,
) -> float:
    if not video_path or not os.path.exists(video_path):
        return 0.0
    if not MOVIEPY_AVAILABLE or np is None:
        return 0.0
    key = (video_path, int(target_duration))
    if key in _HIGHLIGHT_CACHE:
        return _HIGHLIGHT_CACHE[key]
    try:
        clip = VideoFileClip(video_path).without_audio()
        duration = float(clip.duration or 0)
        scan_dur = min(duration, max_scan_sec)
        if scan_dur <= target_duration + 1:
            clip.close()
            _HIGHLIGHT_CACHE[key] = 0.0
            return 0.0
        step = 1.0 / max(sample_fps, 0.5)
        t = 0.0
        prev = None
        scores: List[float] = []
        times: List[float] = []
        while t < scan_dur:
            frame = clip.get_frame(t)
            gray = np.mean(frame, axis=2).astype("float32")
            if prev is not None:
                diff = float(np.mean(np.abs(gray - prev)))
                scores.append(diff)
                times.append(t)
            prev = gray
            t += step
        clip.close()
        if not scores:
            _HIGHLIGHT_CACHE[key] = 0.0
            return 0.0
        window = max(1, int(target_duration * max(sample_fps, 0.5)))
        if window >= len(scores):
            _HIGHLIGHT_CACHE[key] = 0.0
            return 0.0
        # 슬라이딩 윈도우 합계로 가장 역동적인 구간 탐색
        current = sum(scores[:window])
        best = current
        best_idx = 0
        for i in range(window, len(scores)):
            current += scores[i] - scores[i - window]
            if current > best:
                best = current
                best_idx = i - window + 1
        start = times[best_idx] if best_idx < len(times) else 0.0
        max_start = max(duration - target_duration - 0.1, 0.0)
        start = min(max(start, 0.0), max_start)
        _HIGHLIGHT_CACHE[key] = start
        return start
    except Exception:
        _HIGHLIGHT_CACHE[key] = 0.0
        return 0.0


def _open_bg_video(path: str, W: int, H: int) -> Optional["VideoFileClip"]:
    """배경 영상을 열어 세로형(portrait)으로 resize·crop 후 반환."""
    try:
        vid = VideoFileClip(path).without_audio()
        bw, bh = vid.size
        scale = max(W / bw, H / bh)
        vid = vid.resize((int(bw * scale), int(bh * scale)))
        cx = (vid.size[0] - W) // 2
        cy = (vid.size[1] - H) // 2
        return vid.crop(x1=cx, y1=cy, x2=cx + W, y2=cy + H)
    except Exception:
        return None


def render_video(
    config: AppConfig,
    asset_paths: Optional[List[str]],
    texts: List[str],
    tts_audio_path: str,
    output_path: str,
    bgm_path: str | None = None,
    bgm_volume: float = 0.08,
    caption_styles: Optional[List[str]] = None,
    overlay_mode: str = "off",
    bg_video_path: str | None = None,
    bg_video_paths: Optional[List[Optional[str]]] = None,
    bg_image_paths: Optional[List[Optional[str]]] = None,
    caption_texts: Optional[List[str]] = None,
    draw_subtitles: bool = True,
) -> str:
    """
    TTS + 자막 + 에셋 스티커 + 배경영상(or 정적 이미지)으로 숏츠 영상 생성.
    bg_video_paths: 세그먼트별 영상 경로 리스트 (None이면 bg_video_path로 폴백).
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    W, H = config.width, config.height
    audio_clip = AudioFileClip(tts_audio_path)
    max_duration = max(5.0, float(getattr(config, "max_video_duration_sec", 59.0)))
    if audio_clip.duration > max_duration:
        audio_clip = audio_clip.subclip(0, max_duration)
    durations = _estimate_durations(texts, audio_clip.duration)
    caption_texts = caption_texts or texts
    clips = []

    # 경로 → VideoFileClip 캐시 (같은 파일 중복 오픈 방지)
    _vid_cache: Dict[str, Any] = {}
    _highlight_cache: Dict[str, float] = {}

    def _get_vid(path: Optional[str]) -> Optional[Any]:
        if not path or not os.path.exists(path):
            return None
        if path not in _vid_cache:
            _vid_cache[path] = _open_bg_video(path, W, H)
        return _vid_cache[path]

    # 전역 fallback 영상
    global_bg_vid = _get_vid(bg_video_path)

    # 세그먼트별 다른 배경영상 자동 선택:
    # bg_video_path가 폴더 내 여러 파일 중 하나라면, 같은 폴더의 모든 파일을 풀로 사용
    _bg_video_pool: List[str] = []
    if bg_video_path and os.path.exists(bg_video_path):
        _bg_dir = os.path.dirname(bg_video_path)
        _bg_video_pool = [
            os.path.join(_bg_dir, f)
            for f in os.listdir(_bg_dir)
            if f.lower().endswith((".mp4", ".mkv", ".webm"))
            and os.path.getsize(os.path.join(_bg_dir, f)) > 500000
        ]
    if not _bg_video_pool and bg_video_path:
        _bg_video_pool = [bg_video_path]

    vid_offset = 0.0
    for index, text in enumerate(texts):
        asset_path = ""
        if asset_paths:
            asset_path = asset_paths[min(index, len(asset_paths) - 1)]
        dur = durations[index]
        style = (
            caption_styles[index]
            if caption_styles and index < len(caption_styles)
            else "default"
        )
        # 일본 쇼츠 타겟: use_japanese_caption_style이면 default → japanese_variety (노란글씨+검정테두리)
        if getattr(config, "use_japanese_caption_style", False) and style == "default":
            style = "japanese_variety"
        cap_text = caption_texts[index] if index < len(caption_texts) else text

        # 세그먼트별 영상 우선, 없으면 풀에서 랜덤 선택 (매 세그먼트마다 다른 영상으로 다양화)
        seg_path = (bg_video_paths[index] if bg_video_paths and index < len(bg_video_paths) else None)
        if not seg_path and _bg_video_pool:
            seg_path = random.choice(_bg_video_pool)
        bg_vid = _get_vid(seg_path) or global_bg_vid
        highlight_start = None
        if bg_vid is not None and getattr(config, "highlight_clip_enabled", False):
            src_path = seg_path or bg_video_path
            if src_path:
                if src_path not in _highlight_cache:
                    _highlight_cache[src_path] = _find_dynamic_segment_start(
                        src_path,
                        float(getattr(config, "highlight_clip_duration_sec", dur)),
                        float(getattr(config, "highlight_clip_sample_fps", 2.0)),
                        float(getattr(config, "highlight_clip_max_scan_sec", 900.0)),
                    )
                highlight_start = _highlight_cache.get(src_path, 0.0)

        _motion_cfg: Optional[Dict[str, Any]] = None
        _base_bg_arr = None
        if bg_vid is None:
            bg_img_path = (
                bg_image_paths[index]
                if bg_image_paths and index < len(bg_image_paths) and bg_image_paths[index]
                else asset_path
            )
            if not bg_img_path or not os.path.exists(bg_img_path):
                bg_img_path = _ensure_placeholder_image(config)
            bg_img = Image.open(bg_img_path).convert("RGB")
            bg_img = _enhance_bg_image_quality(bg_img)
            bg_img = _fit_image_to_canvas(bg_img, (W, H))
            _base_bg_arr = np.array(bg_img)
            _motion_cfg = {
                "zoom": random.uniform(0.06, 0.11),
                "dir_x": random.choice([-1, 1]),
                "dir_y": random.choice([-1, 1]),
            }

        # 클로저 캡처 (Python for-loop 캡처 이슈 방지)
        _cap = cap_text
        _asset = asset_path
        _font = config.font_path
        _style = style
        _overlay_mode = overlay_mode
        _hold_ratio = float(getattr(config, "caption_hold_ratio", 1.0) or 1.0)
        if not getattr(config, "caption_trim", False):
            _hold_ratio = 1.0
        _use_template = bool(getattr(config, "use_korean_template", False))

        def _render_frame(
            get_frame,
            t,
            __cap=_cap,
            __asset=_asset,
            __font=_font,
            __style=_style,
            __overlay=_overlay_mode,
            __dur=dur,
            __motion_cfg=_motion_cfg,
        ):
            frame = get_frame(t)
            img = Image.fromarray(frame).convert("RGB")
            if __motion_cfg:
                p = _ease_in_out(t / max(__dur, 0.1))
                z = 1.0 + float(__motion_cfg.get("zoom", 0.08)) * p
                sw = max(W + 4, int(W * z))
                sh = max(H + 4, int(H * z))
                enlarged = img.resize((sw, sh), Image.LANCZOS)
                max_x = max(0, sw - W)
                max_y = max(0, sh - H)
                px = p if int(__motion_cfg.get("dir_x", 1)) >= 0 else (1.0 - p)
                py = p if int(__motion_cfg.get("dir_y", 1)) >= 0 else (1.0 - p)
                cx = int(max_x * px)
                cy = int(max_y * py)
                img = enlarged.crop((cx, cy, cx + W, cy + H))
            if _use_template:
                img = _apply_korean_template(img, W, H)
            if draw_subtitles:
                img = _draw_subtitle(
                    img,
                    __cap,
                    __font,
                    W,
                    H,
                    style=__style,
                    t=t,
                    duration=__dur,
                    hold_ratio=_hold_ratio,
                )
            if __asset and os.path.exists(__asset):
                img = _maybe_overlay_sticker(img, __asset, W, H, mode=__overlay, size=260)
            return np.array(img)

        if bg_vid is not None:
            # 배경 영상에서 랜덤 오프셋 구간 추출
            max_start = max(bg_vid.duration - dur - 0.1, 0)
            if highlight_start is not None and max_start > 0:
                seg_start = min(max(highlight_start + vid_offset, 0.0), max_start)
            else:
                seg_start = random.uniform(0, max_start) if max_start > 0 else 0
            seg = bg_vid.subclip(seg_start, seg_start + dur)
            clip = seg.fl(_render_frame).set_duration(dur)
        else:
            # fallback: 정적 이미지 배경 (대본 키워드 이미지)
            if _base_bg_arr is None:
                fallback_img = Image.new("RGB", (W, H), color=(18, 18, 18))
                _base_bg_arr = np.array(fallback_img)
            base_clip = ImageClip(_base_bg_arr).set_duration(dur)
            clip = base_clip.fl(_render_frame)

        # 세그먼트 연결을 부드럽게 (구식 컷 전환 느낌 완화)
        try:
            trans = min(0.12, dur * 0.15)
            clip = clip.fx(vfx.fadein, trans).fx(vfx.fadeout, trans)
        except Exception:
            pass

        clips.append(clip)
        vid_offset += dur

    video = concatenate_videoclips(clips, method="compose").set_fps(config.fps)

    # ── 영상 시작/끝 fade-in / fade-out 효과 ──────────────────────────
    total_dur = video.duration
    fade_sec = min(0.4, total_dur * 0.04)  # 전체 길이의 4% 또는 최대 0.4초
    try:
        video = video.fx(vfx.fadein, fade_sec).fx(vfx.fadeout, fade_sec)
    except Exception:
        pass  # fade 실패 시 원본 유지

    # ── BGM 처리 + Audio Ducking ───────────────────────────────────────
    # Audio Ducking: TTS 발화 구간에서 BGM 볼륨을 자동으로 낮춰 TTS 가독성 확보
    bgm_raw = None
    bgm_full = None
    bgm_ducked = None
    if bgm_path and os.path.exists(bgm_path):
        from moviepy.editor import concatenate_audioclips

        bgm_raw = AudioFileClip(bgm_path)
        # BGM을 TTS 길이만큼 루프
        if bgm_raw.duration < audio_clip.duration:
            n_loops = int(audio_clip.duration / bgm_raw.duration) + 2
            bgm_full = concatenate_audioclips([bgm_raw] * n_loops).subclip(0, audio_clip.duration)
        else:
            bgm_full = bgm_raw.subclip(0, audio_clip.duration)

        # Audio Ducking: TTS 볼륨 기준으로 BGM 볼륨을 동적 감쇠
        # - TTS 발화 시: BGM을 bgm_volume(20%)로 유지
        # - 무음 구간: BGM을 bgm_volume * 1.5(30%)로 살짝 올림 (자연스러운 전환)
        duck_vol = float(bgm_volume)       # 발화 중 BGM 볼륨
        fill_vol = min(duck_vol * 1.5, 0.35)  # 무음 구간 BGM 볼륨

        def _ducking_vol_func(t: float) -> float:
            """TTS 오디오 진폭에 따라 BGM 볼륨 동적 조절 (Audio Ducking)."""
            try:
                chunk = audio_clip.get_frame(t)
                # 모노/스테레오 모두 처리
                amp = float(np.abs(chunk).mean()) if hasattr(chunk, "__len__") else 0.0
                # 진폭이 0.01 이상이면 TTS 발화 중 → ducking 적용
                return duck_vol if amp > 0.01 else fill_vol
            except Exception:
                return duck_vol

        bgm_ducked = bgm_full.fl(lambda gf, t: gf(t) * _ducking_vol_func(t), keep_duration=True)
        audio = CompositeAudioClip([audio_clip, bgm_ducked])
    else:
        audio = audio_clip

    # BGM에도 fade-in/out 적용 (영상 fade와 일치)
    try:
        audio = audio.audio_fadein(fade_sec).audio_fadeout(fade_sec)
    except Exception:
        pass

    video = video.set_audio(audio)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=config.fps,
            threads=max(1, int(getattr(config, "render_threads", 2))),
            preset="ultrafast",
            logger=None,
        )
    except Exception as exc:
        try:
            err_path = "/tmp/auto_shorts_render_error.log"
            with open(err_path, "w", encoding="utf-8") as file:
                file.write(str(exc))
        except Exception:
            pass
        raise
    finally:
        try:
            if bgm_ducked is not None:
                bgm_ducked.close()
        except Exception:
            pass
        try:
            if bgm_full is not None:
                bgm_full.close()
        except Exception:
            pass
        try:
            if bgm_raw is not None:
                bgm_raw.close()
        except Exception:
            pass
        try:
            audio_clip.close()
        except Exception:
            pass
        try:
            video.close()
        except Exception:
            pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass
        for _v in _vid_cache.values():
            try:
                if _v:
                    _v.close()
            except Exception:
                pass
    return output_path


def _list_audio_files(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    items = []
    for name in os.listdir(path):
        if name.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg")):
            items.append(os.path.join(path, name))
    return items


def _list_image_files(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    items = []
    for name in os.listdir(path):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            items.append(os.path.join(path, name))
    return sorted(items)


def _ensure_placeholder_image(config: AppConfig) -> str:
    path = os.path.join("/tmp", "auto_shorts_placeholder.jpg")
    if os.path.exists(path):
        return path
    if not MOVIEPY_AVAILABLE:
        return path
    try:
        img = Image.new("RGB", (config.width, config.height), color=(20, 20, 20))
        img.save(path, "JPEG")
    except Exception:
        pass
    return path


def _get_generated_bg_paths(
    config: AppConfig,
    count: int,
    topic_key: str = "",
) -> List[Optional[str]]:
    global _LAST_GENERATED_BG_TOPIC_KEY
    if count <= 0:
        return []
    topic_norm = (topic_key or "").strip().lower()
    if topic_norm and _LAST_GENERATED_BG_TOPIC_KEY and _LAST_GENERATED_BG_TOPIC_KEY != topic_norm:
        # 주제가 바뀌면 이전 생성 배경은 즉시 폐기 (이전 영상 잔상 방지)
        _clear_dir_cache(config.generated_bg_dir, allowed_ext=(".jpg", ".jpeg", ".png", ".webp"))
    files = _list_image_files(config.generated_bg_dir)
    if not files:
        return []
    # 최근 생성분만 사용해 이전 주제 이미지 재사용(캐시 잔존) 방지
    now_ts = time.time()
    fresh_files: List[str] = []
    for path in files:
        try:
            age = now_ts - os.path.getmtime(path)
            if age <= 60 * 20:  # 20분 이내
                fresh_files.append(path)
        except Exception:
            continue
    if fresh_files:
        files = fresh_files
    else:
        return []
    if len(files) >= count:
        if topic_norm:
            _LAST_GENERATED_BG_TOPIC_KEY = topic_norm
        return files[:count]
    # 부족하면 반복
    repeated: List[Optional[str]] = []
    idx = 0
    while len(repeated) < count:
        repeated.append(files[idx % len(files)])
        idx += 1
    if topic_norm:
        _LAST_GENERATED_BG_TOPIC_KEY = topic_norm
    return repeated


def _apply_primary_photo_override(
    config: AppConfig,
    bg_video_paths: List[Optional[str]],
    bg_image_paths: List[Optional[str]],
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    photo_path = str(getattr(config, "primary_photo_path", "") or "").strip()
    if not photo_path or (not os.path.exists(photo_path)):
        return bg_video_paths, bg_image_paths
    if not bg_video_paths and not bg_image_paths:
        return bg_video_paths, bg_image_paths
    max_len = max(len(bg_video_paths), len(bg_image_paths), 1)
    if len(bg_video_paths) < max_len:
        bg_video_paths = (bg_video_paths + [None] * max_len)[:max_len]
    if len(bg_image_paths) < max_len:
        placeholder = _ensure_placeholder_image(config)
        bg_image_paths = (bg_image_paths + [placeholder] * max_len)[:max_len]
    bg_video_paths[0] = None
    bg_image_paths[0] = photo_path
    return bg_video_paths, bg_image_paths


def pick_bgm_path(config: AppConfig) -> Optional[str]:
    mode = (config.bgm_mode or "off").lower().strip()
    if mode in {"", "off", "none"}:
        return None
    bgm_dir = os.path.join(config.assets_dir, "bgm")
    trending_dir = os.path.join(bgm_dir, "trending")
    trending = _list_audio_files(trending_dir)
    normal = _list_audio_files(bgm_dir)
    if mode in {"trend", "trending"}:
        pool = trending or normal
    elif mode in {"random", "auto"}:
        pool = trending + normal if trending else normal
    else:
        pool = normal
    if not pool:
        return None
    return random.choice(pool)


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
        "caption_variant",
        "background_mode",
        "bgm_file",
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


def _parse_youtube_error(err: Exception) -> Tuple[str, str]:
    reason = ""
    message = str(err)
    content = getattr(err, "content", None)
    if content:
        try:
            text = content.decode("utf-8", errors="ignore") if isinstance(content, (bytes, bytearray)) else str(content)
            data = json.loads(text)
            payload = data.get("error", {})
            errors = payload.get("errors", []) if isinstance(payload, dict) else []
            if errors:
                reason = errors[0].get("reason", "") or reason
                message = errors[0].get("message", "") or message
            if isinstance(payload, dict) and payload.get("message"):
                message = payload.get("message", message)
        except Exception:
            pass
    if not reason and "uploadLimitExceeded" in message:
        reason = "uploadLimitExceeded"
    return reason, message


def _load_pending_uploads(path: str) -> List[Dict[str, Any]]:
    data = _read_json_file(path, {"items": []})
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def _save_pending_uploads(path: str, items: List[Dict[str, Any]]) -> None:
    _write_json_file(path, {"items": items})


def _queue_pending_upload(
    config: AppConfig,
    file_path: str,
    title: str,
    description: str,
    tags: List[str],
    thumb_path: str = "",
    error: str = "",
) -> None:
    if not config.retry_pending_uploads:
        return
    items = _load_pending_uploads(config.pending_uploads_path)
    items.append(
        {
            "file_path": file_path,
            "title": title,
            "description": description,
            "tags": tags,
            "thumb_path": thumb_path,
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "attempts": 0,
            "last_error": error,
        }
    )
    _save_pending_uploads(config.pending_uploads_path, items)


def _try_upload_pending(config: AppConfig, use_streamlit: bool = True, max_items: int = 1) -> None:
    if not config.retry_pending_uploads or not config.enable_youtube_upload:
        return
    items = _load_pending_uploads(config.pending_uploads_path)
    if not items:
        return
    processed = 0
    remain: List[Dict[str, Any]] = []
    for item in items:
        if processed >= max_items:
            remain.append(item)
            continue
        if not os.path.exists(item.get("file_path", "")):
            continue
        _ui_info("대기열 업로드 재시도 중...", use_streamlit)
        result = upload_video(
            config=config,
            file_path=item.get("file_path", ""),
            title=item.get("title", ""),
            description=item.get("description", ""),
            tags=item.get("tags", []),
        )
        error = str(result.get("error", "") or "").strip()
        reason = str(result.get("error_reason", "") or "").strip()
        if error:
            item["attempts"] = int(item.get("attempts", 0)) + 1
            item["last_error"] = error
            remain.append(item)
            if reason == "uploadLimitExceeded":
                _ui_warning("유튜브 업로드 한도 초과로 재시도 보류.", use_streamlit)
                break
        else:
            processed += 1
            video_id = result.get("video_id", "")
            if item.get("thumb_path") and video_id:
                set_video_thumbnail(config, video_id, item["thumb_path"])
            msg = f"✅ 대기열 업로드 완료: {video_id}"
            _telemetry_log(msg, config)
            if config.telegram_bot_token and config.telegram_admin_chat_id:
                send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, msg)
    _save_pending_uploads(config.pending_uploads_path, remain)


def _fetch_youtube_stats(video_id: str, api_key: str) -> Dict[str, int]:
    if not video_id or not api_key:
        return {}
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "statistics", "id": video_id, "key": api_key},
            timeout=15,
        )
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return {}
        stats = items[0].get("statistics", {})
        return {
            "viewCount": int(stats.get("viewCount", 0)),
            "likeCount": int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0)),
        }
    except Exception:
        return {}


def _summarize_ab_tests(config: AppConfig) -> str:
    path = os.path.join(config.output_dir, "ab_tests.jsonl")
    if not os.path.exists(path):
        return "A/B 로그가 없습니다."
    lines = []
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    records = []
    for line in lines[-200:]:
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    if not records:
        return "A/B 로그가 없습니다."
    now = _get_local_now(config)
    cutoff = now - timedelta(days=int(config.ab_report_days))
    filtered = []
    local_tz = now.tzinfo
    for rec in records:
        raw_ts = str(rec.get("date_jst", "") or "").strip()
        try:
            # 기존 로그는 naive 문자열(YYYY-mm-dd HH:MM:SS)이라 로컬 타임존으로 간주.
            if "T" in raw_ts:
                ts = datetime.fromisoformat(raw_ts)
            else:
                ts = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = now
        if local_tz:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=local_tz)
            else:
                ts = ts.astimezone(local_tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        if ts >= cutoff:
            filtered.append(rec)
    if not filtered:
        return "최근 A/B 로그가 없습니다."
    # 최근 N개만 통계
    filtered = filtered[-int(config.ab_report_max_items):]
    stats_by_variant: Dict[str, Dict[str, Any]] = {}
    stats_by_bg: Dict[str, Dict[str, Any]] = {}
    for rec in filtered:
        variant = rec.get("caption_variant", "default")
        stats_by_variant.setdefault(variant, {"count": 0, "views": 0, "likes": 0, "comments": 0})
        stats_by_variant[variant]["count"] += 1
        bg_mode = rec.get("background_mode", "unknown")
        stats_by_bg.setdefault(bg_mode, {"count": 0, "views": 0, "likes": 0, "comments": 0})
        stats_by_bg[bg_mode]["count"] += 1
        if config.youtube_api_key and rec.get("youtube_video_id"):
            s = _fetch_youtube_stats(rec["youtube_video_id"], config.youtube_api_key)
            stats_by_variant[variant]["views"] += s.get("viewCount", 0)
            stats_by_variant[variant]["likes"] += s.get("likeCount", 0)
            stats_by_variant[variant]["comments"] += s.get("commentCount", 0)
            stats_by_bg[bg_mode]["views"] += s.get("viewCount", 0)
            stats_by_bg[bg_mode]["likes"] += s.get("likeCount", 0)
            stats_by_bg[bg_mode]["comments"] += s.get("commentCount", 0)
    lines = ["[A/B 리포트]"]
    lines.append(f"기간: 최근 {int(config.ab_report_days)}일 / 표본 {len(filtered)}개")
    for variant, stat in stats_by_variant.items():
        if config.youtube_api_key:
            avg_views = int(stat["views"] / max(stat["count"], 1))
            lines.append(
                f"- {variant}: {stat['count']}개, 평균 조회 {avg_views}, 좋아요 {stat['likes']}, 댓글 {stat['comments']}"
            )
        else:
            lines.append(f"- {variant}: {stat['count']}개")
    lines.append("")
    lines.append("[배경 모드]")
    for mode, stat in stats_by_bg.items():
        if config.youtube_api_key:
            avg_views = int(stat["views"] / max(stat["count"], 1))
            lines.append(
                f"- {mode}: {stat['count']}개, 평균 조회 {avg_views}, 좋아요 {stat['likes']}, 댓글 {stat['comments']}"
            )
        else:
            lines.append(f"- {mode}: {stat['count']}개")
    if not config.youtube_api_key:
        lines.append("※ 조회수 집계를 위해 YOUTUBE_API_KEY를 설정하세요.")
    return "\n".join(lines)


def _maybe_send_ab_report(config: AppConfig, use_streamlit: bool = True) -> None:
    if not config.ab_report_enabled:
        return
    if not config.telegram_bot_token or not config.telegram_admin_chat_id:
        return
    now = _get_local_now(config)
    if now.hour < int(config.ab_report_hour):
        return
    state = _read_json_file(config.ab_report_state_path, {"last_report_date": ""})
    if state.get("last_report_date") == now.date().isoformat():
        return
    report = _summarize_ab_tests(config)
    send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, report)
    _write_json_file(config.ab_report_state_path, {"last_report_date": now.date().isoformat()})


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
    try:
        response = request.execute()
        video_id = response.get("id", "")
        return {"video_id": video_id, "video_url": f"https://www.youtube.com/watch?v={video_id}"}
    except (HttpError, ResumableUploadError) as err:
        reason, message = _parse_youtube_error(err)
        return {"video_id": "", "video_url": "", "error": message, "error_reason": reason}
    except Exception as err:
        return {"video_id": "", "video_url": "", "error": str(err), "error_reason": "unknown"}


def set_video_thumbnail(config: AppConfig, video_id: str, thumbnail_path: str) -> bool:
    if not video_id or not thumbnail_path or not os.path.exists(thumbnail_path):
        return False
    if not config.youtube_client_id or not config.youtube_client_secret or not config.youtube_refresh_token:
        return False
    try:
        credentials = Credentials(
            token=None,
            refresh_token=config.youtube_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.youtube_client_id,
            client_secret=config.youtube_client_secret,
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )
        youtube = build("youtube", "v3", credentials=credentials)
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        request = youtube.thumbnails().set(videoId=video_id, media_body=media)
        request.execute()
        return True
    except Exception:
        return False


# 댓글 작성에 필요한 scope (토큰 재발급 시 포함 필요)
YOUTUBE_COMMENT_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"


def insert_video_comment(
    config: AppConfig,
    video_id: str,
    comment_text: str,
) -> bool:
    """
    영상에 댓글 작성. (고정은 YouTube Studio에서 수동)
    scope에 youtube.force-ssl 포함 필요.
    """
    if not video_id or not comment_text:
        return False
    if not all([config.youtube_client_id, config.youtube_client_secret, config.youtube_refresh_token]):
        return False
    try:
        credentials = Credentials(
            token=None,
            refresh_token=config.youtube_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.youtube_client_id,
            client_secret=config.youtube_client_secret,
            scopes=[
                "https://www.googleapis.com/auth/youtube.upload",
                YOUTUBE_COMMENT_SCOPE,
            ],
        )
        youtube = build("youtube", "v3", credentials=credentials)
        body = {
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": comment_text[:5000]}},
            }
        }
        request = youtube.commentThreads().insert(part="snippet", body=body)
        request.execute()
        return True
    except Exception:
        return False


def _pick_product_number_for_short(config: AppConfig) -> str:
    """제품 목록에서 랜덤 제품번호 선택. (링크트리 고정댓글용)"""
    try:
        from products import load_products
        products = load_products()
        if products:
            return random.choice(products).number
    except Exception:
        pass
    return ""


def build_pinned_comment_with_voting(
    product_number: str,
    linktree_url: str,
    pinned_base: str = "",
) -> str:
    """
    제품번호 + 링크트리 + 투표 유도 문구로 고정댓글 텍스트 생성.
    프로필 자기소개에 링크트리 URL, DM으로 제품번호 발송 시 링크 반환.
    """
    parts = []
    if product_number and linktree_url:
        search_url = linktree_url.rstrip("/") + (f"?q={product_number}" if "?" not in linktree_url else f"&q={product_number}")
        parts.append(f"📦 제품 [{product_number}] → {search_url}")
        parts.append("DM으로 제품번호 보내시면 링크 전달해드려요!")
    if pinned_base:
        parts.append("")
        parts.append(pinned_base)
    parts.append("")
    parts.append("👍 이 숏츠 어떠세요? 좋아요 눌러주세요!")
    return "\n".join(parts).strip()


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
    if not query or not query.strip():
        raise RuntimeError("검색어가 비어 있습니다.")
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
            "q": "日本 トレンド ハッシュタグ",
            "api_key": config.serpapi_api_key,
            "hl": "ja",
            "gl": "jp",
        }
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        response.raise_for_status()
        news = response.json().get("news_results", [])[:8]
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


def fetch_pexels_image(query: str, api_key: str, output_dir: str) -> Optional[str]:
    try:
        images = collect_images_pexels(
            query=query,
            api_key=api_key,
            output_dir=output_dir,
            limit=1,
            locale="en-US",
        )
        return images[0] if images else None
    except Exception:
        return None


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


def send_telegram_message(token: str, chat_id: str, text: str, silent: bool = False) -> bool:
    """
    텔레그램 메시지 전송. 성공하면 True, 실패하면 False 반환.
    - 4096자 초과 시 자동 분할 전송
    - parse_mode 미사용 (해시태그/특수문자 오류 방지)
    """
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 4000
    chunks = [text[i : i + max_len] for i in range(0, max(len(text), 1), max_len)]
    success = True
    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_notification": bool(silent),
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if not resp.ok:
                print(f"[Telegram 전송 실패] status={resp.status_code} body={resp.text[:300]}")
                success = False
        except Exception as exc:
            print(f"[Telegram 전송 오류] {exc}")
            success = False
    return success


def _build_telegram_preview_video(
    source_video_path: str,
    output_dir: str,
    max_duration_sec: float = 20.0,
    target_width: int = 540,
) -> Optional[str]:
    """
    텔레그램 전송용 경량 프리뷰 영상을 생성한다.
    원본이 크거나 전송 실패 시 사용.
    """
    if not MOVIEPY_AVAILABLE or not VideoFileClip:
        return None
    if not source_video_path or not os.path.exists(source_video_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    preview_name = f"telegram_preview_{int(time.time())}_{random.randint(1000,9999)}.mp4"
    preview_path = os.path.join(output_dir, preview_name)
    clip = None
    sub = None
    resized = None
    try:
        clip = VideoFileClip(source_video_path)
        duration = float(clip.duration or 0.0)
        if duration <= 0.0:
            return None
        end_t = min(duration, max_duration_sec)
        sub = clip.subclip(0, end_t)
        resized = sub
        if getattr(sub, "w", 0) and sub.w > target_width:
            resized = sub.resize(width=target_width)
        fps = int(getattr(clip, "fps", 24) or 24)
        resized.write_videofile(
            preview_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="700k",
            fps=min(24, max(15, fps)),
            preset="veryfast",
            threads=1,
            logger=None,
        )
        if os.path.exists(preview_path):
            return preview_path
    except Exception as exc:
        print(f"[Telegram 프리뷰 생성 실패] {exc}")
    finally:
        for obj in (resized, sub, clip):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass
    return None


def _telemetry_log(message: str, config: Optional[AppConfig] = None) -> bool:
    """
    디버그용 텔레그램 로그 전송 (진행 상황/오류 추적용)
    TELEGRAM_DEBUG_LOGS=1 일 때만 동작.
    """
    token = ""
    chat_id = ""
    enabled = False
    if config:
        token = config.telegram_bot_token
        chat_id = config.telegram_admin_chat_id
        enabled = bool(config.telegram_debug_logs)
        if getattr(config, "telegram_timeline_only", False):
            return False
    else:
        token = _get_secret("TELEGRAM_BOT_TOKEN", "") or ""
        chat_id = _get_secret("TELEGRAM_ADMIN_CHAT_ID", "") or ""
        enabled = _get_bool("TELEGRAM_DEBUG_LOGS", True)
        if _get_bool("TELEGRAM_TIMELINE_ONLY", True):
            return False
    if not enabled or not token or not chat_id:
        return False
    prefix = "[auto-shorts]"
    try:
        return send_telegram_message(token, chat_id, f"{prefix} {message}", silent=True)
    except Exception:
        return False


def _append_bgm_debug(message: str) -> None:
    """BGM 디버그 메시지를 session_state에 저장 (로그 탭 표시용)."""
    try:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        logs = st.session_state.get("bgm_debug_logs", [])
        logs.append(f"{ts} {message}")
        st.session_state["bgm_debug_logs"] = logs[-60:]
    except Exception:
        pass


def _approval_keyboard() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ 업로드 승인", "callback_data": "approve"},
                {"text": "⏸ 업로드 보류", "callback_data": "swap"},
            ]
        ]
    }


def send_telegram_approval_request(
    token: str,
    chat_id: str,
    text: str,
) -> Optional[str]:
    """
    인라인 버튼(✅ 승인 / 🔄 교환)이 포함된 승인 요청 메시지 전송.
    성공 시 message_id 반환, 실패 시 None 반환.
    """
    if not token or not chat_id:
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3800  # 버튼 포함 여유 확보
    body = text[:max_len] + ("..." if len(text) > max_len else "")
    payload = {
        "chat_id": chat_id,
        "text": body,
        "reply_markup": _approval_keyboard(),
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.ok:
            return str(resp.json().get("result", {}).get("message_id", ""))
        print(f"[Telegram 버튼 전송 실패] status={resp.status_code} body={resp.text[:300]}")
    except Exception as exc:
        print(f"[Telegram 버튼 전송 오류] {exc}")
    return None


def send_telegram_video_approval_request(
    token: str,
    chat_id: str,
    video_path: str,
    caption_text: str,
) -> Optional[str]:
    """
    렌더링된 영상 + 대본 캡션 + 승인 버튼을 한 메시지로 전송한다.
    성공 시 해당 메시지의 message_id를 반환한다.
    """
    if not token or not chat_id:
        return None
    body = (caption_text or "").strip()
    caption = body[:980] + ("..." if len(body) > 980 else "")
    keyboard_json = json.dumps(_approval_keyboard(), ensure_ascii=False)

    sent_video = False
    media_path = video_path
    preview_generated = False
    message_id: Optional[str] = None
    try:
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            # Bot API 제한/전송 실패 가능성 대비: 큰 파일은 자동 프리뷰로 축소
            if file_size > 45 * 1024 * 1024:
                preview_path = _build_telegram_preview_video(
                    source_video_path=video_path,
                    output_dir=os.path.dirname(video_path) or ".",
                    max_duration_sec=20.0,
                    target_width=540,
                )
                if preview_path and os.path.exists(preview_path):
                    media_path = preview_path
                    preview_generated = True
    except Exception:
        pass

    if media_path and os.path.exists(media_path):
        send_video_url = f"https://api.telegram.org/bot{token}/sendVideo"
        try:
            with open(media_path, "rb") as handle:
                resp = requests.post(
                    send_video_url,
                    data={
                        "chat_id": chat_id,
                        "caption": (caption + ("\n\n(미리보기 20초)" if preview_generated else ""))[:1024],
                        "supports_streaming": "true",
                        "reply_markup": keyboard_json,
                    },
                    files={"video": handle},
                    timeout=180,
                )
            if resp.ok:
                sent_video = True
                try:
                    message_id = str(resp.json().get("result", {}).get("message_id", "") or "")
                except Exception:
                    message_id = ""
            else:
                print(f"[Telegram 비디오 전송 실패] status={resp.status_code} body={resp.text[:300]}")
        except Exception as exc:
            print(f"[Telegram 비디오 전송 오류] {exc}")

        if not sent_video:
            send_doc_url = f"https://api.telegram.org/bot{token}/sendDocument"
            try:
                with open(media_path, "rb") as handle:
                    resp = requests.post(
                        send_doc_url,
                        data={
                            "chat_id": chat_id,
                            "caption": (caption + ("\n\n(미리보기 파일)" if preview_generated else ""))[:1024],
                            "reply_markup": keyboard_json,
                        },
                        files={"document": handle},
                        timeout=180,
                    )
                if resp.ok:
                    sent_video = True
                    try:
                        message_id = str(resp.json().get("result", {}).get("message_id", "") or "")
                    except Exception:
                        message_id = ""
                else:
                    print(f"[Telegram 문서 전송 실패] status={resp.status_code} body={resp.text[:300]}")
            except Exception as exc:
                print(f"[Telegram 문서 전송 오류] {exc}")

    # 영상/문서 실패 시 텍스트+버튼 단일 메시지로 대체
    if not sent_video:
        request_text = (
            (body + "\n\n" if body else "")
            + "영상 전송에 실패해 텍스트 승인으로 대체합니다. 버튼을 눌러주세요."
        )
        message_id = send_telegram_approval_request(token, chat_id, request_text)

    # 임시 프리뷰 파일 정리
    if preview_generated and media_path and media_path != video_path:
        try:
            if os.path.exists(media_path):
                os.remove(media_path)
        except Exception:
            pass
    return message_id if message_id else None


def _answer_callback_query(token: str, callback_query_id: str, text: str = "") -> None:
    """버튼 클릭 후 로딩 스피너 제거 (answerCallbackQuery)."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass


def _disable_approval_buttons(token: str, chat_id: str, message_id: str, result: str) -> None:
    """버튼 클릭 후 메시지를 결과 텍스트로 교체해 버튼 비활성화."""
    label = "✅ 승인됨" if result == "approve" else "⏸ 보류됨"
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/editMessageReplyMarkup",
            json={
                "chat_id": chat_id,
                "message_id": int(message_id),
                "reply_markup": {"inline_keyboard": [[{"text": label, "callback_data": "done"}]]},
            },
            timeout=10,
        )
    except Exception:
        pass


def get_telegram_updates(token: str, offset: int) -> List[Dict[str, Any]]:
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    # timeout=0 으로 즉시 응답 (Streamlit Cloud long-polling 문제 방지)
    # allowed_updates 에 callback_query 포함해서 버튼 클릭 확실히 수신
    params = {
        "offset": offset,
        "timeout": 0,
        "allowed_updates": ["message", "callback_query"],
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("result", [])
    except Exception as e:
        print(f"[getUpdates 오류] {e}")
        return []


def wait_for_approval(
    config: AppConfig,
    progress,
    status_box,
    approval_message_id: Optional[str] = None,
    stage_label: str = "텔레그램 버튼 응답 대기 중...",
    timeout_decision: str = "approve",
) -> str:
    """
    텔레그램에서 승인/교환 응답 대기.
    - callback_query (버튼 클릭) 우선 처리
    - 텍스트 메시지 fallback 지원 (이전 방식 호환)
    """
    start_time = time.time()
    offset = _load_offset(config.telegram_offset_path)
    approve_set = {kw.lower() for kw in config.approve_keywords}
    swap_set = {kw.lower() for kw in config.swap_keywords}

    while time.time() - start_time < config.telegram_timeout_sec:
        _status_update(progress, status_box, 0.25, stage_label)
        try:
            updates = get_telegram_updates(config.telegram_bot_token, offset)
        except Exception:
            updates = []

        for update in updates:
            update_id = update.get("update_id", 0)
            offset = max(offset, update_id + 1)

            # ── 버튼 클릭 (callback_query) 처리 ──────────────────
            callback = update.get("callback_query")
            if callback:
                cb_data = (callback.get("data") or "").strip().lower()
                cb_id = callback.get("id", "")
                # chat_id: message > chat > id (개인/그룹 모두 커버)
                cb_msg = callback.get("message", {})
                cb_chat_id = str(cb_msg.get("chat", {}).get("id", ""))
                # from.id: 버튼을 누른 사람의 개인 ID
                cb_from_id = str(callback.get("from", {}).get("id", ""))
                # fallback: from.id (개인 채팅일 경우)
                if not cb_chat_id:
                    cb_chat_id = cb_from_id

                print(f"[callback] data={cb_data} chat_id={cb_chat_id} from_id={cb_from_id} admin={config.telegram_admin_chat_id}")

                # 관리자 체크: 그룹 ID 또는 개인 ID 중 하나라도 일치하면 통과
                if config.telegram_admin_chat_id:
                    admin_id = str(config.telegram_admin_chat_id)
                    if cb_chat_id != admin_id and cb_from_id != admin_id:
                        print(f"[callback] 관리자 아님 - 무시 (chat={cb_chat_id}, from={cb_from_id})")
                        continue

                if cb_data in ("approve", "approved"):
                    _answer_callback_query(config.telegram_bot_token, cb_id, "✅ 승인되었습니다!")
                    if approval_message_id:
                        _disable_approval_buttons(
                            config.telegram_bot_token,
                            config.telegram_admin_chat_id,
                            approval_message_id,
                            "approve",
                        )
                    _save_offset(config.telegram_offset_path, offset)
                    return "approve"

                if cb_data in ("swap", "exchange"):
                    _answer_callback_query(config.telegram_bot_token, cb_id, "⏸ 업로드 보류 처리합니다.")
                    if approval_message_id:
                        _disable_approval_buttons(
                            config.telegram_bot_token,
                            config.telegram_admin_chat_id,
                            approval_message_id,
                            "swap",
                        )
                    _save_offset(config.telegram_offset_path, offset)
                    return "swap"

            # ── 텍스트 메시지 fallback ────────────────────────────
            message = update.get("message", {})
            if not message:
                continue
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
        time.sleep(3)  # 5초→3초로 단축해 응답성 향상

    return timeout_decision


def _write_local_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mark_topic_once(path: str, topic: str, title: str = "") -> bool:
    key = (topic or "").strip()
    if not key:
        return False
    used_topics = _load_used_topics(path)
    if _is_used_topic(used_topics, key):
        return False
    _mark_used_topic(path, key, title=title)
    return True


def _write_approved_content_log(
    config: AppConfig,
    *,
    topic_key: str,
    title: str,
    mood: str,
    hashtags: List[str],
    pinned_ja: str,
    pinned_ko: str,
    texts_ja: List[str],
    texts_ko: List[str],
    background_mode: str,
    caption_variant: str,
    video_path: str,
) -> None:
    record = {
        "date_jst": _get_local_now(config).strftime("%Y-%m-%d %H:%M:%S"),
        "topic_key": topic_key,
        "title_ja": title,
        "mood": mood,
        "hashtags_ja": " ".join(hashtags or []),
        "pinned_comment_ja": pinned_ja,
        "pinned_comment_ko": pinned_ko,
        "script_ja": texts_ja or [],
        "script_ko": texts_ko or [],
        "background_mode": background_mode,
        "caption_variant": caption_variant,
        "video_path": video_path,
    }
    _write_local_log(os.path.join(config.output_dir, "approved_contents.jsonl"), record)


def _format_hashtags(tags: List[str]) -> str:
    return " ".join(tags)


def _normalize_ko_lines(texts_ko: List[str], texts_ja: List[str]) -> List[str]:
    if not texts_ko:
        return [_to_ko_literal_tone(line) for line in texts_ja]
    normalized: List[str] = []
    for idx, ja in enumerate(texts_ja):
        ko = texts_ko[idx] if idx < len(texts_ko) else ""
        normalized.append(_to_ko_literal_tone(ko if ko else ja))
    return normalized


_CAPTION_LANG_SEP = "<<KO>>"


def _split_caption_chunks(text: str, max_chars: int = 14, max_lines: int = 3) -> str:
    src = re.sub(r"\s+", " ", str(text or "").strip())
    if not src:
        return ""

    # 쉼표/마침표 기반으로 우선 분리 (요청사항 반영)
    parts = re.split(r"(?<=[,，、.。．!?？！])\s*", src)
    if not parts:
        parts = [src]

    def _char_w(ch: str) -> float:
        return 1.0 if ord(ch) > 127 else 0.55

    def _hard_wrap(segment: str) -> List[str]:
        seg = segment.strip()
        if not seg:
            return []
        out: List[str] = []
        cur = ""
        cur_w = 0.0
        limit = float(max(8, max_chars))
        for ch in seg:
            cw = _char_w(ch)
            if cur and (cur_w + cw) > limit:
                out.append(cur)
                cur = ch
                cur_w = cw
            else:
                cur += ch
                cur_w += cw
        if cur:
            out.append(cur)
        return out

    lines: List[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        wrapped = _hard_wrap(p)
        if not wrapped:
            continue
        lines.extend(wrapped)

    if not lines:
        lines = _hard_wrap(src)
    if not lines:
        return src

    # 줄 수 제한 (세로 영역 과점유 방지)
    lines = lines[: max(1, max_lines)]
    return "\n".join(lines)


def _build_bilingual_caption_texts(
    config: AppConfig,
    texts_ja: List[str],
    texts_ko: List[str],
) -> List[str]:
    ko_norm = _normalize_ko_lines(texts_ko, texts_ja)
    merged: List[str] = []
    max_chars = max(8, int(getattr(config, "caption_max_chars", 14) or 14))
    for idx, ja_src in enumerate(texts_ja):
        ko_src = ko_norm[idx] if idx < len(ko_norm) else ""
        ja_line = _split_caption_chunks(
            re.sub(r"\s+", " ", str(ja_src or "").strip()),
            max_chars=max_chars,
            max_lines=2,
        )
        ko_line = _split_caption_chunks(
            re.sub(r"\s+", " ", str(ko_src or "").strip()),
            max_chars=max_chars + 2,
            max_lines=2,
        )
        if ko_line and ko_line != ja_line:
            merged.append(f"{ja_line}{_CAPTION_LANG_SEP}{ko_line}")
        else:
            merged.append(ja_line)
    return merged


def _guess_ass_font_name(font_path: str) -> str:
    name = os.path.basename(str(font_path or "")).lower()
    if "mplusrounded1c" in name:
        return "M PLUS Rounded 1c"
    if "zenkakugothicnew" in name:
        return "Zen Kaku Gothic New"
    if "bizudpgothic" in name:
        return "BIZ UDPGothic"
    if "notosansjp" in name:
        return "Noto Sans JP"
    return "Noto Sans CJK JP"


def _ass_escape_text(text: str) -> str:
    value = str(text or "")
    value = value.replace(_CAPTION_LANG_SEP, "\n\n")
    value = value.replace("\\", "\\\\")
    value = value.replace("{", r"\{").replace("}", r"\}")
    return value.replace("\n", r"\N")


def _ffmpeg_sub_path_escape(path: str) -> str:
    value = os.path.abspath(path).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace(",", r"\,")
    value = value.replace("[", r"\[")
    value = value.replace("]", r"\]")
    value = value.replace(" ", r"\ ")
    return value


def _build_ass_subtitle_file(
    config: AppConfig,
    ass_path: str,
    texts: List[str],
    caption_texts: List[str],
    caption_styles: Optional[List[str]],
    tts_audio_path: str,
) -> bool:
    if not PYSUBS2_AVAILABLE or pysubs2 is None:
        return False
    if not texts or not caption_texts or not os.path.exists(tts_audio_path):
        return False
    try:
        audio_clip = AudioFileClip(tts_audio_path)
        max_duration = max(5.0, float(getattr(config, "max_video_duration_sec", 59.0)))
        duration = min(float(audio_clip.duration or 0.0), max_duration)
        audio_clip.close()
        if duration <= 0.0:
            return False
        seg_durations = _estimate_durations(texts, duration)
        hold_ratio = float(getattr(config, "caption_hold_ratio", 1.0) or 1.0)
        if not getattr(config, "caption_trim", False):
            hold_ratio = 1.0
        hold_ratio = max(0.45, min(1.0, hold_ratio))

        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = str(int(config.width))
        subs.info["PlayResY"] = str(int(config.height))
        subs.info["WrapStyle"] = "2"
        font_name = _guess_ass_font_name(config.font_path)

        default_style = pysubs2.SSAStyle(
            fontname=font_name,
            fontsize=max(46, int(config.width * 0.053)),
            bold=True,
            italic=False,
            underline=False,
            alignment=2,
            marginl=int(config.width * 0.07),
            marginr=int(config.width * 0.07),
            marginv=int(config.height * 0.34),
            primarycolor=pysubs2.Color(255, 255, 255, 0),
            secondarycolor=pysubs2.Color(255, 255, 255, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 0),
            backcolor=pysubs2.Color(0, 0, 0, 140),
            borderstyle=1,
            outline=2.8,
            shadow=0.8,
            spacing=0,
            angle=0,
        )
        variety_style = default_style.copy()
        variety_style.primarycolor = pysubs2.Color(0, 240, 255, 0)
        variety_style.outline = 4.6
        variety_style.shadow = 1.0

        reaction_style = default_style.copy()
        # 마지막 혼잣말은 고대비(밝은 글자 + 검정 외곽선)로 고정
        reaction_style.primarycolor = pysubs2.Color(120, 248, 255, 0)
        reaction_style.secondarycolor = pysubs2.Color(120, 248, 255, 0)
        reaction_style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
        reaction_style.backcolor = pysubs2.Color(28, 28, 28, 90)
        reaction_style.borderstyle = 1
        reaction_style.outline = 5.2
        reaction_style.shadow = 1.2
        reaction_style.bold = True

        asmr_style = default_style.copy()
        asmr_style.primarycolor = pysubs2.Color(188, 98, 249, 0)
        asmr_style.secondarycolor = pysubs2.Color(188, 98, 249, 0)
        asmr_style.outlinecolor = pysubs2.Color(40, 20, 40, 0)
        asmr_style.backcolor = pysubs2.Color(246, 238, 255, 60)
        asmr_style.borderstyle = 1
        asmr_style.outline = 4.0
        asmr_style.shadow = 0.6
        asmr_style.bold = True

        subs.styles["Default"] = default_style
        subs.styles["Variety"] = variety_style
        subs.styles["Reaction"] = reaction_style
        subs.styles["Asmr"] = asmr_style

        start_sec = 0.0
        for idx, seg_dur in enumerate(seg_durations):
            text_raw = caption_texts[idx] if idx < len(caption_texts) else texts[idx]
            if _CAPTION_LANG_SEP in str(text_raw):
                ja_raw, ko_raw = str(text_raw).split(_CAPTION_LANG_SEP, 1)
                text_raw = f"{ja_raw}\n\n{ko_raw}"
            text_out = _ass_escape_text(text_raw)
            seg_show = max(0.45, seg_dur * hold_ratio)
            end_sec = min(duration, start_sec + seg_show)
            if end_sec <= start_sec:
                end_sec = min(duration, start_sec + 0.45)
            role_style = "Default"
            if caption_styles and idx < len(caption_styles):
                style_name = str(caption_styles[idx] or "").strip().lower()
                if style_name in {"reaction", "outro", "outro_loop"}:
                    role_style = "Reaction"
                elif style_name in {"asmr_tag", "asmr"}:
                    role_style = "Asmr"
                elif style_name == "japanese_variety":
                    role_style = "Variety"
            subs.events.append(
                pysubs2.SSAEvent(
                    start=int(start_sec * 1000),
                    end=int(end_sec * 1000),
                    style=role_style,
                    text=text_out,
                )
            )
            start_sec += seg_dur
        os.makedirs(os.path.dirname(ass_path) or ".", exist_ok=True)
        subs.save(ass_path, encoding="utf-8")
        return True
    except Exception:
        return False


def _burn_ass_subtitles_to_video(
    config: AppConfig,
    input_video_path: str,
    output_video_path: str,
    ass_path: str,
) -> bool:
    if not ass_path or not os.path.exists(ass_path):
        return False
    ffmpeg_bin = "ffmpeg"
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    ass_escaped = _ffmpeg_sub_path_escape(ass_path)
    fonts_dir = os.path.dirname(config.font_path) if config.font_path else ""
    if not fonts_dir:
        fonts_dir = os.path.join(config.assets_dir, "fonts")
    fonts_escaped = _ffmpeg_sub_path_escape(fonts_dir)
    vf_expr = f"subtitles='{ass_escaped}':fontsdir='{fonts_escaped}':charenc=UTF-8"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        vf_expr,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "19",
        "-c:a",
        "copy",
        output_video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and os.path.exists(output_video_path):
            return True
        tail = (proc.stderr or "")[-400:]
        print(f"[ASS 자막 번인 실패] rc={proc.returncode} err={tail}")
        return False
    except Exception as exc:
        print(f"[ASS 자막 번인 예외] {exc}")
        return False


def _render_ass_subtitled_video(
    config: AppConfig,
    input_video_path: str,
    output_video_path: str,
    ass_path: str,
    texts: List[str],
    caption_texts: List[str],
    caption_styles: Optional[List[str]],
    tts_audio_path: str,
) -> bool:
    ok = _build_ass_subtitle_file(
        config=config,
        ass_path=ass_path,
        texts=texts,
        caption_texts=caption_texts,
        caption_styles=caption_styles,
        tts_audio_path=tts_audio_path,
    )
    if not ok:
        return False
    return _burn_ass_subtitles_to_video(
        config=config,
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        ass_path=ass_path,
    )


def _build_upload_report_ko(
    title: str,
    video_url: str,
    hashtags: List[str],
    pinned_comment_ko: str,
    texts_ko: List[str],
    mood: str = "",
) -> str:
    lines = [
        "[업로드 완료] 유튜브 쇼츠",
        f"제목: {title}",
        f"URL: {video_url}",
    ]
    if mood:
        lines.append(f"무드: {mood}")
    if hashtags:
        lines.append(f"해시태그: {_format_hashtags(hashtags)}")
    lines.append(f"고정댓글(한글): {pinned_comment_ko}")
    lines.append("")
    lines.append("[대본(한글)]")
    for idx, line in enumerate(texts_ko, start=1):
        if not line:
            continue
        lines.append(f"{idx}. {line}")
    return "\n".join(lines)


def _build_approval_script_summary(
    texts_ja: List[str],
    texts_ko: List[str],
    roles: Optional[List[str]] = None,
) -> str:
    role_labels = {
        "hook": "Hook",
        "problem": "Problem",
        "failure": "Failure",
        "success": "Twist",
        "point": "Point",
        "reaction": "Reaction",
    }
    lines: List[str] = ["[대본 요약 - 기승전결 전체]"]
    ko_norm = _normalize_ko_lines(texts_ko, texts_ja)
    for idx, ja in enumerate(texts_ja):
        role = roles[idx] if roles and idx < len(roles) else "body"
        role_label = role_labels.get(str(role).lower(), "Body")
        ja_line = re.sub(r"\s+", " ", str(ja or "").strip())
        ko_line = re.sub(r"\s+", " ", str(ko_norm[idx] if idx < len(ko_norm) else "").strip())
        lines.append(f"{idx+1}. [{role_label}] JA: {ja_line}")
        if ko_line:
            lines.append(f"   KO: {ko_line}")
    return "\n".join(lines)


def _build_upload_caption_text(
    title: str,
    hashtags: List[str],
    hook_text: str = "",
) -> str:
    """
    업로드 캡션(설명) 본문 생성.
    고정댓글 문구와 분리해 캡션에는 훅+해시태그만 넣는다.
    """
    first_line = (hook_text or "").strip() or (title or "").strip()
    lines: List[str] = []
    if first_line:
        lines.append(first_line)
    lines.append("")
    if hashtags:
        lines.append(_format_hashtags(hashtags))
    return "\n".join(lines).strip()


def _post_youtube_comment_after_upload(
    config: AppConfig,
    video_id: str,
    pinned_base: str,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    캡션과 별도로 댓글(고정댓글 용도) 작성.
    - 기본: pinned_base 그대로 댓글 작성
    - 선택: ENABLE_PINNED_COMMENT + LINKTREE 설정 시 제품번호 CTA를 앞에 덧붙임
    """
    if not video_id:
        return False
    base_text = (pinned_base or "").strip()
    if not base_text:
        return False
    comment_text = base_text
    if getattr(config, "enable_pinned_comment", False) and config.linktree_url:
        product_number = ""
        if isinstance(meta, dict):
            product_number = str(meta.get("product_number", "") or "").strip()
        if not product_number:
            product_number = _pick_product_number_for_short(config)
        if product_number:
            comment_text = build_pinned_comment_with_voting(
                product_number,
                config.linktree_url,
                pinned_base=base_text,
            )
    return insert_video_comment(config, video_id, comment_text)


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


def _clear_dir_cache(path: str, allowed_ext: Optional[Tuple[str, ...]] = None) -> int:
    if not path or not os.path.exists(path):
        return 0
    removed = 0
    try:
        for root, _, files in os.walk(path, topdown=False):
            for name in files:
                full = os.path.join(root, name)
                if allowed_ext and not name.lower().endswith(allowed_ext):
                    continue
                try:
                    os.remove(full)
                    removed += 1
                except Exception:
                    continue
            if root != path:
                try:
                    if not os.listdir(root):
                        os.rmdir(root)
                except Exception:
                    pass
    except Exception:
        return removed
    return removed


def _clear_background_download_pool(config: AppConfig) -> int:
    backgrounds_dir = os.path.join(config.assets_dir, "backgrounds")
    if not backgrounds_dir or not os.path.exists(backgrounds_dir):
        return 0
    removed = 0
    prefixes = ("minecraft_parkour_", "pixabay_bg_", "pexels_bg_", "context_bg_", "yt_bg_")
    allowed_ext = (".mp4", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".webp")
    try:
        for name in os.listdir(backgrounds_dir):
            lower = name.lower()
            if (not lower.endswith(allowed_ext)) or (not name.startswith(prefixes)):
                continue
            full = os.path.join(backgrounds_dir, name)
            try:
                os.remove(full)
                removed += 1
            except Exception:
                continue
    except Exception:
        return removed
    return removed


def _reset_runtime_caches(config: AppConfig, deep: bool = False) -> None:
    global _HIGHLIGHT_CACHE, _LAST_GENERATED_BG_TOPIC_KEY
    _HIGHLIGHT_CACHE.clear()
    _LAST_GENERATED_BG_TOPIC_KEY = ""
    cache_dirs = [
        config.generated_bg_dir,
        "/tmp/pixabay_images",
        "/tmp/pexels_bg_images",
        "/tmp/serpapi_images",
        "/tmp/wikimedia_images",
        "/tmp/pixabay_bg",
        "/tmp/pexels_bg",
        "/tmp/pexels_bg_videos",
        "/tmp/pixabay_bg_videos",
    ]
    removed_total = 0
    for d in cache_dirs:
        removed_total += _clear_dir_cache(
            d,
            allowed_ext=(".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mkv", ".webm", ".ass"),
        )
    if deep:
        removed_total += _clear_background_download_pool(config)
    if removed_total:
        _telemetry_log(f"런타임 캐시 초기화: {removed_total}개 파일 정리", config)


def _load_recent_run_records(config: AppConfig, limit: int = 120) -> List[Dict[str, Any]]:
    path = os.path.join(config.output_dir, "runs.jsonl")
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()[-max(20, limit * 2):]
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            records.append(rec)
    except Exception:
        return []
    # 최신 순으로 dedupe(video_id 기준)
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for rec in reversed(records):
        vid = str(rec.get("youtube_video_id", "") or "").strip()
        if not vid or vid in seen:
            continue
        seen.add(vid)
        deduped.append(rec)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_growth_terms(text: str) -> List[str]:
    src = str(text or "").lower()
    parts = re.split(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff]+", src)
    stop = {
        "shorts", "youtube", "video", "story", "mystery", "shock", "news",
        "の", "が", "を", "に", "は", "で", "と", "て", "する", "した",
        "これ", "それ", "この", "その", "そして", "でも", "実は",
    }
    return [p for p in parts if len(p) >= 2 and p not in stop]


def _parse_datetime_loose(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _refresh_video_metrics_state(config: AppConfig, limit: int = 120) -> Dict[str, Any]:
    state = _read_json_file(config.video_metrics_state_path, {"videos": {}, "last_refresh": ""})
    if not isinstance(state, dict):
        state = {"videos": {}, "last_refresh": ""}
    videos = state.get("videos", {})
    if not isinstance(videos, dict):
        videos = {}

    records = _load_recent_run_records(config, limit=limit)
    if not records:
        state["videos"] = videos
        return state

    now = _get_local_now(config)
    api_key = getattr(config, "youtube_api_key", "") or ""
    fetch_budget = 30
    for rec in records:
        vid = str(rec.get("youtube_video_id", "") or "").strip()
        if not vid:
            continue
        current = videos.get(vid, {}) if isinstance(videos.get(vid), dict) else {}
        fetched_at = _parse_datetime_loose(str(current.get("fetched_at", "") or ""))
        should_fetch = bool(api_key)
        if should_fetch and fetched_at is not None:
            should_fetch = (now - fetched_at).total_seconds() >= 6 * 3600
        stats: Dict[str, int] = {}
        if should_fetch and fetch_budget > 0:
            stats = _fetch_youtube_stats(vid, api_key)
            fetch_budget -= 1
        elif isinstance(current, dict):
            stats = {
                "viewCount": int(current.get("viewCount", 0) or 0),
                "likeCount": int(current.get("likeCount", 0) or 0),
                "commentCount": int(current.get("commentCount", 0) or 0),
            }
        videos[vid] = {
            "video_id": vid,
            "title": str(rec.get("title_ja", "") or ""),
            "topic_theme": str(rec.get("topic_theme", "") or ""),
            "hashtags_ja": str(rec.get("hashtags_ja", "") or ""),
            "youtube_url": str(rec.get("youtube_url", "") or ""),
            "viewCount": int(stats.get("viewCount", current.get("viewCount", 0) or 0) or 0),
            "likeCount": int(stats.get("likeCount", current.get("likeCount", 0) or 0) or 0),
            "commentCount": int(stats.get("commentCount", current.get("commentCount", 0) or 0) or 0),
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        }

    state["videos"] = videos
    state["last_refresh"] = now.strftime("%Y-%m-%d %H:%M:%S")
    _write_json_file(config.video_metrics_state_path, state)
    return state


def _build_growth_feedback_hint(config: AppConfig) -> str:
    state = _refresh_video_metrics_state(config, limit=120)
    videos = state.get("videos", {})
    if not isinstance(videos, dict) or not videos:
        return ""
    items = [v for v in videos.values() if isinstance(v, dict)]
    items = [v for v in items if int(v.get("viewCount", 0) or 0) > 0]
    if len(items) < 3:
        return ""
    ranked = sorted(
        items,
        key=lambda x: int(x.get("viewCount", 0) or 0) + int(x.get("likeCount", 0) or 0) * 20,
        reverse=True,
    )
    top = ranked[: min(6, len(ranked))]
    bottom = ranked[-min(4, len(ranked)) :]

    top_tags = Counter()
    top_terms = Counter()
    for row in top:
        for t in str(row.get("hashtags_ja", "") or "").split():
            if t.startswith("#"):
                top_tags[t] += 1
        joined = f"{row.get('title','')} {row.get('topic_theme','')}"
        for term in _extract_growth_terms(joined):
            top_terms[term] += 1

    low_terms = Counter()
    for row in bottom:
        joined = f"{row.get('title','')} {row.get('topic_theme','')}"
        for term in _extract_growth_terms(joined):
            low_terms[term] += 1

    best_tags = [tag for tag, _ in top_tags.most_common(4)]
    best_terms = [term for term, _ in top_terms.most_common(5)]
    avoid_terms = [term for term, _ in low_terms.most_common(3) if term not in best_terms]

    avg_views = int(sum(int(r.get("viewCount", 0) or 0) for r in top) / max(1, len(top)))
    avg_like_rate = 0.0
    try:
        avg_like_rate = (
            sum((int(r.get("likeCount", 0) or 0) / max(1, int(r.get("viewCount", 0) or 1))) for r in top)
            / max(1, len(top))
        ) * 100.0
    except Exception:
        avg_like_rate = 0.0

    lines = [
        "[성과 회고 데이터]",
        f"- 상위 성과 평균 조회수: {avg_views}",
        f"- 상위 평균 좋아요율: {avg_like_rate:.2f}%",
    ]
    if best_terms:
        lines.append(f"- 상위 반복 키워드: {', '.join(best_terms[:3])}")
    if best_tags:
        lines.append(f"- 상위 해시태그: {' '.join(best_tags)}")
    if avoid_terms:
        lines.append(f"- 저성과 반복 키워드 회피: {', '.join(avoid_terms)}")
    lines.append("- 이번 주제는 상위 키워드 1~2개를 반영하고, 저성과 키워드는 피해서 작성")
    return "\n".join(lines)


def _get_local_now(config: AppConfig) -> datetime:
    tz_name = (config.auto_run_tz or "Asia/Tokyo").strip()
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.utcnow()


def _normalize_auto_run_time(value: str) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", raw)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _get_auto_run_time_labels(config: AppConfig) -> List[str]:
    labels: List[str] = []
    for item in getattr(config, "auto_run_times", []) or []:
        norm = _normalize_auto_run_time(item)
        if norm and norm not in labels:
            labels.append(norm)
    if not labels:
        fallback = f"{int(getattr(config, 'auto_run_hour', 18)):02d}:00"
        labels.append(fallback)
    return sorted(labels)


def _get_auto_run_slots_for_date(config: AppConfig, base_dt: datetime) -> List[Tuple[str, datetime]]:
    slots: List[Tuple[str, datetime]] = []
    for label in _get_auto_run_time_labels(config):
        hour, minute = [int(part) for part in label.split(":")]
        slot_dt = base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        slots.append((label, slot_dt))
    return slots


def _get_next_run_time(config: AppConfig) -> datetime:
    now = _get_local_now(config)
    today_slots = _get_auto_run_slots_for_date(config, now)
    for _, slot_dt in today_slots:
        if slot_dt > now:
            return slot_dt
    tomorrow = now + timedelta(days=1)
    tomorrow_slots = _get_auto_run_slots_for_date(config, tomorrow)
    if tomorrow_slots:
        return tomorrow_slots[0][1]
    return now + timedelta(hours=24)


def _acquire_run_lock(path: str, ttl_sec: int = 7200) -> bool:
    try:
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < ttl_sec:
                return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(str(int(time.time())))
        return True
    except Exception:
        return True


def _release_run_lock(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _should_auto_run(config: AppConfig) -> bool:
    if not config.auto_run_daily:
        return False
    now = _get_local_now(config)
    state = _read_json_file(config.auto_run_state_path, {})
    if not isinstance(state, dict):
        state = {}
    runs = state.get("runs", {})
    if not isinstance(runs, dict):
        runs = {}
    today_key = now.date().isoformat()
    done_today_raw = runs.get(today_key, [])
    done_today = {
        _normalize_auto_run_time(item) or str(item)
        for item in (done_today_raw if isinstance(done_today_raw, list) else [])
        if str(item).strip()
    }
    for label, slot_dt in _get_auto_run_slots_for_date(config, now):
        if now >= slot_dt and label not in done_today:
            return True
    return False


def _mark_auto_run_done(config: AppConfig) -> None:
    now = _get_local_now(config)
    state = _read_json_file(config.auto_run_state_path, {})
    if not isinstance(state, dict):
        state = {}
    runs = state.get("runs", {})
    if not isinstance(runs, dict):
        runs = {}
    today_key = now.date().isoformat()
    done_today = runs.get(today_key, [])
    if not isinstance(done_today, list):
        done_today = []
    done_set = {str(item) for item in done_today if str(item).strip()}
    # 현재 시각 이전 슬롯은 모두 처리 완료로 마킹 (지연 실행 시 연속 중복 실행 방지)
    for label, slot_dt in _get_auto_run_slots_for_date(config, now):
        if now >= slot_dt:
            done_set.add(label)
    runs[today_key] = sorted(done_set)
    # 상태 파일 과대 방지: 최근 14일만 유지
    keep_days = {(now.date() - timedelta(days=idx)).isoformat() for idx in range(14)}
    runs = {day: value for day, value in runs.items() if day in keep_days}
    state["runs"] = runs
    state["last_run_date"] = today_key
    _write_json_file(config.auto_run_state_path, state)


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


def _load_used_topics(path: str) -> Dict[str, Any]:
    return _read_json_file(path, {"topics": []})


def _is_used_topic(used_data: Dict[str, Any], topic: str) -> bool:
    topics = used_data.get("topics", [])
    topic_key = topic.strip().lower()
    return any(str(item.get("key", "")).lower() == topic_key for item in topics)


def _mark_used_topic(path: str, topic: str, title: str = "") -> None:
    if not topic:
        return
    used_data = _load_used_topics(path)
    topics = used_data.get("topics", [])
    topics.append(
        {
            "key": topic.strip().lower(),
            "topic": topic,
            "title": title,
            "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    used_data["topics"] = topics
    _write_json_file(path, used_data)


def _pick_topic_key(meta: Dict[str, Any]) -> str:
    for key in ("topic_en", "topic", "title_ja", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_offset(path: str) -> int:
    data = _read_json_file(path, {"offset": 0})
    return int(data.get("offset", 0))


def _save_offset(path: str, offset: int) -> None:
    _write_json_file(path, {"offset": offset})


def _clear_inbox_unsaved(config: AppConfig, manifest_items: List[AssetItem]) -> int:
    """
    인박스 폴더에서 라이브러리(manifest)에 저장되지 않은 파일을 삭제.
    session_state의 inbox_current_files 기준으로 처리.
    반환값: 삭제된 파일 수
    """
    saved_paths = {item.path for item in manifest_items}
    prev_files: List[str] = st.session_state.get("inbox_current_files", [])
    deleted = 0
    for file_path in prev_files:
        if file_path not in saved_paths and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted += 1
            except Exception:
                pass
    st.session_state["inbox_current_files"] = []
    return deleted


def _missing_required(config: AppConfig) -> List[str]:
    missing = []
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not config.openai_tts_voices:
        missing.append("OPENAI_TTS_VOICE 또는 OPENAI_TTS_VOICES")
    if not config.sheet_id:
        missing.append("SHEET_ID")
    if not config.google_service_account_json:
        missing.append("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not config.font_path:
        missing.append("FONT_PATH")
    if getattr(config, "enable_instagram_upload", False) and not getattr(config, "jp_youtube_only", False):
        if not getattr(config, "instagram_access_token", ""):
            missing.append("INSTAGRAM_ACCESS_TOKEN")
        if not getattr(config, "instagram_user_id", ""):
            missing.append("INSTAGRAM_USER_ID")
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


def _ui_info(message: str, use_streamlit: bool = True) -> None:
    if use_streamlit:
        try:
            st.info(message)
            return
        except Exception:
            pass
    print(message)


def _ui_warning(message: str, use_streamlit: bool = True) -> None:
    if use_streamlit:
        try:
            st.warning(message)
            return
        except Exception:
            pass
    print(message)


def _ui_error(message: str, use_streamlit: bool = True) -> None:
    if use_streamlit:
        try:
            st.error(message)
            return
        except Exception:
            pass
    print(message)


def _script_plan_text(script: Dict[str, Any]) -> str:
    _meta = script.get("meta", {})
    texts = _script_to_beats(script)
    middle = texts[1] if len(texts) > 1 else (texts[0] if texts else "")
    return (
        f"제목: {_meta.get('title_ja', _meta.get('title', script.get('video_title','')))}\n"
        f"무드: {_meta.get('bgm_mood', script.get('mood','mystery_suspense'))}\n"
        f"훅: {texts[0] if texts else ''}\n"
        f"전개: {middle}\n"
        f"구독유도: {texts[-1] if texts else ''}\n"
        f"해시태그: {' '.join(script.get('hashtags', []))}"
    )


def _auto_jp_flow(
    config: AppConfig,
    progress,
    status_box,
    extra_hint: str = "",
    use_streamlit: bool = True,
) -> bool:
    """
    크롤링 없이 LLM이 주제를 자동 선정해 일본인 타겟 숏츠를 생성하는 메인 플로우.
    텔레그램 승인 → TTS → 영상 렌더링 → 유튜브 업로드.
    """
    if config.require_approval and (not config.telegram_bot_token or not config.telegram_admin_chat_id):
        _ui_error(
            "승인 모드에서는 텔레그램 봇 설정이 필요합니다. TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID를 확인하세요.",
            use_streamlit,
        )
        return False
    _set_run_notifier(RunTimelineNotifier(config, enabled=True))
    _reset_runtime_caches(config, deep=bool(getattr(config, "force_fresh_media_on_start", False)))
    _notify("🚀", "AI 쇼츠 파이프라인 시작!")
    approved_for_publish = not config.require_approval

    # ── 대본 생성 ─────────────────────────────────────────
    if getattr(config, "retry_pending_uploads", False):
        _try_upload_pending(config, use_streamlit=use_streamlit, max_items=1)
    _telemetry_log("자동 생성 시작 — 대본 생성 단계 진입", config)
    _status_update(progress, status_box, 0.10, "AI 대본 생성 중 (주제 자동 선정)...")
    _notify("📝", "기획팀 작업 시작", "GPT-4야, 오늘의 미스터리 대본 써줘")
    script = None
    meta = {}
    content_list = []
    topic_key = ""
    growth_hint = _build_growth_feedback_hint(config)
    base_hint = (extra_hint or "").strip()
    if growth_hint:
        base_hint = (base_hint + "\n\n" + growth_hint).strip() if base_hint else growth_hint
    llm_hint = base_hint
    for attempt in range(3):
        try:
            script = generate_script_jp(config, extra_hint=llm_hint)
            script = _inject_majisho_asmr_beat(script, enabled=getattr(config, "enable_majisho_tag", True))
            _telemetry_log("대본 생성 완료", config)
        except Exception as exc:
            _telemetry_log(f"대본 생성 실패: {exc}", config)
            _ui_error(f"대본 생성 실패: {exc}", use_streamlit)
            _notify("❌", "기획팀 실패", str(exc))
            _reset_runtime_caches(config)
            return False
        meta = script.get("meta", {})
        content_list = _get_story_timeline(script)
        topic_key = _pick_topic_key(meta)
        if topic_key:
            used_topics = _load_used_topics(config.used_topics_path)
            if _is_used_topic(used_topics, topic_key):
                _telemetry_log(f"중복 주제 감지 → 재생성: {topic_key}", config)
                llm_hint = (base_hint + f"\n이전 주제는 제외: {topic_key}").strip()
                continue
        break
    if topic_key:
        used_topics = _load_used_topics(config.used_topics_path)
        if _is_used_topic(used_topics, topic_key):
            msg = f"이미 사용한 주제입니다. 이번 작업을 중단합니다: {topic_key}"
            _telemetry_log(msg, config)
            _ui_warning(msg, use_streamlit)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, msg)
            _notify("❌", "기획팀 실패", "중복 주제로 중단")
            _reset_runtime_caches(config)
            return False

    # ── 새 스키마 필드 추출 ───────────────────────────────
    meta = meta or {}
    video_title = meta.get("title_ja", meta.get("title", script.get("video_title", "ミステリーショーツ")))
    hashtags = meta.get("hashtags", script.get("hashtags", []))
    mood = meta.get("bgm_mood", script.get("mood", "mystery_suspense"))
    pinned = meta.get("pinned_comment", script.get("pinned_comment", ""))
    pinned_ko = _to_ko_literal_tone(meta.get("pinned_comment_ko", pinned))
    pinned_bilingual = _compose_bilingual_text(pinned, pinned_ko)
    if topic_key:
        _mark_topic_once(config.used_topics_path, topic_key, title=video_title)
    if topic_key:
        _notify("📝", "기획팀 완료", f"완성! 오늘 주제: {topic_key}")

    texts = _script_to_beats(script)
    texts_ko = _script_to_beats_ko(script)
    visual_keywords = _script_to_visual_keywords(script)
    roles = _script_to_roles(script)
    caption_styles = _build_caption_styles(roles, len(texts))
    caption_variant = _select_caption_variant(config)
    caption_styles = _apply_caption_variant(caption_styles, caption_variant)
    texts_ko_norm = _normalize_ko_lines(texts_ko, texts)
    caption_texts = _build_bilingual_caption_texts(config, texts, texts_ko_norm)

    # 인스타그램 인기 오디오 스타일: BGM을 fast_exciting 쪽으로 편향
    if getattr(config, "instagram_use_popular_audio", False) and mood == "mystery_suspense":
        if random.random() < 0.5:
            mood = "fast_exciting"
            _telemetry_log("인기 오디오 스타일 적용: fast_exciting", config)

    _ui_info(f"제목: **{video_title}** | 무드: **{mood}** | 자막 변형: **{caption_variant}**", use_streamlit)

    # ── BGM 매칭 ─────────────────────────────────────────
    _telemetry_log(f"BGM 매칭 시작 (mood={mood})", config)
    _status_update(progress, status_box, 0.18, f"BGM 매칭 중 (무드: {mood})")
    bgm_path = match_bgm_by_mood(config, mood)
    bgm_display = os.path.basename(bgm_path) if bgm_path else "없음"
    _telemetry_log(f"BGM 매칭 결과: {bgm_display}", config)
    if not bgm_path or not os.path.exists(bgm_path):
        _ui_warning(
            "BGM 파일을 찾지 못했습니다. "
            "assets/bgm/mystery_suspense 또는 assets/bgm/fast_exciting 폴더에 "
            "BGM 파일을 넣어주세요. (Pixabay 자동 BGM은 PIXABAY_BGM_ENABLED=true일 때만 시도)"
            ,
            use_streamlit,
        )

    # ── 배경 선택: Minecraft 영상 vs 상황 이미지 (A/B) ─────────────
    generated_bg_paths = (
        _get_generated_bg_paths(config, len(texts), topic_key=topic_key)
        if getattr(config, "use_generated_bg_priority", False)
        else []
    )
    bg_video_paths: List[Optional[str]] = []
    bg_image_paths: List[Optional[str]] = []
    forced_bg_mode = _infer_background_mode_by_content(meta, texts, visual_keywords)
    background_mode = _select_background_mode(config, meta=meta, texts=texts, visual_keywords=visual_keywords)
    if forced_bg_mode:
        _telemetry_log(f"배경 모드 강제 선택(콘텐츠 규칙): {background_mode}", config)
    else:
        _telemetry_log(f"배경 모드 선택(AB): {background_mode}", config)
    if generated_bg_paths:
        background_mode = "image"
        bg_video_paths = [None] * len(texts)
        bg_image_paths = generated_bg_paths
        _telemetry_log("생성 배경 이미지 사용", config)
    elif background_mode == "minecraft":
        backgrounds_dir = os.path.join(config.assets_dir, "backgrounds")
        mc_path = fetch_youtube_minecraft_parkour_video(
            backgrounds_dir,
            config.width,
            config.height,
            config=config,
            force_fresh=bool(getattr(config, "force_fresh_minecraft_download", False)),
        )
        if mc_path and _is_video_readable(mc_path):
            bg_video_paths = [mc_path] * len(texts)
            placeholder = _ensure_placeholder_image(config)
            bg_image_paths = [placeholder] * len(texts)
            _telemetry_log("Minecraft Parkour 배경 영상 사용", config)
            _ui_info("배경 영상: Minecraft Parkour (YouTube)", use_streamlit)
        else:
            fallback_video = _fetch_context_video_background(config, visual_keywords)
            if fallback_video:
                background_mode = "context_video"
                bg_video_paths = [fallback_video] * len(texts)
                placeholder = _ensure_placeholder_image(config)
                bg_image_paths = [placeholder] * len(texts)
                _telemetry_log("Minecraft 실패 → 컨텍스트 영상 대체 사용", config)
            else:
                background_mode = "image"
                _notify("⚠️", "배경 영상 실패", "Minecraft 다운로드 실패, 이미지로 대체")
                _telemetry_log("Minecraft 다운로드 실패 → 이미지 모드 전환", config)
    if background_mode == "image":
        bg_video_paths = [None] * len(texts)
        if config.pixabay_api_key or config.pexels_api_key:
            bg_image_paths = fetch_segment_images(config, visual_keywords)
            _telemetry_log("키워드 이미지 수집 완료", config)
            placeholder = _ensure_placeholder_image(config)
            if bg_image_paths and all((not p) or p == placeholder for p in bg_image_paths):
                fallback_video = _fetch_context_video_background(config, visual_keywords)
                if fallback_video:
                    background_mode = "context_video"
                    bg_video_paths = [fallback_video] * len(texts)
                    bg_image_paths = [placeholder] * len(texts)
                    _telemetry_log("이미지 결과 부족 → 컨텍스트 영상 대체", config)
        else:
            placeholder = _ensure_placeholder_image(config)
            bg_image_paths = [placeholder] * len(texts)
    bg_video_paths, bg_image_paths = _apply_primary_photo_override(config, bg_video_paths, bg_image_paths)
    video_count = len([p for p in bg_video_paths if p])
    image_count = len([p for p in bg_image_paths if p])
    _telemetry_log(f"배경 적용 요약: mode={background_mode}, video_segments={video_count}, image_segments={image_count}", config)
    bg_video_paths, bg_image_paths = _apply_majisho_interlude_assets(
        config,
        roles,
        bg_video_paths,
        bg_image_paths,
    )

    # 에셋 스티커는 사용하지 않음 (배경 이미지/영상 중심)
    placeholder = _ensure_placeholder_image(config)
    assets: List[str] = [placeholder] * len(texts)

    # ── YouTube 설명 텍스트 ───────────────────────────────
    description = _build_upload_caption_text(
        title=video_title,
        hashtags=hashtags,
        hook_text=texts[0] if texts else "",
    )

    _telemetry_log("초안 단계 완료 (승인은 업로드 직전에만 수행)", config)

    # ── TTS 생성 ─────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
    voice_id = pick_voice_id(
        config.openai_tts_voices,
        config.openai_tts_voice_preference,
        force_cute=bool(getattr(config, "tts_force_cute_voice", False) or getattr(config, "tts_baby_voice", False)),
    )
    _telemetry_log("TTS 생성 시작", config)
    _status_update(progress, status_box, 0.50, "TTS 생성 중")
    _notify("🎤", "제작팀 작업 시작", "OpenAI TTS야, 이걸 활기찬 목소리로 녹음해줘")
    try:
        tts_generate(
            config,
            "。".join(texts),
            audio_path,
            voice=voice_id,
            segments=_build_tts_segments(texts, roles),
        )
        _telemetry_log("TTS 생성 완료", config)
        if AudioFileClip:
            try:
                aud = AudioFileClip(audio_path)
                dur = int(aud.duration or 0)
                aud.close()
                _notify("🎤", "제작팀 완료", f"{dur}초 분량 음성 파일 저장 완료")
            except Exception:
                _notify("🎤", "제작팀 완료", "음성 파일 저장 완료")
        else:
            _notify("🎤", "제작팀 완료", "음성 파일 저장 완료")
    except Exception as tts_err:
        err_msg = f"❌ TTS 생성 실패: {tts_err}"
        _telemetry_log(err_msg, config)
        _ui_error(err_msg, use_streamlit)
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_msg)
        _notify("❌", "제작팀 실패", str(tts_err))
        _reset_runtime_caches(config)
        return False

    # ── 영상 렌더링 ───────────────────────────────────────
    _telemetry_log("영상 렌더링 시작", config)
    _status_update(progress, status_box, 0.65, "영상 렌더링 중")
    output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
    ass_enabled = bool(getattr(config, "use_ass_subtitles", True) and PYSUBS2_AVAILABLE)
    _notify("🎬", "영상팀 작업 시작", "자막/배경/오디오 합성 중")
    try:
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
            bgm_path=bgm_path,
            bgm_volume=config.bgm_volume,
            bg_video_paths=bg_video_paths,
            bg_image_paths=bg_image_paths,
            caption_styles=caption_styles,
            overlay_mode="off",
            caption_texts=caption_texts,
            draw_subtitles=not ass_enabled,
        )
        if ass_enabled:
            ass_path = os.path.join(config.output_dir, f"subs_{now}.ass")
            ass_output = os.path.join(config.output_dir, f"shorts_ass_{now}.mp4")
            if _render_ass_subtitled_video(
                config=config,
                input_video_path=output_path,
                output_video_path=ass_output,
                ass_path=ass_path,
                texts=texts,
                caption_texts=caption_texts,
                caption_styles=caption_styles,
                tts_audio_path=audio_path,
            ):
                output_path = ass_output
                _telemetry_log("ASS 자막 렌더링 적용 완료", config)
            else:
                _telemetry_log("ASS 자막 렌더링 실패 → Pillow 자막으로 폴백", config)
                render_video(
                    config=config,
                    asset_paths=assets,
                    texts=texts,
                    tts_audio_path=audio_path,
                    output_path=output_path,
                    bgm_path=bgm_path,
                    bgm_volume=config.bgm_volume,
                    bg_video_paths=bg_video_paths,
                    bg_image_paths=bg_image_paths,
                    caption_styles=caption_styles,
                    overlay_mode="off",
                    caption_texts=caption_texts,
                    draw_subtitles=True,
                )
        _telemetry_log("영상 렌더링 완료", config)
        _notify("🎬", "영상팀 완료", f"{config.width}x{config.height} 세로 영상 저장 완료")
    except Exception as render_err:
        _telemetry_log(f"영상 렌더링 실패: {render_err}", config)
        _ui_error(f"영상 렌더링 실패: {render_err}", use_streamlit)
        _notify("❌", "영상팀 실패", str(render_err))
        _reset_runtime_caches(config)
        return False

    # ── 썸네일 생성 ───────────────────────────────────────
    thumb_path = ""
    if config.thumbnail_enabled:
        try:
            thumb_src = _pick_thumbnail_source(
                config=config,
                bg_image_paths=bg_image_paths,
                rendered_video_path=output_path,
                frame_name=f"thumb_frame_{now}.jpg",
            )
            thumb_path = os.path.join(config.output_dir, f"thumb_{now}.jpg")
            thumb_text = _pick_thumbnail_text(meta, texts)
            generate_thumbnail_image(
                config=config,
                bg_image_path=thumb_src,
                title_text=video_title,
                hook_text=thumb_text or video_title,
                output_path=thumb_path,
            )
            _telemetry_log(f"썸네일 생성 완료: {os.path.basename(thumb_path)}", config)
        except Exception as thumb_err:
            _telemetry_log(f"썸네일 생성 실패: {thumb_err}", config)

    # ── 최종 승인: 렌더링된 영상 미리보기 전송 후 업로드 여부 결정 ─────────
    if config.require_approval:
        script_summary = _build_approval_script_summary(texts, texts_ko_norm, roles)
        approval_caption = (
            "[업로드 전 최종 승인]\n"
            f"제목: {video_title}\n"
            f"무드: {mood} / BGM: {bgm_display}\n"
            f"고정댓글 JA: {pinned}\n"
            f"고정댓글 KO 직역: {pinned_ko}\n"
            f"해시태그: {' '.join(hashtags)}\n"
            f"\n{script_summary}\n\n"
            "버튼으로 업로드 여부를 선택해주세요."
        )
        _status_update(progress, status_box, 0.78, "텔레그램에 렌더링 영상 전송 중")
        _telemetry_log("업로드 전 최종 승인 요청(영상 포함) 전송", config)
        approval_msg_id = send_telegram_video_approval_request(
            config.telegram_bot_token,
            config.telegram_admin_chat_id,
            output_path,
            approval_caption,
        )
        decision = wait_for_approval(
            config,
            progress,
            status_box,
            approval_message_id=approval_msg_id,
            stage_label="업로드 승인 대기 중...",
            timeout_decision="swap",
        )
        _telemetry_log(f"최종 승인 결과: {decision}", config)
        if decision != "approve":
            hold_msg = (
                "⏸ 업로드 보류 처리됨.\n"
                f"로컬 파일: {output_path}\n"
                "필요하면 수동 업로드하거나 다시 생성하세요."
            )
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, hold_msg)
            _notify("⏸", "마케팅팀 보류", "승인 전이라 업로드를 중단합니다.")
            _status_update(progress, status_box, 1.0, "보류(업로드 안 함)")
            if use_streamlit:
                st.video(output_path)
            _reset_runtime_caches(config)
            return True
        approved_for_publish = True

    # ── 플랫폼 업로드 (Instagram 우선, YouTube/TikTok) ─────
    video_id = ""
    video_url = ""
    upload_error = ""
    upload_reason = ""
    use_instagram = (
        getattr(config, "enable_instagram_upload", False)
        and not getattr(config, "jp_youtube_only", False)
        and config.instagram_access_token
        and config.instagram_user_id
    )
    if use_instagram:
        _telemetry_log("인스타그램 릴스 업로드 시작", config)
        _status_update(progress, status_box, 0.85, "인스타그램 릴스 업로드")
        _notify("📱", "마케팅팀 작업 시작", "인스타그램 릴스 업로드 중...")
        try:
            from platforms.instagram import add_instagram_comment, upload_instagram_reel
            caption_ig = f"{video_title}\n\n{description}"
            result = upload_instagram_reel(
                access_token=config.instagram_access_token,
                ig_user_id=config.instagram_user_id,
                video_path=output_path,
                caption=caption_ig,
            )
            if result.get("success"):
                video_id = result.get("media_id", "")
                video_url = f"https://www.instagram.com/reel/{video_id}" if video_id else ""
                _telemetry_log(f"인스타그램 릴스 업로드 완료: {video_id}", config)
                _notify("📱", "인스타그램 업로드 완료", "릴스 업로드 완료")
                if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                    product_number = meta.get("product_number", "") or _pick_product_number_for_short(config)
                    if product_number:
                        comment_text = build_pinned_comment_with_voting(
                            product_number, config.linktree_url, pinned_base=pinned_bilingual
                        )
                        if add_instagram_comment(
                            config.instagram_access_token, video_id, comment_text
                        ):
                            _telemetry_log(f"인스타 고정댓글 작성 완료 (제품 {product_number})", config)
            else:
                upload_error = result.get("error", "알 수 없는 오류")
                _telemetry_log(f"인스타그램 업로드 실패: {upload_error}", config)
                send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, f"❌ 인스타 업로드 실패: {upload_error}")
        except Exception as ig_err:
            upload_error = str(ig_err)
            _telemetry_log(f"인스타그램 업로드 예외: {ig_err}", config)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, f"❌ 인스타 업로드 오류: {ig_err}")
            _notify("❌", "인스타그램 업로드 실패", str(ig_err))
    elif config.enable_youtube_upload:
        _telemetry_log("유튜브 업로드 시작", config)
        _status_update(progress, status_box, 0.85, "유튜브 업로드")
        _notify("📱", "마케팅팀 작업 시작", "유튜브 쇼츠 업로드 중...")
        result = upload_video(
            config=config,
            file_path=output_path,
            title=video_title,
            description=description,
            tags=hashtags,
        )
        upload_error = str(result.get("error", "") or "").strip()
        upload_reason = str(result.get("error_reason", "") or "").strip()
        video_id = result.get("video_id", "")
        video_url = result.get("video_url", "")
        if upload_error:
            extra_hint_msg = ""
            if upload_reason == "uploadLimitExceeded":
                extra_hint_msg = "유튜브 업로드 한도 초과(24시간 제한 가능). 잠시 후 다시 시도하거나 다른 계정으로 업로드하세요."
            err_text = f"❌ 유튜브 업로드 실패: {upload_error}"
            if extra_hint_msg:
                err_text += f"\n{extra_hint_msg}"
            _telemetry_log(err_text, config)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_text)
            _queue_pending_upload(
                config,
                file_path=output_path,
                title=video_title,
                description=description,
                tags=hashtags,
                thumb_path=thumb_path,
                error=upload_error,
            )
            _notify("❌", "유튜브 업로드 실패", upload_error)
        else:
            _telemetry_log(f"유튜브 업로드 완료: {video_id}", config)
            _notify("📱", "유튜브 업로드 완료", "업로드 완료")
            if thumb_path and video_id:
                if set_video_thumbnail(config, video_id, thumb_path):
                    _telemetry_log("썸네일 업로드 완료", config)
                else:
                    _telemetry_log("썸네일 업로드 실패", config)
            # 캡션과 별도로 고정댓글(댓글) 작성
            if video_id:
                if _post_youtube_comment_after_upload(
                    config=config,
                    video_id=video_id,
                    pinned_base=pinned_bilingual,
                    meta=meta,
                ):
                    _telemetry_log("고정댓글 작성 완료", config)
                else:
                    _telemetry_log("고정댓글 작성 실패 (youtube.force-ssl scope 필요)", config)
    else:
        _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")
        _telemetry_log("유튜브 업로드 스킵", config)

    # ── 로그 기록 ─────────────────────────────────────────
    log_row = {
        "date_jst": _get_local_now(config).strftime("%Y-%m-%d %H:%M:%S"),
        "title_ja": video_title,
        "topic_theme": video_title,
        "hashtags_ja": " ".join(hashtags),
        "mood": mood,
        "pinned_comment": pinned,
        "voice_id": voice_id,
        "video_path": output_path,
        "youtube_video_id": video_id,
        "youtube_url": video_url,
        "instagram_media_id": video_id if use_instagram else "",
        "platform": "instagram" if use_instagram else "youtube",
        "topic_key": topic_key,
        "approved_for_publish": "1" if approved_for_publish else "0",
        "caption_variant": caption_variant,
        "background_mode": background_mode,
        "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
        "status": "ok" if not upload_error else "error",
        "error": upload_error,
    }
    if approved_for_publish:
        _write_approved_content_log(
            config,
            topic_key=topic_key,
            title=video_title,
            mood=mood,
            hashtags=hashtags,
            pinned_ja=pinned,
            pinned_ko=pinned_ko,
            texts_ja=texts,
            texts_ko=texts_ko_norm,
            background_mode=background_mode,
            caption_variant=caption_variant,
            video_path=output_path,
        )
    try:
        append_publish_log(config, log_row)
    except Exception:
        pass
    _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)
    _write_local_log(
        os.path.join(config.output_dir, "ab_tests.jsonl"),
        {
            "date_jst": log_row["date_jst"],
            "title": video_title,
            "caption_variant": caption_variant,
            "background_mode": background_mode,
            "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
            "mood": mood,
            "youtube_video_id": video_id,
            "youtube_url": video_url,
            "status": log_row["status"],
        },
    )

    _status_update(progress, status_box, 1.0, "완료")
    if use_streamlit:
        st.video(output_path)

    if video_url:
        report = _build_upload_report_ko(
            title=video_title,
            video_url=video_url,
            hashtags=hashtags,
            pinned_comment_ko=pinned_ko,
            texts_ko=texts_ko_norm,
            mood=mood,
        )
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, report)
    else:
        summary_text = (
            f"[완료] 일본인 타겟 숏츠\n"
            f"제목: {video_title}\n"
            f"무드: {mood}\n"
            f"고정댓글 JA: {pinned}\n"
            f"고정댓글 KO 직역: {pinned_ko}\n"
            f"로컬: {output_path}"
        )
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)
    try:
        _refresh_video_metrics_state(config, limit=120)
    except Exception:
        pass
    _notify("📦", "정리 작업", "임시 파일 정리 및 로그 저장 완료")
    _maybe_send_ab_report(config, use_streamlit=use_streamlit)
    if _RUN_NOTIFIER:
        platforms = []
        if video_url:
            platforms.append("YouTube")
        _RUN_NOTIFIER.finish(platforms, _get_next_run_time(config))
    _reset_runtime_caches(config)
    return True


def run_streamlit_app() -> None:
    _telemetry_log("앱 시작: run_streamlit_app()", None)
    st.set_page_config(page_title="숏츠 자동화 스튜디오", layout="wide")
    config = load_config()
    _telemetry_log("앱 설정 로드 완료", config)
    ensure_dirs(
        [
            config.assets_dir,
            os.path.join(config.assets_dir, "images"),
            os.path.join(config.assets_dir, "inbox"),
            os.path.join(config.assets_dir, "bgm"),
            os.path.join(config.assets_dir, "bgm", "trending"),
            # 무드별 BGM 디렉토리 (mystery_suspense / fast_exciting)
            *[os.path.join(config.assets_dir, "bgm", mood) for mood in BGM_MOOD_CATEGORIES],
            os.path.join(config.assets_dir, "bgm", "minecraft"),
            os.path.join(config.assets_dir, "backgrounds"),
            os.path.join(config.assets_dir, "fonts"),
            os.path.join(config.assets_dir, "sfx"),
            os.path.join(config.assets_dir, "bg_videos"),
            os.path.join(config.assets_dir, "branding"),
            config.generated_bg_dir,
            os.path.dirname(config.manifest_path),
            config.output_dir,
        ]
    )

    st.sidebar.title("숏츠 자동화 스튜디오")
    st.sidebar.subheader("상태")
    use_instagram_ui = getattr(config, "enable_instagram_upload", False) and not getattr(config, "jp_youtube_only", False)
    st.sidebar.write(f"인스타그램 업로드: {'켜짐' if use_instagram_ui else '꺼짐'}")
    st.sidebar.write(f"유튜브 업로드: {'켜짐' if config.enable_youtube_upload else '꺼짐'}")
    schedule_labels = ", ".join(_get_auto_run_time_labels(config))
    st.sidebar.write(f"자동 업로드 시간({config.auto_run_tz}): {schedule_labels}")
    st.sidebar.write(f"MoviePy 사용 가능: {'예' if MOVIEPY_AVAILABLE else '아니오'}")
    st.sidebar.write(f"BGM 모드: {config.bgm_mode or 'off'}")
    if config.pixabay_api_key:
        key_tail = config.pixabay_api_key[-4:] if len(config.pixabay_api_key) >= 4 else config.pixabay_api_key
        st.sidebar.write(f"Pixabay BGM: 설정됨 (****{key_tail})")
    else:
        st.sidebar.write("Pixabay BGM: 미설정")
    st.sidebar.write(f"배경 영상 사용: {'예' if config.use_bg_videos else '아니오'}")
    st.sidebar.write(f"렌더 스레드: {config.render_threads}")
    st.sidebar.write(f"템플릿: {'켜짐' if config.use_korean_template else '꺼짐'}")
    st.sidebar.write(f"자막 길이: {config.caption_max_chars}자")
    st.sidebar.write(
        f"ASS 자막 엔진: {'켜짐' if getattr(config, 'use_ass_subtitles', True) else '꺼짐'} / "
        f"{'사용가능' if PYSUBS2_AVAILABLE else '미설치'}"
    )
    st.sidebar.write(f"썸네일: {'켜짐' if config.thumbnail_enabled else '꺼짐'}")
    if not MOVIEPY_AVAILABLE:
        st.sidebar.error(f"MoviePy 오류: {MOVIEPY_ERROR}")

    st.sidebar.subheader("필수 API/설정")
    st.sidebar.markdown(
        "- `OPENAI_API_KEY`\n"
        "- `OPENAI_TTS_VOICE` 또는 `OPENAI_TTS_VOICES`\n"
        "- `SHEET_ID`\n"
        "- `GOOGLE_SERVICE_ACCOUNT_JSON`\n"
        "- `FONT_PATH`"
    )
    st.sidebar.subheader("자동 승인(텔레그램)")
    st.sidebar.markdown(
        "- `TELEGRAM_BOT_TOKEN`\n"
        "- `TELEGRAM_ADMIN_CHAT_ID`\n"
        "- `TELEGRAM_TIMEOUT_SEC`\n"
        "- `TELEGRAM_TIMELINE_ONLY` (6시 로그만)\n"
        "- `TELEGRAM_DEBUG_LOGS` (디버그 로그)"
    )
    st.sidebar.subheader("선택")
    st.sidebar.markdown(
        "- `YOUTUBE_*` (자동 업로드)\n"
        "- `PIXABAY_API_KEY` (배경 이미지/영상)\n"
        "- `PIXABAY_BGM_ENABLED` (Pixabay BGM 자동 다운로드 on/off)\n"
        "- `PEXELS_API_KEY` (이미지 자동 수집)\n"
        "- `SERPAPI_API_KEY` (트렌드 수집)\n"
        "- `OPENAI_VISION_MODEL` (이미지 태그 분석)\n"
        "- `JA_DIALECT_STYLE` (일본어 말투/사투리 스타일)\n"
        "- `BGM_MODE`, `BGM_VOLUME` (배경음악)\n"
        "- `USE_BG_VIDEOS` (배경 영상 사용 여부)\n"
        "- `USE_GENERATED_BG_PRIORITY` (generated_bg 폴더를 배경으로 우선 사용)\n"
        "- `RENDER_THREADS` (렌더링 스레드 수)\n\n"
        "- `USE_KOREAN_TEMPLATE` (한국형 템플릿 오버레이)\n"
        "- `CAPTION_MAX_CHARS` (자막 최대 글자수)\n"
        "- `CAPTION_HOLD_RATIO` (자막 표시 비율)\n"
        "- `CAPTION_TRIM` (자막 잘라쓰기)\n\n"
        "- `BGM_FALLBACK_ENABLED` (BGM 없을 때 합성 BGM 생성)\n\n"
        "- `THUMBNAIL_ENABLED` (썸네일 자동 생성)\n"
        "- `THUMBNAIL_USE_HOOK` (썸네일에 훅 문장 사용)\n"
        "- `THUMBNAIL_MAX_CHARS` (썸네일 문장 길이)\n\n"
        "- `HIGHLIGHT_CLIP_ENABLED` (역동 구간 자동 선택)\n"
        "- `HIGHLIGHT_CLIP_DURATION_SEC` (하이라이트 길이)\n"
        "- `HIGHLIGHT_CLIP_SAMPLE_FPS` (모션 샘플링 FPS)\n\n"
        "- `AB_TEST_ENABLED` (자막/BGM A/B 기록)\n\n"
        "- `BACKGROUND_AB_EPSILON`, `BACKGROUND_AB_LOOKBACK`\n\n"
        "- `RETRY_PENDING_UPLOADS` (업로드 실패 재시도)\n"
        "- `PENDING_UPLOADS_PATH` (대기열 저장 경로)\n"
        "- `AUTO_RUN_LOCK_PATH` (중복 실행 방지)\n"
        "- `YOUTUBE_API_KEY` (A/B 리포트 통계용)\n"
        "- `AB_REPORT_ENABLED`, `AB_REPORT_HOUR`, `AB_REPORT_DAYS`\n\n"
        "- `JP_YOUTUBE_ONLY` (일본 쇼츠 유튜브 전용)\n\n"
        "- `USED_TOPICS_PATH` (중복 주제 저장 경로)\n\n"
        "**BGM 무드 폴더:** `assets/bgm/mystery_suspense/`, `assets/bgm/fast_exciting/`"
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

        st.subheader("일본인 타겟 숏츠 자동 생성 (AI 주제 자동 선정)")
        st.caption("크롤링 없이 LLM이 매번 새로운 주제를 선정합니다. 무드(mystery_suspense/fast_exciting)에 맞게 BGM을 사용합니다.")

        if _should_auto_run(config):
            _telemetry_log("자동 스케줄 실행 트리거", config)
            if _acquire_run_lock(config.auto_run_lock_path):
                try:
                    if _auto_jp_flow(config, progress, status_box, extra_hint=""):
                        _mark_auto_run_done(config)
                finally:
                    _release_run_lock(config.auto_run_lock_path)
            else:
                _telemetry_log("자동 실행 잠금으로 스킵", config)

        extra_hint = st.text_input(
            "주제 힌트 (선택)",
            placeholder="예: 부산 맛집, 서울 명소, K-뷰티 팁 — 비워두면 AI가 알아서 선정",
            help="힌트를 입력하면 LLM이 해당 방향으로 주제를 잡습니다. 비워두면 완전 랜덤.",
        )
        auto_button = st.button("자동 생성 시작", type="primary")
        if auto_button:
            _auto_jp_flow(config, progress, status_box, extra_hint=extra_hint)

        st.divider()
        st.subheader("수동 대본 생성 (힌트 입력 → AI 생성)")
        manual_hint = st.text_area("대본 힌트/아이디어 입력", height=100, placeholder="예: 한국 편의점 신상 음료 Top 5")
        generate_button = st.button("대본 생성")

        if generate_button and manual_hint:
            _status_update(progress, status_box, 0.05, "대본 생성 중")
            try:
                growth_hint = _build_growth_feedback_hint(config)
                hint = manual_hint.strip()
                if growth_hint:
                    hint = (hint + "\n\n" + growth_hint).strip()
                script = generate_script_jp(config, extra_hint=hint)
                script = _inject_majisho_asmr_beat(script, enabled=getattr(config, "enable_majisho_tag", True))
                st.session_state["script_jp"] = script
                _status_update(progress, status_box, 0.2, "대본 생성 완료")
            except Exception as exc:
                st.error(f"대본 생성 실패: {exc}")

        script = st.session_state.get("script_jp")
        if script:
            st.subheader("생성된 대본")
            _meta = script.get("meta", {})
            _content = _get_story_timeline(script)
            _mood_val = _meta.get("bgm_mood", script.get("mood", "mystery_suspense"))
            st.caption(f"무드: **{_mood_val}**")
            video_title_val = st.text_input(
                "유튜브 제목",
                value=_meta.get("title_ja", _meta.get("title", script.get("video_title", ""))),
            )
            hashtags_val = st.text_input(
                "해시태그(공백 구분)",
                value=" ".join(_meta.get("hashtags", script.get("hashtags", []))),
            )

            # 세그먼트별 JA/KO 대본 나란히 표시
            col_ja, col_ko = st.columns(2)
            _texts_ja = _script_to_beats(script)
            _texts_ko = _script_to_beats_ko(script)
            _visual_kws = _script_to_visual_keywords(script)
            with col_ja:
                st.markdown("**🇯🇵 일본어 대본**")
                body_val = st.text_area(
                    "전체 대본 (JA, 줄 구분)",
                    value="\n".join(_texts_ja),
                    height=250,
                    key="body_ja",
                )
            with col_ko:
                st.markdown("**🇰🇷 한국어 대본 (참고용)**")
                st.text_area(
                    "전체 대본 (KO)",
                    value="\n".join(_texts_ko),
                    height=250,
                    key="body_ko",
                    disabled=True,
                )

            # 세그먼트별 배경 키워드 표시
            if _visual_kws:
                with st.expander("🎬 세그먼트별 배경 키워드"):
                    for i, kw in enumerate(_visual_kws):
                        role = _content[i].get("role", "body") if i < len(_content) else "body"
                        st.text(f"[{i+1}] {role}: {kw}")

            pinned_val = st.text_input("고정 댓글", value=_meta.get("pinned_comment", script.get("pinned_comment", "")))
            product_number_val = st.text_input(
                "제품번호 (링크트리/DM용, 비워두면 자동 선택)",
                value=_meta.get("product_number", ""),
                placeholder="예: 001",
            )
            if product_number_val:
                _meta["product_number"] = product_number_val

            render_button = st.button("영상 만들기")
            if render_button:
                if missing:
                    st.error("필수 API/설정이 누락되어 있어 진행할 수 없습니다.")
                    return
                if not MOVIEPY_AVAILABLE:
                    st.error(f"MoviePy가 설치되지 않았습니다: {MOVIEPY_ERROR}")
                    return
                _reset_runtime_caches(config, deep=bool(getattr(config, "force_fresh_media_on_start", False)))
                # UI에서 편집한 텍스트로 texts 재구성
                texts = [l.strip() for l in body_val.split("\n") if l.strip()]
                if not texts:
                    st.error("렌더링할 문장이 없습니다.")
                else:
                    mood = _mood_val
                    topic_key = _pick_topic_key(_meta)
                    if topic_key and _is_used_topic(_load_used_topics(config.used_topics_path), topic_key):
                        st.warning(f"이미 사용한 주제입니다. 다른 주제를 사용하세요: {topic_key}")
                        return
                    if topic_key:
                        _mark_topic_once(config.used_topics_path, topic_key, title=video_title_val)
                    _status_update(progress, status_box, 0.15, f"BGM 매칭 중 ({mood})")
                    bgm_path = match_bgm_by_mood(config, mood)
                    if not bgm_path or not os.path.exists(bgm_path):
                        st.warning(
                            "BGM 파일을 찾지 못했습니다. "
                            "assets/bgm/mystery_suspense 또는 assets/bgm/fast_exciting 폴더에 "
                            "BGM 파일을 넣어주세요. (Pixabay 자동 BGM은 PIXABAY_BGM_ENABLED=true일 때만 시도)"
                        )
                    roles = _script_to_roles(script)
                    caption_styles = _build_caption_styles(roles, len(texts))
                    caption_variant = _select_caption_variant(config)
                    caption_styles = _apply_caption_variant(caption_styles, caption_variant)
                    texts_ko_norm = _normalize_ko_lines(_texts_ko, texts)
                    caption_texts = _build_bilingual_caption_texts(config, texts, texts_ko_norm)
                    pinned_ko = _to_ko_literal_tone(_meta.get("pinned_comment_ko", pinned_val))
                    pinned_bilingual = _compose_bilingual_text(pinned_val, pinned_ko)

                    placeholder = _ensure_placeholder_image(config)
                    assets = [placeholder] * len(texts)

                    # 배경 선택: Minecraft vs 이미지
                    bg_vids_manual: List[Optional[str]] = [None] * len(texts)
                    bg_imgs_manual: List[Optional[str]] = [None] * len(texts)
                    _kws_m = _script_to_visual_keywords(script)
                    _forced_bg_manual = _infer_background_mode_by_content(_meta, texts, _kws_m)
                    background_mode = _select_background_mode(
                        config,
                        meta=_meta,
                        texts=texts,
                        visual_keywords=_kws_m,
                    )
                    gen_bg_manual = (
                        _get_generated_bg_paths(config, len(texts), topic_key=topic_key)
                        if getattr(config, "use_generated_bg_priority", False)
                        else []
                    )
                    if gen_bg_manual:
                        background_mode = "image"
                        bg_imgs_manual = gen_bg_manual
                        _telemetry_log("수동 렌더링: 생성 배경 이미지 사용", config)
                    elif background_mode == "minecraft":
                        if _forced_bg_manual:
                            _telemetry_log("수동 렌더링: 콘텐츠 규칙으로 Minecraft 선택", config)
                        _status_update(progress, status_box, 0.25, "배경 영상 다운로드 중 (Minecraft Parkour)")
                        _bg_dir_m = os.path.join(config.assets_dir, "backgrounds")
                        _mc_m = fetch_youtube_minecraft_parkour_video(
                            _bg_dir_m,
                            config.width,
                            config.height,
                            config=config,
                            force_fresh=bool(getattr(config, "force_fresh_minecraft_download", False)),
                        )
                        if _mc_m and _is_video_readable(_mc_m):
                            bg_vids_manual = [_mc_m] * len(texts)
                            placeholder = _ensure_placeholder_image(config)
                            bg_imgs_manual = [placeholder] * len(texts)
                            _telemetry_log("수동 렌더링: Minecraft Parkour 배경 사용", config)
                        else:
                            fallback_video = _fetch_context_video_background(config, _kws_m)
                            if fallback_video:
                                background_mode = "context_video"
                                bg_vids_manual = [fallback_video] * len(texts)
                                placeholder = _ensure_placeholder_image(config)
                                bg_imgs_manual = [placeholder] * len(texts)
                                _telemetry_log("수동 렌더링: 컨텍스트 영상 대체 사용", config)
                            else:
                                background_mode = "image"
                                _ui_warning("Minecraft 다운로드 실패 — 이미지로 대체", True)
                    if background_mode == "image":
                        if config.pixabay_api_key or config.pexels_api_key:
                            bg_imgs_manual = fetch_segment_images(config, _kws_m)
                            placeholder = _ensure_placeholder_image(config)
                            if bg_imgs_manual and all((not p) or p == placeholder for p in bg_imgs_manual):
                                fallback_video = _fetch_context_video_background(config, _kws_m)
                                if fallback_video:
                                    background_mode = "context_video"
                                    bg_vids_manual = [fallback_video] * len(texts)
                                    bg_imgs_manual = [placeholder] * len(texts)
                                    _telemetry_log("수동 렌더링: 이미지 부족 → 컨텍스트 영상 대체", config)
                        else:
                            placeholder = _ensure_placeholder_image(config)
                            bg_imgs_manual = [placeholder] * len(texts)
                    bg_vids_manual, bg_imgs_manual = _apply_primary_photo_override(config, bg_vids_manual, bg_imgs_manual)
                    _telemetry_log(
                        f"수동 배경 요약: mode={background_mode}, video_segments={len([p for p in bg_vids_manual if p])}, image_segments={len([p for p in bg_imgs_manual if p])}",
                        config,
                    )
                    bg_vids_manual, bg_imgs_manual = _apply_majisho_interlude_assets(
                        config,
                        roles,
                        bg_vids_manual,
                        bg_imgs_manual,
                    )
                    _status_update(progress, status_box, 0.3, "TTS 생성")
                    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
                    voice_id = pick_voice_id(
                        config.openai_tts_voices,
                        config.openai_tts_voice_preference,
                        force_cute=bool(getattr(config, "tts_force_cute_voice", False) or getattr(config, "tts_baby_voice", False)),
                    )
                    tts_generate(
                        config,
                        "。".join(texts),
                        audio_path,
                        voice=voice_id,
                        segments=_build_tts_segments(texts, roles),
                    )
                    _status_update(progress, status_box, 0.6, "영상 렌더링")
                    output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
                    ass_enabled = bool(getattr(config, "use_ass_subtitles", True) and PYSUBS2_AVAILABLE)
                    try:
                        render_video(
                            config=config,
                            asset_paths=assets,
                            texts=texts,
                            tts_audio_path=audio_path,
                            output_path=output_path,
                            bgm_path=bgm_path,
                            bgm_volume=config.bgm_volume,
                            bg_video_paths=bg_vids_manual,
                            bg_image_paths=bg_imgs_manual,
                            caption_styles=caption_styles,
                            overlay_mode="off",
                            caption_texts=caption_texts,
                            draw_subtitles=not ass_enabled,
                        )
                        if ass_enabled:
                            ass_path = os.path.join(config.output_dir, f"subs_{now}.ass")
                            ass_output = os.path.join(config.output_dir, f"shorts_ass_{now}.mp4")
                            if _render_ass_subtitled_video(
                                config=config,
                                input_video_path=output_path,
                                output_video_path=ass_output,
                                ass_path=ass_path,
                                texts=texts,
                                caption_texts=caption_texts,
                                caption_styles=caption_styles,
                                tts_audio_path=audio_path,
                            ):
                                output_path = ass_output
                                _telemetry_log("수동 렌더링: ASS 자막 적용 완료", config)
                            else:
                                _telemetry_log("수동 렌더링: ASS 실패 → Pillow 자막 폴백", config)
                                render_video(
                                    config=config,
                                    asset_paths=assets,
                                    texts=texts,
                                    tts_audio_path=audio_path,
                                    output_path=output_path,
                                    bgm_path=bgm_path,
                                    bgm_volume=config.bgm_volume,
                                    bg_video_paths=bg_vids_manual,
                                    bg_image_paths=bg_imgs_manual,
                                    caption_styles=caption_styles,
                                    overlay_mode="off",
                                    caption_texts=caption_texts,
                                    draw_subtitles=True,
                                )
                        _telemetry_log("수동 렌더링 완료", config)
                    except Exception as render_err:
                        _telemetry_log(f"수동 렌더링 실패: {render_err}", config)
                        st.error(f"영상 렌더링 실패: {render_err}")
                        _reset_runtime_caches(config)
                        return
                    # 썸네일 생성
                    thumb_path = ""
                    if config.thumbnail_enabled:
                        try:
                            thumb_src = _pick_thumbnail_source(
                                config=config,
                                bg_image_paths=bg_imgs_manual,
                                rendered_video_path=output_path,
                                frame_name=f"thumb_frame_{now}.jpg",
                            )
                            thumb_path = os.path.join(config.output_dir, f"thumb_{now}.jpg")
                            thumb_text = _pick_thumbnail_text(_meta, texts)
                            generate_thumbnail_image(
                                config=config,
                                bg_image_path=thumb_src,
                                title_text=video_title_val,
                                hook_text=thumb_text or video_title_val,
                                output_path=thumb_path,
                            )
                        except Exception as thumb_err:
                            _telemetry_log(f"수동 썸네일 생성 실패: {thumb_err}", config)
                    should_upload = True
                    if config.require_approval:
                        script_summary = _build_approval_script_summary(texts, texts_ko_norm, roles)
                        approval_caption = (
                            "[업로드 전 최종 승인]\n"
                            f"제목: {video_title_val}\n"
                            f"무드: {mood}\n"
                            f"고정댓글 JA: {pinned_val}\n"
                            f"고정댓글 KO 직역: {pinned_ko}\n"
                            f"해시태그: {hashtags_val}\n"
                            f"\n{script_summary}\n\n"
                            "버튼으로 업로드 여부를 선택해주세요."
                        )
                        _status_update(progress, status_box, 0.78, "텔레그램에 렌더링 영상 전송 중")
                        approval_msg_id = send_telegram_video_approval_request(
                            config.telegram_bot_token,
                            config.telegram_admin_chat_id,
                            output_path,
                            approval_caption,
                        )
                        decision = wait_for_approval(
                            config,
                            progress,
                            status_box,
                            approval_message_id=approval_msg_id,
                            stage_label="업로드 승인 대기 중...",
                            timeout_decision="swap",
                        )
                        if decision != "approve":
                            should_upload = False
                            send_telegram_message(
                                config.telegram_bot_token,
                                config.telegram_admin_chat_id,
                                f"⏸ 수동 업로드 보류 처리됨.\n로컬 파일: {output_path}",
                            )
                    caption_yt = _build_upload_caption_text(
                        title=video_title_val,
                        hashtags=hashtags_val.split(),
                        hook_text=texts[0] if texts else "",
                    )
                    video_id = ""
                    video_url = ""
                    upload_error = ""
                    upload_reason = ""
                    if not should_upload:
                        upload_error = "approval_hold"
                        upload_reason = "approval_hold"
                    use_instagram = (
                        getattr(config, "enable_instagram_upload", False)
                        and not getattr(config, "jp_youtube_only", False)
                        and config.instagram_access_token
                        and config.instagram_user_id
                    )
                    if should_upload and use_instagram:
                        _status_update(progress, status_box, 0.85, "인스타그램 릴스 업로드")
                        try:
                            from platforms.instagram import add_instagram_comment, upload_instagram_reel
                            cap_ig = f"{video_title_val}\n\n{pinned_bilingual}\n\n{hashtags_val}"
                            result = upload_instagram_reel(
                                config.instagram_access_token,
                                config.instagram_user_id,
                                output_path,
                                caption=cap_ig,
                            )
                            if result.get("success"):
                                video_id = result.get("media_id", "")
                                video_url = f"https://www.instagram.com/reel/{video_id}" if video_id else ""
                                if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                                    pn = _meta.get("product_number", "") or _pick_product_number_for_short(config)
                                    if pn:
                                        comment_text = build_pinned_comment_with_voting(pn, config.linktree_url, pinned_base=pinned_bilingual)
                                        add_instagram_comment(config.instagram_access_token, video_id, comment_text)
                            else:
                                upload_error = result.get("error", "알 수 없는 오류")
                        except Exception as e:
                            upload_error = str(e)
                    elif should_upload and config.enable_youtube_upload:
                        _status_update(progress, status_box, 0.85, "유튜브 업로드")
                        result = upload_video(
                            config=config,
                            file_path=output_path,
                            title=video_title_val,
                            description=caption_yt,
                            tags=hashtags_val.split(),
                        )
                        upload_error = str(result.get("error", "") or "").strip()
                        upload_reason = str(result.get("error_reason", "") or "").strip()
                        video_id = result.get("video_id", "")
                        video_url = result.get("video_url", "")
                        if upload_error:
                            extra_hint_msg = ""
                            if upload_reason == "uploadLimitExceeded":
                                extra_hint_msg = "유튜브 업로드 한도 초과(24시간 제한 가능). 잠시 후 다시 시도하세요."
                            err_text = f"❌ 유튜브 업로드 실패: {upload_error}"
                            if extra_hint_msg:
                                err_text += f"\n{extra_hint_msg}"
                            _telemetry_log(err_text, config)
                            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_text)
                            _queue_pending_upload(
                                config,
                                file_path=output_path,
                                title=video_title_val,
                                description=caption_yt,
                                tags=hashtags_val.split(),
                                thumb_path=thumb_path,
                                error=upload_error,
                            )
                        else:
                            if thumb_path and video_id:
                                if set_video_thumbnail(config, video_id, thumb_path):
                                    _telemetry_log("수동 썸네일 업로드 완료", config)
                                else:
                                    _telemetry_log("수동 썸네일 업로드 실패", config)
                            if video_id:
                                _post_youtube_comment_after_upload(
                                    config=config,
                                    video_id=video_id,
                                    pinned_base=pinned_bilingual,
                                    meta=_meta,
                                )
                    else:
                        if not should_upload:
                            _status_update(progress, status_box, 0.85, "승인 보류(업로드 안 함)")
                            st.info("승인 보류로 업로드를 건너뛰었습니다.")
                        else:
                            _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")
                    if video_url:
                        report = _build_upload_report_ko(
                            title=video_title_val,
                            video_url=video_url,
                            hashtags=hashtags_val.split(),
                            pinned_comment_ko=pinned_ko,
                            texts_ko=texts_ko_norm,
                            mood=mood,
                        )
                        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, report)
                    log_row = {
                        "date_jst": _get_local_now(config).strftime("%Y-%m-%d %H:%M:%S"),
                        "title_ja": video_title_val,
                        "topic_theme": video_title_val,
                        "hashtags_ja": hashtags_val,
                        "mood": mood,
                        "pinned_comment": pinned_val,
                        "voice_id": voice_id,
                        "video_path": output_path,
                        "youtube_video_id": video_id,
                        "youtube_url": video_url,
                        "topic_key": topic_key,
                        "approved_for_publish": "1" if should_upload else "0",
                        "caption_variant": caption_variant,
                        "background_mode": background_mode,
                        "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
                        "status": "ok" if not upload_error else "error",
                        "error": upload_error,
                    }
                    if should_upload:
                        _write_approved_content_log(
                            config,
                            topic_key=topic_key,
                            title=video_title_val,
                            mood=mood,
                            hashtags=hashtags_val.split(),
                            pinned_ja=pinned_val,
                            pinned_ko=pinned_ko,
                            texts_ja=texts,
                            texts_ko=texts_ko_norm,
                            background_mode=background_mode,
                            caption_variant=caption_variant,
                            video_path=output_path,
                        )
                    try:
                        append_publish_log(config, log_row)
                    except Exception:
                        pass
                    _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)
                    _write_local_log(
                        os.path.join(config.output_dir, "ab_tests.jsonl"),
                        {
                            "date_jst": log_row["date_jst"],
                            "title": video_title_val,
                            "caption_variant": caption_variant,
                            "background_mode": background_mode,
                            "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
                            "mood": mood,
                            "youtube_video_id": video_id,
                            "youtube_url": video_url,
                            "status": log_row["status"],
                        },
                    )
                    try:
                        _refresh_video_metrics_state(config, limit=120)
                    except Exception:
                        pass
                    _reset_runtime_caches(config)
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

        st.subheader("BGM 업로드")
        bgm_files = st.file_uploader(
            "오디오 업로드",
            type=["mp3", "wav", "m4a", "aac", "ogg"],
            accept_multiple_files=True,
            key="bgm_upload",
        )
        mood_labels = list(BGM_MOOD_CATEGORIES.keys())
        bgm_target = st.selectbox(
            "저장 위치",
            ["일반 BGM"] + [f"무드: {m}" for m in mood_labels],
        )
        if st.button("BGM 저장") and bgm_files:
            if bgm_target.startswith("무드: "):
                mood_name = bgm_target.replace("무드: ", "")
                target_dir = os.path.join(config.assets_dir, "bgm", mood_name)
            else:
                target_dir = os.path.join(config.assets_dir, "bgm")
            os.makedirs(target_dir, exist_ok=True)
            for file in bgm_files:
                save_path = os.path.join(target_dir, file.name)
                with open(save_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
            st.success("BGM이 저장되었습니다.")

        st.subheader("Whisk/AI 배경 이미지 업로드")
        st.caption("Whisk에서 생성한 이미지나 AI 배경을 업로드하면, 대사 전환마다 자동으로 배경이 바뀝니다.")
        bg_files = st.file_uploader(
            "배경 이미지 업로드",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="bg_upload",
        )
        if st.button("배경 이미지 저장") and bg_files:
            os.makedirs(config.generated_bg_dir, exist_ok=True)
            for file in bg_files:
                save_path = os.path.join(config.generated_bg_dir, file.name)
                with open(save_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
            st.success("배경 이미지가 저장되었습니다.")
            previews = _list_image_files(config.generated_bg_dir)[:6]
            if previews:
                st.caption("미리보기 (최대 6장)")
                st.image(previews, width=120)

        st.subheader("AI 이미지 수집(SerpAPI)")
        collect_query = st.text_input("검색어")
        collect_count = st.slider("수집 개수", 4, 20, 8)
        if st.button("인박스로 수집"):
            try:
                if not collect_query.strip():
                    st.error("검색어를 입력하세요.")
                    return
                # 새 검색어로 수집 시 이전 인박스 초기화
                prev_query = st.session_state.get("inbox_source_key", "")
                if f"serpapi_{collect_query}" != prev_query:
                    _clear_inbox_unsaved(config, manifest_items)
                    st.session_state["inbox_source_key"] = f"serpapi_{collect_query}"
                    st.session_state["ai_tag_map"] = {}
                    st.session_state["ai_category_map"] = {}
                inbox_dir = os.path.join(config.assets_dir, "inbox")
                collected = collect_images_serpapi(
                    query=collect_query,
                    api_key=config.serpapi_api_key,
                    output_dir=inbox_dir,
                    limit=collect_count,
                )
                st.session_state["inbox_current_files"] = collected
                st.success(f"{len(collected)}개 이미지를 인박스에 저장했습니다.")
            except Exception as exc:
                st.error(f"수집 실패: {exc}")

        st.subheader("URL 목록으로 이미지 수집(권한 보유 필수)")
        st.caption("저작권이 있는 이미지/밈은 권한이 있을 때만 사용하세요.")
        url_text = st.text_area("이미지 URL 목록 (줄바꿈)", height=120, key="url_import")
        url_limit = st.slider("URL 최대 수집 개수", 1, 50, 20, key="url_import_limit")
        url_confirm = st.checkbox("이 URL의 이미지 사용 권한을 보유했습니다.")
        if st.button("URL로 인박스 저장"):
            if not url_confirm:
                st.error("권한 확인에 동의해야 합니다.")
            else:
                urls = [line.strip() for line in url_text.splitlines() if line.strip()]
                if not urls:
                    st.error("URL을 입력하세요.")
                else:
                    try:
                        new_url_key = f"url_{str(sorted(urls))}"
                        if new_url_key != st.session_state.get("inbox_source_key", ""):
                            _clear_inbox_unsaved(config, manifest_items)
                            st.session_state["inbox_source_key"] = new_url_key
                            st.session_state["ai_tag_map"] = {}
                            st.session_state["ai_category_map"] = {}
                        inbox_dir = os.path.join(config.assets_dir, "inbox")
                        collected = download_images_from_urls(urls, inbox_dir, limit=url_limit)
                        st.session_state["inbox_current_files"] = collected
                        st.success(f"{len(collected)}개 이미지를 인박스에 저장했습니다.")
                    except Exception as exc:
                        st.error(f"URL 수집 실패: {exc}")

        st.subheader("네이버 블로그 포스트 이미지 수집(권한 보유 필수)")
        st.caption("퍼가기가 허용된 포스트 URL만 넣어주세요.")
        post_text = st.text_area("포스트 URL 목록 (줄바꿈)", height=120, key="naver_post_import")
        post_limit = st.slider("포스트 이미지 최대 수집 개수", 1, 100, 40, key="naver_post_limit")
        post_confirm = st.checkbox("해당 포스트 이미지 사용 권한을 보유했습니다.")
        if st.button("포스트에서 인박스 저장"):
            if not post_confirm:
                st.error("권한 확인에 동의해야 합니다.")
            else:
                post_urls = [line.strip() for line in post_text.splitlines() if line.strip()]
                if not post_urls:
                    st.error("포스트 URL을 입력하세요.")
                else:
                    # 새 URL 입력 시 이전 수집 파일 초기화 (라이브러리에 저장된 건 보호)
                    new_source_key = str(sorted(post_urls))
                    prev_source_key = st.session_state.get("inbox_source_key", "")
                    if new_source_key != prev_source_key:
                        prev_files = st.session_state.get("inbox_current_files", [])
                        saved_paths = {item.path for item in load_manifest(config.manifest_path)}
                        deleted_count = 0
                        for old_file in prev_files:
                            if old_file not in saved_paths and os.path.exists(old_file):
                                try:
                                    os.remove(old_file)
                                    deleted_count += 1
                                except Exception:
                                    pass
                        if deleted_count:
                            st.info(f"이전 수집 사진 {deleted_count}개를 초기화했습니다.")
                        st.session_state["inbox_current_files"] = []
                        st.session_state["inbox_source_key"] = new_source_key
                        # AI 태그/카테고리 캐시도 초기화
                        st.session_state["ai_tag_map"] = {}
                        st.session_state["ai_category_map"] = {}
                    try:
                        inbox_dir = os.path.join(config.assets_dir, "inbox")
                        total_urls, downloaded_count = collect_images_from_post_urls(
                            post_urls=post_urls,
                            output_dir=inbox_dir,
                            limit=post_limit,
                        )
                        if total_urls == 0:
                            st.warning("이미지 URL을 찾지 못했습니다.")
                        else:
                            # 방금 수집된 파일만 session_state에 기록
                            inbox_dir_path = os.path.join(config.assets_dir, "inbox")
                            all_inbox = [
                                os.path.join(inbox_dir_path, f)
                                for f in os.listdir(inbox_dir_path)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                            ]
                            st.session_state["inbox_current_files"] = sorted(
                                all_inbox, key=os.path.getmtime, reverse=True
                            )[:downloaded_count]
                            st.success(f"발견 {total_urls}개 / 저장 {downloaded_count}개")
                    except Exception as exc:
                        st.error(f"포스트 수집 실패: {exc}")

        st.subheader("일본 트렌드 자동 수집(Pexels)")
        st.caption("SerpAPI로 일본 트렌드 키워드를 만들고, Pexels에서 이미지를 자동 수집합니다.")
        auto_count = st.slider("자동 수집 개수", 4, 30, 12, key="auto_collect_count")
        auto_queries = st.slider("검색어 개수", 2, 6, 4, key="auto_collect_queries")
        if st.button("일본 트렌드 자동 수집"):
            # 새 수집 시 이전 인박스 초기화
            _clear_inbox_unsaved(config, manifest_items)
            try:
                inbox_dir = os.path.join(config.assets_dir, "inbox")
                collected, queries = collect_images_auto_trend(
                    config=config,
                    output_dir=inbox_dir,
                    total_count=auto_count,
                    max_queries=auto_queries,
                )
                st.session_state["inbox_current_files"] = collected
                st.session_state["inbox_source_key"] = f"trend_{datetime.utcnow().isoformat()}"
                st.session_state["ai_tag_map"] = {}
                st.session_state["ai_category_map"] = {}
                st.write("사용 검색어:", ", ".join(queries))
                st.success(f"{len(collected)}개 이미지를 인박스에 저장했습니다.")
            except Exception as exc:
                st.error(f"자동 수집 실패: {exc}")

        # 인박스 표시: session_state 기반 (현재 수집분만)
        inbox_files = [
            f for f in st.session_state.get("inbox_current_files", [])
            if os.path.exists(f)
        ]
        # 하위 호환: session_state가 비어있으면 폴더 전체 표시
        if not inbox_files:
            inbox_dir = os.path.join(config.assets_dir, "inbox")
            if os.path.exists(inbox_dir):
                inbox_files = [
                    os.path.join(inbox_dir, f)
                    for f in os.listdir(inbox_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
        if inbox_files:
            st.subheader("인박스")
            inbox_tags = st.text_input("인박스 태그(쉼표 구분)")
            preset_map = {
                "충격/반전": ["shock", "wow"],
                "웃김/비꼼": ["laugh"],
                "어이없음/당황": ["awkward", "facepalm"],
                "짜증/분노": ["angry"],
                "귀여움/힐링": ["cute"],
                "전개/스토리": ["plot"],
                "엔딩/마무리": ["ending"],
            }
            preset_choices = list(preset_map.keys())
            selected_presets = st.multiselect("상황 프리셋", options=preset_choices)
            add_pepe_tag = st.checkbox("페페 기본 태그 추가", value=False)
            select_all = st.checkbox("전체 선택", value=False)
            ai_apply = st.checkbox("AI 태그 자동 적용", value=False)

            # NEW: 콘텐츠 카테고리 AI 자동 적용 옵션
            ai_category_apply = st.checkbox("AI 콘텐츠 카테고리 자동 적용 (태그로 저장)", value=False)

            selected_files: List[str] = []
            for file_path in inbox_files:
                st.image(file_path, width=200)
                if select_all or st.checkbox(f"선택: {os.path.basename(file_path)}", key=f"select_{file_path}"):
                    selected_files.append(file_path)
                ai_tag_map = st.session_state.get("ai_tag_map", {})
                if file_path in ai_tag_map:
                    st.caption(f"AI 태그: {', '.join(ai_tag_map[file_path])}")
                # NEW: 카테고리 표시
                ai_cat_map = st.session_state.get("ai_category_map", {})
                if file_path in ai_cat_map:
                    st.caption(f"AI 카테고리: {ai_cat_map[file_path]}")

            if st.button("선택한 짤 AI 태그 분석"):
                if not config.openai_api_key:
                    st.error("OPENAI_API_KEY가 필요합니다.")
                else:
                    targets = selected_files or inbox_files
                    tag_map = st.session_state.get("ai_tag_map", {})
                    cat_map = st.session_state.get("ai_category_map", {})
                    progress_bar = st.progress(0.0)
                    for index, path in enumerate(targets):
                        try:
                            tags = analyze_image_tags(config, path)
                            tag_map[path] = tags
                            # NEW: 카테고리도 함께 분석
                            cat = analyze_image_content_category(config, path)
                            cat_map[path] = cat
                        except Exception:
                            continue
                        progress_bar.progress((index + 1) / max(len(targets), 1))
                    st.session_state["ai_tag_map"] = tag_map
                    st.session_state["ai_category_map"] = cat_map
                    st.success("AI 태그 + 카테고리 분석 완료")

            if st.button("선택한 짤 저장"):
                base_tags = tags_from_text(inbox_tags)
                for preset in selected_presets:
                    base_tags.extend(preset_map.get(preset, []))
                base_tags = list(set(base_tags))
                ai_cat_map = st.session_state.get("ai_category_map", {})
                for file_path in selected_files:
                    tags = list(base_tags)
                    if add_pepe_tag:
                        tags.append("pepe")
                    if ai_apply:
                        ai_tag_map = st.session_state.get("ai_tag_map", {})
                        tags.extend(ai_tag_map.get(file_path, []))
                    # NEW: 카테고리를 태그로 저장
                    if ai_category_apply and file_path in ai_cat_map:
                        tags.append(ai_cat_map[file_path])
                    tags = list(set(tags))
                    add_asset(config.manifest_path, file_path, tags)
                # 저장된 파일은 인박스 목록에서 제거 (다음 초기화 때 삭제 대상에서 제외)
                saved_set = set(selected_files)
                current = st.session_state.get("inbox_current_files", [])
                st.session_state["inbox_current_files"] = [f for f in current if f not in saved_set]
                st.success(f"선택한 짤 {len(selected_files)}개가 라이브러리에 추가되었습니다.")
                st.rerun()

        st.subheader("라이브러리")
        if manifest_items:
            selected_tag = st.selectbox("태그 필터", options=["(전체)"] + all_tags)
            filtered = manifest_items if selected_tag == "(전체)" else filter_assets_by_tags(manifest_items, [selected_tag])
            lib_select_all = st.checkbox("라이브러리 전체 선택", value=False, key="lib_select_all")
            delete_files = st.checkbox("선택 항목 파일도 삭제", value=False, key="lib_delete_files")
            keep_tags = st.checkbox("AI 태그 적용 시 기존 태그 유지", value=True, key="lib_ai_keep")
            selected_assets: List[AssetItem] = []
            for item in filtered:
                st.image(item.path, width=200, caption=",".join(item.tags))
                if lib_select_all or st.checkbox(f"선택: {item.asset_id}", key=f"lib_select_{item.asset_id}"):
                    selected_assets.append(item)
                lib_ai_map = st.session_state.get("library_ai_tag_map", {})
                if item.asset_id in lib_ai_map:
                    st.caption(f"AI 태그: {', '.join(lib_ai_map[item.asset_id])}")
                # NEW: 라이브러리 카테고리 표시
                lib_cat_map = st.session_state.get("library_ai_cat_map", {})
                if item.asset_id in lib_cat_map:
                    st.caption(f"AI 카테고리: {lib_cat_map[item.asset_id]}")

            if st.button("선택한 에셋 AI 태그 분석"):
                if not config.openai_api_key:
                    st.error("OPENAI_API_KEY가 필요합니다.")
                else:
                    targets = selected_assets or filtered
                    tag_map = st.session_state.get("library_ai_tag_map", {})
                    cat_map = st.session_state.get("library_ai_cat_map", {})
                    progress_bar = st.progress(0.0)
                    for index, item in enumerate(targets):
                        try:
                            tags = analyze_image_tags(config, item.path)
                            tag_map[item.asset_id] = tags
                            # NEW: 카테고리도 함께 분석
                            cat = analyze_image_content_category(config, item.path)
                            cat_map[item.asset_id] = cat
                        except Exception:
                            continue
                        progress_bar.progress((index + 1) / max(len(targets), 1))
                    st.session_state["library_ai_tag_map"] = tag_map
                    st.session_state["library_ai_cat_map"] = cat_map
                    st.success("AI 태그 + 카테고리 분석 완료")

            if st.button("AI 태그로 분류 저장"):
                tag_map = st.session_state.get("library_ai_tag_map", {})
                cat_map = st.session_state.get("library_ai_cat_map", {})
                if not tag_map and not cat_map:
                    st.error("AI 태그가 없습니다. 먼저 분석하세요.")
                else:
                    ids = [item.asset_id for item in (selected_assets or filtered)]
                    # 태그 + 카테고리 합쳐서 저장
                    apply_map: Dict[str, List[str]] = {}
                    for asset_id in ids:
                        combined = list(tag_map.get(asset_id, []))
                        if asset_id in cat_map:
                            combined.append(cat_map[asset_id])
                        apply_map[asset_id] = combined
                    updated = update_asset_tags(config.manifest_path, apply_map, keep_existing=keep_tags)
                    st.success(f"{updated}개 에셋 태그+카테고리가 업데이트되었습니다.")
                    st.rerun()

            if st.button("선택한 에셋 삭제"):
                ids = [item.asset_id for item in selected_assets]
                if not ids:
                    st.error("선택된 에셋이 없습니다.")
                else:
                    removed = remove_assets(config.manifest_path, ids, delete_files=delete_files)
                    st.success(f"{removed}개 에셋이 삭제되었습니다.")
                    st.rerun()
        else:
            st.info("아직 에셋이 없습니다.")

    if page == "로그":
        st.header("로그")

        # ── 텔레그램 연결 진단 ──────────────────────────────────────
        st.subheader("텔레그램 연결 테스트")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"BOT TOKEN: `{'설정됨' if config.telegram_bot_token else '❌ 없음'}`")
            st.write(f"ADMIN CHAT ID: `{config.telegram_admin_chat_id or '❌ 없음'}`")
        with col2:
            if st.button("🔔 테스트 메시지 전송"):
                if not config.telegram_bot_token:
                    st.error("TELEGRAM_BOT_TOKEN이 없습니다.")
                elif not config.telegram_admin_chat_id:
                    st.error("TELEGRAM_ADMIN_CHAT_ID가 없습니다.")
                else:
                    ok = send_telegram_message(
                        config.telegram_bot_token,
                        config.telegram_admin_chat_id,
                        "✅ 숏츠 자동화 스튜디오 텔레그램 연결 테스트 메시지입니다.\n승인: 승인\n교환: 교환",
                    )
                    if ok:
                        st.success("전송 성공! 텔레그램에서 메시지를 확인하세요.")
                    else:
                        st.error("전송 실패! 아래 BOT 상태를 확인하세요.")

            if st.button("🤖 BOT 상태 확인"):
                if not config.telegram_bot_token:
                    st.error("TELEGRAM_BOT_TOKEN이 없습니다.")
                else:
                    try:
                        resp = requests.get(
                            f"https://api.telegram.org/bot{config.telegram_bot_token}/getMe",
                            timeout=10,
                        )
                        data = resp.json()
                        if data.get("ok"):
                            bot = data["result"]
                            st.success(f"BOT 정상: @{bot.get('username')} ({bot.get('first_name')})")
                        else:
                            st.error(f"BOT 오류: {data.get('description', '알 수 없는 오류')}")
                    except Exception as exc:
                        st.error(f"BOT 확인 실패: {exc}")

            if st.button("💬 내 CHAT ID 확인"):
                if not config.telegram_bot_token:
                    st.error("TELEGRAM_BOT_TOKEN이 없습니다.")
                else:
                    try:
                        resp = requests.get(
                            f"https://api.telegram.org/bot{config.telegram_bot_token}/getUpdates",
                            params={"limit": 5},
                            timeout=10,
                        )
                        data = resp.json()
                        updates = data.get("result", [])
                        if not updates:
                            st.warning("수신된 메시지가 없습니다.\n먼저 봇에게 아무 메시지나 보내고 다시 확인하세요.")
                        else:
                            for upd in updates:
                                chat = upd.get("message", {}).get("chat", {})
                                cid = chat.get("id")
                                cname = chat.get("first_name") or chat.get("title") or ""
                                st.info(f"CHAT ID: `{cid}`  이름: {cname}")
                    except Exception as exc:
                        st.error(f"CHAT ID 확인 실패: {exc}")

        st.divider()

        st.subheader("Pixabay BGM 디버그")
        if st.button("🎵 Pixabay BGM 테스트"):
            if not config.pixabay_api_key:
                st.error("PIXABAY_API_KEY가 없습니다.")
            elif not getattr(config, "pixabay_bgm_enabled", False):
                st.warning("PIXABAY_BGM_ENABLED=false 입니다. true로 바꾼 뒤 테스트하세요.")
            else:
                test_path = fetch_bgm_from_pixabay(
                    api_key=config.pixabay_api_key,
                    category="mystery_suspense",
                    output_dir=os.path.join(config.assets_dir, "bgm", "mystery_suspense"),
                    custom_query="suspense",
                    config=config,
                )
                if test_path and os.path.exists(test_path):
                    st.success(f"테스트 성공: {os.path.basename(test_path)}")
                    st.audio(test_path)
                else:
                    st.error("테스트 실패: Pixabay 응답이 없거나 다운로드 실패")

        bgm_logs = st.session_state.get("bgm_debug_logs", [])
        if bgm_logs:
            st.caption("최근 BGM 디버그 로그")
            st.code("\n".join(bgm_logs[-20:]))

        st.divider()

        st.subheader("최근 에러 로그")
        err_paths = [
            "/tmp/auto_shorts_error.log",
            "/tmp/auto_shorts_render_error.log",
            "/tmp/auto_shorts_llm_output.log",
        ]
        shown = False
        for ep in err_paths:
            if os.path.exists(ep):
                shown = True
                st.caption(ep)
                try:
                    with open(ep, "r", encoding="utf-8") as file:
                        st.code(file.read()[-4000:])
                except Exception as exc:
                    st.error(f"로그 읽기 실패: {exc}")
        if not shown:
            st.info("에러 로그 파일이 없습니다.")

        # ── 실행 로그 ────────────────────────────────────────────────
        local_log_path = os.path.join(config.output_dir, "runs.jsonl")
        if os.path.exists(local_log_path):
            with open(local_log_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[-50:]
            records = [json.loads(line) for line in lines]
            st.dataframe(pd.DataFrame(records))
        else:
            st.info("아직 로그가 없습니다.")


def run_batch(count: int, seed: str = "", beats: int = 7) -> None:
    config = load_config()
    growth_hint = _build_growth_feedback_hint(config)
    base_hint = (seed or "").strip()
    for index in range(count):
        _reset_runtime_caches(config, deep=bool(getattr(config, "force_fresh_media_on_start", False)))
        script = None
        topic_key = ""
        llm_hint = base_hint
        if growth_hint:
            llm_hint = (llm_hint + "\n\n" + growth_hint).strip() if llm_hint else growth_hint
        for attempt in range(3):
            script = generate_script_jp(config, extra_hint=llm_hint)
            script = _inject_majisho_asmr_beat(script, enabled=getattr(config, "enable_majisho_tag", True))
            _meta_tmp = script.get("meta", {})
            topic_key = _pick_topic_key(_meta_tmp)
            if topic_key and _is_used_topic(_load_used_topics(config.used_topics_path), topic_key):
                llm_hint = (base_hint + f"\n이전 주제는 제외: {topic_key}").strip()
                if growth_hint:
                    llm_hint = (llm_hint + "\n\n" + growth_hint).strip()
                continue
            break
        if topic_key and _is_used_topic(_load_used_topics(config.used_topics_path), topic_key):
            _reset_runtime_caches(config)
            continue
        _meta_b = script.get("meta", {})
        if topic_key:
            _mark_topic_once(
                config.used_topics_path,
                topic_key,
                title=_meta_b.get("title_ja", _meta_b.get("title", "")),
            )
        mood = _meta_b.get("bgm_mood", script.get("mood", "mystery_suspense"))
        texts = _script_to_beats(script)
        texts_ko = _script_to_beats_ko(script)
        visual_kws = _script_to_visual_keywords(script)
        roles = _script_to_roles(script)
        caption_styles = _build_caption_styles(roles, len(texts))
        caption_variant = _select_caption_variant(config)
        caption_styles = _apply_caption_variant(caption_styles, caption_variant)
        caption_texts = _build_bilingual_caption_texts(config, texts, texts_ko)
        placeholder = _ensure_placeholder_image(config)
        assets: List[str] = [placeholder] * len(texts)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}_{index}.mp3")
        voice_id = pick_voice_id(
            config.openai_tts_voices,
            config.openai_tts_voice_preference,
            force_cute=bool(getattr(config, "tts_force_cute_voice", False) or getattr(config, "tts_baby_voice", False)),
        )
        tts_generate(
            config,
            "。".join(texts),
            audio_path,
            voice=voice_id,
            segments=_build_tts_segments(texts, roles),
        )
        output_path = os.path.join(config.output_dir, f"shorts_{now}_{index}.mp4")
        bgm_path = match_bgm_by_mood(config, mood)
        # 배경 선택: Minecraft vs 이미지
        bg_vids_b: List[Optional[str]] = [None] * len(texts)
        bg_imgs_b: List[Optional[str]] = [None] * len(texts)
        background_mode = _select_background_mode(
            config,
            meta=_meta_b,
            texts=texts,
            visual_keywords=visual_kws,
        )
        gen_bg_b = (
            _get_generated_bg_paths(config, len(texts), topic_key=topic_key)
            if getattr(config, "use_generated_bg_priority", False)
            else []
        )
        if gen_bg_b:
            background_mode = "image"
            bg_imgs_b = gen_bg_b
        elif background_mode == "minecraft":
            _bg_dir = os.path.join(config.assets_dir, "backgrounds")
            _mc_b = fetch_youtube_minecraft_parkour_video(
                _bg_dir,
                config.width,
                config.height,
                config=config,
                force_fresh=bool(getattr(config, "force_fresh_minecraft_download", False)),
            )
            if _mc_b and _is_video_readable(_mc_b):
                bg_vids_b = [_mc_b] * len(texts)
                placeholder = _ensure_placeholder_image(config)
                bg_imgs_b = [placeholder] * len(texts)
            else:
                fallback_video = _fetch_context_video_background(config, visual_kws)
                if fallback_video:
                    background_mode = "context_video"
                    bg_vids_b = [fallback_video] * len(texts)
                    placeholder = _ensure_placeholder_image(config)
                    bg_imgs_b = [placeholder] * len(texts)
                else:
                    background_mode = "image"
        if background_mode == "image":
            if config.pixabay_api_key or config.pexels_api_key:
                bg_imgs_b = fetch_segment_images(config, visual_kws)
                placeholder = _ensure_placeholder_image(config)
                if bg_imgs_b and all((not p) or p == placeholder for p in bg_imgs_b):
                    fallback_video = _fetch_context_video_background(config, visual_kws)
                    if fallback_video:
                        background_mode = "context_video"
                        bg_vids_b = [fallback_video] * len(texts)
                        bg_imgs_b = [placeholder] * len(texts)
            else:
                placeholder = _ensure_placeholder_image(config)
                bg_imgs_b = [placeholder] * len(texts)
        bg_vids_b, bg_imgs_b = _apply_primary_photo_override(config, bg_vids_b, bg_imgs_b)
        _telemetry_log(
            f"배치 배경 요약: mode={background_mode}, video_segments={len([p for p in bg_vids_b if p])}, image_segments={len([p for p in bg_imgs_b if p])}",
            config,
        )
        bg_vids_b, bg_imgs_b = _apply_majisho_interlude_assets(
            config,
            roles,
            bg_vids_b,
            bg_imgs_b,
        )
        ass_enabled = bool(getattr(config, "use_ass_subtitles", True) and PYSUBS2_AVAILABLE)
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
            bgm_path=bgm_path,
            bgm_volume=config.bgm_volume,
            bg_video_paths=bg_vids_b,
            bg_image_paths=bg_imgs_b,
            caption_styles=caption_styles,
            overlay_mode="off",
            caption_texts=caption_texts,
            draw_subtitles=not ass_enabled,
        )
        if ass_enabled:
            ass_path = os.path.join(config.output_dir, f"subs_{now}_{index}.ass")
            ass_output = os.path.join(config.output_dir, f"shorts_ass_{now}_{index}.mp4")
            if _render_ass_subtitled_video(
                config=config,
                input_video_path=output_path,
                output_video_path=ass_output,
                ass_path=ass_path,
                texts=texts,
                caption_texts=caption_texts,
                caption_styles=caption_styles,
                tts_audio_path=audio_path,
            ):
                output_path = ass_output
            else:
                render_video(
                    config=config,
                    asset_paths=assets,
                    texts=texts,
                    tts_audio_path=audio_path,
                    output_path=output_path,
                    bgm_path=bgm_path,
                    bgm_volume=config.bgm_volume,
                    bg_video_paths=bg_vids_b,
                    bg_image_paths=bg_imgs_b,
                    caption_styles=caption_styles,
                    overlay_mode="off",
                    caption_texts=caption_texts,
                    draw_subtitles=True,
                )
        thumb_path = ""
        if config.thumbnail_enabled:
            try:
                thumb_src = _pick_thumbnail_source(
                    config=config,
                    bg_image_paths=bg_imgs_b,
                    rendered_video_path=output_path,
                    frame_name=f"thumb_frame_{now}_{index}.jpg",
                )
                thumb_path = os.path.join(config.output_dir, f"thumb_{now}_{index}.jpg")
                thumb_text = _pick_thumbnail_text(_meta_b, texts)
                generate_thumbnail_image(
                    config=config,
                    bg_image_path=thumb_src,
                    title_text=_meta_b.get("title_ja", _meta_b.get("title", "")),
                    hook_text=thumb_text or _meta_b.get("title_ja", ""),
                    output_path=thumb_path,
                )
            except Exception:
                thumb_path = ""
        should_upload = True
        if config.require_approval:
            script_summary = _build_approval_script_summary(texts, texts_ko, roles)
            approval_caption = (
                "[업로드 전 최종 승인]\n"
                f"제목: {_meta_b.get('title_ja', _meta_b.get('title', ''))}\n"
                f"무드: {mood}\n"
                f"\n{script_summary}\n\n"
                "배치 생성 영상입니다. 업로드하려면 승인 버튼을 눌러주세요."
            )
            approval_msg_id = send_telegram_video_approval_request(
                config.telegram_bot_token,
                config.telegram_admin_chat_id,
                output_path,
                approval_caption,
            )
            decision = wait_for_approval(
                config,
                None,
                None,
                approval_message_id=approval_msg_id,
                stage_label="업로드 승인 대기 중...",
                timeout_decision="swap",
            )
            if decision != "approve":
                should_upload = False
                send_telegram_message(
                    config.telegram_bot_token,
                    config.telegram_admin_chat_id,
                    f"⏸ 배치 업로드 보류 처리됨.\n로컬 파일: {output_path}",
                )
        video_id = ""
        video_url = ""
        _title_b = _meta_b.get("title_ja", _meta_b.get("title", script.get("video_title", "")))
        _hashtags_b = _meta_b.get("hashtags", script.get("hashtags", []))
        _pinned_b = _meta_b.get("pinned_comment", script.get("pinned_comment", ""))
        _pinned_ko_b = _to_ko_literal_tone(_meta_b.get("pinned_comment_ko", _pinned_b))
        _pinned_bilingual_b = _compose_bilingual_text(_pinned_b, _pinned_ko_b)
        _caption_b = _build_upload_caption_text(
            title=_title_b,
            hashtags=_hashtags_b,
            hook_text=texts[0] if texts else "",
        )
        upload_error = ""
        upload_reason = ""
        use_instagram = (
            getattr(config, "enable_instagram_upload", False)
            and not getattr(config, "jp_youtube_only", False)
            and config.instagram_access_token
            and config.instagram_user_id
        )
        if should_upload and use_instagram:
            try:
                from platforms.instagram import add_instagram_comment, upload_instagram_reel
                cap_b = f"{_title_b}\n\n{_pinned_bilingual_b}\n\n" + " ".join(_hashtags_b)
                result = upload_instagram_reel(
                    config.instagram_access_token,
                    config.instagram_user_id,
                    output_path,
                    caption=cap_b,
                )
                if result.get("success"):
                    video_id = result.get("media_id", "")
                    video_url = f"https://www.instagram.com/reel/{video_id}" if video_id else ""
                    if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                        pn = _meta_b.get("product_number", "") or _pick_product_number_for_short(config)
                        if pn:
                            ct = build_pinned_comment_with_voting(pn, config.linktree_url, pinned_base=_pinned_bilingual_b)
                            add_instagram_comment(config.instagram_access_token, video_id, ct)
                else:
                    upload_error = result.get("error", "알 수 없는 오류")
            except Exception as e:
                upload_error = str(e)
        elif should_upload and config.enable_youtube_upload:
            result = upload_video(
                config=config,
                file_path=output_path,
                title=_title_b,
                description=_caption_b,
                tags=_hashtags_b,
            )
            upload_error = str(result.get("error", "") or "").strip()
            upload_reason = str(result.get("error_reason", "") or "").strip()
            video_id = result.get("video_id", "")
            video_url = result.get("video_url", "")
            if upload_error:
                extra_hint_msg = ""
                if upload_reason == "uploadLimitExceeded":
                    extra_hint_msg = "유튜브 업로드 한도 초과(24시간 제한 가능)."
                err_text = f"❌ 유튜브 업로드 실패: {upload_error}"
                if extra_hint_msg:
                    err_text += f"\n{extra_hint_msg}"
                _telemetry_log(err_text, config)
                send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_text)
                _queue_pending_upload(
                    config,
                    file_path=output_path,
                    title=_title_b,
                    description=_caption_b,
                    tags=_hashtags_b,
                    thumb_path=thumb_path,
                    error=upload_error,
                )
            else:
                if thumb_path and video_id:
                    set_video_thumbnail(config, video_id, thumb_path)
                if video_id:
                    _post_youtube_comment_after_upload(
                        config=config,
                        video_id=video_id,
                        pinned_base=_pinned_bilingual_b,
                        meta=_meta_b,
                    )
        elif not should_upload:
            upload_error = "approval_hold"
            upload_reason = "approval_hold"
        log_row = {
            "date_jst": _get_local_now(config).strftime("%Y-%m-%d %H:%M:%S"),
            "title_ja": _title_b,
            "topic_theme": _title_b,
            "hashtags_ja": " ".join(_hashtags_b),
            "mood": mood,
            "pinned_comment": _pinned_b,
            "voice_id": voice_id,
            "video_path": output_path,
            "youtube_video_id": video_id,
            "youtube_url": video_url,
            "topic_key": topic_key,
            "approved_for_publish": "1" if should_upload else "0",
            "caption_variant": caption_variant,
            "background_mode": background_mode,
            "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
            "status": "ok" if not upload_error else "error",
            "error": upload_error,
        }
        if should_upload:
            _write_approved_content_log(
                config,
                topic_key=topic_key,
                title=_title_b,
                mood=mood,
                hashtags=_hashtags_b,
                pinned_ja=_pinned_b,
                pinned_ko=_pinned_ko_b,
                texts_ja=texts,
                texts_ko=texts_ko,
                background_mode=background_mode,
                caption_variant=caption_variant,
                video_path=output_path,
            )
        try:
            append_publish_log(config, log_row)
        except Exception:
            pass
        _write_local_log(os.path.join(config.output_dir, "runs.jsonl"), log_row)
        _write_local_log(
            os.path.join(config.output_dir, "ab_tests.jsonl"),
            {
                "date_jst": log_row["date_jst"],
                "title": _title_b,
                "caption_variant": caption_variant,
                "background_mode": background_mode,
                "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
                "mood": mood,
                "youtube_video_id": video_id,
                "youtube_url": video_url,
                "status": log_row["status"],
            },
        )
        try:
            _refresh_video_metrics_state(config, limit=120)
        except Exception:
            pass
        _reset_runtime_caches(config)


def _run_streamlit_app_safe() -> None:
    try:
        _telemetry_log("앱 부팅 시작", None)
        run_streamlit_app()
    except Exception:
        import traceback

        err = traceback.format_exc()
        try:
            st.error("앱 실행 중 오류가 발생했습니다. 아래 로그를 확인하세요.")
            st.code(err)
        except Exception:
            pass
        try:
            with open("/tmp/auto_shorts_error.log", "w", encoding="utf-8") as file:
                file.write(err)
        except Exception:
            pass
        try:
            _telemetry_log(f"앱 크래시:\n{err[-3500:]}", None)
        except Exception:
            pass


if os.getenv("RUN_BATCH") == "1":
    run_batch(
        count=int(os.getenv("BATCH_COUNT", "2")),
        seed=os.getenv("BATCH_SEED", "일본어 밈 숏츠"),
        beats=int(os.getenv("BATCH_BEATS", "7")),
    )
else:
    _run_streamlit_app_safe()
