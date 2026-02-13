from __future__ import annotations

import json
import ast
import base64
import mimetypes
import os
import random
import re
import textwrap
import time
from html import unescape
from urllib.parse import urljoin, urlencode, urlparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
from googleapiclient.errors import HttpError, ResumableUploadError

MOVIEPY_AVAILABLE = True
MOVIEPY_ERROR = ""
try:
    import numpy as np
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
    auto_run_tz: str
    auto_run_state_path: str
    auto_run_lock_path: str
    used_topics_path: str
    generated_bg_dir: str
    telegram_bot_token: str
    telegram_admin_chat_id: str
    telegram_timeout_sec: int
    telegram_offset_path: str
    telegram_debug_logs: bool
    telegram_timeline_only: bool
    approve_keywords: List[str]
    swap_keywords: List[str]
    pixabay_api_key: str
    use_bg_videos: bool
    render_threads: int
    use_korean_template: bool
    caption_max_chars: int
    caption_hold_ratio: float
    caption_trim: bool
    thumbnail_enabled: bool
    thumbnail_use_hook: bool
    thumbnail_max_chars: int
    use_minecraft_parkour_bg: bool
    tts_provider: str
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
    pixabay_api_key = (_get_secret("PIXABAY_API_KEY", "") or "").strip()
    pexels_api_key = (_get_secret("PEXELS_API_KEY", "") or "").strip()
    return AppConfig(
        openai_api_key=_get_secret("OPENAI_API_KEY", "") or "",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
        openai_vision_model=_get_secret("OPENAI_VISION_MODEL", "") or "",
        openai_tts_voices=_get_list("OPENAI_TTS_VOICES")
        or ([v for v in [_get_secret("OPENAI_TTS_VOICE", "alloy")] if v]),
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
        ja_dialect_style=_get_secret("JA_DIALECT_STYLE", "") or "",
        bgm_mode=_get_secret("BGM_MODE", "auto") or "auto",  # 기본값 auto: BGM 자동 선택
        bgm_volume=float(_get_secret("BGM_VOLUME", "0.2") or 0.2),  # 일본 쇼츠: TTS 가독성 위해 20%
        bgm_fallback_enabled=_get_bool("BGM_FALLBACK_ENABLED", True),
        asset_overlay_mode=_get_secret("ASSET_OVERLAY_MODE", "off") or "off",
        max_video_duration_sec=float(_get_secret("MAX_VIDEO_DURATION_SEC", "59") or 59),
        require_approval=_get_bool("REQUIRE_APPROVAL", False),
        auto_run_daily=_get_bool("AUTO_RUN_DAILY", True),
        auto_run_hour=int(_get_secret("AUTO_RUN_HOUR", "18") or 18),
        auto_run_tz=_get_secret("AUTO_RUN_TZ", "Asia/Seoul") or "Asia/Seoul",
        auto_run_state_path=auto_run_state_path,
        auto_run_lock_path=auto_run_lock_path,
        used_topics_path=used_topics_path,
        generated_bg_dir=_get_secret("GENERATED_BG_DIR", "data/assets/generated_bg") or "data/assets/generated_bg",
        telegram_bot_token=_get_secret("TELEGRAM_BOT_TOKEN", "") or "",
        telegram_admin_chat_id=_get_secret("TELEGRAM_ADMIN_CHAT_ID", "") or "",
        telegram_timeout_sec=int(_get_secret("TELEGRAM_TIMEOUT_SEC", "600") or 600),
        telegram_offset_path=telegram_offset_path,
        telegram_debug_logs=_get_bool("TELEGRAM_DEBUG_LOGS", False),
        telegram_timeline_only=_get_bool("TELEGRAM_TIMELINE_ONLY", True),
        approve_keywords=_get_list("APPROVE_KEYWORDS") or ["승인", "approve", "ok", "yes"],
        swap_keywords=_get_list("SWAP_KEYWORDS") or ["교환", "swap", "change", "next"],
        pixabay_api_key=pixabay_api_key,
        use_bg_videos=_get_bool("USE_BG_VIDEOS", True),  # 기본값 True: 배경영상 항상 활성화
        render_threads=int(_get_secret("RENDER_THREADS", "2") or 2),
        use_korean_template=_get_bool("USE_KOREAN_TEMPLATE", True),
        caption_max_chars=int(_get_secret("CAPTION_MAX_CHARS", "18") or 18),
        caption_hold_ratio=float(_get_secret("CAPTION_HOLD_RATIO", "1.0") or 1.0),
        caption_trim=_get_bool("CAPTION_TRIM", False),
        thumbnail_enabled=_get_bool("THUMBNAIL_ENABLED", True),
        thumbnail_use_hook=_get_bool("THUMBNAIL_USE_HOOK", True),
        thumbnail_max_chars=int(_get_secret("THUMBNAIL_MAX_CHARS", "22") or 22),
        use_minecraft_parkour_bg=_get_bool("USE_MINECRAFT_PARKOUR_BG", True),
        tts_provider=(_get_secret("TTS_PROVIDER", "") or "openai").strip().lower() or "openai",
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


def _check_story_quality(items: List[Dict[str, Any]]) -> List[str]:
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
    if not re.search(r"(衝撃|ヤバ|マジ|嘘|閲覧注意|知らない|やべ|危険|99%)", hook):
        issues.append("Hook 임팩트 부족")
    # Problem: 裏側/問題系
    if not re.search(r"(問題|裏|秘密|闇|危険|違和感|炎上|黒歴史)", problem):
        issues.append("Problem 내용 약함")
    # Failure: 실패/비판
    if not re.search(r"(失敗|炎上|批判|最悪|叩かれ|伸びなかった|滑った)", failure):
        issues.append("Failure 내용 약함")
    # Success: 반전/성공
    if not re.search(r"(でも|ところが|実は|まさか|逆転|成功|大ヒット)", success):
        issues.append("Success 반전 부족")
    # Point: 핵심
    if not re.search(r"(ポイント|理由|核心|決め手)", point):
        issues.append("Point 문장 약함")
    # Reaction: 1인칭
    if not re.search(r"(俺|私|やば|マジ|草|w|笑)", reaction):
        issues.append("Reaction 1인칭/감정 부족")

    return issues


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
        # Pixabay music API endpoint
        music_params = {
            "key": api_key,
            "q": query,
            "per_page": 10,
            "safesearch": "true",
        }
        music_response = requests.get(
            "https://pixabay.com/api/music/",
            params=music_params,
            timeout=30,
        )
        if music_response.status_code != 200:
            _bgm_log(f"Pixabay BGM API 실패: status={music_response.status_code}")
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
    "コカ・コーラ初期レシピにコカイン成分が含まれていた時代",
    "マクドナルド兄弟とレイ・クロックの契約裏切り",
    "タイレノール青酸事件とブランド危機管理",
    "DBクーパー航空機ハイジャック未解決事件",
    "アポロ13号 酸素タンク爆発と奇跡の帰還",
    "フォルクスワーゲン ディーゼルゲート不正",
    "ペプシチャレンジのブラインドテスト反転結果",
    "エベレスト1996年遭難と判断ミス",
    "ボストン糖蜜大洪水(1919)の意外な惨事",
    "メアリー・セレスト号の幽霊船ミステリー",
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

[ストーリー構成: 6段階]
1) Hook: 目を止める挑発。例「まだこれ使ってるの？」
2) Problem: 裏側の問題／違和感を提示。
3) Failure: 最初は失敗・炎上・批判。
4) Success: でも/ところが/実は…で逆転。
5) Point: 成功の核心を一言で言い切る。
6) Reaction: 1人称の独り言で締める。

[注意]
- 句読点少なめ・テンポ重視
- 長文禁止、1行を短く
- script_jaは必ず日本語。script_koは確認用の韓国語訳

[JSON出力フォーマット]
{
  "meta": {
    "topic_en": "テーマ英語",
    "title_ja": "日本語タイトル(釣り/衝撃系)",
    "hashtags": ["#雑学", "#裏話", "#都市伝説", "#衝撃", "#ミステリー"],
    "pinned_comment": "視聴者参加を促す日本語質問",
    "pinned_comment_ko": "上の日本語コメントの韓国語訳",
    "bgm_mood": "mystery_suspense" または "fast_exciting"
  },
  "story_timeline": [
    {"order": 1, "role": "hook", "script_ja": "日本語タメ口", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"},
    {"order": 2, "role": "problem", "script_ja": "日本語タメ口", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"},
    {"order": 3, "role": "failure", "script_ja": "日本語タメ口", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"},
    {"order": 4, "role": "success", "script_ja": "日本語タメ口", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"},
    {"order": 5, "role": "point", "script_ja": "日本語タメ口", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"},
    {"order": 6, "role": "reaction", "script_ja": "日本語タメ口の独り言", "script_ko": "韓国語訳", "visual_search_keyword": "英語検索キーワード"}
  ]
}
"""


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

    theme_pool = "\n".join(f"- {t}" for t in JP_CONTENT_THEMES)
    user_text = (
        "以下のテーマ例を参考に、日本人向けの現代ミステリー/裏話ショート台本を作成してください。\n"
        "Hook → Problem → Failure → Success → Point → Reaction の6段構成を守り、タメ口でテンポよく書いてください。\n"
        "必ずJSONのみを出力してください。\n\n"
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
        meta["pinned_comment_ko"] = meta.get("pinned_comment_ko") or meta["pinned_comment"]
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
            visual_kw = raw.get("visual_search_keyword") or raw.get("visual_keyword_en") or ""
            visual_kw = str(visual_kw).strip()
            visual_kw = _refine_visual_keyword(visual_kw, meta.get("topic_en", ""), role, script_ja)
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
        issues = _check_story_quality(normalized)
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
        return [
            item.get("visual_search_keyword")
            or item.get("visual_keyword_en")
            or "archival newspaper headline close up"
            for item in timeline
        ]
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


def _build_caption_styles(roles: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if not roles:
        return ["default"] * count
    styles: List[str] = []
    for i in range(count):
        role = roles[i] if i < len(roles) else ""
        styles.append("reaction" if role == "reaction" else "default")
    return styles


def _select_caption_variant(config: AppConfig) -> str:
    if not getattr(config, "ab_test_enabled", False):
        return "default"
    return random.choice(["default", "japanese_variety"])


def _apply_caption_variant(styles: List[str], variant: str) -> List[str]:
    if not styles:
        return styles
    if variant == "japanese_variety":
        return ["reaction" if s == "reaction" else "japanese_variety" for s in styles]
    if variant == "default":
        return ["reaction" if s == "reaction" else "default_plain" for s in styles]
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


def _select_background_mode(config: AppConfig) -> str:
    modes = ["minecraft", "image"]
    if not getattr(config, "use_minecraft_parkour_bg", True) or not getattr(config, "use_bg_videos", True):
        return "image"
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
}


def _refine_visual_keyword(keyword: str, topic_en: str, role: str, script_ja: str) -> str:
    kw = (keyword or "").strip()
    topic = (topic_en or "").strip()
    if not kw:
        kw = topic
    # 너무 일반적이거나 짧으면 주제 키워드를 강제 접두
    tokens = {t for t in re.split(r"[^a-zA-Z0-9]+", kw.lower()) if t}
    is_generic = (not tokens) or tokens.issubset(_GENERIC_VIDEO_TOKENS) or len(tokens) <= 2
    if topic and (is_generic or topic.lower() not in kw.lower()):
        kw = f"{topic} {kw}".strip()
    # 훅은 시선을 잡는 아카이브/헤드라인류 키워드 보강
    if role == "hook" and not any(
        w in kw.lower()
        for w in ("archive", "archival", "headline", "newspaper", "photo", "footage", "courtroom", "police", "factory")
    ):
        kw = f"{kw} archival photo".strip()
    # 최후 보정
    return kw or "archival newspaper headline close up"


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

    # Pixabay 자동 다운로드 (API 키 있을 때만)
    if config.pixabay_api_key:
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
        _append_bgm_debug("PIXABAY_API_KEY 미설정")
        _telemetry_log("PIXABAY_API_KEY 미설정: BGM 자동 다운로드 생략", config)

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
            return out_path
        except Exception:
            pass
    return None


ENERGETIC_VOICE_ORDER = ["nova", "alloy", "shimmer", "echo", "fable", "onyx"]


def pick_voice_id(voice_ids: List[str], preference: Optional[List[str]] = None) -> str:
    if not voice_ids:
        return ""
    pref_list = [v for v in (preference or []) if v in voice_ids]
    if pref_list:
        return random.choice(pref_list)
    energetic = [v for v in ENERGETIC_VOICE_ORDER if v in voice_ids]
    return random.choice(energetic or voice_ids)


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


def tts_generate(
    config: AppConfig,
    text: str,
    output_path: str,
    voice: str = "",
) -> str:
    """
    TTS 생성. tts_provider에 따라 OpenAI TTS 또는 gTTS 선택.
    일본어 텍스트 처리 가능.
    """
    provider = (getattr(config, "tts_provider", "") or "openai").strip().lower()
    if provider == "gtts":
        return tts_gtts(text, output_path, lang="ja")
    return tts_openai(config, text, output_path, voice=voice or "alloy")


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
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Bold.otf",
] + _SYSTEM_FONT_CANDIDATES


def _ensure_japanese_font(assets_dir: str) -> str:
    """
    ./assets/fonts/ 경로에서 일본어 폰트를 불러오거나, 없으면 Google Fonts에서
    Noto Sans JP Bold를 다운로드합니다. 일본어 자막 깨짐 방지용.
    """
    fonts_dir = os.path.join(assets_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    for name in ("NotoSansJP-Bold.otf", "NotoSansJP-Bold.ttf", "NotoSansCJKjp-Bold.otf"):
        p = os.path.join(fonts_dir, name)
        if os.path.exists(p):
            return p
    try:
        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Bold.otf"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 10000:
            path = os.path.join(fonts_dir, "NotoSansJP-Bold.otf")
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
    except Exception:
        pass
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


def _wrap_cjk_text(text: str, max_width_px: int, font_size: int) -> List[str]:
    """CJK(일본어·한국어) 문자를 픽셀 폭 기준으로 줄바꿈."""
    # 영문은 ~0.55배, CJK는 ~1배 폭 차지
    def char_w(c: str) -> float:
        return 1.0 if ord(c) > 127 else 0.55

    max_chars = max(6, int(max_width_px / (font_size * 0.95)))
    lines: List[str] = []
    current = ""
    current_w = 0.0
    for ch in text:
        cw = char_w(ch)
        if current_w + cw > max_chars:
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
    head = re.split(r"[。．.!?？！\\n、]", text, maxsplit=1)[0].strip()
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


def _apply_korean_template(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
    """한국형 쇼츠 템플릿 느낌의 오버레이 (상·하단 그라디언트 + 얇은 프레임)."""
    overlay = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grad_h = int(canvas_height * 0.18)
    # 상단 그라디언트
    for y in range(grad_h):
        alpha = int(160 * (1 - y / max(1, grad_h)))
        draw.line([(0, y), (canvas_width, y)], fill=(0, 0, 0, alpha))
    # 하단 그라디언트
    for y in range(grad_h):
        alpha = int(200 * (y / max(1, grad_h)))
        draw.line(
            [(0, canvas_height - grad_h + y), (canvas_width, canvas_height - grad_h + y)],
            fill=(0, 0, 0, alpha),
        )
    # 얇은 프레임
    frame_margin = max(6, int(canvas_width * 0.006))
    draw.rectangle(
        [frame_margin, frame_margin, canvas_width - frame_margin, canvas_height - frame_margin],
        outline=(255, 255, 255, 40),
        width=2,
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

    # 폰트 크기: 1080px 기준 78px (일본어 가독성 위해 기존 72px에서 상향)
    font_size = max(58, canvas_width // 14)
    font = _load_font(font_path, font_size)
    pad_x = int(canvas_width * 0.05)
    max_text_w = canvas_width - pad_x * 2
    lines = _wrap_cjk_text(text, max_text_w, font_size)
    line_h = font_size + 14
    total_h = line_h * len(lines) + 20
    # ── Shorts 안전 영역: 화면 55% 지점을 자막 중앙으로 ──
    # 하단 UI 안전선: canvas_height * 0.68 이하
    safe_bottom = int(canvas_height * 0.68)
    box_y = safe_bottom - total_h
    box_y = max(int(canvas_height * 0.45), box_y)  # 최소 45% 아래 유지
    # 스타일 결정 (reaction은 말풍선, japanese_variety는 일본 예능 스타일: 노란색+검정테두리)
    style_key = (style or "").strip().lower()
    is_reaction = style_key in {"reaction", "outro", "outro_loop"}
    is_japanese_variety = style_key == "japanese_variety"
    if is_reaction:
        box_fill = (255, 232, 92, 220)
        text_fill = (25, 20, 0)
        stroke_fill = (0, 0, 0)
        stroke_width = 3
    elif is_japanese_variety:
        # 일본 예능 자막: 순백+노란색 그라데이션 글씨 + 강화된 검정 테두리
        box_fill = (0, 0, 0, 0)   # 배경 박스 없음 (자막만)
        text_fill = (255, 240, 0)  # 선명한 노란색
        stroke_fill = (0, 0, 0)
        stroke_width = 10          # 기존 6 → 10으로 강화 (일본 예능 TV 스타일)
    else:
        box_fill = (0, 0, 0, 185)
        text_fill = (255, 255, 255)
        stroke_fill = (0, 0, 0)
        stroke_width = 4

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
        if hasattr(box_draw, "rounded_rectangle") and is_reaction:
            box_draw.rounded_rectangle(box_rect, radius=24, fill=box_fill)
        else:
            box_draw.rectangle(box_rect, fill=box_fill)

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
    for line in lines:
        scaled_size = max(20, int(font_size * scale))
        scaled_font = _load_font(font_path, scaled_size)
        try:
            lw = scaled_font.getbbox(line)[2]
        except Exception:
            lw = len(line) * scaled_size
        lx = max(pad_x, (canvas_width - lw) // 2)

        # japanese_variety 스타일: glow(발광) 레이어 먼저 그려서 텍스트 존재감 강화
        if is_japanese_variety and glow_alpha > 0:
            glow_size = max(20, scaled_size + 4)
            glow_font = _load_font(font_path, glow_size)
            draw.text(
                (lx - 1, y + y_bounce - 1),
                line,
                font=glow_font,
                fill=(255, 255, 150, glow_alpha),
                stroke_width=stroke_width + 4,
                stroke_fill=(255, 220, 0, glow_alpha),
            )

        draw.text(
            (lx, y + y_bounce),
            line,
            font=scaled_font,
            fill=(text_fill[0], text_fill[1], text_fill[2], int(255 * alpha_mul)),
            stroke_width=stroke_width,
            stroke_fill=(stroke_fill[0], stroke_fill[1], stroke_fill[2], int(255 * alpha_mul)),
        )
        y += line_h
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
            params = {
                "key": api_key,
                "q": q,
                "image_type": "photo",
                "orientation": "vertical",
                "per_page": 10,
                "safesearch": "true",
            }
            resp = requests.get("https://pixabay.com/api/", params=params, timeout=30)
            if resp.status_code != 200:
                return None
            hits = resp.json().get("hits", []) or []
            if not hits:
                return None
            random.shuffle(hits)
            for hit in hits[:6]:
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

    result = _try_q(query)
    if result:
        return result
    for fbq in _GLOBAL_BG_FALLBACK_QUERIES:
        result = _try_q(fbq)
        if result:
            return result
    return None


def fetch_segment_images(
    config: AppConfig,
    keywords: List[str],
) -> List[Optional[str]]:
    if not keywords:
        return []
    unique_kws = [kw for kw in dict.fromkeys(keywords) if kw]
    kw_to_path: Dict[str, Optional[str]] = {}
    for kw in unique_kws:
        path: Optional[str] = None
        if config.pixabay_api_key:
            path = fetch_pixabay_image(kw, config.pixabay_api_key, "/tmp/pixabay_images")
        if not path and config.pexels_api_key:
            path = fetch_pexels_image(kw, config.pexels_api_key, "/tmp/pexels_bg_images")
        if not path:
            for fbq in _GLOBAL_BG_FALLBACK_QUERIES:
                if config.pixabay_api_key:
                    path = fetch_pixabay_image(fbq, config.pixabay_api_key, "/tmp/pixabay_images")
                if not path and config.pexels_api_key:
                    path = fetch_pexels_image(fbq, config.pexels_api_key, "/tmp/pexels_bg_images")
                if path:
                    break
        kw_to_path[kw] = path

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
) -> Optional[str]:
    """
    YouTube에서 Minecraft Parkour No Copyright 영상을 yt-dlp로 다운로드합니다.
    - 검색어를 여러 개 순환하여 다양한 영상 확보 (세그먼트마다 다른 영상 사용 가능)
    - 1080p 이상 우선 다운로드 (쇼츠 9:16 크롭 시 화질 손실 최소화)
    - output_dir에 target_count개 이상 파일이 있으면 스킵
    - 반환: 다운로드된 파일 경로 또는 None
    """
    os.makedirs(output_dir, exist_ok=True)

    # 기존 파일 목록 확인
    existing_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith((".mp4", ".mkv", ".webm"))
        and os.path.getsize(os.path.join(output_dir, f)) > 500000  # 500KB 이상만 유효
    ]
    if len(existing_files) >= target_count:
        # 이미 충분한 파일이 있으면 랜덤 반환
        return random.choice(existing_files)
    if existing_files:
        # 일부 있으면 첫 번째 반환하되 추가 다운로드는 계속 진행
        result_path = existing_files[0]
    else:
        result_path = None

    try:
        import yt_dlp
    except ImportError:
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
        except Exception:
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
        except Exception:
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

    return result_path


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
        send_telegram_message(self.config.telegram_bot_token, self.config.telegram_admin_chat_id, msg)

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
    asset_paths: List[str],
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

        def _render_frame(get_frame, t, __cap=_cap, __asset=_asset, __font=_font, __style=_style, __overlay=_overlay_mode, __dur=dur):
            frame = get_frame(t)
            img = Image.fromarray(frame).convert("RGB")
            if _use_template:
                img = _apply_korean_template(img, W, H)
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
            bg_img_path = (
                bg_image_paths[index]
                if bg_image_paths and index < len(bg_image_paths) and bg_image_paths[index]
                else asset_path
            )
            bg_img = Image.open(bg_img_path).convert("RGB")
            bg_img = _fit_image_to_canvas(bg_img, (W, H))
            base_clip = ImageClip(np.array(bg_img)).set_duration(dur)
            clip = base_clip.fl(_render_frame)
            clip = clip.fx(vfx.resize, lambda t, d=dur: 1 + 0.02 * (t / max(d, 0.1)))

        clips.append(clip)
        vid_offset += dur

    # 캐시된 모든 배경 영상 클립 닫기
    for _v in _vid_cache.values():
        try:
            if _v:
                _v.close()
        except Exception:
            pass

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
            audio_clip.close()
        except Exception:
            pass
        try:
            video.close()
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


def _get_generated_bg_paths(config: AppConfig, count: int) -> List[Optional[str]]:
    if count <= 0:
        return []
    files = _list_image_files(config.generated_bg_dir)
    if not files:
        return []
    if len(files) >= count:
        return files[:count]
    # 부족하면 반복
    repeated: List[Optional[str]] = []
    idx = 0
    while len(repeated) < count:
        repeated.append(files[idx % len(files)])
        idx += 1
    return repeated


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
    for rec in records:
        try:
            ts = datetime.strptime(rec.get("date_jst", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = now
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


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
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
        payload = {"chat_id": chat_id, "text": chunk}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if not resp.ok:
                print(f"[Telegram 전송 실패] status={resp.status_code} body={resp.text[:300]}")
                success = False
        except Exception as exc:
            print(f"[Telegram 전송 오류] {exc}")
            success = False
    return success


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
        return send_telegram_message(token, chat_id, f"{prefix} {message}")
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
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "✅ 승인", "callback_data": "approve"},
                    {"text": "🔄 교환", "callback_data": "swap"},
                ]
            ]
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.ok:
            return str(resp.json().get("result", {}).get("message_id", ""))
        print(f"[Telegram 버튼 전송 실패] status={resp.status_code} body={resp.text[:300]}")
    except Exception as exc:
        print(f"[Telegram 버튼 전송 오류] {exc}")
    return None


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
    label = "✅ 승인됨" if result == "approve" else "🔄 교환됨"
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
        _status_update(progress, status_box, 0.25, "텔레그램 버튼 응답 대기 중...")
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
                    _answer_callback_query(config.telegram_bot_token, cb_id, "🔄 교환 처리합니다.")
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

    return "approve"


def _write_local_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_hashtags(tags: List[str]) -> str:
    return " ".join(tags)


def _normalize_ko_lines(texts_ko: List[str], texts_ja: List[str]) -> List[str]:
    if not texts_ko:
        return texts_ja
    normalized: List[str] = []
    for idx, ja in enumerate(texts_ja):
        ko = texts_ko[idx] if idx < len(texts_ko) else ""
        normalized.append(ko if ko else ja)
    return normalized


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


def _get_local_now(config: AppConfig) -> datetime:
    tz_name = (config.auto_run_tz or "Asia/Seoul").strip()
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.utcnow()


def _get_next_run_time(config: AppConfig) -> datetime:
    now = _get_local_now(config)
    target = now.replace(hour=int(config.auto_run_hour), minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


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
    if now.hour < int(config.auto_run_hour):
        return False
    state = _read_json_file(config.auto_run_state_path, {"last_run_date": ""})
    last_date = state.get("last_run_date", "")
    return last_date != now.date().isoformat()


def _mark_auto_run_done(config: AppConfig) -> None:
    now = _get_local_now(config)
    _write_json_file(config.auto_run_state_path, {"last_run_date": now.date().isoformat()})


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
    manifest_items = load_manifest(config.manifest_path)
    _notify("🚀", "AI 쇼츠 파이프라인 시작!")

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
    for attempt in range(3):
        try:
            script = generate_script_jp(config, extra_hint=extra_hint)
            _telemetry_log("대본 생성 완료", config)
        except Exception as exc:
            _telemetry_log(f"대본 생성 실패: {exc}", config)
            _ui_error(f"대본 생성 실패: {exc}", use_streamlit)
            _notify("❌", "기획팀 실패", str(exc))
            return False
        meta = script.get("meta", {})
        content_list = _get_story_timeline(script)
        topic_key = _pick_topic_key(meta)
        if topic_key:
            used_topics = _load_used_topics(config.used_topics_path)
            if _is_used_topic(used_topics, topic_key):
                _telemetry_log(f"중복 주제 감지 → 재생성: {topic_key}", config)
                extra_hint = (extra_hint + f"\n이전 주제는 제외: {topic_key}").strip()
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
            return False

    # ── 새 스키마 필드 추출 ───────────────────────────────
    meta = meta or {}
    video_title = meta.get("title_ja", meta.get("title", script.get("video_title", "ミステリーショーツ")))
    hashtags = meta.get("hashtags", script.get("hashtags", []))
    mood = meta.get("bgm_mood", script.get("mood", "mystery_suspense"))
    pinned = meta.get("pinned_comment", script.get("pinned_comment", ""))
    pinned_ko = meta.get("pinned_comment_ko", pinned)
    if topic_key:
        _notify("📝", "기획팀 완료", f"완성! 오늘 주제: {topic_key}")

    texts = _script_to_beats(script)
    texts_ko = _script_to_beats_ko(script)
    visual_keywords = _script_to_visual_keywords(script)
    roles = _script_to_roles(script)
    caption_styles = _build_caption_styles(roles, len(texts))
    caption_variant = _select_caption_variant(config)
    caption_styles = _apply_caption_variant(caption_styles, caption_variant)
    caption_texts = _get_caption_texts(config, texts)
    texts_ko_norm = _normalize_ko_lines(texts_ko, texts)

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
            "BGM 파일을 넣어주세요. (PIXABAY_API_KEY가 있으면 자동 다운로드 시도)"
            ,
            use_streamlit,
        )

    # ── 배경 선택: Minecraft 영상 vs 상황 이미지 (A/B) ─────────────
    generated_bg_paths = _get_generated_bg_paths(config, len(texts))
    unique_kws = list(dict.fromkeys(visual_keywords))
    bg_video_paths: List[Optional[str]] = []
    bg_image_paths: List[Optional[str]] = []
    background_mode = _select_background_mode(config)
    if generated_bg_paths:
        background_mode = "image"
        bg_video_paths = [None] * len(texts)
        bg_image_paths = generated_bg_paths
        _telemetry_log("생성 배경 이미지 사용", config)
    elif background_mode == "minecraft":
        backgrounds_dir = os.path.join(config.assets_dir, "backgrounds")
        mc_path = fetch_youtube_minecraft_parkour_video(backgrounds_dir, config.width, config.height)
        if mc_path:
            bg_video_paths = [mc_path] * len(texts)
            placeholder = _ensure_placeholder_image(config)
            bg_image_paths = [placeholder] * len(texts)
            _telemetry_log("Minecraft Parkour 배경 영상 사용", config)
            _ui_info("배경 영상: Minecraft Parkour (YouTube)", use_streamlit)
        else:
            background_mode = "image"
            _notify("⚠️", "배경 영상 실패", "Minecraft 다운로드 실패, 이미지로 대체")
            _telemetry_log("Minecraft 다운로드 실패 → 이미지 모드 전환", config)
    if background_mode == "image":
        bg_video_paths = [None] * len(texts)
        if config.pixabay_api_key or config.pexels_api_key:
            bg_image_paths = fetch_segment_images(config, visual_keywords)
            _telemetry_log("키워드 이미지 수집 완료", config)
        else:
            placeholder = _ensure_placeholder_image(config)
            bg_image_paths = [placeholder] * len(texts)

    # ── 에셋 선택 ─────────────────────────────────────────
    mood_to_cat = {
        "mystery_suspense": "shocking",
        "fast_exciting": "exciting",
        "mystery": "shocking",
        "suspense": "shocking",
        "exciting": "exciting",
        "informative": "humor",
        "emotional": "touching",
    }
    content_category = mood_to_cat.get(mood, "exciting")
    assets: List[str] = []
    if manifest_items:
        for _ in texts:
            asset = pick_asset_by_category(manifest_items, content_category)
            if not asset:
                asset = random.choice(manifest_items)
            assets.append(asset.path)
    else:
        # 에셋이 없어도 배경 이미지/영상만으로 렌더링 가능하도록 플레이스홀더 사용
        placeholder = _ensure_placeholder_image(config)
        assets = [placeholder] * len(texts)

    # ── YouTube 설명 텍스트 ───────────────────────────────
    description_lines = [pinned, "", " ".join(hashtags)]
    description = "\n".join(description_lines)

    # ── 텔레그램 미리보기 (한글+일본어 대본) ─────────────
    body_preview = ""
    for i, (ja_line, ko_line) in enumerate(zip(texts, texts_ko)):
        role = content_list[i].get("role", "body") if i < len(content_list) else "body"
        kw = visual_keywords[i] if i < len(visual_keywords) else ""
        body_preview += f"  [{role}] JA: {ja_line}\n          KO: {ko_line}\n          🎬 {kw}\n"

    request_text = (
        f"[ 승인 요청 ] 일본인 타겟 숏츠\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"무드: {mood}  |  BGM: {bgm_display}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"[제목 JA] {video_title}\n"
        f"[해시태그] {' '.join(hashtags)}\n\n"
        f"[대본 (총 {len(texts)}개 세그먼트)]\n{body_preview}"
        f"[고정댓글] {pinned}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"아래 버튼으로 응답해주세요."
    )

    # 대표 에셋 미리보기 전송
    if assets and os.path.exists(assets[0]):
        try:
            photo_api = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendPhoto"
            with open(assets[0], "rb") as photo_file:
                requests.post(
                    photo_api,
                    data={"chat_id": config.telegram_admin_chat_id, "caption": "대표 사진 미리보기"},
                    files={"photo": photo_file},
                    timeout=30,
                )
        except Exception:
            pass

    if config.require_approval:
        _telemetry_log("텔레그램 승인 요청 전송", config)
        _status_update(progress, status_box, 0.30, "텔레그램 승인 요청 전송")
        approval_msg_id = send_telegram_approval_request(
            config.telegram_bot_token, config.telegram_admin_chat_id, request_text
        )
        decision = wait_for_approval(config, progress, status_box, approval_message_id=approval_msg_id)
        if decision == "swap":
            _telemetry_log("승인 요청 결과: 교환", config)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, "🔄 교환 처리됨. 새 주제로 다시 생성합니다.")
            _auto_jp_flow(config, progress, status_box, extra_hint=extra_hint)
            return False
        _telemetry_log(f"승인 요청 결과: {decision}", config)
    else:
        _telemetry_log("승인 단계 생략 (자동 진행)", config)

    # ── TTS 생성 ─────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
    voice_id = pick_voice_id(config.openai_tts_voices, config.openai_tts_voice_preference)
    _telemetry_log("TTS 생성 시작", config)
    _status_update(progress, status_box, 0.50, "TTS 생성 중")
    _notify("🎤", "제작팀 작업 시작", "OpenAI TTS야, 이걸 활기찬 목소리로 녹음해줘")
    try:
        tts_generate(config, "。".join(texts), audio_path, voice=voice_id)
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
        return False

    # ── 영상 렌더링 ───────────────────────────────────────
    _telemetry_log("영상 렌더링 시작", config)
    _status_update(progress, status_box, 0.65, "영상 렌더링 중")
    output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
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
            overlay_mode=config.asset_overlay_mode,
            caption_texts=caption_texts,
        )
        _telemetry_log("영상 렌더링 완료", config)
        _notify("🎬", "영상팀 완료", f"{config.width}x{config.height} 세로 영상 저장 완료")
    except Exception as render_err:
        _telemetry_log(f"영상 렌더링 실패: {render_err}", config)
        _ui_error(f"영상 렌더링 실패: {render_err}", use_streamlit)
        _notify("❌", "영상팀 실패", str(render_err))
        return False

    # ── 썸네일 생성 ───────────────────────────────────────
    thumb_path = ""
    if config.thumbnail_enabled:
        try:
            thumb_src = next((p for p in bg_image_paths if p), None)
            if not thumb_src and assets:
                thumb_src = assets[0]
            if not thumb_src:
                thumb_src = _ensure_placeholder_image(config)
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
                            product_number, config.linktree_url, pinned_base=pinned
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
            # 고정댓글 + 투표 유도 (제품번호, 링크트리)
            if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                product_number = meta.get("product_number", "") or _pick_product_number_for_short(config)
                if product_number:
                    comment_text = build_pinned_comment_with_voting(
                        product_number, config.linktree_url, pinned_base=pinned
                    )
                    if insert_video_comment(config, video_id, comment_text):
                        _telemetry_log(f"고정댓글 작성 완료 (제품 {product_number})", config)
                    else:
                        _telemetry_log("고정댓글 작성 실패 (youtube.force-ssl scope 필요)", config)
    else:
        _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")
        _telemetry_log("유튜브 업로드 스킵", config)

    # ── 로그 기록 ─────────────────────────────────────────
    log_row = {
        "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
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
        "caption_variant": caption_variant,
        "background_mode": background_mode,
        "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
        "status": "ok" if not upload_error else "error",
        "error": upload_error,
    }
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
            f"고정댓글: {pinned}\n"
            f"로컬: {output_path}"
        )
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)
    if topic_key:
        _mark_used_topic(config.used_topics_path, topic_key, title=video_title)
    _notify("📦", "정리 작업", "임시 파일 정리 및 로그 저장 완료")
    _maybe_send_ab_report(config, use_streamlit=use_streamlit)
    if _RUN_NOTIFIER:
        platforms = []
        if video_url:
            platforms.append("YouTube")
        _RUN_NOTIFIER.finish(platforms, _get_next_run_time(config))
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
        "- `PIXABAY_API_KEY` (BGM 자동 다운로드)\n"
        "- `PEXELS_API_KEY` (이미지 자동 수집)\n"
        "- `SERPAPI_API_KEY` (트렌드 수집)\n"
        "- `OPENAI_VISION_MODEL` (이미지 태그 분석)\n"
        "- `BGM_MODE`, `BGM_VOLUME` (배경음악)\n"
        "- `USE_BG_VIDEOS` (배경 영상 사용 여부)\n"
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
                script = generate_script_jp(config, extra_hint=manual_hint)
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
                if not manifest_items:
                    st.warning("에셋이 없어도 배경 이미지로 렌더링을 진행합니다.")

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
                    _status_update(progress, status_box, 0.15, f"BGM 매칭 중 ({mood})")
                    bgm_path = match_bgm_by_mood(config, mood)
                    if not bgm_path or not os.path.exists(bgm_path):
                        st.warning(
                            "BGM 파일을 찾지 못했습니다. "
                            "assets/bgm/mystery_suspense 또는 assets/bgm/fast_exciting 폴더에 "
                            "BGM 파일을 넣어주세요. (PIXABAY_API_KEY가 있으면 자동 다운로드 시도)"
                        )
                    roles = _script_to_roles(script)
                    caption_styles = _build_caption_styles(roles, len(texts))
                    caption_variant = _select_caption_variant(config)
                    caption_styles = _apply_caption_variant(caption_styles, caption_variant)
                    caption_texts = _get_caption_texts(config, texts)
                    texts_ko_norm = _normalize_ko_lines(_texts_ko, texts)
                    pinned_ko = _meta.get("pinned_comment_ko", pinned_val)

                    mood_to_cat = {
                        "mystery_suspense": "shocking",
                        "fast_exciting": "exciting",
                        "mystery": "shocking",
                        "suspense": "shocking",
                        "exciting": "exciting",
                        "informative": "humor",
                        "emotional": "touching",
                    }
                    cat = mood_to_cat.get(mood, "exciting")
                    assets = []
                    if manifest_items:
                        for _ in texts:
                            asset = pick_asset_by_category(manifest_items, cat)
                            if not asset:
                                asset = random.choice(manifest_items)
                            assets.append(asset.path)
                    else:
                        placeholder = _ensure_placeholder_image(config)
                        assets = [placeholder] * len(texts)

                    # 배경 선택: Minecraft vs 이미지
                    bg_vids_manual: List[Optional[str]] = [None] * len(texts)
                    bg_imgs_manual: List[Optional[str]] = [None] * len(texts)
                    background_mode = _select_background_mode(config)
                    gen_bg_manual = _get_generated_bg_paths(config, len(texts))
                    if gen_bg_manual:
                        background_mode = "image"
                        bg_imgs_manual = gen_bg_manual
                        _telemetry_log("수동 렌더링: 생성 배경 이미지 사용", config)
                    elif background_mode == "minecraft":
                        _status_update(progress, status_box, 0.25, "배경 영상 다운로드 중 (Minecraft Parkour)")
                        _bg_dir_m = os.path.join(config.assets_dir, "backgrounds")
                        _mc_m = fetch_youtube_minecraft_parkour_video(_bg_dir_m, config.width, config.height)
                        if _mc_m:
                            bg_vids_manual = [_mc_m] * len(texts)
                            placeholder = _ensure_placeholder_image(config)
                            bg_imgs_manual = [placeholder] * len(texts)
                            _telemetry_log("수동 렌더링: Minecraft Parkour 배경 사용", config)
                        else:
                            background_mode = "image"
                            _ui_warning("Minecraft 다운로드 실패 — 이미지로 대체", True)
                    if background_mode == "image":
                        if config.pixabay_api_key or config.pexels_api_key:
                            _kws_m = _script_to_visual_keywords(script)
                            bg_imgs_manual = fetch_segment_images(config, _kws_m)
                        else:
                            placeholder = _ensure_placeholder_image(config)
                            bg_imgs_manual = [placeholder] * len(texts)
                    _status_update(progress, status_box, 0.3, "TTS 생성")
                    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
                    voice_id = pick_voice_id(config.openai_tts_voices, config.openai_tts_voice_preference)
                    tts_generate(config, "。".join(texts), audio_path, voice=voice_id)
                    _status_update(progress, status_box, 0.6, "영상 렌더링")
                    output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
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
                            overlay_mode=config.asset_overlay_mode,
                            caption_texts=caption_texts,
                        )
                        _telemetry_log("수동 렌더링 완료", config)
                    except Exception as render_err:
                        _telemetry_log(f"수동 렌더링 실패: {render_err}", config)
                        st.error(f"영상 렌더링 실패: {render_err}")
                        return
                    # 썸네일 생성
                    thumb_path = ""
                    if config.thumbnail_enabled:
                        try:
                            thumb_src = next((p for p in bg_imgs_manual if p), None)
                            if not thumb_src and assets:
                                thumb_src = assets[0]
                            if not thumb_src:
                                thumb_src = _ensure_placeholder_image(config)
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
                        _status_update(progress, status_box, 0.85, "인스타그램 릴스 업로드")
                        try:
                            from platforms.instagram import add_instagram_comment, upload_instagram_reel
                            cap_ig = f"{video_title_val}\n\n{pinned_val}\n\n{hashtags_val}"
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
                                        comment_text = build_pinned_comment_with_voting(pn, config.linktree_url, pinned_base=pinned_val)
                                        add_instagram_comment(config.instagram_access_token, video_id, comment_text)
                            else:
                                upload_error = result.get("error", "알 수 없는 오류")
                        except Exception as e:
                            upload_error = str(e)
                    elif config.enable_youtube_upload:
                        _status_update(progress, status_box, 0.85, "유튜브 업로드")
                        result = upload_video(
                            config=config,
                            file_path=output_path,
                            title=video_title_val,
                            description=pinned_val + "\n\n" + hashtags_val,
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
                                description=pinned_val + "\n\n" + hashtags_val,
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
                            if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                                product_number = _meta.get("product_number", "") or _pick_product_number_for_short(config)
                                if product_number:
                                    comment_text = build_pinned_comment_with_voting(
                                        product_number, config.linktree_url, pinned_base=pinned_val
                                    )
                                    insert_video_comment(config, video_id, comment_text)
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
                        "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "title_ja": video_title_val,
                        "topic_theme": video_title_val,
                        "hashtags_ja": hashtags_val,
                        "mood": mood,
                        "pinned_comment": pinned_val,
                        "voice_id": voice_id,
                        "video_path": output_path,
                        "youtube_video_id": video_id,
                        "youtube_url": video_url,
                        "caption_variant": caption_variant,
                        "background_mode": background_mode,
                        "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
                        "status": "ok" if not upload_error else "error",
                        "error": upload_error,
                    }
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
                    topic_key = _pick_topic_key(_meta)
                    if topic_key:
                        _mark_used_topic(config.used_topics_path, topic_key, title=video_title_val)
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
    manifest_items = load_manifest(config.manifest_path)
    for index in range(count):
        script = None
        topic_key = ""
        for attempt in range(3):
            script = generate_script_jp(config, extra_hint=seed)
            _meta_tmp = script.get("meta", {})
            topic_key = _pick_topic_key(_meta_tmp)
            if topic_key and _is_used_topic(_load_used_topics(config.used_topics_path), topic_key):
                continue
            break
        if topic_key and _is_used_topic(_load_used_topics(config.used_topics_path), topic_key):
            continue
        _meta_b = script.get("meta", {})
        mood = _meta_b.get("bgm_mood", script.get("mood", "mystery_suspense"))
        texts = _script_to_beats(script)
        visual_kws = _script_to_visual_keywords(script)
        roles = _script_to_roles(script)
        caption_styles = _build_caption_styles(roles, len(texts))
        caption_variant = _select_caption_variant(config)
        caption_styles = _apply_caption_variant(caption_styles, caption_variant)
        caption_texts = _get_caption_texts(config, texts)
        mood_to_cat = {
            "mystery_suspense": "shocking",
            "fast_exciting": "exciting",
            "mystery": "shocking",
            "suspense": "shocking",
            "exciting": "exciting",
            "informative": "humor",
            "emotional": "touching",
        }
        cat = mood_to_cat.get(mood, "exciting")
        assets: List[str] = []
        if manifest_items:
            for _ in texts:
                asset = pick_asset_by_category(manifest_items, cat)
                if not asset:
                    asset = random.choice(manifest_items)
                assets.append(asset.path)
        else:
            placeholder = _ensure_placeholder_image(config)
            assets = [placeholder] * len(texts)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}_{index}.mp3")
        voice_id = pick_voice_id(config.openai_tts_voices, config.openai_tts_voice_preference)
        tts_generate(config, "。".join(texts), audio_path, voice=voice_id)
        output_path = os.path.join(config.output_dir, f"shorts_{now}_{index}.mp4")
        bgm_path = match_bgm_by_mood(config, mood)
        # 배경 선택: Minecraft vs 이미지
        bg_vids_b: List[Optional[str]] = [None] * len(texts)
        bg_imgs_b: List[Optional[str]] = [None] * len(texts)
        background_mode = _select_background_mode(config)
        gen_bg_b = _get_generated_bg_paths(config, len(texts))
        if gen_bg_b:
            background_mode = "image"
            bg_imgs_b = gen_bg_b
        elif background_mode == "minecraft":
            _bg_dir = os.path.join(config.assets_dir, "backgrounds")
            _mc_b = fetch_youtube_minecraft_parkour_video(_bg_dir, config.width, config.height)
            if _mc_b:
                bg_vids_b = [_mc_b] * len(texts)
                placeholder = _ensure_placeholder_image(config)
                bg_imgs_b = [placeholder] * len(texts)
            else:
                background_mode = "image"
        if background_mode == "image":
            if config.pixabay_api_key or config.pexels_api_key:
                bg_imgs_b = fetch_segment_images(config, visual_kws)
            else:
                placeholder = _ensure_placeholder_image(config)
                bg_imgs_b = [placeholder] * len(texts)
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
            overlay_mode=config.asset_overlay_mode,
            caption_texts=caption_texts,
        )
        thumb_path = ""
        if config.thumbnail_enabled:
            try:
                thumb_src = next((p for p in bg_imgs_b if p), None)
                if not thumb_src and assets:
                    thumb_src = assets[0]
                if not thumb_src:
                    thumb_src = _ensure_placeholder_image(config)
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
        video_id = ""
        video_url = ""
        _title_b = _meta_b.get("title_ja", _meta_b.get("title", script.get("video_title", "")))
        _hashtags_b = _meta_b.get("hashtags", script.get("hashtags", []))
        _pinned_b = _meta_b.get("pinned_comment", script.get("pinned_comment", ""))
        upload_error = ""
        upload_reason = ""
        use_instagram = (
            getattr(config, "enable_instagram_upload", False)
            and not getattr(config, "jp_youtube_only", False)
            and config.instagram_access_token
            and config.instagram_user_id
        )
        if use_instagram:
            try:
                from platforms.instagram import add_instagram_comment, upload_instagram_reel
                cap_b = f"{_title_b}\n\n{_pinned_b}\n\n" + " ".join(_hashtags_b)
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
                            ct = build_pinned_comment_with_voting(pn, config.linktree_url, pinned_base=_pinned_b)
                            add_instagram_comment(config.instagram_access_token, video_id, ct)
                else:
                    upload_error = result.get("error", "알 수 없는 오류")
            except Exception as e:
                upload_error = str(e)
        elif config.enable_youtube_upload:
            result = upload_video(
                config=config,
                file_path=output_path,
                title=_title_b,
                description=_pinned_b + "\n\n" + " ".join(_hashtags_b),
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
                    description=_pinned_b + "\n\n" + " ".join(_hashtags_b),
                    tags=_hashtags_b,
                    thumb_path=thumb_path,
                    error=upload_error,
                )
            else:
                if thumb_path and video_id:
                    set_video_thumbnail(config, video_id, thumb_path)
                if getattr(config, "enable_pinned_comment", False) and config.linktree_url and video_id:
                    product_number = _meta_b.get("product_number", "") or _pick_product_number_for_short(config)
                    if product_number:
                        comment_text = build_pinned_comment_with_voting(
                            product_number, config.linktree_url, pinned_base=_pinned_b
                        )
                        insert_video_comment(config, video_id, comment_text)
        log_row = {
            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "title_ja": _title_b,
            "topic_theme": _title_b,
            "hashtags_ja": " ".join(_hashtags_b),
            "mood": mood,
            "pinned_comment": _pinned_b,
            "voice_id": voice_id,
            "video_path": output_path,
            "youtube_video_id": video_id,
            "youtube_url": video_url,
            "caption_variant": caption_variant,
            "background_mode": background_mode,
            "bgm_file": os.path.basename(bgm_path) if bgm_path else "",
            "status": "ok" if not upload_error else "error",
            "error": upload_error,
        }
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
        if topic_key:
            _mark_used_topic(config.used_topics_path, topic_key, title=_title_b)


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
