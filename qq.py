from __future__ import annotations

import json
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
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    # Pillow 10.0+에서 ANTIALIAS 제거됨 → MoviePy 내부 호환성 패치
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        concatenate_videoclips,
        vfx,
    )
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
    pixabay_api_key: str  # NEW: Pixabay Audio API


def load_config() -> AppConfig:
    assets_dir = _get_secret("ASSETS_DIR", "data/assets")
    manifest_path = _get_secret("MANIFEST_PATH", "data/manifests/assets.json")
    output_dir = _get_secret("OUTPUT_DIR", "data/output")
    return AppConfig(
        openai_api_key=_get_secret("OPENAI_API_KEY", "") or "",
        openai_model=_get_secret("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
        openai_vision_model=_get_secret("OPENAI_VISION_MODEL", "") or "",
        openai_tts_voices=_get_list("OPENAI_TTS_VOICES")
        or ([v for v in [_get_secret("OPENAI_TTS_VOICE", "alloy")] if v]),
        openai_tts_model=_get_secret("OPENAI_TTS_MODEL", "tts-1") or "tts-1",
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
        bgm_mode=_get_secret("BGM_MODE", "off") or "off",
        bgm_volume=float(_get_secret("BGM_VOLUME", "0.08") or 0.08),
        telegram_bot_token=_get_secret("TELEGRAM_BOT_TOKEN", "") or "",
        telegram_admin_chat_id=_get_secret("TELEGRAM_ADMIN_CHAT_ID", "") or "",
        telegram_timeout_sec=int(_get_secret("TELEGRAM_TIMEOUT_SEC", "600") or 600),
        telegram_offset_path=_get_secret("TELEGRAM_OFFSET_PATH", "data/state/telegram_offset.json")
        or "data/state/telegram_offset.json",
        bboom_list_url=_get_secret("BBOOM_LIST_URL", "https://m.bboom.naver.com/")
        or "https://m.bboom.naver.com/",
        bboom_max_fetch=int(_get_secret("BBOOM_MAX_FETCH", "30") or 30),
        used_links_path=_get_secret("USED_LINKS_PATH", "data/state/used_links.json")
        or "data/state/used_links.json",
        trend_query=_get_secret("TREND_QUERY", "日本 トレンド ハッシュタグ") or "日本 トレンド ハッシュタグ",
        trend_max_results=int(_get_secret("TREND_MAX_RESULTS", "8") or 8),
        approve_keywords=_get_list("APPROVE_KEYWORDS") or ["승인", "approve", "ok", "yes"],
        swap_keywords=_get_list("SWAP_KEYWORDS") or ["교환", "swap", "change", "next"],
        pixabay_api_key=_get_secret("PIXABAY_API_KEY", "") or "",  # NEW
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

def fetch_bgm_from_pixabay(
    api_key: str,
    category: str,
    output_dir: str,
    custom_query: str = "",  # AI가 생성한 BGM 검색 쿼리 (우선 사용)
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
    try:
        params = {
            "key": api_key,
            "q": query,
            "media_type": "music",
            "per_page": 10,
            "safesearch": "true",
        }
        response = requests.get(
            "https://pixabay.com/api/videos/",  # music endpoint
            params=params,
            timeout=30,
        )
        # Pixabay music API endpoint
        music_params = {
            "key": api_key,
            "q": query,
            "per_page": 10,
        }
        music_response = requests.get(
            "https://pixabay.com/api/music/",
            params=music_params,
            timeout=30,
        )
        if music_response.status_code != 200:
            return None
        hits = music_response.json().get("hits", [])
        if not hits:
            return None
        # 랜덤으로 하나 선택
        hit = random.choice(hits[:5])
        audio_url = hit.get("audio", {}).get("url") if isinstance(hit.get("audio"), dict) else hit.get("audio")
        if not audio_url:
            # 다른 필드 탐색
            audio_url = hit.get("url") or hit.get("previewURL")
        if not audio_url:
            return None
        audio_response = requests.get(audio_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        audio_response.raise_for_status()
        filename = f"pixabay_{category}_{random.randint(10000, 99999)}.mp3"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(audio_response.content)
        return file_path
    except Exception:
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
        )
        if path:
            return path
    # fallback: 기존 bgm 디렉토리
    return pick_bgm_path(config)


# ─────────────────────────────────────────────
# NEW: 한국 양산형 숏츠 대본 생성 (분위기별)
# ─────────────────────────────────────────────

KOREAN_SHORTS_STYLE_PROMPTS: Dict[str, str] = {
    "shock_twist": (
        "한국 숏츠 양산형 '충격/반전' 스타일로 작성하세요. "
        "첫 3초 훅은 '이거 실화임?' '결말이...' 식으로 궁금증을 유발하고, "
        "마지막 비트는 반전/충격 결말로 끝내세요. "
        "전체 구조: 훅 → 상황설명 → 고조 → 반전 결말"
    ),
    "empathy_touching": (
        "한국 숏츠 양산형 '공감/감동' 스타일로 작성하세요. "
        "첫 3초 훅은 '이런 사람 주변에 있으면...' '나만 이런 거 아니지?' 식 공감 유발, "
        "마지막 비트는 따뜻하거나 눈물나는 결말로 끝내세요. "
        "전체 구조: 공감훅 → 상황공감 → 감정고조 → 감동결말"
    ),
}

CATEGORY_TO_SCRIPT_STYLE: Dict[str, str] = {
    "humor":        "shock_twist",
    "shocking":     "shock_twist",
    "cringe":       "shock_twist",
    "exciting":     "shock_twist",
    "anger":        "shock_twist",
    "touching":     "empathy_touching",
    "heartwarming": "empathy_touching",
    "sad":          "empathy_touching",
}


def generate_script(
    config: AppConfig,
    seed_text: str,
    language: str = "ja",
    beats_count: int = 7,
    allowed_tags: List[str] | None = None,
    trend_context: str = "",
    dialect_style: str = "",
    content_category: str = "",
    source_type: str = "",  # "news" | "" 뉴스 소스일 때 프롬프트 강화
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

    # 카테고리에 따른 스타일 프롬프트 선택
    script_style_key = CATEGORY_TO_SCRIPT_STYLE.get(content_category, "shock_twist")
    korean_style_prompt = KOREAN_SHORTS_STYLE_PROMPTS[script_style_key]

    # 뉴스 전용 추가 지시
    news_extra = ""
    if source_type == "news":
        news_extra = (
            "SPECIAL INSTRUCTION - This is a Korean NEWS article:\n"
            "- Format title as: '韓国で話題！○○' or '韓国人が衝撃！○○' or '【韓国】○○が大炎上' style\n"
            "- Beat 1 (hook): Start with '韓国で今...' or '韓国人が...' to establish Korean context\n"
            "- Focus on the SURPRISING or EMOTIONAL core of the news\n"
            "- Include reactions/responses if mentioned in source (e.g. netizen reactions, expert opinions)\n"
            "- Last beat: Japanese viewer's perspective or call-to-action ('あなたはどう思う？' style)\n"
            "- hashtags_ja MUST include #韓国 #韓国ニュース #韓国情報 and topic-specific tags\n"
        )

    system_text = (
        "You are a short-form video scriptwriter specializing in Korean viral content adapted for Japanese audiences.\n"
        "Your job: read the Korean source content carefully and create a script that FAITHFULLY captures its actual story, joke, or reaction — do NOT invent unrelated content.\n\n"
        + (news_extra + "\n" if news_extra else "")
        + "Return ONLY valid JSON with these keys:\n"
        "  title_ko        : Korean title (punchy, short-form style)\n"
        "  title_ja        : Japanese title — for news use '韓国で話題！○○' style\n"
        "  description_ja  : Japanese description (1-2 sentences)\n"
        "  hashtags_ja     : array of 3-6 Japanese hashtags (# included)\n"
        "  bgm_query       : 1-3 English keywords for royalty-free BGM that fits the mood (e.g. 'funny quirky ukulele')\n"
        "  beats           : array of beat objects\n\n"
        "Each beat object:\n"
        "  text_ko : Korean version of the line (natural Korean short-form style, 1 punchy line)\n"
        "  text    : Japanese translation of text_ko\n"
        "  tag     : one reaction tag from the allowed list\n\n"
        "Rules:\n"
        "- beats count: decide naturally based on the content — typically 5~9 beats. "
        "  Short jokes → 5 beats. Complex stories → up to 9 beats. Never force a fixed number.\n"
        "- First beat: hook (grab attention in 3 seconds)\n"
        "- Last beat: loop-friendly ending or satisfying punchline\n"
        "- CRITICAL: Every beat must be directly based on the actual source content. "
        "  Do NOT add unrelated details or hallucinate story elements not present in the source.\n"
        "- Keep each beat short (1 line, 10-20 words in Korean)\n"
        "- No emojis in text or text_ko\n"
        "- title_ko and text_ko must be in Korean. title_ja, text, description_ja, hashtags_ja in Japanese.\n"
        "Output JSON only, no markdown."
    )
    style_line = (
        f"Korean viral style instruction: {korean_style_prompt} "
        "Translate and adapt the Korean viral style into natural Japanese. "
    )
    if dialect_style:
        style_line += (
            f"Use {dialect_style} dialect for ALL Japanese text. "
            "Keep it friendly and natural, avoid offensive stereotypes. "
        )

    user_text = (
        f"=== SOURCE CONTENT (Korean) ===\n{seed_text}\n\n"
        f"=== METADATA ===\n"
        f"Content mood category: {content_category or 'humor'}\n"
        f"Language output: {language}\n"
        f"Allowed reaction tags: {tag_list}\n"
        f"Trend context: {trend_context}\n\n"
        f"=== STYLE INSTRUCTION ===\n{style_line}\n\n"
        "IMPORTANT: Read the source content carefully. "
        "The script beats MUST reflect the actual story/joke/reaction in the source. "
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
    result["content_category"] = content_category  # 카테고리 보존
    return result


def pick_voice_id(voice_ids: List[str]) -> str:
    if not voice_ids:
        return ""
    return random.choice(voice_ids)


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


def _list_audio_files(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    items = []
    for name in os.listdir(path):
        if name.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg")):
            items.append(os.path.join(path, name))
    return items


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


# ══════════════════════════════════════════════════════════════════
#  다중 커뮤니티 소스 크롤러
# ══════════════════════════════════════════════════════════════════

def fetch_natepann_list(max_fetch: int = 30) -> List[Dict[str, str]]:
    """네이트판 실시간 베스트 글 목록 수집."""
    base = "https://pann.nate.com"
    urls_to_try = [
        "https://pann.nate.com/talk/ranking/",
        "https://pann.nate.com/talk/ranking/d",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://pann.nate.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    items: List[Dict[str, str]] = []
    seen: set = set()
    for list_url in urls_to_try:
        try:
            r = requests.get(list_url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "/talk/t/" not in href and "/talk/r/" not in href:
                    continue
                full_url = href if href.startswith("http") else urljoin(base, href)
                full_url = full_url.split("?")[0]
                if full_url in seen:
                    continue
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 3:
                    p = a.find_parent(["li", "div", "article"])
                    if p:
                        title = p.get_text(" ", strip=True)[:80]
                if not title:
                    continue
                seen.add(full_url)
                items.append({"url": full_url, "title": title, "source": "네이트판"})
                if len(items) >= max_fetch:
                    break
        except Exception as e:
            print(f"[네이트판] 수집 오류: {e}")
        if len(items) >= max_fetch:
            break
    return items


def fetch_natepann_post(url: str) -> Dict[str, str]:
    """네이트판 글 본문 + 댓글 수집."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://pann.nate.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = ""
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        # 본문
        content_area = soup.find("div", class_=lambda c: c and any(k in c for k in ("post-ct", "talk-ct", "content", "view-content")))
        if not content_area:
            content_area = soup
        SKIP = {"더보기", "닫기", "공유", "신고", "좋아요", "댓글", "팔로우", "loading"}
        blocks, seen_t = [], set()
        for tag in content_area.find_all(["p", "span", "div", "li"]):
            if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
                continue
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 5 or len(t) > 400 or any(k in t for k in SKIP) or t in seen_t:
                continue
            seen_t.add(t)
            blocks.append(t)
            if len(blocks) >= 30:
                break
        # 댓글
        cmt_blocks = []
        for area in soup.find_all(["div", "ul"], class_=lambda c: c and any(k in c for k in ("comment", "reply", "cmt"))):
            for tag in area.find_all(["p", "span", "li"]):
                t = tag.get_text(" ", strip=True)
                if t and 5 < len(t) < 150 and t not in seen_t:
                    seen_t.add(t)
                    cmt_blocks.append(t)
                if len(cmt_blocks) >= 10:
                    break
        content = "\n".join(blocks)
        if cmt_blocks:
            content += "\n\n[댓글 반응]\n" + "\n".join(cmt_blocks)
        return {"title": title, "content": content, "source": "네이트판"}
    except Exception as e:
        return {"title": "", "content": "", "source": "네이트판", "error": str(e)}


def fetch_fmkorea_list(max_fetch: int = 30) -> List[Dict[str, str]]:
    """에펨코리아 베스트 글 목록 수집."""
    list_urls = [
        "https://www.fmkorea.com/best",
        "https://www.fmkorea.com/humor",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.fmkorea.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    items: List[Dict[str, str]] = []
    seen: set = set()
    for list_url in list_urls:
        try:
            r = requests.get(list_url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if not re.search(r"/\d{7,}", href):
                    continue
                full_url = href if href.startswith("http") else urljoin("https://www.fmkorea.com", href)
                full_url = full_url.split("?")[0]
                if full_url in seen or "fmkorea.com" not in full_url:
                    continue
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 3:
                    continue
                seen.add(full_url)
                items.append({"url": full_url, "title": title, "source": "에펨코리아"})
                if len(items) >= max_fetch:
                    break
        except Exception as e:
            print(f"[에펨코리아] 수집 오류: {e}")
        if len(items) >= max_fetch:
            break
    return items


def fetch_fmkorea_post(url: str) -> Dict[str, str]:
    """에펨코리아 글 본문 수집."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.fmkorea.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = ""
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        content_area = soup.find("div", class_=lambda c: c and any(k in c for k in ("xe_content", "rd_body", "content")))
        if not content_area:
            content_area = soup
        SKIP = {"더보기", "닫기", "공유", "신고", "좋아요", "댓글", "팔로우", "loading", "광고"}
        blocks, seen_t = [], set()
        for tag in content_area.find_all(["p", "span", "div", "li"]):
            if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
                continue
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 5 or len(t) > 400 or any(k in t for k in SKIP) or t in seen_t:
                continue
            seen_t.add(t)
            blocks.append(t)
            if len(blocks) >= 30:
                break
        cmt_blocks = []
        for area in soup.find_all(["div", "ul"], class_=lambda c: c and any(k in c for k in ("comment", "reply", "cmt"))):
            for tag in area.find_all(["p", "span", "li"]):
                t = tag.get_text(" ", strip=True)
                if t and 5 < len(t) < 150 and t not in seen_t:
                    seen_t.add(t)
                    cmt_blocks.append(t)
                if len(cmt_blocks) >= 10:
                    break
        content = "\n".join(blocks)
        if cmt_blocks:
            content += "\n\n[댓글 반응]\n" + "\n".join(cmt_blocks)
        return {"title": title, "content": content, "source": "에펨코리아"}
    except Exception as e:
        return {"title": "", "content": "", "source": "에펨코리아", "error": str(e)}


def fetch_dcinside_list(max_fetch: int = 30) -> List[Dict[str, str]]:
    """DC인사이드 개념글 목록 수집."""
    list_urls = [
        "https://gall.dcinside.com/board/lists/?id=hit&exception_mode=recommend",
        "https://gall.dcinside.com/board/lists/?id=humorworld",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://gall.dcinside.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    items: List[Dict[str, str]] = []
    seen: set = set()
    for list_url in list_urls:
        try:
            r = requests.get(list_url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "no=" not in href or "lists" in href:
                    continue
                full_url = href if href.startswith("http") else urljoin("https://gall.dcinside.com", href)
                full_url = full_url.split("&page=")[0]
                if full_url in seen:
                    continue
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 3:
                    continue
                seen.add(full_url)
                items.append({"url": full_url, "title": title, "source": "DC인사이드"})
                if len(items) >= max_fetch:
                    break
        except Exception as e:
            print(f"[DC인사이드] 수집 오류: {e}")
        if len(items) >= max_fetch:
            break
    return items


def fetch_dcinside_post(url: str) -> Dict[str, str]:
    """DC인사이드 글 본문 수집."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://gall.dcinside.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = ""
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        content_area = soup.find("div", class_=lambda c: c and any(k in c for k in ("write_div", "view-content", "content")))
        if not content_area:
            content_area = soup
        SKIP = {"더보기", "닫기", "공유", "신고", "좋아요", "댓글", "팔로우", "loading", "광고", "갤로그"}
        blocks, seen_t = [], set()
        for tag in content_area.find_all(["p", "span", "div", "li"]):
            if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
                continue
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 5 or len(t) > 400 or any(k in t for k in SKIP) or t in seen_t:
                continue
            seen_t.add(t)
            blocks.append(t)
            if len(blocks) >= 30:
                break
        cmt_blocks = []
        for area in soup.find_all(["div", "ul"], class_=lambda c: c and any(k in c for k in ("comment", "reply", "cmt", "dccon"))):
            for tag in area.find_all(["p", "span", "li", "em"]):
                t = tag.get_text(" ", strip=True)
                if t and 5 < len(t) < 150 and t not in seen_t:
                    seen_t.add(t)
                    cmt_blocks.append(t)
                if len(cmt_blocks) >= 10:
                    break
        content = "\n".join(blocks)
        if cmt_blocks:
            content += "\n\n[댓글 반응]\n" + "\n".join(cmt_blocks)
        return {"title": title, "content": content, "source": "DC인사이드"}
    except Exception as e:
        return {"title": "", "content": "", "source": "DC인사이드", "error": str(e)}


def fetch_bobaedream_list(max_fetch: int = 30) -> List[Dict[str, str]]:
    """보배드림 베스트 글 목록 수집."""
    list_urls = [
        "https://www.bobaedream.co.kr/list?code=best",
        "https://www.bobaedream.co.kr/list?code=humor",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.bobaedream.co.kr/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    items: List[Dict[str, str]] = []
    seen: set = set()
    for list_url in list_urls:
        try:
            r = requests.get(list_url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "view.php" not in href and "No=" not in href:
                    continue
                full_url = href if href.startswith("http") else urljoin("https://www.bobaedream.co.kr", href)
                full_url = full_url.split("&page=")[0]
                if full_url in seen:
                    continue
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 3:
                    continue
                seen.add(full_url)
                items.append({"url": full_url, "title": title, "source": "보배드림"})
                if len(items) >= max_fetch:
                    break
        except Exception as e:
            print(f"[보배드림] 수집 오류: {e}")
        if len(items) >= max_fetch:
            break
    return items


def fetch_bobaedream_post(url: str) -> Dict[str, str]:
    """보배드림 글 본문 수집."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.bobaedream.co.kr/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = ""
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        content_area = soup.find("div", class_=lambda c: c and any(k in c for k in ("bodyCont", "post_content", "content", "view_cont")))
        if not content_area:
            content_area = soup
        SKIP = {"더보기", "닫기", "공유", "신고", "좋아요", "댓글", "팔로우", "loading", "광고"}
        blocks, seen_t = [], set()
        for tag in content_area.find_all(["p", "span", "div", "li"]):
            if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
                continue
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 5 or len(t) > 400 or any(k in t for k in SKIP) or t in seen_t:
                continue
            seen_t.add(t)
            blocks.append(t)
            if len(blocks) >= 30:
                break
        cmt_blocks = []
        for area in soup.find_all(["div", "ul"], class_=lambda c: c and any(k in c for k in ("comment", "reply", "cmt"))):
            for tag in area.find_all(["p", "span", "li"]):
                t = tag.get_text(" ", strip=True)
                if t and 5 < len(t) < 150 and t not in seen_t:
                    seen_t.add(t)
                    cmt_blocks.append(t)
                if len(cmt_blocks) >= 10:
                    break
        content = "\n".join(blocks)
        if cmt_blocks:
            content += "\n\n[댓글 반응]\n" + "\n".join(cmt_blocks)
        return {"title": title, "content": content, "source": "보배드림"}
    except Exception as e:
        return {"title": "", "content": "", "source": "보배드림", "error": str(e)}


# 소스별 크롤러 매핑
# ── 네이트뉴스 실시간 랭킹 ──────────────────────────────────────

# 뉴스 필터: 제목에 이 키워드가 포함되면 스킵 (정치/날씨/증시 등 재미없는 것들)
NEWS_SKIP_KEYWORDS = [
    # 정치
    "대통령", "국회", "여당", "야당", "민주당", "국민의힘", "대선", "총선", "탄핵",
    "내각", "장관", "의원", "정치", "헌재", "헌법재판", "법원", "검찰", "수사",
    "선거", "투표", "의석", "국정", "외교", "정부", "청와대", "용산",
    "계엄", "특검", "공수처", "의회", "입법",
    # 경제/날씨/주식
    "코스피", "코스닥", "주가", "환율", "금리", "증시", "주식", "경제지표",
    "날씨", "미세먼지", "태풍", "폭설", "황사", "기온", "강수",
    "GDP", "CPI", "무역수지", "수출입", "통계청", "물가", "기준금리",
    # 부동산/건설
    "아파트값", "집값", "전셋값", "분양", "재건축", "재개발",
    # 군사/외교
    "북한", "핵", "미사일", "국방", "전쟁", "군사", "병역",
    # 기업/산업 딱딱한 것
    "삼성전자", "반도체", "배터리", "수주", "실적발표", "영업이익",
]

# K-POP/한류 팬이라면 관심 가질만한 필수 키워드 화이트리스트
# 제목에 하나라도 포함되어야 통과 (없으면 전량 스킵)
# ★ 의도적으로 "세계/글로벌/해외/드라마/배우/가수" 등 모호한 단어 제외 — 정치기사 유입 방지
NEWS_MUST_KEYWORDS = [
    # K-POP 그룹 / 팀명
    "아이돌", "걸그룹", "보이그룹", "케이팝", "K-POP", "KPOP", "한류",
    "BTS", "방탄소년단", "블랙핑크", "BLACKPINK",
    "뉴진스", "NewJeans", "에스파", "aespa",
    "트와이스", "TWICE", "세븐틴", "SEVENTEEN",
    "스트레이키즈", "Stray Kids", "NCT", "IVE", "아이브",
    "르세라핌", "LE SSERAFIM", "엑소", "EXO",
    "빅뱅", "BIGBANG", "샤이니", "SHINee", "2NE1", "소녀시대",
    "(여자)아이들", "오마이걸", "레드벨벳", "있지", "ITZY",
    "txt", "투모로우바이투게더", "엔하이픈", "ENHYPEN",
    "지코", "태민", "청하", "효린", "청하",
    # BTS 멤버
    "지민", "뷔", "정국", "슈가", "RM", "제이홉",
    # 블랙핑크 멤버
    "제니", "리사", "로제", "지수",
    # 에스파 멤버
    "카리나", "윈터", "닝닝", "지젤",
    # 기획사 (연예 맥락)
    "하이브", "HYBE", "SM엔터테인먼트", "JYP엔터테인먼트", "YG엔터테인먼트",
    # K-POP 전용 용어
    "컴백", "뮤직비디오", "앨범", "음원", "팬미팅", "콘서트", "월드투어",
    "빌보드", "그래미", "데뷔", "활동재개",
    # K-드라마 (구체적 제목·장르명)
    "넷플릭스 한국", "K드라마", "오징어게임",
    # 한국 연예 행사
    "연예인", "멜론", "가요대전", "음악방송",
    # 해외 반응 (구체적 표현)
    "해외반응", "외국인 반응",
    # K뷰티/K패션 (구체적)
    "K뷰티", "K패션", "한복",
]

# 뉴스 통과: 제목에 이 키워드가 있으면 우선순위 상승
NEWS_HOT_KEYWORDS = [
    # K-POP 최고 우선
    "빌보드", "그래미", "월드투어", "컴백", "신곡", "뮤직비디오",
    "역대급", "최초", "최고", "1위",
    # 화제성
    "충격", "논란", "폭로", "화제", "난리", "반전",
    # 글로벌 관심
    "해외반응", "외국인 반응", "세계", "글로벌",
    # 연애/스캔들 (팬들 관심)
    "열애", "결혼", "이혼", "열애설", "불륜", "스캔들",
    # 사건
    "사망", "체포", "갑질", "탈세",
    # SNS 바이럴
    "유튜브", "틱톡", "SNS",
]


def fetch_natenews_list(max_fetch: int = 30) -> List[Dict[str, str]]:
    """네이트뉴스 실시간 인기 뉴스 수집 (K-POP/한류/연예 화이트리스트 필터링)."""
    list_urls = [
        "https://news.nate.com/rank/interest?sc=en",    # 연예 전용
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://news.nate.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    items: List[Dict[str, str]] = []
    seen: set = set()

    for list_url in list_urls:
        try:
            r = requests.get(list_url, headers=headers, timeout=15)
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                title = a.get_text(strip=True)
                # /view/ 패턴 링크만
                if "/view/" not in href:
                    continue
                # URL 정규화
                # 케이스1: //news.nate.com/view/... (프로토콜 상대 URL)
                # 케이스2: /view/... (경로 상대 URL)
                # 케이스3: https://... (절대 URL)
                if href.startswith("//"):
                    full_url = "https:" + href
                elif href.startswith("http"):
                    full_url = href
                else:
                    full_url = "https://news.nate.com" + href
                full_url = full_url.split("?")[0]
                if full_url in seen or not title or len(title) < 10:
                    continue
                # 1차: 정치/경제 등 하드 블록
                if any(kw in title for kw in NEWS_SKIP_KEYWORDS):
                    continue
                # 2차: K-POP/한류/연예 화이트리스트 — 하나도 없으면 스킵
                if not any(kw in title for kw in NEWS_MUST_KEYWORDS):
                    continue
                seen.add(full_url)
                # 핫 키워드 포함 여부로 점수 매기기
                hot_score = sum(1 for kw in NEWS_HOT_KEYWORDS if kw in title)
                items.append({"url": full_url, "title": title, "source": "네이트뉴스 🔥", "hot_score": str(hot_score)})
        except Exception as e:
            print(f"[네이트뉴스] 수집 오류 ({list_url}): {e}")

    # 핫 점수 높은 순으로 정렬
    items.sort(key=lambda x: int(x.get("hot_score", "0")), reverse=True)
    return items[:max_fetch]


def fetch_natenews_post(url: str) -> Dict[str, str]:
    """네이트뉴스 기사 본문 수집."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://news.nate.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")

        # 제목
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].replace(" : 네이트", "").replace(" - 네이트뉴스", "").strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)

        # og:description (AI 요약 or 첫 문장)
        og_desc = soup.find("meta", property="og:description")
        desc = og_desc["content"].strip() if og_desc and og_desc.get("content") else ""

        # 본문 영역
        SKIP_UI = {"NATE", "스포츠", "뉴스", "연예", "판", "날씨", "검색", "최신뉴스",
                   "공유하기", "원문", "기사전송", "AI 챗", "AI 요약", "작게", "보통", "크게"}
        content_area = soup.find("div", class_=lambda c: c and any(
            k in c for k in ("news_text", "artText", "article_body", "news-article", "article-body", "view_text")
        ))
        if not content_area:
            # id 기반 fallback
            content_area = soup.find("div", id=lambda i: i and any(
                k in i for k in ("articleBodyContents", "articeBody", "newsEndContents")
            ))
        if not content_area:
            content_area = soup

        blocks, seen_t = [], set()
        for tag in content_area.find_all(["p", "span", "div", "li"]):
            if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
                continue
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 10 or len(t) > 500:
                continue
            if t in SKIP_UI or any(ui in t for ui in SKIP_UI):
                continue
            if t in seen_t:
                continue
            seen_t.add(t)
            blocks.append(t)
            if len(blocks) >= 25:
                break

        content = "\n".join(blocks)
        if not content.strip():
            content = desc

        return {"title": title, "content": content, "desc": desc, "source": "네이트뉴스 🔥"}
    except Exception as e:
        return {"title": "", "content": "", "source": "네이트뉴스 🔥", "error": str(e)}


SOURCE_CRAWLERS = {
    "네이트뉴스 🔥": (fetch_natenews_list, fetch_natenews_post),  # 핫 이슈 뉴스
    "네이버 뿜": None,   # fetch_bboom_list / fetch_bboom_post_text (기존)
    "네이트판":  (fetch_natepann_list,   fetch_natepann_post),
    "에펨코리아": (fetch_fmkorea_list,   fetch_fmkorea_post),
    "DC인사이드": (fetch_dcinside_list,  fetch_dcinside_post),
    "보배드림":   (fetch_bobaedream_list, fetch_bobaedream_post),
}


def fetch_all_sources(config: AppConfig, selected_sources: List[str]) -> List[Dict[str, str]]:
    """
    선택된 소스에서 글 목록을 수집해 하나의 리스트로 합산.
    각 아이템: {"url", "title", "source"}
    """
    all_items: List[Dict[str, str]] = []
    per_source = max(10, config.bboom_max_fetch // max(len(selected_sources), 1))
    for src in selected_sources:
        try:
            if src == "네이버 뿜":
                items = fetch_bboom_list(config)
                for it in items:
                    it.setdefault("source", "네이버 뿜")
            else:
                crawlers = SOURCE_CRAWLERS.get(src)
                if not crawlers:
                    continue
                list_fn, _ = crawlers
                items = list_fn(per_source)
            all_items.extend(items)
            print(f"[소스수집] {src}: {len(items)}개")
        except Exception as e:
            print(f"[소스수집] {src} 오류: {e}")
    # 중복 URL 제거
    seen: set = set()
    deduped = []
    for it in all_items:
        u = it.get("url", "")
        if u and u not in seen:
            seen.add(u)
            deduped.append(it)
    random.shuffle(deduped)
    return deduped


def fetch_post_by_source(source: str, url: str, config: AppConfig) -> Dict[str, str]:
    """소스에 맞는 크롤러로 글 본문 수집."""
    if source == "네이버 뿜":
        return fetch_bboom_post_text(url)
    # 정확한 키 매칭 우선
    crawlers = SOURCE_CRAWLERS.get(source)
    if not crawlers:
        # 이모지 등 불일치 대비: 소스 이름 포함 여부로 fallback 매칭
        for key, val in SOURCE_CRAWLERS.items():
            if val and (source in key or key in source):
                crawlers = val
                break
    if crawlers:
        _, post_fn = crawlers
        return post_fn(url)
    return {"title": "", "content": ""}


def fetch_bboom_list(config: AppConfig) -> List[Dict[str, str]]:
    _BBOOM_BASE = "https://m.bboom.naver.com"
    # 메인 페이지 URL 자동 보정 (/best 는 404 → / 로)
    list_url = config.bboom_list_url
    if list_url.rstrip("/").endswith("/best"):
        list_url = _BBOOM_BASE + "/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.naver.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    response = requests.get(list_url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        # 현재 뿜 게시글 URL 패턴: /best/get?boardNo=...&postNo=...
        if "postNo=" not in href and "postno=" not in href.lower():
            continue
        # 앵커(#) 제거
        href = href.split("#")[0]
        full_url = urljoin(_BBOOM_BASE, href) if href.startswith("/") else href
        if full_url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        title = unescape(title).strip()
        # 제목 없으면 상위 부모에서 탐색
        if not title:
            parent = anchor.find_parent(["li", "div", "article"])
            if parent:
                title = parent.get_text(" ", strip=True)[:80].strip()
        if not title:
            title = full_url
        seen.add(full_url)
        items.append({"url": full_url, "title": title})
        if len(items) >= config.bboom_max_fetch:
            break

    if items:
        return items

    # fallback: 정규식으로 postNo 포함 경로 추출
    for match in re.findall(r'href=["\']([^"\']*postNo=\d+[^"\']*)["\']', html, flags=re.I):
        href = match.split("#")[0]
        full_url = urljoin(_BBOOM_BASE, href) if href.startswith("/") else href
        if full_url in seen:
            continue
        seen.add(full_url)
        items.append({"url": full_url, "title": full_url})
        if len(items) >= config.bboom_max_fetch:
            break

    return items


def fetch_bboom_post_text(url: str) -> Dict[str, str]:
    """
    네이버 뿜 게시글 본문 + 댓글/반응 텍스트 최대한 수집.
    seed_text 로 그대로 OpenAI 에 넘기므로 맥락이 풍부할수록 좋음.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://m.bboom.naver.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # ── 제목 ──────────────────────────────────────────────
    title = ""
    meta_title = soup.find("meta", property="og:title")
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    if not title and soup.title:
        title = soup.title.get_text(strip=True)

    # ── og:description (미리보기 한줄 요약) ──────────────
    meta_desc = soup.find("meta", property="og:description")
    desc = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""

    # ── 본문: 짧은 텍스트 노드 제거, 불필요 UI 텍스트 필터 ──
    SKIP_KEYWORDS = {"더보기", "닫기", "공유", "신고", "좋아요", "댓글", "팔로우", "구독", "loading"}
    text_blocks = []
    seen_texts: set = set()

    # 본문 영역 후보 (class 명 기반 우선)
    body_candidates = soup.find_all(
        ["div", "section", "article"],
        class_=lambda c: c and any(k in c for k in ("content", "body", "post", "text", "article", "view"))
    )
    search_root = body_candidates[0] if body_candidates else soup

    for tag in search_root.find_all(["p", "span", "div", "li", "td", "blockquote"]):
        # 자식 태그가 많은 컨테이너 div는 건너뜀 (텍스트 중복 방지)
        if tag.name == "div" and len(tag.find_all(["p", "span", "li"])) > 3:
            continue
        text = tag.get_text(" ", strip=True)
        if not text or len(text) < 4 or len(text) > 300:
            continue
        if any(kw in text for kw in SKIP_KEYWORDS):
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        text_blocks.append(text)
        if len(text_blocks) >= 30:
            break

    content = "\n".join(text_blocks)
    if not content:
        content = desc

    # ── 댓글/반응 섹션 별도 수집 ────────────────────────
    comment_blocks = []
    comment_area = soup.find_all(
        ["div", "ul", "section"],
        class_=lambda c: c and any(k in c for k in ("comment", "reply", "reaction", "cmt"))
    )
    for area in comment_area[:2]:
        for tag in area.find_all(["p", "span", "li"]):
            t = tag.get_text(" ", strip=True)
            if t and 4 < len(t) < 150 and t not in seen_texts:
                seen_texts.add(t)
                comment_blocks.append(t)
            if len(comment_blocks) >= 10:
                break

    comments_text = "\n".join(comment_blocks)

    # ── 최종 조합 ────────────────────────────────────────
    full_content = content
    if comments_text:
        full_content += f"\n\n[반응/댓글]\n{comments_text}"

    return {"title": title, "content": full_content, "desc": desc}


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
    category = script.get("content_category", "")
    return (
        f"제목(안): {script.get('title_ja','')}\n"
        f"분위기: {category}\n"
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
        _status_update(progress, status_box, 0.10, "글 내용 분석")
        try:
            post = fetch_bboom_post_text(url)
        except Exception:
            post = {"title": item.get("title", ""), "content": ""}
        seed = f"{post.get('title','')}\n{post.get('content','')}"

        # 글 분위기 카테고리 분석
        _status_update(progress, status_box, 0.15, "글 분위기 분석 중")
        content_category = analyze_content_category(config, seed)
        st.info(f"감지된 분위기 카테고리: **{content_category}**")

        # 스크립트 생성 (beats 수, BGM 쿼리 등 AI가 자동 결정)
        _status_update(progress, status_box, 0.22, "AI 스크립트 생성 중...")
        script = generate_script(
            config=config,
            seed_text=seed,
            trend_context=trend_context,
            dialect_style=config.ja_dialect_style,
            content_category=content_category,
        )

        # AI가 제안한 BGM 쿼리로 BGM 다운로드 (스크립트 생성 후 처리)
        ai_bgm_query = script.get("bgm_query")
        if isinstance(ai_bgm_query, list):
            ai_bgm_query = " ".join(ai_bgm_query)
        ai_bgm_query = ai_bgm_query or ""  # None 방어
        _status_update(progress, status_box, 0.30, f"BGM 선정 중 (키워드: {ai_bgm_query or content_category})")
        bgm_path = get_or_download_bgm(config, content_category, custom_query=ai_bgm_query)
        if bgm_path:
            st.info(f"BGM: {os.path.basename(bgm_path)} (쿼리: {ai_bgm_query or content_category})")
        else:
            bgm_path = pick_bgm_path(config)

        texts = [beat.get("text", "") for beat in script.get("beats", [])]
        beat_tags = [beat.get("tag", "") for beat in script.get("beats", [])]

        # 에셋 미리 선택 (미리보기에 포함) - (asset_path, category_tags) 쌍으로 저장
        assets = []          # str: 파일 경로
        asset_cats = []      # str: 해당 에셋의 카테고리/태그 요약
        for tag in beat_tags:
            asset = pick_asset(manifest_items, [tag])
            if not asset:
                asset = pick_asset_by_category(manifest_items, content_category)
            if asset:
                assets.append(asset.path)
                # 태그 정보 요약 (없으면 content_category)
                tag_summary = ", ".join(asset.tags) if asset.tags else content_category
                asset_cats.append(tag_summary)

        if not assets:
            st.error("태그에 맞는 에셋이 없습니다.")
            return

        # ── 텔레그램 미리보기 메시지 구성 ────────────────────────
        beats = script.get("beats", [])
        beats_preview = ""
        for i, beat in enumerate(beats, 1):
            tag = beat.get("tag", "")
            txt_ko = beat.get("text_ko", "")   # 한글 원문
            txt_ja = beat.get("text", "")       # 일본어 번역
            cat_label = asset_cats[min(i - 1, len(asset_cats) - 1)] if asset_cats else content_category
            beats_preview += (
                f"  {i}. [{tag}]\n"
                f"     KO: {txt_ko}\n"
                f"     JA: {txt_ja}\n"
                f"     사진 카테고리: {cat_label}\n"
            )

        beats_count_actual = len(beats)
        bgm_display = os.path.basename(bgm_path) if bgm_path else "없음"
        bgm_query_display = ai_bgm_query if ai_bgm_query else content_category
        request_text = (
            f"[ 승인 요청 ]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"출처 글: {post.get('title', '')}\n"
            f"링크: {url}\n"
            f"분위기: {content_category}  |  beats: {beats_count_actual}개\n"
            f"BGM: {bgm_display}  (키워드: {bgm_query_display})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"제목 KO: {script.get('title_ko', '')}\n"
            f"제목 JA: {script.get('title_ja', '')}\n"
            f"해시태그: {' '.join(script.get('hashtags_ja', []))}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"대본 미리보기 ({beats_count_actual}컷)\n"
            f"{beats_preview}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"아래 버튼으로 응답해주세요."
        )

        # 첫 번째 에셋 사진을 미리보기로 전송
        if assets and os.path.exists(assets[0]):
            try:
                photo_url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendPhoto"
                with open(assets[0], "rb") as photo_file:
                    requests.post(
                        photo_url,
                        data={"chat_id": config.telegram_admin_chat_id, "caption": "대표 사진 미리보기"},
                        files={"photo": photo_file},
                        timeout=30,
                    )
            except Exception:
                pass

        approval_msg_id = send_telegram_approval_request(
            config.telegram_bot_token, config.telegram_admin_chat_id, request_text
        )
        decision = wait_for_approval(config, progress, status_box, approval_message_id=approval_msg_id)
        if decision == "swap":
            _mark_used_link(config.used_links_path, url, "swap", post.get("title", ""))
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, "🔄 교환 처리됨. 다음 인기글로 진행합니다.")
            continue
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
        voice_id = pick_voice_id(config.openai_tts_voices)
        try:
            tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
        except Exception as tts_err:
            err_msg = f"❌ TTS 생성 실패: {tts_err}\n\nOpenAI API 키가 잘못되었거나 크레딧이 부족할 수 있습니다."
            st.error(err_msg)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_msg)
            _mark_used_link(config.used_links_path, url, "error", post.get("title", ""))
            continue

        _status_update(progress, status_box, 0.6, "영상 렌더링")
        output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
            bgm_path=bgm_path,
            bgm_volume=config.bgm_volume,
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

        summary_text = f"[완료]\n제목: {script.get('title_ja','')}\n분위기: {content_category}\n요약: {script.get('description_ja','')}\n"
        if video_url:
            summary_text += f"유튜브 링크: {video_url}"
        else:
            summary_text += f"로컬 파일: {output_path}"
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)
        return

    st.warning("사용 가능한 인기글이 더 이상 없습니다.")


def _auto_content_flow(config: AppConfig, progress, status_box, selected_sources: List[str]) -> None:
    """다중 소스(네이트판, 에펨코리아, DC인사이드, 보배드림, 네이버 뿜)에서 글을 수집해 숏츠 자동 생성."""
    if not config.telegram_bot_token or not config.telegram_admin_chat_id:
        st.error("텔레그램 봇 설정이 필요합니다. TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID를 확인하세요.")
        return
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        st.error("에셋이 없습니다. 먼저 이미지를 추가하세요.")
        return

    src_label = ", ".join(selected_sources)
    _status_update(progress, status_box, 0.05, f"글 수집 중 ({src_label})")
    try:
        items = fetch_all_sources(config, selected_sources)
    except Exception as exc:
        st.error(f"글 수집 실패: {exc}")
        return
    if not items:
        st.error("가져올 글이 없습니다.")
        return

    st.info(f"총 {len(items)}개 글 수집 완료 (소스: {src_label})")
    used_data = _load_used_links(config.used_links_path)
    trend_context = get_trend_context(config)

    skipped_used = 0
    skipped_short = 0

    for item in items:
        url = item.get("url", "")
        source = item.get("source", "")
        if not url:
            continue
        if _is_used_link(used_data, url):
            skipped_used += 1
            print(f"[스킵-이미사용] {url}")
            continue

        _status_update(progress, status_box, 0.10, f"[{source}] 글 내용 분석 중... ({url[-30:]})")
        try:
            post = fetch_post_by_source(source, url, config)
        except Exception as e:
            print(f"[본문수집오류] {url} → {e}")
            post = {"title": item.get("title", ""), "content": "", "source": source}

        seed = f"{post.get('title','')}\n{post.get('content','')}"
        print(f"[본문확인] source={source} len={len(seed.strip())} title={post.get('title','')[:30]}")

        # 본문이 너무 짧으면 스킵 (품질 필터)
        if len(seed.strip()) < 30:
            skipped_short += 1
            print(f"[스킵-짧음] len={len(seed.strip())} url={url}")
            if skipped_short >= 3:
                st.warning(f"본문 수집 실패가 많습니다. 수집된 {len(items)}개 중 {skipped_used}개 이미사용, {skipped_short}개 본문없음")
            continue

        # 글 분위기 카테고리 분석
        _status_update(progress, status_box, 0.15, "글 분위기 분석 중")
        content_category = analyze_content_category(config, seed)
        st.info(f"[{source}] {item.get('title','')[:40]}... → 분위기: **{content_category}**")

        # 스크립트 생성 (뉴스 소스면 news 프롬프트 강화)
        is_news = "뉴스" in source
        _status_update(progress, status_box, 0.22, "AI 스크립트 생성 중...")
        script = generate_script(
            config=config,
            seed_text=seed,
            trend_context=trend_context,
            dialect_style=config.ja_dialect_style,
            content_category=content_category,
            source_type="news" if is_news else "",
        )

        # BGM 선정
        ai_bgm_query = script.get("bgm_query")
        if isinstance(ai_bgm_query, list):
            ai_bgm_query = " ".join(ai_bgm_query)
        ai_bgm_query = ai_bgm_query or ""
        _status_update(progress, status_box, 0.30, f"BGM 선정 중 (키워드: {ai_bgm_query or content_category})")
        bgm_path = get_or_download_bgm(config, content_category, custom_query=ai_bgm_query)
        if bgm_path:
            st.info(f"BGM: {os.path.basename(bgm_path)} (쿼리: {ai_bgm_query or content_category})")
        else:
            bgm_path = pick_bgm_path(config)

        texts = [beat.get("text", "") for beat in script.get("beats", [])]
        beat_tags = [beat.get("tag", "") for beat in script.get("beats", [])]

        # 에셋 선택
        assets = []
        asset_cats = []
        for tag in beat_tags:
            asset = pick_asset(manifest_items, [tag])
            if not asset:
                asset = pick_asset_by_category(manifest_items, content_category)
            if asset:
                assets.append(asset.path)
                tag_summary = ", ".join(asset.tags) if asset.tags else content_category
                asset_cats.append(tag_summary)

        if not assets:
            st.error("태그에 맞는 에셋이 없습니다.")
            return

        # 텔레그램 미리보기 구성
        beats = script.get("beats", [])
        beats_preview = ""
        for idx, beat in enumerate(beats, 1):
            tag = beat.get("tag", "")
            txt_ko = beat.get("text_ko", "")
            txt_ja = beat.get("text", "")
            cat_label = asset_cats[min(idx - 1, len(asset_cats) - 1)] if asset_cats else content_category
            beats_preview += (
                f"  {idx}. [{tag}]\n"
                f"     KO: {txt_ko}\n"
                f"     JA: {txt_ja}\n"
                f"     사진 카테고리: {cat_label}\n"
            )

        beats_count_actual = len(beats)
        bgm_display = os.path.basename(bgm_path) if bgm_path else "없음"
        bgm_query_display = ai_bgm_query if ai_bgm_query else content_category
        request_text = (
            f"[ 승인 요청 ] [{source}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"출처 글: {post.get('title', '')}\n"
            f"링크: {url}\n"
            f"분위기: {content_category}  |  beats: {beats_count_actual}개\n"
            f"BGM: {bgm_display}  (키워드: {bgm_query_display})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"제목 KO: {script.get('title_ko', '')}\n"
            f"제목 JA: {script.get('title_ja', '')}\n"
            f"해시태그: {' '.join(script.get('hashtags_ja', []))}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"대본 미리보기 ({beats_count_actual}컷)\n"
            f"{beats_preview}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"아래 버튼으로 응답해주세요."
        )

        # 대표 사진 미리보기 전송
        if assets and os.path.exists(assets[0]):
            try:
                photo_api = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendPhoto"
                with open(assets[0], "rb") as photo_file:
                    requests.post(
                        photo_api,
                        data={"chat_id": config.telegram_admin_chat_id, "caption": f"[{source}] 대표 사진 미리보기"},
                        files={"photo": photo_file},
                        timeout=30,
                    )
            except Exception:
                pass

        approval_msg_id = send_telegram_approval_request(
            config.telegram_bot_token, config.telegram_admin_chat_id, request_text
        )
        decision = wait_for_approval(config, progress, status_box, approval_message_id=approval_msg_id)
        if decision == "swap":
            _mark_used_link(config.used_links_path, url, "swap", post.get("title", ""))
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, "🔄 교환 처리됨. 다음 인기글로 진행합니다.")
            continue

        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
        voice_id = pick_voice_id(config.openai_tts_voices)
        try:
            tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
        except Exception as tts_err:
            err_msg = f"❌ TTS 생성 실패: {tts_err}\n\nOpenAI API 키가 잘못되었거나 크레딧이 부족할 수 있습니다."
            st.error(err_msg)
            send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_msg)
            _mark_used_link(config.used_links_path, url, "error", post.get("title", ""))
            continue

        _status_update(progress, status_box, 0.6, "영상 렌더링")
        output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
            bgm_path=bgm_path,
            bgm_volume=config.bgm_volume,
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
            "source": source,
            "source_url": url,
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

        summary_text = (
            f"[완료] [{source}]\n"
            f"제목: {script.get('title_ja','')}\n"
            f"분위기: {content_category}\n"
            f"요약: {script.get('description_ja','')}\n"
        )
        if video_url:
            summary_text += f"유튜브 링크: {video_url}"
        else:
            summary_text += f"로컬 파일: {output_path}"
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)
        return

    st.warning(f"사용 가능한 글이 없습니다. (수집: {len(items)}개 / 이미사용: {skipped_used}개 / 본문없음: {skipped_short}개) — Streamlit Cloud 로그를 확인해주세요.")


def run_streamlit_app() -> None:
    st.set_page_config(page_title="숏츠 자동화 스튜디오", layout="wide")
    config = load_config()
    ensure_dirs(
        [
            config.assets_dir,
            os.path.join(config.assets_dir, "images"),
            os.path.join(config.assets_dir, "inbox"),
            os.path.join(config.assets_dir, "bgm"),
            os.path.join(config.assets_dir, "bgm", "trending"),
            # NEW: 카테고리별 BGM 디렉토리
            *[os.path.join(config.assets_dir, "bgm", cat) for cat in CONTENT_CATEGORIES],
            os.path.join(config.assets_dir, "sfx"),
            os.path.dirname(config.manifest_path),
            config.output_dir,
        ]
    )

    st.sidebar.title("숏츠 자동화 스튜디오")
    st.sidebar.subheader("상태")
    st.sidebar.write(f"자동 업로드: {'켜짐' if config.enable_youtube_upload else '꺼짐'}")
    st.sidebar.write(f"MoviePy 사용 가능: {'예' if MOVIEPY_AVAILABLE else '아니오'}")
    st.sidebar.write(f"BGM 모드: {config.bgm_mode or 'off'}")
    st.sidebar.write(f"Pixabay BGM: {'연결됨' if config.pixabay_api_key else '미설정'}")
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
        "- `TELEGRAM_TIMEOUT_SEC`"
    )
    st.sidebar.subheader("선택")
    st.sidebar.markdown(
        "- `YOUTUBE_*` (자동 업로드)\n"
        "- `SERPAPI_API_KEY` (트렌드 수집)\n"
        "- `PEXELS_API_KEY` (트렌드 이미지 자동 수집)\n"
        "- `PIXABAY_API_KEY` (BGM 자동 다운로드) ← NEW\n"
        "- `JA_DIALECT_STYLE` (일본어 사투리 스타일)\n"
        "- `OPENAI_VISION_MODEL` (이미지 태그 분석 모델)\n"
        "- `BGM_MODE`, `BGM_VOLUME` (배경음악 자동 선택)"
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

        st.subheader("한국 핫이슈 자동 생성 (승인 포함)")
        all_sources = list(SOURCE_CRAWLERS.keys())
        # 기본값: 네이트뉴스만 선택
        default_sources = ["네이트뉴스 🔥"]
        selected_sources = st.multiselect(
            "수집할 소스 선택 (복수 선택 가능)",
            options=all_sources,
            default=default_sources,
            help="네이트뉴스: K-POP/한류/연예 화이트리스트 필터 적용 (정치·경제·날씨 자동 제외)\n커뮤니티: 네이트판/에펨코리아/DC/보배드림 유머글",
        )
        auto_button = st.button("자동 생성 시작", type="primary")
        if auto_button:
            if not selected_sources:
                st.warning("소스를 하나 이상 선택해주세요.")
            else:
                _auto_content_flow(config, progress, status_box, selected_sources)

        st.divider()
        seed_text = st.text_area("아이디어/요약 입력", height=120)
        beats_count = st.slider("문장(비트) 수", 5, 9, 7)
        tag_filter = st.multiselect("허용 태그", options=all_tags, default=all_tags[:5])

        # NEW: 수동 생성 시 카테고리 선택
        manual_category = st.selectbox(
            "콘텐츠 분위기 카테고리",
            options=["(자동 감지)"] + CONTENT_CATEGORIES,
            help="'자동 감지'를 선택하면 AI가 입력 텍스트를 분석해 카테고리를 결정합니다.",
        )
        generate_button = st.button("스크립트 생성")

        if generate_button and seed_text:
            _status_update(progress, status_box, 0.05, "스크립트 생성 중")
            # 카테고리 자동/수동 결정
            if manual_category == "(자동 감지)":
                detected_cat = analyze_content_category(config, seed_text)
                st.info(f"감지된 카테고리: **{detected_cat}**")
            else:
                detected_cat = manual_category
            script = generate_script(
                config=config,
                seed_text=seed_text,
                beats_count=beats_count,
                allowed_tags=tag_filter or all_tags,
                trend_context=get_trend_context(config),
                dialect_style=config.ja_dialect_style,
                content_category=detected_cat,
            )
            st.session_state["script"] = script
            st.session_state["detected_category"] = detected_cat
            _status_update(progress, status_box, 0.2, "스크립트 생성 완료")

        script = st.session_state.get("script")
        if script:
            st.subheader("스크립트")
            st.caption(f"분위기 카테고리: **{script.get('content_category', '-')}**")
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
                    beat_tags = [beat.get("tag", "") for beat in beats]
                    detected_cat = st.session_state.get("detected_category", script.get("content_category", "humor"))

                    # NEW: 카테고리 기반 BGM 선택
                    _status_update(progress, status_box, 0.15, f"BGM 선정 중 ({detected_cat})")
                    bgm_path = get_or_download_bgm(config, detected_cat)
                    if not bgm_path:
                        bgm_path = pick_bgm_path(config)

                    # NEW: 카테고리 기반 에셋 선택
                    assets = []
                    for tag in beat_tags:
                        asset = pick_asset(manifest_items, [tag])
                        if not asset:
                            asset = pick_asset_by_category(manifest_items, detected_cat)
                        if asset:
                            assets.append(asset.path)

                    if not assets:
                        st.error("태그에 맞는 에셋이 없습니다.")
                    else:
                        _status_update(progress, status_box, 0.3, "TTS 생성")
                        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
                        voice_id = pick_voice_id(config.openai_tts_voices)
                        tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
                        _status_update(progress, status_box, 0.6, "영상 렌더링")
                        output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
                        render_video(
                            config=config,
                            asset_paths=assets,
                            texts=texts,
                            tts_audio_path=audio_path,
                            output_path=output_path,
                            bgm_path=bgm_path,
                            bgm_volume=config.bgm_volume,
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

        st.subheader("BGM 업로드")
        bgm_files = st.file_uploader(
            "오디오 업로드",
            type=["mp3", "wav", "m4a", "aac", "ogg"],
            accept_multiple_files=True,
            key="bgm_upload",
        )
        bgm_target = st.selectbox(
            "저장 위치",
            ["일반 BGM", "트렌드 BGM"] + [f"카테고리: {cat}" for cat in CONTENT_CATEGORIES],
        )
        if st.button("BGM 저장") and bgm_files:
            if bgm_target == "트렌드 BGM":
                target_dir = os.path.join(config.assets_dir, "bgm", "trending")
            elif bgm_target.startswith("카테고리: "):
                cat_name = bgm_target.replace("카테고리: ", "")
                target_dir = os.path.join(config.assets_dir, "bgm", cat_name)
            else:
                target_dir = os.path.join(config.assets_dir, "bgm")
            os.makedirs(target_dir, exist_ok=True)
            for file in bgm_files:
                save_path = os.path.join(target_dir, file.name)
                with open(save_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
            st.success("BGM이 저장되었습니다.")

        # NEW: Pixabay BGM 수동 다운로드
        st.subheader("Pixabay BGM 수동 다운로드")
        if not config.pixabay_api_key:
            st.warning("PIXABAY_API_KEY가 설정되지 않았습니다. `.streamlit/secrets.toml`에 추가하세요.")
        else:
            pixabay_cat = st.selectbox("BGM 카테고리", CONTENT_CATEGORIES, key="pixabay_cat")
            pixabay_count = st.slider("다운로드 개수", 1, 5, 3, key="pixabay_count")
            if st.button("Pixabay에서 BGM 다운로드"):
                bgm_out_dir = os.path.join(config.assets_dir, "bgm", pixabay_cat)
                downloaded_bgms = []
                for _ in range(pixabay_count):
                    path = fetch_bgm_from_pixabay(
                        api_key=config.pixabay_api_key,
                        category=pixabay_cat,
                        output_dir=bgm_out_dir,
                    )
                    if path:
                        downloaded_bgms.append(path)
                if downloaded_bgms:
                    st.success(f"{len(downloaded_bgms)}개 BGM을 `bgm/{pixabay_cat}/`에 저장했습니다.")
                    for p in downloaded_bgms:
                        st.audio(p)
                else:
                    st.error("BGM 다운로드에 실패했습니다. API 키나 네트워크를 확인하세요.")

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

        # ── 실행 로그 ────────────────────────────────────────────────
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
        content_category = analyze_content_category(config, seed)
        script = generate_script(
            config,
            seed,
            beats_count=beats,
            trend_context=get_trend_context(config),
            dialect_style=config.ja_dialect_style,
            content_category=content_category,
        )
        beats_list = script.get("beats", [])
        texts = [beat.get("text", "") for beat in beats_list]
        beat_tags = [beat.get("tag", "") for beat in beats_list]
        assets = []
        for tag in beat_tags:
            asset = pick_asset(manifest_items, [tag])
            if not asset:
                asset = pick_asset_by_category(manifest_items, content_category)
            if asset:
                assets.append(asset.path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}_{index}.mp3")
        voice_id = pick_voice_id(config.openai_tts_voices)
        tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
        output_path = os.path.join(config.output_dir, f"shorts_{now}_{index}.mp4")
        bgm_path = get_or_download_bgm(config, content_category)
        if not bgm_path:
            bgm_path = pick_bgm_path(config)
        render_video(
            config=config,
            asset_paths=assets,
            texts=texts,
            tts_audio_path=audio_path,
            output_path=output_path,
            bgm_path=bgm_path,
            bgm_volume=config.bgm_volume,
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