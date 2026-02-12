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
    approve_keywords: List[str]
    swap_keywords: List[str]
    pixabay_api_key: str


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
        approve_keywords=_get_list("APPROVE_KEYWORDS") or ["승인", "approve", "ok", "yes"],
        swap_keywords=_get_list("SWAP_KEYWORDS") or ["교환", "swap", "change", "next"],
        pixabay_api_key=_get_secret("PIXABAY_API_KEY", "") or "",
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
# 일본인 타겟 숏츠 대본 생성 시스템
# ─────────────────────────────────────────────

# BGM 무드 카테고리 (mystery / exciting / informative)
BGM_MOOD_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "mystery": {
        "description": "미스터리/긴장감 — 충격 폭로, 이면 폭로, 공포 계열",
        "pixabay_queries": ["suspense dramatic", "mystery tension", "thriller cinematic"],
        "folder": "mystery",
    },
    "exciting": {
        "description": "템포 빠른 신나는 비트 — 맛집, 여행, 랭킹, 에너지",
        "pixabay_queries": ["energetic upbeat", "hype electronic", "fun pop beat"],
        "folder": "exciting",
    },
    "informative": {
        "description": "감성 브이로그 톤 — 정보/팁, 일상, 가이드 계열",
        "pixabay_queries": ["chill lofi", "acoustic vlog", "soft background"],
        "folder": "informative",
    },
}

# 롤렛 주제 풀 — LLM이 이 리스트를 참고해 매번 새 주제 생성
JP_CONTENT_THEMES: List[str] = [
    "한국 여행 꿀팁 / 가성비 맛집 랭킹",
    "일본인이 놀라는 한국 문화 충격",
    "한국 편의점 신상품 리뷰",
    "서울/부산/제주 숨겨진 명소",
    "한국인만 아는 초가성비 맛집",
    "한국 최근 논란·핫이슈",
    "K-뷰티 꿀팁·추천 아이템",
    "한국 길거리 음식 BEST",
    "한국과 일본의 문화 차이",
    "한국 여행에서 절대 하면 안 되는 실수",
]

# 시스템 프롬프트 (LLM에 직접 전달)
JP_SHORTS_SYSTEM_PROMPT: str = """당신은 일본인 유튜브 숏츠 채널 전문 작가입니다.
채널 컨셉: '한국 여행·문화·음식·이슈'를 일본인 시청자에게 소개하는 숏츠.
목표: 조회수 1위, 극강 어그로, 시청 지속 시간 극대화.

[주제 선정 규칙]
- 매번 새롭고 자극적인 주제를 스스로 선정할 것 (롤렛 방식)
- 예시: 가성비 맛집 랭킹, 숨겨진 명소, 충격 이슈, 문화 차이, 여행 실수 주의
- 지역 타겟팅 활용: '부산', '서울 홍대', '제주도' 등 구체적일수록 좋음
- Top5/Top3 랭킹 주제일 경우 top5_info 필드에 실제 업소명·주소·한줄 설명 포함 (필수)

[필수 출력 규칙]
1. hook_3_sec + hook_3_sec_ko: 일본어 + 한국어 훅 각 1문장
2. body_script + body_script_ko: 일본어/한국어 각 6~8개 문장 배열 (순서 일치)
3. cta_outro + cta_outro_ko: 일본어/한국어 구독 유도 각 1문장
4. pinned_comment: 댓글 폭발 유도 일본어 질문 1개
5. mood: "mystery" | "exciting" | "informative" 중 1개 (BGM 자동 매칭용)
6. video_title: 극강 어그로 일본어 유튜브 제목
7. hashtags: 일본 조회수 터지는 해시태그 정확히 5개 (# 포함)
8. top5_info (주제가 랭킹/맛집/명소일 때 필수): 업소 정보 배열
   각 항목: {"rank": 1, "name_ko": "가게명", "area": "지역구", "address_hint": "역 근처 등 힌트", "desc_ko": "한줄 설명", "desc_ja": "일본어 설명"}
9. bg_search_query: Pexels 배경영상 검색용 영어 키워드 (예: "Seoul street food market")

[출력 형식 — 반드시 순수 JSON만 출력, 마크다운 금지]
{
  "topic_theme": "주제 테마",
  "video_title": "극강 어그로 일본어 제목",
  "hashtags": ["#태그1","#태그2","#태그3","#태그4","#태그5"],
  "hook_3_sec": "일본어 훅",
  "hook_3_sec_ko": "한국어 훅",
  "body_script": ["일본어 문장1","문장2","..."],
  "body_script_ko": ["한국어 문장1","문장2","..."],
  "cta_outro": "일본어 구독 유도",
  "cta_outro_ko": "한국어 구독 유도",
  "pinned_comment": "댓글 유도 일본어 질문",
  "mood": "mystery | exciting | informative 중 1개",
  "bg_search_query": "pexels 검색 영어 키워드",
  "top5_info": [
    {"rank":1,"name_ko":"가게명","area":"지역","address_hint":"힌트","desc_ko":"한줄설명","desc_ja":"日本語説明"},
    ...
  ]
}"""


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
        "아래 주제 풀에서 영감을 받아, 오늘 가장 조회수가 터질 것 같은 주제를 스스로 선택하거나 새로 창작하세요.\n\n"
        f"[주제 풀 예시]\n{theme_pool}\n\n"
        + (f"[추가 힌트]\n{extra_hint}\n\n" if extra_hint else "")
        + "위 시스템 프롬프트의 규칙을 완벽히 지켜 JSON만 출력하세요."
    )

    client = OpenAI(api_key=config.openai_api_key)
    response = client.responses.create(
        model=config.openai_model,
        input=[
            {"role": "system", "content": JP_SHORTS_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    output_text = getattr(response, "output_text", "") or ""
    result = extract_json(output_text)
    if not result:
        raise RuntimeError("LLM JSON 파싱 실패")

    # 해시태그 정규화
    result["hashtags"] = normalize_hashtags(result.get("hashtags", []))
    # mood 검증
    if result.get("mood") not in BGM_MOOD_CATEGORIES:
        result["mood"] = "exciting"
    # body_script 리스트 확인
    if not isinstance(result.get("body_script"), list):
        body = result.get("body_script", "")
        result["body_script"] = [s.strip() for s in str(body).split("。") if s.strip()]
    # body_script_ko fallback
    if not isinstance(result.get("body_script_ko"), list):
        result["body_script_ko"] = result["body_script"]  # 일본어 그대로 복사 (최후 fallback)
    # top5_info 기본값
    if not isinstance(result.get("top5_info"), list):
        result["top5_info"] = []
    # bg_search_query 기본값
    if not result.get("bg_search_query"):
        result["bg_search_query"] = "korea street city"
    return result


def _script_to_beats(script: Dict[str, Any]) -> List[str]:
    """generate_script_jp 결과를 TTS/영상용 텍스트 리스트로 변환."""
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


def match_bgm_by_mood(config: AppConfig, mood: str) -> Optional[str]:
    """
    mood(mystery/exciting/informative)에 맞는 BGM 파일 반환.
    1) assets/bgm/{mood}/ 폴더에서 랜덤 선택
    2) 없으면 Pixabay에서 다운로드 시도
    3) 그래도 없으면 기존 pick_bgm_path() fallback
    """
    mood_info = BGM_MOOD_CATEGORIES.get(mood, BGM_MOOD_CATEGORIES["exciting"])
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

    # Pixabay 다운로드 시도
    if config.pixabay_api_key:
        queries = mood_info.get("pixabay_queries", [])
        query = random.choice(queries) if queries else folder_name
        path = fetch_bgm_from_pixabay(config.pixabay_api_key, mood, bgm_dir, custom_query=query)
        if path:
            return path

    # fallback: pick_bgm_path (기존 폴더)
    existing_any = pick_bgm_path(config)
    if existing_any:
        return existing_any

    # 최후 fallback: numpy로 ambient BGM 자동 생성 (저작권 없음)
    fallback_path = os.path.join(bgm_dir, f"generated_{mood}.wav")
    if not os.path.exists(fallback_path):
        try:
            _generate_bgm_fallback(fallback_path, duration=120.0, mood=mood)
        except Exception:
            return None
    return fallback_path if os.path.exists(fallback_path) else None


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


def _draw_subtitle(
    image: Image.Image,
    text: str,
    font_path: str,
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """화면 하단 자막 영역에 반투명 배경 + 흰색 테두리 텍스트를 그립니다."""
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    font_size = max(48, canvas_width // 18)
    font = _load_font(font_path, font_size)
    pad_x = int(canvas_width * 0.05)
    max_text_w = canvas_width - pad_x * 2
    lines = _wrap_cjk_text(text, max_text_w, font_size)
    line_h = font_size + 10
    total_h = line_h * len(lines) + 20
    box_y = canvas_height - total_h - 80
    # 반투명 배경 박스
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(overlay)
    box_draw.rectangle(
        [pad_x - 20, box_y - 10, canvas_width - pad_x + 20, canvas_height - 60],
        fill=(0, 0, 0, 160),
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    y = box_y
    for line in lines:
        try:
            lw = font.getbbox(line)[2]
        except Exception:
            lw = len(line) * font_size
        lx = max(pad_x, (canvas_width - lw) // 2)
        draw.text((lx, y), line, font=font, fill=(255, 255, 255),
                  stroke_width=3, stroke_fill=(0, 0, 0))
        y += line_h
    return image


def _overlay_sticker(
    image: Image.Image,
    asset_path: str,
    canvas_width: int,
    canvas_height: int,
    size: int = 220,
) -> Image.Image:
    """에셋 이미지를 이모티콘처럼 우하단에 작게 붙입니다."""
    if not os.path.exists(asset_path):
        return image
    try:
        sticker = Image.open(asset_path).convert("RGBA")
        sticker = sticker.resize((size, size), Image.LANCZOS)
        margin = 40
        x = canvas_width - size - margin
        y = canvas_height - size - 200  # 자막 위
        base = image.convert("RGBA")
        base.paste(sticker, (x, y), sticker)
        return base.convert("RGB")
    except Exception:
        return image


def _compose_frame(
    asset_path: str,
    text: str,
    size: Tuple[int, int],
    font_path: str,
) -> Image.Image:
    """정적 이미지 배경 프레임 생성 (배경영상 없을 때 fallback)."""
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    base = Image.open(asset_path).convert("RGB")
    background = _make_background(base, size)
    composed = background.copy()
    width, height = size
    composed = _draw_subtitle(composed, text, font_path, width, height)
    composed = _overlay_sticker(composed, asset_path, width, height, size=200)
    return composed


def fetch_pexels_video(
    query: str,
    api_key: str,
    output_dir: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
) -> Optional[str]:
    """Pexels에서 세로형(portrait) royalty-free 영상을 검색·다운로드."""
    if not api_key:
        return None
    os.makedirs(output_dir, exist_ok=True)
    try:
        headers = {"Authorization": api_key}
        params = {"query": query, "per_page": 15, "orientation": "portrait"}
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
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
                    fpath = os.path.join(output_dir, fname)
                    with open(fpath, "wb") as vf_out:
                        for chunk in vresp.iter_content(chunk_size=65536):
                            vf_out.write(chunk)
                    return fpath
                except Exception:
                    continue
    except Exception:
        pass
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
    }
    freqs = mood_chords.get(mood, mood_chords["exciting"])
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
    bg_video_path: str | None = None,
) -> str:
    """
    TTS + 자막 + 에셋 스티커 + 배경영상(or 정적 이미지)으로 숏츠 영상 생성.
    bg_video_path가 있으면 영상 배경, 없으면 정적 이미지 배경 사용.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(f"MoviePy/PIL not available: {MOVIEPY_ERROR}")
    W, H = config.width, config.height
    audio_clip = AudioFileClip(tts_audio_path)
    durations = _estimate_durations(texts, audio_clip.duration)
    clips = []

    bg_vid = None
    if bg_video_path and os.path.exists(bg_video_path):
        try:
            bg_vid = VideoFileClip(bg_video_path).without_audio()
            # 세로형으로 resize (비율 유지 → crop)
            bw, bh = bg_vid.size
            scale = max(W / bw, H / bh)
            bg_vid = bg_vid.resize((int(bw * scale), int(bh * scale)))
            cx = (bg_vid.size[0] - W) // 2
            cy = (bg_vid.size[1] - H) // 2
            bg_vid = bg_vid.crop(x1=cx, y1=cy, x2=cx + W, y2=cy + H)
        except Exception:
            bg_vid = None

    vid_offset = 0.0
    for index, text in enumerate(texts):
        asset_path = asset_paths[min(index, len(asset_paths) - 1)]
        dur = durations[index]

        if bg_vid is not None:
            # 배경 영상에서 랜덤 오프셋 구간 추출
            max_start = max(bg_vid.duration - dur - 0.1, 0)
            seg_start = random.uniform(0, max_start) if max_start > 0 else 0
            seg = bg_vid.subclip(seg_start, seg_start + dur)

            # 클로저 캡처 (Python for-loop 캡처 이슈 방지)
            _text = text
            _asset = asset_path
            _font = config.font_path

            def _make_frame(frame, __text=_text, __asset=_asset, __font=_font):
                img = Image.fromarray(frame).convert("RGB")
                img = _draw_subtitle(img, __text, __font, W, H)
                img = _overlay_sticker(img, __asset, W, H, size=200)
                return np.array(img)

            clip = seg.fl_image(_make_frame).set_duration(dur)
        else:
            # fallback: 정적 이미지 배경
            frame_img = _compose_frame(asset_path, text, (W, H), config.font_path)
            clip = ImageClip(np.array(frame_img)).set_duration(dur)
            clip = clip.fx(vfx.resize, lambda t, d=dur: 1 + 0.02 * (t / max(d, 0.1)))

        clips.append(clip)
        vid_offset += dur

    if bg_vid:
        bg_vid.close()

    video = concatenate_videoclips(clips, method="compose").set_fps(config.fps)

    # BGM 처리
    if bgm_path and os.path.exists(bgm_path):
        bgm_clip = AudioFileClip(bgm_path).volumex(bgm_volume)
        if bgm_clip.duration < audio_clip.duration:
            bgm_clip = bgm_clip.loop(duration=audio_clip.duration)
        bgm_clip = bgm_clip.set_duration(audio_clip.duration)
        audio = CompositeAudioClip([audio_clip, bgm_clip])
    else:
        audio = audio_clip

    video = video.set_audio(audio)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=config.fps,
        threads=4,
        logger=None,
    )
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
    body = script.get("body_script", [])
    middle = body[0] if body else ""
    return (
        f"주제: {script.get('topic_theme','')}\n"
        f"제목: {script.get('video_title','')}\n"
        f"무드: {script.get('mood','')}\n"
        f"훅: {script.get('hook_3_sec','')}\n"
        f"전개: {middle}\n"
        f"구독유도: {script.get('cta_outro','')}\n"
        f"해시태그: {' '.join(script.get('hashtags', []))}"
    )


def _auto_jp_flow(config: AppConfig, progress, status_box, extra_hint: str = "") -> None:
    """
    크롤링 없이 LLM이 주제를 자동 선정해 일본인 타겟 숏츠를 생성하는 메인 플로우.
    텔레그램 승인 → TTS → 영상 렌더링 → 유튜브 업로드.
    """
    if not config.telegram_bot_token or not config.telegram_admin_chat_id:
        st.error("텔레그램 봇 설정이 필요합니다. TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID를 확인하세요.")
        return
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        st.error("에셋이 없습니다. 먼저 이미지를 추가하세요.")
        return

    # ── 대본 생성 ─────────────────────────────────────────
    _status_update(progress, status_box, 0.10, "AI 대본 생성 중 (주제 자동 선정)...")
    try:
        script = generate_script_jp(config, extra_hint=extra_hint)
    except Exception as exc:
        st.error(f"대본 생성 실패: {exc}")
        return

    topic_theme = script.get("topic_theme", "")
    video_title = script.get("video_title", "")
    hashtags = script.get("hashtags", [])
    hook_ja = script.get("hook_3_sec", "")
    hook_ko = script.get("hook_3_sec_ko", hook_ja)
    body_ja = script.get("body_script", [])
    body_ko = script.get("body_script_ko", body_ja)
    cta_ja = script.get("cta_outro", "")
    cta_ko = script.get("cta_outro_ko", cta_ja)
    pinned = script.get("pinned_comment", "")
    mood = script.get("mood", "exciting")
    bg_query = script.get("bg_search_query", "korea city street")
    top5_info = script.get("top5_info", [])

    st.info(f"주제: **{topic_theme}** | 무드: **{mood}**")

    # ── BGM 매칭 ─────────────────────────────────────────
    _status_update(progress, status_box, 0.18, f"BGM 매칭 중 (무드: {mood})")
    bgm_path = match_bgm_by_mood(config, mood)
    bgm_display = os.path.basename(bgm_path) if bgm_path else "자동생성(ambient)"

    # ── Pexels 배경 영상 다운로드 ─────────────────────────
    bg_video_path: Optional[str] = None
    if config.pexels_api_key:
        _status_update(progress, status_box, 0.22, f"배경 영상 다운로드 중 ({bg_query})")
        vid_dir = os.path.join(config.assets_dir, "bg_videos")
        bg_video_path = fetch_pexels_video(bg_query, config.pexels_api_key, vid_dir, config.width, config.height)
        if bg_video_path:
            st.info(f"배경 영상: {os.path.basename(bg_video_path)}")
        else:
            st.warning("배경 영상 다운로드 실패 — 정적 이미지 배경으로 대체")

    # ── TTS용 텍스트 리스트 ───────────────────────────────
    texts = _script_to_beats(script)

    # ── 에셋 선택 ─────────────────────────────────────────
    mood_to_cat = {"mystery": "shocking", "exciting": "exciting", "informative": "humor"}
    content_category = mood_to_cat.get(mood, "exciting")
    assets: List[str] = []
    for _ in texts:
        asset = pick_asset_by_category(manifest_items, content_category)
        if not asset:
            asset = random.choice(manifest_items)
        assets.append(asset.path)

    # ── Top5 설명 조립 ────────────────────────────────────
    top5_desc = ""
    if top5_info:
        lines = []
        for item in top5_info:
            rank = item.get("rank", "")
            name = item.get("name_ko", "")
            area = item.get("area", "")
            hint = item.get("address_hint", "")
            desc = item.get("desc_ko", "")
            desc_ja = item.get("desc_ja", "")
            lines.append(f"#{rank} {name} ({area}) — {hint}\n  KO: {desc}\n  JA: {desc_ja}")
        top5_desc = "\n".join(lines)

    # ── YouTube 설명 텍스트 ───────────────────────────────
    description_lines = [pinned, ""]
    if top5_info:
        description_lines += ["📍 Top5 정보", top5_desc, ""]
    description_lines += [" ".join(hashtags)]
    description = "\n".join(description_lines)

    # ── 텔레그램 미리보기 (한글+일본어 대본) ─────────────
    body_preview = ""
    max_lines = max(len(body_ja), len(body_ko))
    for i in range(max_lines):
        ja_line = body_ja[i] if i < len(body_ja) else ""
        ko_line = body_ko[i] if i < len(body_ko) else ""
        body_preview += f"  {i+1}. JA: {ja_line}\n      KO: {ko_line}\n"

    top5_preview = ""
    if top5_desc:
        top5_preview = f"\n━━ Top5 정보 ━━\n{top5_desc}\n"

    request_text = (
        f"[ 승인 요청 ] 일본인 타겟 숏츠\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"주제: {topic_theme}\n"
        f"무드: {mood}  |  BGM: {bgm_display}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"[제목 JA] {video_title}\n"
        f"[해시태그] {' '.join(hashtags)}\n\n"
        f"[훅 3초]\nJA: {hook_ja}\nKO: {hook_ko}\n\n"
        f"[본문]\n{body_preview}"
        f"[구독유도]\nJA: {cta_ja}\nKO: {cta_ko}\n\n"
        f"[고정댓글] {pinned}\n"
        f"{top5_preview}"
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

    _status_update(progress, status_box, 0.30, "텔레그램 승인 요청 전송")
    approval_msg_id = send_telegram_approval_request(
        config.telegram_bot_token, config.telegram_admin_chat_id, request_text
    )
    decision = wait_for_approval(config, progress, status_box, approval_message_id=approval_msg_id)
    if decision == "swap":
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, "🔄 교환 처리됨. 새 주제로 다시 생성합니다.")
        _auto_jp_flow(config, progress, status_box, extra_hint=extra_hint)
        return

    # ── TTS 생성 ─────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
    voice_id = pick_voice_id(config.openai_tts_voices)
    _status_update(progress, status_box, 0.50, "TTS 생성 중")
    try:
        tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
    except Exception as tts_err:
        err_msg = f"❌ TTS 생성 실패: {tts_err}"
        st.error(err_msg)
        send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, err_msg)
        return

    # ── 영상 렌더링 ───────────────────────────────────────
    _status_update(progress, status_box, 0.65, "영상 렌더링 중")
    output_path = os.path.join(config.output_dir, f"shorts_{now}.mp4")
    render_video(
        config=config,
        asset_paths=assets,
        texts=texts,
        tts_audio_path=audio_path,
        output_path=output_path,
        bgm_path=bgm_path,
        bgm_volume=config.bgm_volume,
        bg_video_path=bg_video_path,
    )

    # ── 유튜브 업로드 ─────────────────────────────────────
    video_id = ""
    video_url = ""
    if config.enable_youtube_upload:
        _status_update(progress, status_box, 0.85, "유튜브 업로드")
        result = upload_video(
            config=config,
            file_path=output_path,
            title=video_title,
            description=description,
            tags=hashtags,
        )
        video_id = result.get("video_id", "")
        video_url = result.get("video_url", "")
    else:
        _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")

    # ── 로그 기록 ─────────────────────────────────────────
    log_row = {
        "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "title_ja": video_title,
        "topic_theme": topic_theme,
        "hashtags_ja": " ".join(hashtags),
        "mood": mood,
        "pinned_comment": pinned,
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

    summary_text = (
        f"[완료] 일본인 타겟 숏츠\n"
        f"주제: {topic_theme}\n"
        f"제목: {video_title}\n"
        f"무드: {mood}\n"
        f"고정댓글: {pinned}\n"
    )
    if video_url:
        summary_text += f"유튜브: {video_url}"
    else:
        summary_text += f"로컬: {output_path}"
    send_telegram_message(config.telegram_bot_token, config.telegram_admin_chat_id, summary_text)


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
            # 무드별 BGM 디렉토리 (mystery / exciting / informative)
            *[os.path.join(config.assets_dir, "bgm", mood) for mood in BGM_MOOD_CATEGORIES],
            os.path.join(config.assets_dir, "sfx"),
            os.path.join(config.assets_dir, "bg_videos"),
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
        "- `PIXABAY_API_KEY` (BGM 자동 다운로드)\n"
        "- `PEXELS_API_KEY` (이미지 자동 수집)\n"
        "- `SERPAPI_API_KEY` (트렌드 수집)\n"
        "- `OPENAI_VISION_MODEL` (이미지 태그 분석)\n"
        "- `BGM_MODE`, `BGM_VOLUME` (배경음악)\n\n"
        "**BGM 무드 폴더:** `assets/bgm/mystery/`, `assets/bgm/exciting/`, `assets/bgm/informative/`"
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
        st.caption("크롤링 없이 LLM이 매번 새로운 주제를 선정합니다. 무드(mystery/exciting/informative)에 맞게 BGM도 자동 매칭됩니다.")

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
            st.caption(f"주제: **{script.get('topic_theme', '-')}** | 무드: **{script.get('mood', '-')}**")
            video_title_val = st.text_input("유튜브 제목", value=script.get("video_title", ""))
            hashtags_val = st.text_input(
                "해시태그(공백 구분)",
                value=" ".join(script.get("hashtags", [])),
            )

            # 한글/일본어 대본 나란히 표시
            col_ja, col_ko = st.columns(2)
            with col_ja:
                st.markdown("**🇯🇵 일본어 대본**")
                hook_val = st.text_input("3초 훅 (JA)", value=script.get("hook_3_sec", ""), key="hook_ja")
                body_val = st.text_area(
                    "본문 (JA, 줄 구분)",
                    value="\n".join(script.get("body_script", [])),
                    height=200,
                    key="body_ja",
                )
                cta_val = st.text_input("구독 유도 (JA)", value=script.get("cta_outro", ""), key="cta_ja")
            with col_ko:
                st.markdown("**🇰🇷 한국어 대본 (참고용)**")
                st.text_input("3초 훅 (KO)", value=script.get("hook_3_sec_ko", ""), key="hook_ko", disabled=True)
                st.text_area(
                    "본문 (KO)",
                    value="\n".join(script.get("body_script_ko", [])),
                    height=200,
                    key="body_ko",
                    disabled=True,
                )
                st.text_input("구독 유도 (KO)", value=script.get("cta_outro_ko", ""), key="cta_ko", disabled=True)

            pinned_val = st.text_input("고정 댓글", value=script.get("pinned_comment", ""))

            # Top5 정보 표시
            top5_info = script.get("top5_info", [])
            if top5_info:
                st.markdown("**📍 Top5 정보 (설명란 자동 포함)**")
                for item in top5_info:
                    st.markdown(
                        f"**#{item.get('rank')} {item.get('name_ko','')}** "
                        f"({item.get('area','')}) — {item.get('address_hint','')}\n\n"
                        f">{item.get('desc_ko','')} / {item.get('desc_ja','')}"
                    )

            render_button = st.button("영상 만들기")
            if render_button:
                if missing:
                    st.error("필수 API/설정이 누락되어 있어 진행할 수 없습니다.")
                    return
                if not MOVIEPY_AVAILABLE:
                    st.error(f"MoviePy가 설치되지 않았습니다: {MOVIEPY_ERROR}")
                    return
                if not manifest_items:
                    st.error("에셋이 없습니다. 먼저 이미지를 추가하세요.")
                else:
                    body_lines = [l.strip() for l in body_val.split("\n") if l.strip()]
                    texts = [hook_val] + body_lines + ([cta_val] if cta_val else [])
                    if not texts:
                        st.error("렌더링할 문장이 없습니다.")
                    else:
                        mood = script.get("mood", "exciting")
                        _status_update(progress, status_box, 0.15, f"BGM 매칭 중 ({mood})")
                        bgm_path = match_bgm_by_mood(config, mood)

                        mood_to_cat = {"mystery": "shocking", "exciting": "exciting", "informative": "humor"}
                        cat = mood_to_cat.get(mood, "exciting")
                        assets = []
                        for _ in texts:
                            asset = pick_asset_by_category(manifest_items, cat)
                            if not asset:
                                asset = random.choice(manifest_items)
                            assets.append(asset.path)

                        # Pexels 배경 영상 다운로드 시도
                        bg_vid_manual: Optional[str] = None
                        if config.pexels_api_key:
                            _status_update(progress, status_box, 0.25, "배경 영상 다운로드 중")
                            bg_query_m = script.get("bg_search_query", "korea city")
                            vid_dir_m = os.path.join(config.assets_dir, "bg_videos")
                            bg_vid_manual = fetch_pexels_video(bg_query_m, config.pexels_api_key, vid_dir_m, config.width, config.height)

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
                            bg_video_path=bg_vid_manual,
                        )
                        video_id = ""
                        video_url = ""
                        if config.enable_youtube_upload:
                            _status_update(progress, status_box, 0.85, "유튜브 업로드")
                            result = upload_video(
                                config=config,
                                file_path=output_path,
                                title=video_title_val,
                                description=pinned_val + "\n\n" + hashtags_val,
                                tags=hashtags_val.split(),
                            )
                            video_id = result.get("video_id", "")
                            video_url = result.get("video_url", "")
                        else:
                            _status_update(progress, status_box, 0.85, "유튜브 업로드(스킵)")
                        log_row = {
                            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "title_ja": video_title_val,
                            "topic_theme": script.get("topic_theme", ""),
                            "hashtags_ja": hashtags_val,
                            "mood": mood,
                            "pinned_comment": pinned_val,
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

        st.subheader("Pixabay BGM 수동 다운로드")
        if not config.pixabay_api_key:
            st.warning("PIXABAY_API_KEY가 설정되지 않았습니다. `.streamlit/secrets.toml`에 추가하세요.")
        else:
            pixabay_mood = st.selectbox("BGM 무드", mood_labels, key="pixabay_mood")
            pixabay_count = st.slider("다운로드 개수", 1, 5, 3, key="pixabay_count")
            if st.button("Pixabay에서 BGM 다운로드"):
                mood_info = BGM_MOOD_CATEGORIES[pixabay_mood]
                bgm_out_dir = os.path.join(config.assets_dir, "bgm", pixabay_mood)
                query = random.choice(mood_info["pixabay_queries"])
                downloaded_bgms = []
                for _ in range(pixabay_count):
                    path = fetch_bgm_from_pixabay(
                        api_key=config.pixabay_api_key,
                        category=pixabay_mood,
                        output_dir=bgm_out_dir,
                        custom_query=query,
                    )
                    if path:
                        downloaded_bgms.append(path)
                if downloaded_bgms:
                    st.success(f"{len(downloaded_bgms)}개 BGM을 `bgm/{pixabay_mood}/`에 저장했습니다.")
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


def run_batch(count: int, seed: str = "", beats: int = 7) -> None:
    config = load_config()
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        raise RuntimeError("에셋이 없습니다. 먼저 이미지를 추가하세요.")
    for index in range(count):
        script = generate_script_jp(config, extra_hint=seed)
        mood = script.get("mood", "exciting")
        texts = _script_to_beats(script)
        mood_to_cat = {"mystery": "shocking", "exciting": "exciting", "informative": "humor"}
        cat = mood_to_cat.get(mood, "exciting")
        assets = []
        for _ in texts:
            asset = pick_asset_by_category(manifest_items, cat)
            if not asset:
                asset = random.choice(manifest_items)
            assets.append(asset.path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.output_dir, f"tts_{now}_{index}.mp3")
        voice_id = pick_voice_id(config.openai_tts_voices)
        tts_openai(config, "。".join(texts), audio_path, voice=voice_id)
        output_path = os.path.join(config.output_dir, f"shorts_{now}_{index}.mp4")
        bgm_path = match_bgm_by_mood(config, mood)
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
                title=script.get("video_title", ""),
                description=script.get("pinned_comment", "") + "\n\n" + " ".join(script.get("hashtags", [])),
                tags=script.get("hashtags", []),
            )
            video_id = result.get("video_id", "")
            video_url = result.get("video_url", "")
        log_row = {
            "date_jst": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "title_ja": script.get("video_title", ""),
            "topic_theme": script.get("topic_theme", ""),
            "hashtags_ja": " ".join(script.get("hashtags", [])),
            "mood": mood,
            "pinned_comment": script.get("pinned_comment", ""),
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