from __future__ import annotations

import json
import os
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    concatenate_videoclips,
    vfx,
)
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gspread
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


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
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


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
) -> Dict[str, Any]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
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
    user_text = (
        f"Seed: {seed_text}\n"
        f"Language: {language}\n"
        f"Beats: {beats_count}\n"
        f"Allowed tags: {tag_list}\n"
        "Style: Japanese, fast, comedic, meme-like. "
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
        raise RuntimeError("ELEVENLABS_API_KEY is missing")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID(S) is missing")
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
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _fit_image_to_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
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
        raise RuntimeError("SHEET_ID is missing")
    if not config.google_service_account_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is missing")
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
        raise RuntimeError("YouTube OAuth credentials are missing")
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


def collect_images_serpapi(
    query: str,
    api_key: str,
    output_dir: str,
    limit: int = 12,
) -> List[str]:
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is missing")
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


def _write_local_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_hashtags(tags: List[str]) -> str:
    return " ".join(tags)


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Shorts Auto Studio", layout="wide")
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

    st.sidebar.title("Shorts Auto Studio")
    page = st.sidebar.radio("Menu", ["Generate", "Assets", "Logs"])

    manifest_items = load_manifest(config.manifest_path)
    all_tags = list_tags(manifest_items)

    if page == "Generate":
        st.header("Generate")
        seed_text = st.text_area("Seed idea or summary", height=120)
        beats_count = st.slider("Beats count", 5, 9, 7)
        tag_filter = st.multiselect("Allowed tags", options=all_tags, default=all_tags[:5])
        generate_button = st.button("Generate Script")

        if generate_button and seed_text:
            script = generate_script(
                config=config,
                seed_text=seed_text,
                beats_count=beats_count,
                allowed_tags=tag_filter or all_tags,
            )
            st.session_state["script"] = script

        script = st.session_state.get("script")
        if script:
            st.subheader("Script")
            title = st.text_input("Title (JA)", value=script.get("title_ja", ""))
            description = st.text_area("Description (JA)", value=script.get("description_ja", ""), height=80)
            hashtags = st.text_input(
                "Hashtags (space separated)",
                value=_format_hashtags(script.get("hashtags_ja", [])),
            )
            beats_df = pd.DataFrame(script.get("beats", []))
            edited_beats = st.data_editor(beats_df, num_rows="fixed")

            render_button = st.button("Render Video")
            if render_button:
                beats = edited_beats.to_dict(orient="records")
                if not beats:
                    st.error("No beats to render")
                elif not manifest_items:
                    st.error("No assets in manifest. Add images first.")
                else:
                    texts = [beat.get("text", "") for beat in beats]
                    tags = [beat.get("tag", "") for beat in beats]
                    assets = []
                    for tag in tags:
                        asset = pick_asset(manifest_items, [tag])
                        if asset:
                            assets.append(asset.path)
                    if not assets:
                        st.error("No assets matched tags")
                    else:
                        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        audio_path = os.path.join(config.output_dir, f"tts_{now}.mp3")
                        voice_id = pick_voice_id(config.elevenlabs_voice_ids)
                        tts_elevenlabs(config, "。".join(texts), audio_path, voice_id=voice_id)
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
                            result = upload_video(
                                config=config,
                                file_path=output_path,
                                title=title,
                                description=description + "\n\n" + hashtags,
                                tags=hashtags.split(),
                            )
                            video_id = result.get("video_id", "")
                            video_url = result.get("video_url", "")
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
                        st.video(output_path)
                        if video_url:
                            st.success(video_url)

    if page == "Assets":
        st.header("Assets")
        upload_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        tag_input = st.text_input("Tags for uploaded images (comma separated)")
        if st.button("Save uploaded assets") and upload_files:
            for file in upload_files:
                save_path = os.path.join(config.assets_dir, "images", file.name)
                with open(save_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
                add_asset(config.manifest_path, save_path, tags_from_text(tag_input))
            st.success("Uploaded assets saved")

        st.subheader("AI Collect (SerpAPI)")
        collect_query = st.text_input("Search query")
        collect_count = st.slider("Number of images", 4, 20, 8)
        if st.button("Collect to Inbox"):
            try:
                inbox_dir = os.path.join(config.assets_dir, "inbox")
                collected = collect_images_serpapi(
                    query=collect_query,
                    api_key=config.serpapi_api_key,
                    output_dir=inbox_dir,
                    limit=collect_count,
                )
                st.success(f"Collected {len(collected)} images to inbox")
            except Exception as exc:
                st.error(str(exc))

        inbox_dir = os.path.join(config.assets_dir, "inbox")
        inbox_files = [
            os.path.join(inbox_dir, f)
            for f in os.listdir(inbox_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if inbox_files:
            st.subheader("Inbox")
            inbox_tags = st.text_input("Tags for selected inbox files (comma separated)")
            for file_path in inbox_files:
                st.image(file_path, width=200)
                if st.button(f"Add to library: {os.path.basename(file_path)}", key=file_path):
                    add_asset(config.manifest_path, file_path, tags_from_text(inbox_tags))
                    st.success("Added to manifest")

        st.subheader("Library")
        if manifest_items:
            selected_tag = st.selectbox("Filter by tag", options=["(all)"] + all_tags)
            filtered = manifest_items if selected_tag == "(all)" else filter_assets_by_tags(manifest_items, [selected_tag])
            for item in filtered:
                st.image(item.path, width=200, caption=",".join(item.tags))
        else:
            st.info("No assets yet")

    if page == "Logs":
        st.header("Logs")
        local_log_path = os.path.join(config.output_dir, "runs.jsonl")
        if os.path.exists(local_log_path):
            with open(local_log_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[-50:]
            records = [json.loads(line) for line in lines]
            st.dataframe(pd.DataFrame(records))
        else:
            st.info("No local logs yet")


def run_batch(count: int, seed: str, beats: int) -> None:
    config = load_config()
    manifest_items = load_manifest(config.manifest_path)
    if not manifest_items:
        raise RuntimeError("No assets in manifest")
    for index in range(count):
        script = generate_script(config, seed, beats_count=beats)
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
        seed=os.getenv("BATCH_SEED", "Japanese meme short"),
        beats=int(os.getenv("BATCH_BEATS", "7")),
    )
else:
    run_streamlit_app()
