# Shorts Auto Streamlit

일본어 양산형 숏츠를 자동으로 만들기 위한 Streamlit 앱입니다.

## 1) 설치

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

영상 렌더링은 `ffmpeg`가 필요합니다. macOS에서는 `brew install ffmpeg`로 설치할 수 있습니다.

## 2) 시크릿 설정

`.streamlit/secrets.toml.example`를 복사해 `.streamlit/secrets.toml`로 만들고 값을 채우세요.

필수:
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `SHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `FONT_PATH` (일본어 폰트 경로)

선택:
- `YOUTUBE_*` (업로드 활성화)
- `SERPAPI_API_KEY` (이미지 자동 수집)

## 3) 실행

```
streamlit run qq.py
```

## 4) 자동 실행(하루 2개)

```
RUN_BATCH=1 BATCH_COUNT=2 BATCH_SEED="일본어 숏츠 주제" python qq.py
```

이 명령을 크론이나 스케줄러에 등록하면 완전 자동화가 됩니다.

## 5) 에셋 준비

- `Assets` 탭에서 이미지 업로드 후 태그를 붙입니다.
- 태그는 `shock, laugh, awkward, wow` 같은 감정 키워드로 정리하면 자동 매칭이 잘 됩니다.
- `AI Collect` 기능은 SerpAPI 키가 있을 때만 작동합니다.

## 6) 유튜브 업로드

`YOUTUBE_UPLOAD_ENABLED=true`로 설정하면 자동 업로드합니다.
설정이 없으면 로컬 MP4만 생성합니다.
