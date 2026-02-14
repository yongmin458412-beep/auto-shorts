# Shorts Auto Streamlit

일본어 양산형 숏츠를 자동으로 만들기 위한 Streamlit 앱입니다.

## 1) 설치

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

영상 렌더링은 `ffmpeg`가 필요합니다. macOS에서는 `brew install ffmpeg`로 설치할 수 있습니다.
Streamlit Cloud에서는 `packages.txt`에 `ffmpeg` 한 줄을 넣어 설치됩니다.
Streamlit Cloud에서 파이썬 버전 문제로 `moviepy`가 설치 실패하면 `runtime.txt`에 `python-3.11`를 지정하세요.

## 2) 시크릿 설정

`.streamlit/secrets.toml.example`를 복사해 `.streamlit/secrets.toml`로 만들고 값을 채우세요.

필수:
- `OPENAI_API_KEY`
- `OPENAI_VISION_MODEL` (이미지 태그 분석용, 비워두면 `OPENAI_MODEL` 사용)
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID` 또는 `ELEVENLABS_VOICE_IDS`
- `SHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON` (JSON은 TOML에서 `'''`로 감싸서 붙여넣기)
- `FONT_PATH` (일본어 폰트 경로)

자동 승인(텔레그램):
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ADMIN_CHAT_ID`
- `TELEGRAM_TIMEOUT_SEC`

선택:
- `YOUTUBE_*` (업로드 활성화)
- `PINTEREST_ACCESS_TOKEN`, `PINTEREST_AD_ACCOUNT_ID` (Pinterest 이미지 검색)
- `IMAGE_SEARCH_SOURCE_PRIORITY` (예: `pinterest,serpapi,pixabay,pexels,wikimedia`)
- `FREEPIK_API_KEY` (레거시 호환)
- `SERPAPI_API_KEY` (트렌드 수집)
- `PEXELS_API_KEY` (트렌드 이미지 자동 수집)
- `BBOOM_LIST_URL` (네이버 뿜 인기글 목록 URL)
- `BBOOM_MAX_FETCH`
- `USED_LINKS_PATH`
- `TREND_QUERY`, `TREND_MAX_RESULTS`
- `JA_DIALECT_STYLE` (예: 関西弁, 博多弁)
- `BGM_MODE`, `BGM_VOLUME` (배경음악 자동 선택)
- `APPROVE_KEYWORDS`, `SWAP_KEYWORDS`
- `GOOGLE_SERVICE_ACCOUNT_JSON_B64` (대체 옵션)

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

- `에셋` 탭에서 이미지 업로드 후 태그를 붙입니다.
- 태그는 `shock, laugh, awkward, wow` 같은 감정 키워드로 정리하면 자동 매칭이 잘 됩니다.
- `AI 이미지 수집(SerpAPI)` 기능은 SerpAPI 키가 있을 때만 작동합니다.
- `일본 트렌드 자동 수집(Pexels)`은 Pexels 키가 필요합니다.
- 인박스에서 체크 후 **`선택한 짤 저장`** 버튼으로 라이브러리에 넣습니다.
- 인박스에서 `페페 기본 태그 추가`를 켜면 자동으로 `pepe` 태그가 붙습니다.
- 인박스에서 `상황 프리셋`을 고르면 자동으로 태그가 추가됩니다. (충격/반전, 웃김/비꼼 등)
- 인박스에서 `AI 태그 자동 적용`을 켜면 이미지 분석 태그가 자동 추가됩니다.
- 라이브러리에서 `전체 선택`, `삭제`, `AI 태그 분석/분류 저장`이 가능합니다.
- `BGM 업로드`에서 오디오를 올리면 자동 선택에 사용됩니다.
- `BGM_MODE=trending`이면 `bgm/trending` 폴더의 음악을 우선 사용합니다.
- `URL 목록으로 이미지 수집`은 사용 권한이 있는 이미지 URL만 넣어야 합니다.
- `네이버 블로그 포스트 이미지 수집`은 퍼가기가 허용된 포스트만 사용하세요.

## 6) 유튜브 업로드

`YOUTUBE_UPLOAD_ENABLED=true`로 설정하면 자동 업로드합니다.
설정이 없으면 로컬 MP4만 생성합니다.

## 7) 진행상황 확인

- `생성` 탭에서 스크립트 생성 → TTS → 렌더 → 업로드 순서가 진행되며,
  상단 진행바와 상태 메시지로 진행상황을 확인할 수 있습니다.

## 8) 유튜브 리프레시 토큰 발급(앱 내)

- 사이드바 `토큰` 탭에서 OAuth 정보를 입력하고 승인 URL을 생성합니다.
- 승인 후 code를 붙여넣으면 refresh token이 표시됩니다.

## 9) 네이버 뿜 자동 생성(승인 포함)

- `생성` 탭의 **“뿜 인기글로 자동 생성 시작”** 버튼을 누르면,
  뿜 인기글을 수집 → 텔레그램 승인 요청 → 승인 시 제작/업로드까지 자동 진행합니다.
- 10분 내 응답이 없으면 자동 승인됩니다.
- 완료 후 텔레그램으로 요약과 유튜브 링크가 전송됩니다.
