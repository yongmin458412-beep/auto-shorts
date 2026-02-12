#!/usr/bin/env python3
"""
auto-shorts 메인 진입점 (일본 쇼츠 시장 타겟팅 버전)

실행 방법:
  python main.py                    # Streamlit UI 실행
  RUN_BATCH=1 BATCH_COUNT=2 python main.py   # 자동 배치 생성 (2개)
  RUN_BATCH=1 BATCH_SEED="주제" python main.py  # 특정 주제로 배치 생성

또는:
  streamlit run qq.py
"""
import os

if __name__ == "__main__":
    from qq import run_batch, _run_streamlit_app_safe

    if os.getenv("RUN_BATCH") == "1":
        run_batch(
            count=int(os.getenv("BATCH_COUNT", "2")),
            seed=os.getenv("BATCH_SEED", "일본어 밈 숏츠"),
            beats=int(os.getenv("BATCH_BEATS", "7")),
        )
    else:
        _run_streamlit_app_safe()
