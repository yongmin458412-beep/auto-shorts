from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from qq import (
    load_config,
    _should_auto_run,
    _mark_auto_run_done,
    _auto_jp_flow,
    _acquire_run_lock,
    _release_run_lock,
    _maybe_send_ab_report,
    RunTimelineNotifier,
    _set_run_notifier,
)

# 예시 크론 (매일 18:00 실행):
# 0 18 * * * /usr/bin/python3 /Users/himynameisttochi/Documents/auto-shorts-new/scripts/run_daily.py >> /tmp/auto_shorts_cron.log 2>&1


def main() -> None:
    config = load_config()
    if not _should_auto_run(config):
        print("AUTO_RUN_DAILY 조건이 충족되지 않아 종료합니다.")
        return
    if not _acquire_run_lock(config.auto_run_lock_path):
        print("이미 실행 중이라 종료합니다.")
        return
    try:
        notifier = RunTimelineNotifier(config, enabled=True)
        _set_run_notifier(notifier)
        notifier.send("🕙", "스케줄러 깨어남")
        ok = _auto_jp_flow(config, progress=None, status_box=None, extra_hint="", use_streamlit=False)
        if ok:
            _mark_auto_run_done(config)
            print("자동 실행 완료")
        else:
            print("자동 실행 실패")
        _maybe_send_ab_report(config, use_streamlit=False)
    finally:
        _set_run_notifier(None)
        _release_run_lock(config.auto_run_lock_path)


if __name__ == "__main__":
    main()
