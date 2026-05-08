from pathlib import Path


def test_healthbot_schedule_is_weekly_wednesday():
    workflow = Path(".github/workflows/healthbot.yml").read_text(encoding="utf-8")
    assert "cron: '0 0 * * 3'" in workflow
