"""
README.md 자동 업데이트 스크립트.
- 최근 커밋 섹션 갱신
- 실험 기록 섹션: submission 파일 감지 후 새 행 자동 추가 (점수/메모는 수동 입력)
git commit 후 자동 실행됨.
"""
import subprocess
import re
import glob
import os
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"
SUBMISSION_DIR = REPO_ROOT / "2026-ssafy-ai-15-2"


def get_git_log(n=10):
    result = subprocess.run(
        ["git", "log", f"-{n}", "--pretty=format:| %ad | %s | %an |", "--date=short"],
        capture_output=True, cwd=REPO_ROOT
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def get_submissions():
    """submission_*.csv 파일 목록을 수정 시간 순으로 반환."""
    files = sorted(
        glob.glob(str(SUBMISSION_DIR / "submission_*.csv")),
        key=os.path.getmtime
    )
    result = []
    for f in files:
        name = os.path.basename(f)
        date = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d")
        result.append((name, date))
    return result


def parse_existing_experiment_rows(content):
    """기존 실험 기록 테이블에서 파일명 → (점수, 메모) 매핑 추출."""
    saved = {}
    pattern = r"\|\s*(submission_[^\|]+?)\s*\|[^\|]*\|[^\|]*\|\s*([^\|]*?)\s*\|\s*([^\|]*?)\s*\|"
    for m in re.finditer(pattern, content):
        fname = m.group(1).strip()
        score = m.group(2).strip()
        memo  = m.group(3).strip()
        saved[fname] = (score, memo)
    return saved


def build_experiment_section(submissions, saved):
    rows = []
    for fname, date in submissions:
        score, memo = saved.get(fname, ("-", ""))
        rows.append(f"| {fname} | {date} | | {score} | {memo} |")

    table = "\n".join(rows) if rows else "| (아직 없음) | | | | |"
    return (
        "\n## 실험 기록\n\n"
        "> 점수/메모 열은 직접 채워주세요.\n\n"
        "| 파일명 | 날짜 | 주요 변경 | 점수 | 메모 |\n"
        "|--------|------|-----------|------|------|\n"
        f"{table}\n"
    )


def update_readme():
    content = README.read_text(encoding="utf-8")

    # 1. 실험 기록 섹션 업데이트
    submissions = get_submissions()
    saved = parse_existing_experiment_rows(content)
    exp_section = build_experiment_section(submissions, saved)

    exp_pattern = r"\n## 실험 기록\n[\s\S]*?(?=\n## |\Z)"
    if re.search(exp_pattern, content):
        content = re.sub(exp_pattern, lambda _: exp_section, content)
    else:
        # 최근 커밋 섹션 앞에 삽입, 없으면 파일 끝
        if "\n## 최근 커밋" in content:
            content = content.replace("\n## 최근 커밋", exp_section + "\n## 최근 커밋")
        else:
            content = content.rstrip() + "\n" + exp_section

    # 2. 최근 커밋 섹션 업데이트
    log = get_git_log()
    commit_section = (
        "\n## 최근 커밋\n\n"
        "| 날짜 | 내용 | 작성자 |\n"
        "|------|------|--------|\n"
        f"{log}\n"
    )
    commit_pattern = r"\n## 최근 커밋\n[\s\S]*?(?=\n## |\Z)"
    if re.search(commit_pattern, content):
        content = re.sub(commit_pattern, lambda _: commit_section, content)
    else:
        content = content.rstrip() + "\n" + commit_section

    README.write_text(content, encoding="utf-8")
    print(f"README.md updated. submissions={len(submissions)}, commits={log.count(chr(10))+1}")


if __name__ == "__main__":
    update_readme()
