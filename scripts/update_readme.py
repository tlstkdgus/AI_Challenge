"""
README.md의 '최근 커밋' 섹션을 최신 git log로 자동 업데이트하는 스크립트.
git commit 후 자동 실행됨.
"""
import subprocess
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"

def get_git_log(n=10):
    result = subprocess.run(
        ["git", "log", f"-{n}", "--pretty=format:| %ad | %s | %an |", "--date=short"],
        capture_output=True, cwd=REPO_ROOT
    )
    return result.stdout.decode("utf-8", errors="replace").strip()

def update_readme():
    content = README.read_text(encoding="utf-8")
    log = get_git_log()

    section = (
        "\n## 최근 커밋\n\n"
        "| 날짜 | 내용 | 작성자 |\n"
        "|------|------|--------|\n"
        f"{log}\n"
    )

    # 기존 섹션 교체 또는 파일 끝에 추가
    pattern = r"\n## 최근 커밋\n[\s\S]*?(?=\n## |\Z)"
    if re.search(pattern, content):
        content = re.sub(pattern, lambda _: section, content)
    else:
        content = content.rstrip() + "\n" + section

    README.write_text(content, encoding="utf-8")
    print(f"README.md updated with latest {log.count(chr(10))+1} commits.")

if __name__ == "__main__":
    update_readme()
