"""下载并校验公开来源的 1998 年人民日报标注语料归档。"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "199801.zip"
DEST = RAW / "199801"
SOURCE = "https://raw.githubusercontent.com/chenhui-bupt/PeopleDaily1998/4020ecc558c554c6f3c9bd81a606d2197e28e628/199801.zip"
SHA256 = "17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2"
MEMBERS = [f"199801/1998{month:02d}.txt" for month in range(1, 7)] + ["199801/shengming.doc"]


def digest(path):
    check = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            check.update(chunk)
    return check.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        temporary = RAW / "199801.zip.download"
        try:
            with urllib.request.urlopen(SOURCE, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if digest(temporary) != SHA256:
                raise ValueError("下载文件的 SHA-256 与固定版本不符")
            temporary.replace(ARCHIVE)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(ARCHIVE) != SHA256:
        raise ValueError("本地语料归档的 SHA-256 与固定版本不符")
    DEST.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in MEMBERS:
            if name not in archive.namelist():
                raise ValueError(f"语料归档缺少 {name}")
            target = DEST / Path(name).name
            if not target.exists() or target.stat().st_size != archive.getinfo(name).file_size:
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print(f"语料已校验：{ARCHIVE}；解压 {len(MEMBERS)} 个文件", flush=True)


if __name__ == "__main__":
    main()
