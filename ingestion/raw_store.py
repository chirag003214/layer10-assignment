import json
import os
from datetime import datetime
from pathlib import Path


class RawStore:
    def __init__(self, base_dir: Path = Path("data/raw")):
        self.base_dir = base_dir
        self.issues_dir = base_dir / "issues"
        self.comments_dir = base_dir / "comments"
        self.timeline_dir = base_dir / "timeline"
        for d in (self.issues_dir, self.comments_dir, self.timeline_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _write_atomic(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def issue_exists(self, number: int) -> bool:
        return (self.issues_dir / f"{number}.json").exists()

    def save_issue(self, number: int, data: dict) -> None:
        self._write_atomic(self.issues_dir / f"{number}.json", data)

    def save_comments(self, number: int, data: list) -> None:
        self._write_atomic(self.comments_dir / f"{number}.json", data)

    def save_timeline(self, number: int, data: list) -> None:
        self._write_atomic(self.timeline_dir / f"{number}.json", data)

    def load_issue(self, number: int) -> dict:
        return json.loads((self.issues_dir / f"{number}.json").read_text(encoding="utf-8"))

    def load_comments(self, number: int) -> list:
        path = self.comments_dir / f"{number}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def load_timeline(self, number: int) -> list:
        path = self.timeline_dir / f"{number}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def list_all_numbers(self) -> list[int]:
        numbers = [int(p.stem) for p in self.issues_dir.glob("*.json")]
        return sorted(numbers, reverse=True)

    def update_meta(self, **kwargs) -> None:
        path = self.base_dir / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        meta.update({"repo": "ArduPilot/ardupilot", **kwargs})
        self._write_atomic(path, meta)
