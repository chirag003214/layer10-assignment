import time
from typing import Iterator

import requests


class GitHubClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None, repo: str = "ArduPilot/ardupilot"):
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: dict | None = None, accept: str | None = None) -> list | dict:
        url = f"{self.BASE_URL}{path}"
        headers = {"Accept": accept} if accept else {}

        while True:
            resp = self.session.get(url, params=params, headers=headers)

            # Rate limit handling
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))

            if resp.status_code in (403, 429) or remaining == 0:
                sleep_for = max(reset_ts - time.time(), 0) + 1
                print(f"Rate limited — sleeping {sleep_for:.0f}s")
                time.sleep(sleep_for)
                continue

            resp.raise_for_status()
            return resp.json()

    def iter_issues(self, state: str = "all", per_page: int = 100) -> Iterator[dict]:
        path = f"/repos/{self.repo}/issues"
        page = 1
        while True:
            items = self._get(path, params={"state": state, "per_page": per_page, "page": page})
            if not items:
                break
            for issue in items:
                if "pull_request" not in issue:
                    yield issue
            page += 1

    def _paginate(self, path: str, accept: str | None = None) -> list:
        results = []
        page = 1
        while True:
            items = self._get(path, params={"per_page": 100, "page": page}, accept=accept)
            if not items:
                break
            results.extend(items)
            page += 1
        return results

    def get_comments(self, issue_number: int) -> list[dict]:
        path = f"/repos/{self.repo}/issues/{issue_number}/comments"
        return self._paginate(path)

    def get_timeline(self, issue_number: int) -> list[dict]:
        path = f"/repos/{self.repo}/issues/{issue_number}/timeline"
        return self._paginate(path, accept="application/vnd.github.mockingbird-preview+json")
