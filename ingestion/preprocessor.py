from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ingestion.raw_store import RawStore

KEEP_EVENTS = {
    "closed", "reopened", "labeled", "unlabeled", "assigned", "unassigned",
    "renamed", "milestoned", "demilestoned", "referenced", "cross-referenced",
    "locked", "unlocked", "merged",
}


class RawDocument(BaseModel):
    doc_id: str
    doc_type: Literal["issue", "comment", "timeline_event"]
    issue_number: int
    source_url: str
    author: str
    created_at: str
    updated_at: str | None = None

    # issue-only
    title: str | None = None
    body: str | None = None
    state: str | None = None
    state_reason: str | None = None
    labels: list[str] = []
    assignees: list[str] = []
    milestone: str | None = None
    closed_at: str | None = None

    # comment-only
    comment_id: int | None = None

    # timeline-only
    event_id: int | None = None
    event_type: str | None = None
    event_detail: dict = {}


class Preprocessor:
    def __init__(self, store: RawStore, out_dir: Path = Path("data/processed")):
        self.store = store
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _make_issue_doc(self, issue: dict) -> RawDocument:
        number = issue["number"]
        return RawDocument(
            doc_id=f"ardupilot/ardupilot#{number}",
            doc_type="issue",
            issue_number=number,
            source_url=issue.get("html_url", ""),
            author=(issue.get("user") or {}).get("login", ""),
            created_at=issue.get("created_at", ""),
            updated_at=issue.get("updated_at"),
            title=issue.get("title"),
            body=issue.get("body"),
            state=issue.get("state"),
            state_reason=issue.get("state_reason"),
            labels=[lbl["name"] for lbl in issue.get("labels", [])],
            assignees=[a["login"] for a in issue.get("assignees", [])],
            milestone=(issue.get("milestone") or {}).get("title"),
            closed_at=issue.get("closed_at"),
        )

    def _make_comment_doc(self, issue_number: int, comment: dict) -> RawDocument:
        cid = comment["id"]
        return RawDocument(
            doc_id=f"ardupilot/ardupilot#{issue_number}/c/{cid}",
            doc_type="comment",
            issue_number=issue_number,
            source_url=comment.get("html_url", ""),
            author=(comment.get("user") or {}).get("login", ""),
            created_at=comment.get("created_at", ""),
            updated_at=comment.get("updated_at"),
            comment_id=cid,
            body=comment.get("body"),
        )

    def _make_timeline_doc(self, issue_number: int, event: dict) -> RawDocument | None:
        event_type = event.get("event")
        if event_type not in KEEP_EVENTS:
            return None

        detail: dict = {}
        if event_type in ("labeled", "unlabeled"):
            detail = {"label": (event.get("label") or {}).get("name")}
        elif event_type == "renamed":
            detail = {"from": (event.get("rename") or {}).get("from"), "to": (event.get("rename") or {}).get("to")}
        elif event_type in ("assigned", "unassigned"):
            detail = {"assignee": (event.get("assignee") or {}).get("login")}
        elif event_type in ("milestoned", "demilestoned"):
            detail = {"milestone": (event.get("milestone") or {}).get("title")}
        elif event_type == "cross-referenced":
            src = event.get("source") or {}
            issue_ref = src.get("issue") or {}
            detail = {"referenced_url": issue_ref.get("html_url")}

        eid = event.get("id")
        actor = (event.get("actor") or {}).get("login", "")
        return RawDocument(
            doc_id=f"ardupilot/ardupilot#{issue_number}/t/{eid}",
            doc_type="timeline_event",
            issue_number=issue_number,
            source_url=event.get("url", ""),
            author=actor,
            created_at=event.get("created_at") or event.get("submitted_at", ""),
            updated_at=None,
            event_id=eid,
            event_type=event_type,
            event_detail=detail,
        )

    def process_issue(self, number: int) -> list[RawDocument]:
        docs: list[RawDocument] = []
        docs.append(self._make_issue_doc(self.store.load_issue(number)))
        for comment in self.store.load_comments(number):
            docs.append(self._make_comment_doc(number, comment))
        for event in self.store.load_timeline(number):
            doc = self._make_timeline_doc(number, event)
            if doc is not None:
                docs.append(doc)
        return docs

    def process_all(self, numbers: list[int] | None = None) -> int:
        if numbers is None:
            numbers = self.store.list_all_numbers()
        out_path = self.out_dir / "documents.jsonl"
        total = 0
        with out_path.open("w", encoding="utf-8") as f:
            for number in numbers:
                for doc in self.process_issue(number):
                    f.write(doc.model_dump_json() + "\n")
                    total += 1
        return total


if __name__ == "__main__":
    from ingestion.raw_store import RawStore
    store = RawStore()
    p = Preprocessor(store)
    total = p.process_all()
    print(f"Wrote {total} documents to data/processed/documents.jsonl")
