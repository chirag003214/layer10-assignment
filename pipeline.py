import os
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def run_ingestion(max_issues: int = 1000, force_refetch: bool = False) -> None:
    """
    Phase 1: fetch raw data from GitHub.
    Phase 2: normalize to data/processed/documents.jsonl.
    """
    from ingestion.github_client import GitHubClient
    from ingestion.preprocessor import Preprocessor
    from ingestion.raw_store import RawStore

    client = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
    store = RawStore()
    preprocessor = Preprocessor(store)

    fetched = 0
    skipped = 0
    for issue in tqdm(client.iter_issues(), desc="Fetching issues"):
        number = issue["number"]
        if not force_refetch and store.issue_exists(number):
            skipped += 1
            continue
        store.save_issue(number, issue)
        store.save_comments(number, client.get_comments(number))
        store.save_timeline(number, client.get_timeline(number))
        fetched += 1
        if fetched >= max_issues:
            break

    store.update_meta(total_fetched=fetched + skipped, last_run=datetime.utcnow().isoformat())
    doc_count = preprocessor.process_all()
    print(f"Ingestion complete: {fetched} new, {skipped} cached → {doc_count} documents")


if __name__ == "__main__":
    run_ingestion()
