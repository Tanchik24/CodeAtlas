import argparse
import json
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--out", default="preindexed.json")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    result = requests.post(f"{base}/repos/ensure", json={"github_url": args.repo_url}, timeout=1800)
    result.raise_for_status()
    data = result.json()

    out = {
        "base_url": base,
        "repo_id": int(data["id"]),
        "github_url": data.get("github_url"),
        "project_name": data.get("project_name"),
        "collection_name": data.get("collection_name"),
        "is_indexed": bool(data.get("is_indexed", True)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("OK:", out)

if __name__ == "__main__":
    main()
