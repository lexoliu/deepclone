"""Route successful main CI to a release PR or a merged release's publication."""

import json
import os
import subprocess
from pathlib import Path


def route(run, pulls, current_sha, repository):
    if (
        run["conclusion"] != "success"
        or run["event"] != "push"
        or run["head_branch"] != "main"
        or run["head_repository"]["full_name"] != repository
    ):
        return "skip"
    for pr in pulls:
        if (
            pr["merged_at"]
            and pr["merge_commit_sha"] == run["head_sha"]
            and pr["base"]["ref"] == "main"
            and pr["head"]["repo"]["full_name"] == repository
            and pr["head"]["ref"].startswith("release-plz-")
        ):
            return "release"
    return "pr" if current_sha == run["head_sha"] else "skip"


def main():
    run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"]
    repository = os.environ["GITHUB_REPOSITORY"]

    def query(path):
        return json.loads(subprocess.check_output(["gh", "api", path], text=True))

    pulls = query(f"repos/{repository}/commits/{run['head_sha']}/pulls")
    current = query(f"repos/{repository}/commits/main")["sha"]
    kind = route(run, pulls, current, repository)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"kind={kind}\n")


if __name__ == "__main__":
    main()
