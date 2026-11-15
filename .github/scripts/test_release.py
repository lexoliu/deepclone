"""Release routing must distinguish tested code from a merged release PR."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from check_release_artifacts import check
from release_assets import collect
from release_gate import route


class ReleaseRouting(unittest.TestCase):
    def test_installer_and_combined_list_must_pin_the_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "binary.tar.xz"
            archive.write_bytes(b"archive fixture")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (root / "binary.tar.xz.sha256").write_text(digest)
            (root / "sha256.sum").write_text(digest)
            (root / "deepclone-installer.sh").write_text(digest)
            check(root)
            (root / "deepclone-installer.sh").write_text("")
            with self.assertRaisesRegex(ValueError, "Installer omits"):
                check(root)
            (root / "sha256.sum").write_text("")
            with self.assertRaisesRegex(ValueError, "Combined checksum"):
                check(root)

    def setUp(self):
        self.repository = "lexoliu/deepclone"
        self.run = {
            "conclusion": "success",
            "event": "push",
            "head_branch": "main",
            "head_repository": {"full_name": self.repository},
            "head_sha": "tested-commit",
        }
        self.pr = {
            "merged_at": "merged",
            "merge_commit_sha": "tested-commit",
            "base": {"ref": "main"},
            "head": {
                "ref": "release-plz-version",
                "repo": {"full_name": self.repository},
            },
        }

    def test_only_the_merged_release_commit_can_publish(self):
        self.assertEqual(
            route(self.run, [self.pr], "newer", self.repository), "release"
        )
        for field, value in [("merged_at", None), ("merge_commit_sha", "other")]:
            pr = copy.deepcopy(self.pr)
            pr[field] = value
            self.assertEqual(route(self.run, [pr], "newer", self.repository), "skip")
        for ref in ["feature", "release-candidate"]:
            pr = copy.deepcopy(self.pr)
            pr["head"]["ref"] = ref
            self.assertEqual(
                route(self.run, [pr], "tested-commit", self.repository), "pr"
            )

    def test_failed_or_untrusted_ci_cannot_create_pr_or_release(self):
        for field, value in [
            ("conclusion", "failure"),
            ("event", "pull_request"),
            ("event", "workflow_dispatch"),
            ("head_branch", "feature"),
            ("head_repository", {"full_name": "fork/deepclone"}),
        ]:
            run = {**self.run, field: value}
            self.assertEqual(
                route(run, [self.pr], "tested-commit", self.repository), "skip"
            )

    def test_stale_normal_ci_cannot_rewind_release_pr(self):
        self.assertEqual(route(self.run, [], "tested-commit", self.repository), "pr")
        self.assertEqual(route(self.run, [], "newer", self.repository), "skip")

    def test_assets_include_only_declared_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary.tar.xz"
            installer = root / "installer.sh"
            binary.write_bytes(b"archive fixture")
            installer.write_text("installer fixture")
            (root / "private.wav").write_bytes(b"private fixture")
            collect(
                [{"upload_files": [str(binary)]}, {"upload_files": [str(installer)]}],
                root / "out",
            )
            self.assertEqual(
                {p.name for p in (root / "out").iterdir()},
                {binary.name, installer.name},
            )


if __name__ == "__main__":
    unittest.main()
