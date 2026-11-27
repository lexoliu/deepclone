"""Path-filtered CI routing must keep full coverage for every touched surface."""

import unittest

from ci_changes import JOBS, classify


class CiChangeRouting(unittest.TestCase):
    def test_unknown_or_dispatch_runs_every_job(self):
        self.assertEqual(classify(None), {job: True for job in JOBS})

    def test_ci_workflow_edit_runs_every_job(self):
        for trigger in (
            ".github/workflows/ci.yml",
            ".github/scripts/ci_changes.py",
            ".github/scripts/test_ci_changes.py",
        ):
            self.assertEqual(classify([trigger]), {job: True for job in JOBS})

    def test_web_only_change_runs_only_browser(self):
        expected = {job: job == "browser" for job in JOBS}
        self.assertEqual(classify(["web/src/runtime/audio.ts"]), expected)

    def test_export_only_change_runs_only_export(self):
        expected = {job: job == "export" for job in JOBS}
        self.assertEqual(classify(["export/convert.py"]), expected)

    def test_rust_source_change_covers_all_rust_and_browser_lanes(self):
        expected = {
            "dependencies": False,
            "workflows": False,
            "rust_lint": True,
            "rust_macos": True,
            "burn_linux": True,
            "export": False,
            "release_preview": True,
            "browser": True,
        }
        self.assertEqual(classify(["src/burn/layers.rs"]), expected)

    def test_rust_tests_only_skip_release_and_browser_builds(self):
        expected = {
            "dependencies": False,
            "workflows": False,
            "rust_lint": True,
            "rust_macos": True,
            "burn_linux": True,
            "export": False,
            "release_preview": False,
            "browser": False,
        }
        self.assertEqual(classify(["tests/burn_cloning.rs"]), expected)

    def test_cargo_manifest_change_covers_dependencies_and_all_rust_lanes(self):
        expected = {job: job != "export" for job in JOBS}
        self.assertEqual(classify(["Cargo.lock"]), expected)

    def test_release_config_and_scripts_trigger_workflows_and_release_preview(self):
        expected = {job: job in {"workflows", "release_preview"} for job in JOBS}
        self.assertEqual(classify(["dist-workspace.toml"]), expected)
        self.assertEqual(classify([".github/scripts/package_release.py"]), expected)

    def test_web_audit_script_triggers_workflows_and_browser(self):
        expected = {job: job in {"workflows", "browser"} for job in JOBS}
        self.assertEqual(classify([".github/scripts/audit_web.ts"]), expected)

    def test_docs_only_change_runs_no_heavy_jobs(self):
        self.assertEqual(classify(["AGENTS.md"]), {job: False for job in JOBS})


if __name__ == "__main__":
    unittest.main()
