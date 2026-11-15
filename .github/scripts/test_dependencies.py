import copy
import unittest

from check_dependencies import validate


class DependencyPolicy(unittest.TestCase):
    def setUp(self):
        self.active = set()
        self.audit = {
            "vulnerabilities": {
                "list": [
                    {
                        "package": {"name": "zip", "version": "0.6.6"},
                        "advisory": {"id": "RUSTSEC-2026-0429"},
                    }
                ]
            },
            "warnings": {},
        }

    def test_dormant_advisory_cannot_become_active(self):
        validate(self.active, self.audit)
        self.active.add(("zip", "0.6.6"))
        with self.assertRaisesRegex(ValueError, "Unresolved advisory"):
            validate(self.active, self.audit)

    def test_new_advisory_is_not_implicitly_accepted(self):
        audit = copy.deepcopy(self.audit)
        audit["vulnerabilities"]["list"][0]["advisory"]["id"] = "new-advisory"
        with self.assertRaisesRegex(ValueError, "Unresolved advisory"):
            validate(self.active, audit)

    def test_libtorch_is_rejected_even_without_advisories(self):
        self.active.add(("tch", "0.22.0"))
        with self.assertRaisesRegex(ValueError, "LibTorch"):
            validate(self.active, self.audit)
