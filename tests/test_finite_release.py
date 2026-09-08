"""CPU-only finite ArtFID release regression checks."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_finite_release import check

class FiniteReleaseTests(unittest.TestCase):
    def test_current_table_and_evidence(self):
        self.assertEqual(check(ROOT)["status"], "passed")

if __name__ == "__main__":
    unittest.main()
