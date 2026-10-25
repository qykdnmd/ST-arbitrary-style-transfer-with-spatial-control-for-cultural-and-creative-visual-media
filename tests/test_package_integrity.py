"""Exercise corruption, omission and path-validation failure paths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_package.py'
spec = importlib.util.spec_from_file_location('package_checker', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'provenance').mkdir()
        (self.root/'fixture.txt').write_bytes(b'expected bytes')
        self.row = {'path':'fixture.txt','bytes':14,'sha256':hashlib.sha256(b'expected bytes').hexdigest()}
        self.manifest([self.row])
        rows = [{'model_id':f'model_{i}', 'path':f'checkpoints/model_{i}.pth', 'bytes':1,
                 'sha256':hashlib.sha256(b'x').hexdigest()} for i in range(5)]
        (self.root/'provenance/checkpoints.json').write_text(json.dumps(rows))

    def manifest(self, rows):
        (self.root/'MANIFEST.json').write_text(json.dumps({'files':rows}))

    def test_code_pass_does_not_claim_weight_recovery(self):
        r=checker.verify(self.root)
        self.assertTrue(r['pass'])
        self.assertFalse(r['all_five_weights_verified'])

    def test_strict_mode_fails_for_missing_weights(self):
        r=checker.verify(self.root, True)
        self.assertTrue(r['code_data_integrity_pass'])
        self.assertFalse(r['pass'])

    def test_same_size_corruption_fails(self):
        (self.root/'fixture.txt').write_bytes(b'corrupt! bytes')
        self.assertFalse(checker.verify(self.root)['pass'])

    def test_missing_manifest_file_fails(self):
        (self.root/'fixture.txt').unlink()
        self.assertFalse(checker.verify(self.root)['pass'])

    def test_traversal_and_duplicate_are_rejected(self):
        for rows in ([dict(self.row,path='../escape')], [self.row,self.row]):
            self.manifest(rows)
            with self.assertRaises(ValueError):
                checker.verify(self.root)


if __name__ == '__main__':
    unittest.main()
