"""Exercise download integrity and publication with tiny responses, without HF."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import download_channel


class ChannelDownload(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zs-hf-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "package"
        self.payloads = [b"field fixture", b"mesh settings\n", b""]
        self.records = [
            {"path": "restart_files/init_1/rs6tcf0.f00001", "bytes": len(self.payloads[0]),
             "sha256": hashlib.sha256(self.payloads[0]).hexdigest()},
            {"path": "environment_config.yaml", "bytes": len(self.payloads[1]),
             "git_blob_sha1": hashlib.sha1(b"blob " + str(len(self.payloads[1])).encode() + b"\0" + self.payloads[1]).hexdigest()},
            {"path": ".NEK5000", "bytes": 0,
             "git_blob_sha1": hashlib.sha1(b"blob 0\0").hexdigest()},
        ]
        self.manifest = {"repository": "test/dataset", "revision": "pinned-commit",
                         "environment": "TCFlarge_3D_Re207", "files": self.records}

    def download(self, payloads=None):
        with patch.object(download_channel, "urlopen", side_effect=[io.BytesIO(b) for b in
                          (self.payloads if payloads is None else payloads)]) as request:
            result = download_channel.download_package(self.manifest, self.root)
        return result, request

    def test_full_package_uses_pinned_urls_and_exact_bytes(self):
        results, request = self.download()
        self.assertEqual(len(results), 3)
        for record, payload, call in zip(self.records, self.payloads, request.call_args_list):
            path = self.root / record["path"]
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertIn("/resolve/pinned-commit/TCFlarge_3D_Re207/", call.args[0])

    def test_second_call_verifies_existing_files_without_network(self):
        self.download()
        with patch.object(download_channel, "urlopen") as request:
            result = download_channel.download_package(self.manifest, self.root)
        request.assert_not_called()
        self.assertEqual({r["acquisition"] for r in result}, {"verified_existing"})

    def test_bad_responses_are_not_published(self):
        for payload in [b"short", b"x" * len(self.payloads[0]), b"too long" * 10]:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                self.download([payload])
            self.assertFalse((self.root / self.records[0]["path"]).exists())
            self.assertFalse(list(self.root.rglob(".download-*")))

    def test_corrupt_existing_file_is_not_replaced(self):
        self.download()
        path = self.root / "environment_config.yaml"
        path.write_bytes(b"keep damaged evidence")
        with patch.object(download_channel, "urlopen") as request, self.assertRaises(RuntimeError):
            download_channel.download_package(self.manifest, self.root)
        request.assert_not_called()
        self.assertEqual(path.read_bytes(), b"keep damaged evidence")

    def test_symlink_destination_cannot_write_to_reference(self):
        outside = self.root.parent / "reference"
        outside.mkdir()
        self.root.symlink_to(outside, target_is_directory=True)
        with patch.object(download_channel, "urlopen") as request, self.assertRaisesRegex(RuntimeError, "symlink"):
            download_channel.download_package(self.manifest, self.root)
        request.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_parent_traversal_is_rejected(self):
        self.manifest["files"][0]["path"] = "../reference"
        with patch.object(download_channel, "urlopen") as request, self.assertRaises(ValueError):
            download_channel.download_package(self.manifest, self.root)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
