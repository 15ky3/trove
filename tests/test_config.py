"""Paths, repo-id validation and settings persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import config


class TestValidRepoId:
    @pytest.mark.parametrize(
        "repo_id",
        ["gpt2", "org/name", "Qwen/Qwen3-8B", "a.b/c-d_e", "unsloth/Llama-3.1-8B-GGUF"],
    )
    def test_accepts(self, repo_id):
        assert config.valid_repo_id(repo_id)

    @pytest.mark.parametrize(
        "repo_id",
        [
            "",
            "/name",
            "org/",
            "org/name/extra",
            "../etc",
            "org/../etc",
            ".hidden/name",
            "org/name with space",
            "org/name;rm",
        ],
    )
    def test_rejects(self, repo_id):
        assert not config.valid_repo_id(repo_id)


class TestPaths:
    def test_type_root_per_type(self):
        assert config.type_root("model").name == "models"
        assert config.type_root("dataset").name == "datasets"
        assert config.type_root("space").name == "spaces"

    def test_type_root_rejects_unknown(self):
        with pytest.raises(ValueError):
            config.type_root("nope")

    def test_local_dir_layout(self):
        path = config.local_dir_for("model", "org/name")
        assert path.parent.name == "org"
        assert path.name == "name"
        assert path.parent.parent.name == "models"

    def test_local_dir_without_org(self):
        assert config.local_dir_for("model", "gpt2").name == "gpt2"

    def test_local_dir_rejects_traversal(self):
        with pytest.raises(ValueError):
            config.local_dir_for("model", "../../etc")

    def test_inside_data_dir(self):
        assert config.inside_data_dir(config.DATA_DIR)
        assert config.inside_data_dir(config.DATA_DIR / "models" / "a")
        assert not config.inside_data_dir(Path("/etc"))
        assert not config.inside_data_dir(config.DATA_DIR.parent)

    def test_ensure_dirs_creates_everything(self):
        config.ensure_dirs()
        for repo_type in config.REPO_TYPES:
            assert config.type_root(repo_type).is_dir()
        assert config.CONFIG_DIR.is_dir()


class TestSettings:
    def test_defaults(self):
        data = config.settings.all()
        assert data["max_concurrent"] == config.DEFAULT_SETTINGS["max_concurrent"]
        assert data["max_workers"] == config.DEFAULT_SETTINGS["max_workers"]

    def test_files_at_once_default_is_four(self):
        # Deliberately below huggingface_hub's own 8: it multiplies with
        # max_concurrent and every file in flight costs memory.
        assert config.DEFAULT_SETTINGS["max_workers"] == 4

    @pytest.mark.parametrize(
        ("key", "given", "expected"),
        [
            ("max_concurrent", 0, 1),
            ("max_concurrent", 999, 16),
            ("max_concurrent", 3, 3),
            ("max_workers", 0, 1),
            ("max_workers", 999, 32),
            ("max_workers", 7, 7),
        ],
    )
    def test_numbers_are_clamped(self, key, given, expected):
        assert config.settings.update({key: given})[key] == expected

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (0, 0.0),
            (-50, 0.0),
            (25, 25.0),
            (2.46, 2.5),
            ("40", 40.0),
            (999_999, config.MAX_DOWNLOAD_MBIT),
        ],
    )
    def test_the_speed_limit_is_clamped(self, given, expected):
        assert config.settings.update({"max_download_mbit": given})["max_download_mbit"] == expected

    def test_the_speed_limit_is_off_by_default(self):
        assert config.DEFAULT_SETTINGS["max_download_mbit"] == 0.0

    def test_a_speed_limit_that_is_not_a_number_is_ignored(self):
        config.settings.update({"max_download_mbit": 12})
        config.settings.update({"max_download_mbit": "quick"})
        assert config.settings.get("max_download_mbit") == 12.0

    def test_non_numeric_is_ignored(self):
        before = config.settings.get("max_workers")
        config.settings.update({"max_workers": "many"})
        assert config.settings.get("max_workers") == before

    def test_booleans_are_coerced(self):
        assert config.settings.update({"auto_clear_done": 1})["auto_clear_done"] is True
        assert config.settings.update({"auto_clear_done": ""})["auto_clear_done"] is False

    def test_unknown_keys_are_dropped(self):
        result = config.settings.update({"nonsense": "x"})
        assert "nonsense" not in result

    def test_strings_are_stripped(self):
        assert config.settings.update({"endpoint": "  https://hub  "})["endpoint"] == "https://hub"

    def test_saved_to_disk_and_reloadable(self):
        config.settings.update({"max_workers": 9, "endpoint": "https://mirror"})
        stored = json.loads(config.SETTINGS_FILE.read_text())
        assert stored["max_workers"] == 9
        assert stored["endpoint"] == "https://mirror"

    def test_settings_file_is_not_world_readable(self):
        config.settings.update({"hf_token": "hf_secret"})
        assert config.SETTINGS_FILE.stat().st_mode & 0o077 == 0

    def test_public_masks_the_token(self):
        config.settings.update({"hf_token": "hf_abcdefghijklmnop"})
        public = config.settings.public()
        assert "hf_token" not in public
        assert public["token_set"] is True
        assert "abcdefghijklmnop" not in public["token_hint"]
        assert public["token_hint"].startswith("hf_")

    def test_public_without_a_token(self):
        public = config.settings.public()
        assert public["token_set"] is False
        assert public["token_hint"] == ""


class TestSessionSecret:
    def test_is_stable_across_calls(self):
        assert config.session_secret() == config.session_secret()

    def test_key_file_is_private(self):
        config.session_secret()
        assert config.SECRET_FILE.stat().st_mode & 0o077 == 0
