#!/usr/bin/env python3
"""Executable CLI for semantic version derivation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Pattern, Sequence, Union

import yaml


DEFAULT_CONFIG = {
    "tag_prefix": "v",
    "tag_regex": r"^v?(?P<version_core>\d+\.\d+\.\d+)(?:-(?P<pre_release>[0-9A-Za-z.-]+))?(?:\+(?P<build_metadata>[0-9A-Za-z.-]+))?$",
    "root_commit": None,
    "default_version": {
        "major": 0,
        "minor": 0,
        "patch": 0,
    },
    "versioning": {
        "minor_strategy": "merges",
        "default_branch": "main",
    },
    "pre_release": {
        "sanitize_pattern": r"[^0-9A-Za-z.-]",
        "separator": ".",
    },
    "build_metadata": {
        "date_time": True,
    },
}

DEFAULT_CONFIG_PATH = Path(".semantic-version/config.yaml")
REQUIRED_TAG_REGEX_GROUPS = ("version_core", "pre_release", "build_metadata")
ENVIRONMENT_VARIABLE_PREFIX = "SEMVER_"
ENVIRONMENT_VARIABLE_NAMES = {
    "tag_regex": "SEMVER_TAG_REGEX",
    "default_version.major": "SEMVER_DEFAULT_VERSION_MAJOR",
    "default_version.minor": "SEMVER_DEFAULT_VERSION_MINOR",
    "default_version.patch": "SEMVER_DEFAULT_VERSION_PATCH",
    "versioning.minor_strategy": "SEMVER_VERSIONING_MINOR_STRATEGY",
    "versioning.default_branch": "SEMVER_VERSIONING_DEFAULT_BRANCH",
    "pre_release.sanitize_pattern": "SEMVER_PRE_RELEASE_SANITIZE_PATTERN",
    "pre_release.separator": "SEMVER_PRE_RELEASE_SEPARATOR",
    "root_commit": "SEMVER_ROOT_COMMIT",
    "tag_prefix": "SEMVER_TAG_PREFIX",
    "build_metadata.date_time": "SEMVER_BUILD_METADATA_DATE_TIME",
}


@dataclass(frozen=True)
class SemanticTag:
    """Normalized semantic tag fields."""

    original_tag: str
    normalized_tag: str
    version_core: str
    major: int
    minor: int
    patch: int
    pre_release: Optional[str]
    build_metadata: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        """Return a deterministic dictionary representation."""

        return {
            "original_tag": self.original_tag,
            "normalized_tag": self.normalized_tag,
            "version_core": self.version_core,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "pre_release": self.pre_release,
            "build_metadata": self.build_metadata,
        }


class GitCommandError(RuntimeError):
    """Raised when a git subprocess command fails."""

    def __init__(
        self,
        command_args: Sequence[str],
        cwd: Optional[Path],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Store deterministic git failure context."""

        super().__init__()
        self.command_args = tuple(command_args)
        self.cwd = cwd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        """Return an actionable failure message with command context."""

        command_text = _format_git_command(self.command_args)
        location = str(self.cwd) if self.cwd is not None else str(Path.cwd())
        details = [
            f"Git command failed with exit code {self.returncode}: {command_text}",
            f"Repository location: {location}",
        ]
        if self.stderr:
            details.append(f"stderr: {self.stderr}")
        if self.stdout:
            details.append(f"stdout: {self.stdout}")
        if not self.stderr and not self.stdout:
            details.append("Git returned no stdout or stderr output.")
        return "\n".join(details)


def run_git(args: Sequence[str], cwd: Optional[Union[str, Path]] = None) -> str:
    """Run a git command deterministically and return trimmed stdout."""

    normalized_args = _normalize_git_args(args)
    resolved_cwd = Path(cwd) if cwd is not None else None

    try:
        completed = subprocess.run(
            ["git", *normalized_args],
            cwd=resolved_cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except OSError as exc:
        command_text = _format_git_command(normalized_args)
        location = str(resolved_cwd) if resolved_cwd is not None else str(Path.cwd())
        raise RuntimeError(
            f"Unable to execute git command {command_text} in {location}: {exc}"
        ) from exc

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise GitCommandError(
            command_args=normalized_args,
            cwd=resolved_cwd,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return stdout


def get_current_branch(cwd: Optional[Union[str, Path]] = None) -> str:
    """Return the currently checked out branch name for HEAD."""

    return run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def is_working_tree_dirty(cwd: Optional[Union[str, Path]] = None) -> bool:
    """Return whether the working tree has tracked or untracked changes."""

    return bool(run_git(["status", "--porcelain"], cwd=cwd))


def get_short_commit_hash(cwd: Optional[Union[str, Path]] = None) -> str:
    """Return the abbreviated commit hash for HEAD."""

    return run_git(["rev-parse", "--short", "HEAD"], cwd=cwd)


def get_merge_base(
    default_branch: str,
    cwd: Optional[Union[str, Path]] = None,
) -> str:
    """Return the merge-base between HEAD and the configured default branch."""

    normalized_branch = _require_non_empty_string(default_branch, "default_branch")
    return run_git(["merge-base", "HEAD", normalized_branch], cwd=cwd)


def count_commits(
    commit_range: str,
    cwd: Optional[Union[str, Path]] = None,
) -> int:
    """Return the number of commits in a git revision range."""

    normalized_range = _require_non_empty_string(commit_range, "commit_range")
    output = run_git(["rev-list", normalized_range, "--count"], cwd=cwd)
    return _parse_git_int(output, f"commit count for range '{normalized_range}'")


def count_merge_commits(
    commit_range: str,
    cwd: Optional[Union[str, Path]] = None,
) -> int:
    """Return the number of merge commits in a git revision range."""

    normalized_range = _require_non_empty_string(commit_range, "commit_range")
    output = run_git(["rev-list", "--merges", normalized_range, "--count"], cwd=cwd)
    return _parse_git_int(output, f"merge commit count for range '{normalized_range}'")


def list_tags_on_head(cwd: Optional[Union[str, Path]] = None) -> tuple[str, ...]:
    """Return all tag names that point at HEAD in sorted order."""

    return _list_tags_pointing_at("HEAD", cwd=cwd)


def list_nearest_reachable_semantic_tag_candidates(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    target_ref: str = "HEAD",
) -> tuple[SemanticTag, ...]:
    """Return semantic tags on the nearest reachable tagged commit from a target ref."""

    active_config = validate_config(DEFAULT_CONFIG if config is None else config)
    root_commit = _resolve_root_commit(
        active_config.get("root_commit"),
        descendant_ref=target_ref,
        cwd=cwd,
    )
    for commit in _list_commits_from_ref(
        head_ref=target_ref,
        root_commit=root_commit,
        cwd=cwd,
    ):
        semantic_tags = _parse_semantic_tags(
            _list_tags_pointing_at(commit, cwd=cwd),
            config=active_config,
        )
        if semantic_tags:
            return semantic_tags

    return ()


def load_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    defaults: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load and validate semantic version configuration."""

    resolved_defaults = deepcopy(DEFAULT_CONFIG if defaults is None else defaults)
    validate_config(resolved_defaults, source="defaults")

    merged_config = resolved_defaults
    resolved_path = Path(config_path)
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as handle:
            loaded_config = yaml.safe_load(handle)

        if loaded_config is None:
            loaded_config = {}

        if not isinstance(loaded_config, Mapping):
            raise ValueError(
                f"Config file '{resolved_path}' must contain a mapping at the top level."
            )

        merged_config = deep_merge(merged_config, loaded_config, path="config")
        validate_config(merged_config, source=str(resolved_path))

    env_config = load_env_config(env=env)
    if env_config:
        merged_config = deep_merge(merged_config, env_config, path="environment")
        validate_config(merged_config, source="environment")

    return merged_config


def load_env_config(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Load semantic version overrides from environment variables."""

    resolved_env = os.environ if env is None else env
    override_config: Dict[str, Any] = {}

    env_specs = (
        (ENVIRONMENT_VARIABLE_NAMES["tag_regex"], ("tag_regex",), _parse_env_string),
        (ENVIRONMENT_VARIABLE_NAMES["default_version.major"], ("default_version", "major"), _parse_env_non_negative_int),
        (ENVIRONMENT_VARIABLE_NAMES["default_version.minor"], ("default_version", "minor"), _parse_env_non_negative_int),
        (ENVIRONMENT_VARIABLE_NAMES["default_version.patch"], ("default_version", "patch"), _parse_env_non_negative_int),
        (ENVIRONMENT_VARIABLE_NAMES["versioning.minor_strategy"], ("versioning", "minor_strategy"), _parse_env_minor_strategy),
        (ENVIRONMENT_VARIABLE_NAMES["versioning.default_branch"], ("versioning", "default_branch"), _parse_env_string),
        (ENVIRONMENT_VARIABLE_NAMES["pre_release.sanitize_pattern"], ("pre_release", "sanitize_pattern"), _parse_env_string),
        (ENVIRONMENT_VARIABLE_NAMES["pre_release.separator"], ("pre_release", "separator"), _parse_env_string),
        (ENVIRONMENT_VARIABLE_NAMES["root_commit"], ("root_commit",), _parse_env_non_empty_string),
        (ENVIRONMENT_VARIABLE_NAMES["tag_prefix"], ("tag_prefix",), _parse_env_string),
        (ENVIRONMENT_VARIABLE_NAMES["build_metadata.date_time"], ("build_metadata", "date_time"), _parse_env_bool),
    )

    for env_name, key_path, parser in env_specs:
        raw_value = resolved_env.get(env_name)
        if raw_value is None:
            continue
        _set_nested_config_value(override_config, key_path, parser(raw_value, env_name))

    return override_config


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    path: str = "config",
) -> Dict[str, Any]:
    """Deep-merge override values over base values."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        current_path = f"{path}.{key}"
        base_value = merged.get(key)

        if isinstance(base_value, Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"{current_path} must be a mapping.")
            merged[key] = deep_merge(base_value, value, path=current_path)
            continue

        merged[key] = deepcopy(value)

    return merged


def validate_config(config: Mapping[str, Any], source: str = "config") -> Dict[str, Any]:
    """Validate semantic version configuration shape and values."""

    if not isinstance(config, Mapping):
        raise ValueError(f"{source} must be a mapping.")

    _require_string(config, "tag_prefix", source)
    tag_regex = _require_string(config, "tag_regex", source)
    _require_optional_string(config.get("root_commit"), f"{source}.root_commit")
    default_version = _require_mapping(config, "default_version", source)
    versioning = _require_mapping(config, "versioning", source)
    pre_release = _require_mapping(config, "pre_release", source)
    build_metadata = _require_mapping(config, "build_metadata", source)

    _require_non_negative_int(default_version, "major", f"{source}.default_version")
    _require_non_negative_int(default_version, "minor", f"{source}.default_version")
    _require_non_negative_int(default_version, "patch", f"{source}.default_version")

    minor_strategy = _require_string(versioning, "minor_strategy", f"{source}.versioning")
    if minor_strategy not in {"commits", "merges"}:
        raise ValueError(
            f"{source}.versioning.minor_strategy must be either 'commits' or 'merges'."
        )
    _require_string(versioning, "default_branch", f"{source}.versioning")

    _require_string(pre_release, "sanitize_pattern", f"{source}.pre_release")
    _require_string(pre_release, "separator", f"{source}.pre_release")
    _require_bool(build_metadata, "date_time", f"{source}.build_metadata")

    compile_tag_regex(tag_regex)
    return deepcopy(dict(config))


def compile_tag_regex(tag_regex: str) -> Pattern[str]:
    """Compile and validate the configured tag regex."""

    if not isinstance(tag_regex, str):
        raise ValueError("config.tag_regex must be a string.")

    try:
        compiled_pattern = re.compile(tag_regex)
    except re.error as exc:
        raise ValueError(f"Invalid tag_regex: {exc}") from exc

    missing_groups = [
        group_name
        for group_name in REQUIRED_TAG_REGEX_GROUPS
        if group_name not in compiled_pattern.groupindex
    ]
    if missing_groups:
        missing_group_list = ", ".join(missing_groups)
        raise ValueError(
            "tag_regex must define named groups: "
            f"{', '.join(REQUIRED_TAG_REGEX_GROUPS)}. Missing: {missing_group_list}"
        )

    return compiled_pattern


def parse_semantic_tag(
    tag: str,
    config: Optional[Mapping[str, Any]] = None,
) -> SemanticTag:
    """Parse a semantic tag into normalized semantic version fields."""

    normalized_input = _normalize_tag_input(tag)
    active_config = validate_config(DEFAULT_CONFIG if config is None else config)
    tag_pattern = compile_tag_regex(active_config["tag_regex"])
    match = _match_tag(tag_pattern, normalized_input)

    if match is None:
        raise ValueError(
            f"Tag '{tag}' does not match the configured semantic tag pattern."
        )

    version_core = match.group("version_core")
    major, minor, patch = _parse_version_core(version_core, tag)
    pre_release = _optional_group(match.group("pre_release"))
    build_metadata = _optional_group(match.group("build_metadata"))
    normalized_tag = build_normalized_tag(
        major=major,
        minor=minor,
        patch=patch,
        pre_release=pre_release,
        build_metadata=build_metadata,
    )

    return SemanticTag(
        original_tag=tag,
        normalized_tag=normalized_tag,
        version_core=version_core,
        major=major,
        minor=minor,
        patch=patch,
        pre_release=pre_release,
        build_metadata=build_metadata,
    )


def normalize_semantic_tag(
    tag: str,
    config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the canonical semantic version string for a tag."""

    return parse_semantic_tag(tag, config=config).normalized_tag


def build_normalized_tag(
    major: int,
    minor: int,
    patch: int,
    pre_release: Optional[str] = None,
    build_metadata: Optional[str] = None,
) -> str:
    """Build a canonical semantic version string without a leading v."""

    normalized_tag = f"{major}.{minor}.{patch}"
    if pre_release:
        normalized_tag = f"{normalized_tag}-{pre_release}"
    if build_metadata:
        normalized_tag = f"{normalized_tag}+{build_metadata}"
    return normalized_tag


def sanitize_branch_name(branch_name: str, config: Mapping[str, Any]) -> str:
    """Sanitize a branch name for deterministic pre-release identifiers."""

    normalized_branch_name = _require_non_empty_string(branch_name, "branch_name")
    pre_release = _require_mapping(config, "pre_release", "config")
    separator = _require_string(pre_release, "separator", "config.pre_release")
    sanitize_pattern = _require_string(
        pre_release,
        "sanitize_pattern",
        "config.pre_release",
    )

    sanitized_branch_name = normalized_branch_name.replace("/", separator)
    sanitized_branch_name = re.sub(sanitize_pattern, separator, sanitized_branch_name)
    repeated_separator_pattern = re.escape(separator) + r"{2,}"
    sanitized_branch_name = re.sub(
        repeated_separator_pattern,
        separator,
        sanitized_branch_name,
    )
    return sanitized_branch_name.strip(separator)


def resolve_semver(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve a deterministic semantic version for HEAD."""

    active_config = validate_config(DEFAULT_CONFIG if config is None else config)
    root_commit = _resolve_root_commit(active_config.get("root_commit"), cwd=cwd)
    short_hash = get_short_commit_hash(cwd=cwd)
    build_metadata = _build_commit_build_metadata(short_hash, active_config)
    is_dirty = is_working_tree_dirty(cwd=cwd)

    head_tags = _parse_semantic_tags(list_tags_on_head(cwd=cwd), config=active_config)
    if head_tags:
        head_tag = head_tags[0]
        pre_release = "dirty" if is_dirty else None
        return build_normalized_tag(
            major=head_tag.major,
            minor=head_tag.minor,
            patch=head_tag.patch,
            pre_release=pre_release,
            build_metadata=build_metadata,
        )

    current_branch = get_current_branch(cwd=cwd)
    sanitized_branch = sanitize_branch_name(current_branch, active_config)
    default_branch = active_config["versioning"]["default_branch"]
    merge_base = get_merge_base(default_branch, cwd=cwd)
    distance_from_default_branch = _count_commits_since_ref(
        merge_base,
        root_commit=root_commit,
        cwd=cwd,
    )

    base_tag = _select_base_semantic_tag(active_config, cwd=cwd)
    default_version = active_config["default_version"]
    minor_strategy = active_config["versioning"]["minor_strategy"]

    if minor_strategy not in {"commits", "merges"}:
        raise ValueError(
            "config.versioning.minor_strategy must be either 'commits' or 'merges'."
        )

    base_major = base_tag.major if base_tag is not None else default_version["major"]
    base_minor = base_tag.minor if base_tag is not None else default_version["minor"]

    commits_since_base = _count_commits_since_ref(
        base_tag.original_tag if base_tag is not None else None,
        root_commit=root_commit,
        cwd=cwd,
    )
    merge_commits_since_base = _count_merge_commits_since_ref(
        base_tag.original_tag if base_tag is not None else None,
        root_commit=root_commit,
        cwd=cwd,
    )

    if minor_strategy == "commits":
        derived_minor = base_minor + commits_since_base
        derived_patch = 0
    else:
        derived_minor = base_minor + merge_commits_since_base
        if current_branch == default_branch:
            last_merge_commit = _get_last_merge_commit(root_commit=root_commit, cwd=cwd)
            if last_merge_commit is None:
                derived_patch = commits_since_base
            else:
                derived_patch = _count_commits_since_ref(
                    last_merge_commit,
                    root_commit=root_commit,
                    cwd=cwd,
                )
        else:
            derived_patch = commits_since_base

    pre_release = _build_derived_pre_release(
        current_branch=current_branch,
        default_branch=default_branch,
        sanitized_branch=sanitized_branch,
        distance_from_default_branch=distance_from_default_branch,
        is_dirty=is_dirty,
    )
    return build_normalized_tag(
        major=base_major,
        minor=derived_minor,
        patch=derived_patch,
        pre_release=pre_release,
        build_metadata=build_metadata,
    )


def resolve_compute_tag(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    target_ref: Optional[str] = None,
) -> str:
    """Resolve the computed release tag core for HEAD or a specific target ref."""

    active_config = validate_config(DEFAULT_CONFIG if config is None else config)
    resolved_target_ref = resolve_target_commit(
        target_ref=target_ref,
        cwd=cwd,
        ref_name="compute-tag",
    )
    head_tags = _parse_semantic_tags(
        _list_tags_pointing_at(resolved_target_ref, cwd=cwd),
        config=active_config,
    )
    if head_tags:
        version_core = head_tags[0].version_core
        return f"{active_config['tag_prefix']}{version_core}"

    root_commit = _resolve_root_commit(
        active_config.get("root_commit"),
        descendant_ref=resolved_target_ref,
        cwd=cwd,
    )
    base_tag = _select_base_semantic_tag(
        active_config,
        cwd=cwd,
        target_ref=resolved_target_ref,
    )
    default_version = active_config["default_version"]
    minor_strategy = active_config["versioning"]["minor_strategy"]
    default_branch = active_config["versioning"]["default_branch"]

    base_major = base_tag.major if base_tag is not None else default_version["major"]
    base_minor = base_tag.minor if base_tag is not None else default_version["minor"]

    commits_since_base = _count_commits_since_ref(
        base_tag.original_tag if base_tag is not None else None,
        head_ref=resolved_target_ref,
        root_commit=root_commit,
        cwd=cwd,
    )
    merge_commits_since_base = _count_merge_commits_since_ref(
        base_tag.original_tag if base_tag is not None else None,
        head_ref=resolved_target_ref,
        root_commit=root_commit,
        cwd=cwd,
    )

    if minor_strategy == "commits":
        derived_minor = base_minor + commits_since_base
        derived_patch = 0
    else:
        derived_minor = base_minor + merge_commits_since_base
        if _is_commit_on_first_parent_chain(
            resolved_target_ref,
            default_branch,
            cwd=cwd,
        ):
            last_merge_commit = _get_last_merge_commit(
                head_ref=resolved_target_ref,
                root_commit=root_commit,
                cwd=cwd,
            )
            if last_merge_commit is None:
                derived_patch = commits_since_base
            else:
                derived_patch = _count_commits_since_ref(
                    last_merge_commit,
                    head_ref=resolved_target_ref,
                    root_commit=root_commit,
                    cwd=cwd,
                )
        else:
            derived_patch = commits_since_base

    version_core = f"{base_major}.{derived_minor}.{_clamp_distance(derived_patch)}"
    return f"{active_config['tag_prefix']}{version_core}"


def resolve_target_commit(
    target_ref: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
    *,
    ref_name: str = "target",
) -> str:
    """Resolve an optional target ref to a commit SHA."""

    normalized_target_ref = "HEAD" if target_ref is None else _require_non_empty_string(target_ref, ref_name)
    try:
        return run_git(
            ["rev-parse", "--verify", f"{normalized_target_ref}^{{commit}}"],
            cwd=cwd,
        )
    except GitCommandError as exc:
        raise ValueError(
            f"{ref_name} ref '{normalized_target_ref}' could not be resolved to a commit or tag."
        ) from exc


def create_lightweight_tag(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    target_ref: Optional[str] = None,
) -> str:
    """Create a lightweight tag at a target commit using the computed tag name."""

    computed_tag, target_commit = plan_lightweight_tag(
        config=config,
        cwd=cwd,
        target_ref=target_ref,
    )

    try:
        run_git(["tag", computed_tag, target_commit], cwd=cwd)
    except GitCommandError as exc:
        if _tag_exists(computed_tag, cwd=cwd):
            raise ValueError(f"tag '{computed_tag}' already exists.") from exc
        raise

    return computed_tag


def plan_lightweight_tag(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    target_ref: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve the computed tag name and target commit without creating a tag."""

    target_commit = resolve_target_commit(target_ref=target_ref, cwd=cwd, ref_name="tag")
    computed_tag = resolve_compute_tag(config=config, cwd=cwd, target_ref=target_commit)
    if _tag_exists(computed_tag, cwd=cwd):
        raise ValueError(f"tag '{computed_tag}' already exists.")
    return computed_tag, target_commit


def resolve_next_tag(
    config: Optional[Mapping[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    target_ref: Optional[str] = None,
) -> str:
    """Backward-compatible alias for resolve_compute_tag."""

    return resolve_compute_tag(config=config, cwd=cwd, target_ref=target_ref)


def build_cli_override_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration overrides sourced from CLI arguments."""

    override_config: Dict[str, Any] = {}

    if args.tag_regex is not None:
        override_config["tag_regex"] = args.tag_regex
    if args.tag_prefix is not None:
        override_config["tag_prefix"] = args.tag_prefix
    if args.root_commit is not None:
        override_config["root_commit"] = args.root_commit

    default_version: Dict[str, Any] = {}
    if args.default_major is not None:
        default_version["major"] = args.default_major
    if args.default_minor is not None:
        default_version["minor"] = args.default_minor
    if args.default_patch is not None:
        default_version["patch"] = args.default_patch
    if default_version:
        override_config["default_version"] = default_version

    versioning: Dict[str, Any] = {}
    if args.minor_strategy is not None:
        versioning["minor_strategy"] = args.minor_strategy
    if args.default_branch is not None:
        versioning["default_branch"] = args.default_branch
    if versioning:
        override_config["versioning"] = versioning

    pre_release: Dict[str, Any] = {}
    if args.sanitize_pattern is not None:
        pre_release["sanitize_pattern"] = args.sanitize_pattern
    if args.separator is not None:
        pre_release["separator"] = args.separator
    if pre_release:
        override_config["pre_release"] = pre_release

    build_metadata: Dict[str, Any] = {}
    if args.build_metadata_date_time is not None:
        build_metadata["date_time"] = args.build_metadata_date_time
    if build_metadata:
        override_config["build_metadata"] = build_metadata

    return override_config


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""

    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed_value


def _match_tag(pattern: Pattern[str], tag: str) -> Optional[re.Match[str]]:
    """Match a tag, tolerating an optional leading v."""

    candidates = [tag]
    alternate_tag = tag[1:] if tag.startswith("v") else f"v{tag}"
    if alternate_tag != tag:
        candidates.append(alternate_tag)

    for candidate in candidates:
        match = pattern.fullmatch(candidate)
        if match is not None:
            return match

    return None


def _extract_version_core(version: str) -> str:
    """Extract the major.minor.patch core from a normalized semver string."""

    normalized_version = _require_non_empty_string(version, "version")
    match = re.fullmatch(r"(?P<core>\d+\.\d+\.\d+)(?:[-+].*)?", normalized_version)
    if match is None:
        raise RuntimeError(f"Resolved semantic version has invalid format: {version!r}")
    return match.group("core")


def _select_base_semantic_tag(
    config: Mapping[str, Any],
    cwd: Optional[Union[str, Path]] = None,
    target_ref: str = "HEAD",
) -> Optional[SemanticTag]:
    """Return the nearest reachable semantic tag, if one exists."""

    candidates = list_nearest_reachable_semantic_tag_candidates(
        config=config,
        cwd=cwd,
        target_ref=target_ref,
    )
    return candidates[0] if candidates else None


def _normalize_tag_input(tag: str) -> str:
    """Validate and normalize raw tag input."""

    if not isinstance(tag, str):
        raise ValueError("Tag must be a string.")

    normalized_tag = tag.strip()
    if not normalized_tag:
        raise ValueError("Tag must be a non-empty string.")

    return normalized_tag


def _count_commits_since_ref(
    git_ref: Optional[str],
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> int:
    """Count commits from an optional ref to a target ref."""

    return _count_revisions(
        exclude_refs=() if git_ref is None else (git_ref,),
        head_ref=head_ref,
        root_commit=root_commit,
        cwd=cwd,
    )


def _count_merge_commits_since_ref(
    git_ref: Optional[str],
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> int:
    """Count merge commits from an optional ref to a target ref."""

    return _count_revisions(
        exclude_refs=() if git_ref is None else (git_ref,),
        head_ref=head_ref,
        root_commit=root_commit,
        cwd=cwd,
        merges_only=True,
    )


def _get_last_merge_commit(
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Return the most recent reachable merge commit, if present."""

    output = run_git(
        [
            "rev-list",
            "--merges",
            "-n",
            "1",
            *_build_target_revision_args(
                head_ref=head_ref,
                root_commit=root_commit,
                cwd=cwd,
            ),
        ],
        cwd=cwd,
    )
    return output or None


def _count_revisions(
    exclude_refs: Sequence[str] = (),
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
    merges_only: bool = False,
) -> int:
    """Count revisions reachable from a target ref with optional exclusions."""

    args = ["rev-list", "--count"]
    if merges_only:
        args.append("--merges")
    args.extend(
        _build_target_revision_args(
            exclude_refs=exclude_refs,
            head_ref=head_ref,
            root_commit=root_commit,
            cwd=cwd,
        )
    )
    output = run_git(args, cwd=cwd)
    description = "merge commit count" if merges_only else "commit count"
    return _parse_git_int(output, description)


def _build_target_revision_args(
    exclude_refs: Sequence[str] = (),
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> list[str]:
    """Build git rev-list arguments for a target ref and optional exclusions."""

    revision_args = [_require_non_empty_string(head_ref, "head_ref")]
    for git_ref in exclude_refs:
        revision_args.append(f"^{_require_non_empty_string(git_ref, 'git_ref')}")
    if root_commit is not None:
        for parent_ref in _get_parent_refs(root_commit, cwd=cwd):
            revision_args.append(f"^{parent_ref}")
    return revision_args


def _list_commits_from_ref(
    head_ref: str = "HEAD",
    root_commit: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> tuple[str, ...]:
    """List commits reachable from a target ref honoring root bounds."""

    commit_history = run_git(
        [
            "rev-list",
            *_build_target_revision_args(
                head_ref=head_ref,
                root_commit=root_commit,
                cwd=cwd,
            ),
        ],
        cwd=cwd,
    )
    if not commit_history:
        return ()

    return tuple(commit for commit in commit_history.splitlines() if commit)


def _resolve_root_commit(
    root_commit: Optional[str],
    descendant_ref: str = "HEAD",
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolve and validate an optional root commit bound."""

    if root_commit is None:
        return None

    normalized_root_commit = _require_non_empty_string(root_commit, "root_commit")
    try:
        resolved_root_commit = run_git(
            ["rev-parse", "--verify", f"{normalized_root_commit}^{{commit}}"],
            cwd=cwd,
        )
    except GitCommandError as exc:
        raise ValueError(
            f"root_commit '{normalized_root_commit}' could not be resolved to a commit or tag."
        ) from exc

    if not _is_ancestor(resolved_root_commit, descendant_ref, cwd=cwd):
        raise ValueError(
            f"root_commit '{normalized_root_commit}' must be an ancestor of {descendant_ref}."
        )

    return resolved_root_commit


def _is_ancestor(
    ancestor_ref: str,
    descendant_ref: str,
    cwd: Optional[Union[str, Path]] = None,
) -> bool:
    """Return whether one ref is an ancestor of another."""

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_ref, descendant_ref],
        cwd=Path(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise GitCommandError(
        command_args=("merge-base", "--is-ancestor", ancestor_ref, descendant_ref),
        cwd=Path(cwd) if cwd is not None else None,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _is_commit_on_first_parent_chain(
    target_ref: str,
    branch_ref: str,
    cwd: Optional[Union[str, Path]] = None,
) -> bool:
    """Return whether a target commit appears on a branch's first-parent history."""

    try:
        resolved_target_ref = run_git(
            ["rev-parse", "--verify", f"{_require_non_empty_string(target_ref, 'target_ref')}^{{commit}}"],
            cwd=cwd,
        )
        first_parent_history = run_git(
            ["rev-list", "--first-parent", _require_non_empty_string(branch_ref, "branch_ref")],
            cwd=cwd,
        )
    except GitCommandError:
        return False

    return resolved_target_ref in {line for line in first_parent_history.splitlines() if line}


def _get_parent_refs(
    git_ref: str,
    cwd: Optional[Union[str, Path]] = None,
) -> tuple[str, ...]:
    """Return parent refs for a commit."""

    output = run_git(["rev-list", "--parents", "-n", "1", git_ref], cwd=cwd)
    parts = [part for part in output.split() if part]
    if not parts:
        raise RuntimeError(f"Git returned no commit data for ref '{git_ref}'.")
    return tuple(parts[1:])


def _clamp_distance(distance: int) -> int:
    """Clamp any derived distance to zero or greater."""

    return max(0, distance)


def _parse_version_core(version_core: str, original_tag: str) -> tuple[int, int, int]:
    """Parse the semantic version core into integer components."""

    parts = version_core.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Tag '{original_tag}' has invalid version_core '{version_core}'."
        )

    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(
            f"Tag '{original_tag}' has a non-integer version_core '{version_core}'."
        ) from exc


def _optional_group(value: Optional[str]) -> Optional[str]:
    """Normalize empty regex groups to None."""

    return value if value else None


def _build_commit_build_metadata(short_hash: str, config: Mapping[str, Any]) -> str:
    """Build commit metadata with an optional local datetime suffix."""

    build_metadata_config = _require_mapping(config, "build_metadata", "config")
    include_date_time = _require_bool(
        build_metadata_config,
        "date_time",
        "config.build_metadata",
    )
    if not include_date_time:
        return short_hash

    return f"{short_hash}.{_current_build_metadata_datetime()}"


def _current_build_metadata_datetime() -> str:
    """Return local time formatted for semver build metadata."""

    return datetime.now().strftime("%Y%m%d.%H%M")


def _build_derived_pre_release(
    current_branch: str,
    default_branch: str,
    sanitized_branch: str,
    distance_from_default_branch: int,
    is_dirty: bool,
) -> Optional[str]:
    """Build derived pre-release text for non-tagged versions."""

    parts: list[str] = []
    if not (
        current_branch == default_branch and distance_from_default_branch == 0
    ):
        parts.append(f"{sanitized_branch}.{distance_from_default_branch}")
    if is_dirty:
        parts.append("dirty")
    return ".".join(parts) or None


def _normalize_git_args(args: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize git subprocess arguments."""

    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise ValueError("Git arguments must be provided as a sequence of strings.")

    normalized_args = tuple(args)
    if not normalized_args:
        raise ValueError("Git arguments must contain at least one entry.")
    if any(not isinstance(arg, str) for arg in normalized_args):
        raise ValueError("Each git argument must be a string.")

    return normalized_args


def _format_git_command(args: Sequence[str]) -> str:
    """Format a git command for error messages."""

    return "git " + " ".join(repr(arg) for arg in args)


def _list_tags_pointing_at(
    git_ref: str,
    cwd: Optional[Union[str, Path]] = None,
) -> tuple[str, ...]:
    """Return sorted tags that point at a specific git ref."""

    normalized_ref = _require_non_empty_string(git_ref, "git_ref")
    output = run_git(["tag", "--points-at", normalized_ref], cwd=cwd)
    if not output:
        return ()

    return tuple(sorted(line for line in output.splitlines() if line))


def _tag_exists(
    tag_name: str,
    cwd: Optional[Union[str, Path]] = None,
) -> bool:
    """Return whether a tag ref already exists."""

    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{_require_non_empty_string(tag_name, 'tag_name')}"],
        cwd=Path(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise GitCommandError(
        command_args=("rev-parse", "--verify", "--quiet", f"refs/tags/{tag_name}"),
        cwd=Path(cwd) if cwd is not None else None,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _parse_semantic_tags(
    tags: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[SemanticTag, ...]:
    """Parse and sort semantic tags, skipping non-semantic candidates."""

    semantic_tags = []
    for tag in tags:
        try:
            semantic_tags.append(parse_semantic_tag(tag, config=config))
        except ValueError:
            continue

    semantic_tags.sort(key=lambda tag: (tag.normalized_tag, tag.original_tag))
    return tuple(semantic_tags)


def _parse_git_int(raw_value: str, description: str) -> int:
    """Parse an integer returned by git and raise actionable errors on failure."""

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Git returned a non-integer {description}: {raw_value!r}"
        ) from exc

    if parsed_value < 0:
        raise RuntimeError(
            f"Git returned a negative {description}: {raw_value!r}"
        )

    return parsed_value


def _set_nested_config_value(
    config: Dict[str, Any],
    key_path: Sequence[str],
    value: Any,
) -> None:
    """Assign a nested configuration value using a key path."""

    current: Dict[str, Any] = config
    for key in key_path[:-1]:
        existing_value = current.get(key)
        if not isinstance(existing_value, dict):
            existing_value = {}
            current[key] = existing_value
        current = existing_value
    current[key_path[-1]] = value


def _parse_env_string(raw_value: str, env_name: str) -> str:
    """Return an environment string value unchanged."""

    if not isinstance(raw_value, str):
        raise ValueError(f"Environment variable {env_name} must be a string.")
    return raw_value


def _parse_env_non_empty_string(raw_value: str, env_name: str) -> str:
    """Parse a required non-empty environment string value."""

    try:
        return _require_non_empty_string(raw_value, f"Environment variable {env_name}")
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a non-empty string.") from exc


def _parse_env_non_negative_int(raw_value: str, env_name: str) -> int:
    """Parse a non-negative integer from an environment variable."""

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a non-negative integer. Got: {raw_value!r}."
        ) from exc

    if parsed_value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be a non-negative integer. Got: {raw_value!r}."
        )
    return parsed_value


def _parse_env_minor_strategy(raw_value: str, env_name: str) -> str:
    """Parse a valid minor strategy from an environment variable."""

    if raw_value not in {"commits", "merges"}:
        raise ValueError(
            f"Environment variable {env_name} must be one of: commits, merges. Got: {raw_value!r}."
        )
    return raw_value


def _parse_env_bool(raw_value: str, env_name: str) -> bool:
    """Parse a boolean from an environment variable."""

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "Environment variable "
        f"{env_name} must be a boolean (one of: 1, 0, true, false, yes, no, on, off). "
        f"Got: {raw_value!r}."
    )


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    """Require a nested mapping value."""

    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a mapping.")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    """Require a string value."""

    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string.")
    return value


def _require_non_empty_string(value: Any, context: str) -> str:
    """Require a non-empty string value."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be a non-empty string.")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{context} must be a non-empty string.")

    return normalized_value


def _require_non_negative_int(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> int:
    """Require a non-negative integer value."""

    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer.")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    """Require a boolean value."""

    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean.")
    return value


def _require_optional_string(value: Any, context: str) -> Optional[str]:
    """Require an optional string value when present."""

    if value is None:
        return None
    return _require_non_empty_string(value, context)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for semantic version derivation."""

    parser = argparse.ArgumentParser(description="Derive a semantic version for the current repository.")
    parser.add_argument("--tag-regex")
    parser.add_argument("--tag-prefix")
    parser.add_argument("--default-major", type=non_negative_int)
    parser.add_argument("--default-minor", type=non_negative_int)
    parser.add_argument("--default-patch", type=non_negative_int)
    parser.add_argument("--minor-strategy", choices=("commits", "merges"))
    parser.add_argument("--default-branch")
    parser.add_argument("--root-commit")
    parser.add_argument("--sanitize-pattern")
    parser.add_argument("--separator")
    date_time_group = parser.add_mutually_exclusive_group()
    date_time_group.add_argument(
        "--date-time",
        dest="build_metadata_date_time",
        action="store_true",
        default=None,
        help="Include the local datetime suffix in build metadata.",
    )
    date_time_group.add_argument(
        "--no-date-time",
        dest="build_metadata_date_time",
        action="store_false",
        default=None,
        help="Suppress the local datetime suffix in build metadata.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compute-tag",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="sha_or_ref",
        help="Print the computed release tag core without pre-release or build metadata.",
    )
    mode_group.add_argument(
        "--tag",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="sha_or_ref",
        help="Create a lightweight tag using the computed release tag core.",
    )
    mode_group.add_argument(
        "--next-tag",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="sha_or_ref",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the tag that would be created without creating it. Requires --tag.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the semantic version CLI."""

    repo_root = Path.cwd()

    try:
        args = parse_args(argv)
        if args.dry_run and args.tag is None:
            raise ValueError("--dry-run requires --tag.")

        config = load_config(repo_root / DEFAULT_CONFIG_PATH)
        override_config = build_cli_override_config(args)
        if override_config:
            config = deep_merge(config, override_config, path="cli")
        config = validate_config(config)

        compute_tag_target = args.compute_tag if args.compute_tag is not None else args.next_tag
        if compute_tag_target is not None:
            print(resolve_compute_tag(config=config, cwd=repo_root, target_ref=compute_tag_target))
        elif args.tag is not None:
            if args.dry_run:
                computed_tag, target_commit = plan_lightweight_tag(
                    config=config,
                    cwd=repo_root,
                    target_ref=args.tag,
                )
                print(f"Would create tag {computed_tag} at {target_commit}")
            else:
                print(create_lightweight_tag(config=config, cwd=repo_root, target_ref=args.tag))
        else:
            print(resolve_semver(config=config, cwd=repo_root))
    except Exception as exc:
        print(f"Error deriving semantic version: {exc}", file=sys.stderr)
        return 1

    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "ENVIRONMENT_VARIABLE_NAMES",
    "ENVIRONMENT_VARIABLE_PREFIX",
    "GitCommandError",
    "SemanticTag",
    "build_normalized_tag",
    "build_cli_override_config",
    "count_commits",
    "count_merge_commits",
    "compile_tag_regex",
    "deep_merge",
    "create_lightweight_tag",
    "get_current_branch",
    "get_merge_base",
    "get_short_commit_hash",
    "is_working_tree_dirty",
    "list_nearest_reachable_semantic_tag_candidates",
    "list_tags_on_head",
    "load_config",
    "load_env_config",
    "main",
    "normalize_semantic_tag",
    "non_negative_int",
    "parse_args",
    "parse_semantic_tag",
    "plan_lightweight_tag",
    "resolve_compute_tag",
    "resolve_target_commit",
    "resolve_next_tag",
    "resolve_semver",
    "run_git",
    "sanitize_branch_name",
    "validate_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
