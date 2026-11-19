#!/usr/bin/env python3
"""
Generic Unsloth GGUF updater.

Features:
- Automatically discover locally installed UD quantizations.
- Discover model names from GGUF filenames.
- Compare with corresponding Unsloth Hugging Face repositories.
- Persistent SHA-256 cache using inode and file metadata.
- Per-file hashing progress.
- --fast: size-only comparison, without hashing.
- --rehash: recompute every SHA-256, ignoring cached values.
- --apply: download and verify changed shards.
- Resume interrupted downloads.
- Hard-link identical shards when possible.
- Preserve originals and Maya preprocessing directories.

Requires: huggingface_hub, hf CLI
Default model directory: /models

Important:
  A repository update does not necessarily indicate a new
  Unsloth Dynamic quantization algorithm version.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


ROOT_DEFAULT = Path("/models")

QUANT_RE = re.compile(r"^UD-[A-Za-z0-9_]+$")

GGUF_RE = re.compile(
    r"^(?P<model>.+)-(?P<quant>UD-[A-Za-z0-9_]+)"
    r"(?:-\d{5}-of-\d{5})?\.gguf$"
)

SHARD_RE = re.compile(
    r"-(\d{5})-of-(\d{5})\.gguf$"
)


@dataclass
class Target:
    model: str
    directory: Path
    repo: str
    quant: str
    local_files: list[Path] = field(default_factory=list)


@dataclass
class RemoteFile:
    name: str
    size: int
    sha256: str

    @property
    def basename(self):
        return Path(self.name).name


def human_size(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def sha256_with_progress(path):
    """Hash one file while reporting read progress."""

    total = path.stat().st_size
    processed = 0
    digest = hashlib.sha256()

    started = time.monotonic()
    last_report = started
    interactive = sys.stdout.isatty()
    interval = 2.0 if interactive else 15.0

    print(
        f"    HASHING {path.name} "
        f"({human_size(total)})",
        flush=True,
    )

    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(16 * 1024 * 1024)
                if not block:
                    break

                digest.update(block)
                processed += len(block)

                now = time.monotonic()
                if now - last_report >= interval:
                    elapsed = max(now - started, 0.001)
                    percent = (
                        processed / total * 100
                        if total else 100.0
                    )
                    rate = processed / elapsed

                    message = (
                        f"      {percent:6.2f}%  "
                        f"{human_size(processed)} / "
                        f"{human_size(total)}  "
                        f"{human_size(rate)}/s"
                    )

                    if interactive:
                        print(
                            "\r" + message.ljust(90),
                            end="",
                            flush=True,
                        )
                    else:
                        print(message, flush=True)

                    last_report = now

    except BaseException:
        if interactive:
            print()
        raise

    elapsed = max(time.monotonic() - started, 0.001)
    rate = processed / elapsed

    if interactive:
        print("\r" + " " * 90, end="\r")

    print(
        f"      100.00%  {human_size(processed)} "
        f"in {elapsed:.1f}s "
        f"({human_size(rate)}/s)",
        flush=True,
    )

    return digest.hexdigest()


class HashCache:
    """
    Persistent SHA-256 cache.

    Cached hashes are reusable only when:
    - Device and inode match.
    - File size matches.
    - mtime_ns matches.
    - ctime_ns matches.

    This assumes ordinary filesystem metadata integrity.
    --rehash bypasses the persistent cache.
    """

    VERSION = 1

    def __init__(self, path, fast=False, rehash=False):
        self.path = path
        self.fast = fast
        self.rehash = rehash

        self.entries = {}
        self.rehashed_this_run = set()

        self.load()

    def load(self):
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text())

            if data.get("version") != self.VERSION:
                print(
                    "Hash cache version changed; "
                    "starting a new cache."
                )
                return

            self.entries = data.get("entries", {})

            print(
                f"Loaded {len(self.entries)} cached hashes"
            )

        except (OSError, ValueError, TypeError) as exc:
            print(
                f"WARNING: Cannot load hash cache: {exc}",
                file=sys.stderr,
            )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": self.VERSION,
            "entries": self.entries,
        }

        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )

        with temporary.open("w") as stream:
            json.dump(data, stream)

        os.replace(temporary, self.path)

    @staticmethod
    def signature(st):
        return [
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        ]

    def matches(self, path, size, digest):
        if not path.is_file():
            return False

        before = path.stat()

        if before.st_size != size:
            return False

        if self.fast:
            return True

        signature = self.signature(before)

        # Inode-based keys allow hard-linked files to share
        # hashes, even when located in different directories.
        key = f"{before.st_dev}:{before.st_ino}"

        cached = self.entries.get(key)

        cache_valid = (
            cached is not None
            and cached.get("signature") == signature
        )

        force = (
            self.rehash
            and key not in self.rehashed_this_run
        )

        if cache_valid and not force:
            result = cached["sha256"] == digest

            print(
                f"    CACHE {'MATCH' if result else 'DIFF '} "
                f"{path.name}"
            )

            return result

        calculated = sha256_with_progress(path)

        after = path.stat()

        if self.signature(after) != signature:
            raise RuntimeError(
                f"File changed while hashing: {path}"
            )

        self.entries[key] = {
            "signature": signature,
            "sha256": calculated,
        }

        self.rehashed_this_run.add(key)

        # Persist after EACH successfully hashed file.
        # An interrupted scan retains completed work.
        self.save()

        result = calculated == digest

        print(
            f"    SHA256 {'MATCH' if result else 'DIFF '} "
            f"{path.name}"
        )

        return result


def repo_for_model(model):
    if model.endswith("-GGUF"):
        return f"unsloth/{model}"

    return f"unsloth/{model}-GGUF"


def discover(root):
    """
    Discover installed UD quantizations.

    Determine model names from actual GGUF filenames,
    not directory names. This also supports directories
    such as Qwen3.6-27B-MTP.
    """

    targets = {}

    for base in sorted(root.iterdir()):
        if not base.is_dir():
            continue

        candidates = list(base.glob("*.gguf"))

        for child in base.iterdir():
            if (
                child.is_dir()
                and QUANT_RE.fullmatch(child.name)
            ):
                candidates.extend(child.glob("*.gguf"))

        for path in sorted(candidates):
            match = GGUF_RE.fullmatch(path.name)

            if match is None:
                continue

            model = match.group("model")
            quant = match.group("quant")

            if (
                path.parent != base
                and path.parent.name != quant
            ):
                continue

            repo = repo_for_model(model)
            key = (str(base), repo, quant)

            if key not in targets:
                targets[key] = Target(
                    model=model,
                    directory=base,
                    repo=repo,
                    quant=quant,
                )

            targets[key].local_files.append(path)

    return sorted(
        targets.values(),
        key=lambda t: (t.model, t.quant),
    )


def remote_metadata(api, target):
    info = api.model_info(
        target.repo,
        files_metadata=True,
    )

    files = []

    for sibling in info.siblings:
        name = sibling.rfilename

        in_quant_dir = (
            name.startswith(target.quant + "/")
            and name.count("/") == 1
        )

        in_root = (
            "/" not in name
            and GGUF_RE.fullmatch(name) is not None
            and GGUF_RE.fullmatch(name).group("model")
                == target.model
            and GGUF_RE.fullmatch(name).group("quant")
                == target.quant
        )

        if not (in_quant_dir or in_root):
            continue

        if not name.endswith(".gguf"):
            continue

        lfs = sibling.lfs

        if isinstance(lfs, dict):
            digest = lfs.get("sha256")
        else:
            digest = getattr(lfs, "sha256", None)

        if not digest or sibling.size is None:
            raise RuntimeError(
                f"Missing LFS SHA-256 metadata: {name}"
            )

        files.append(
            RemoteFile(
                name=name,
                size=sibling.size,
                sha256=digest,
            )
        )

    files.sort(key=lambda f: f.name)

    if not files:
        return info.sha, []

    # Ensure that no two remote paths would collide
    # when placed in the destination directory.
    basenames = [f.basename for f in files]

    if len(basenames) != len(set(basenames)):
        raise RuntimeError(
            "Duplicate remote GGUF basenames"
        )

    shard_matches = [
        SHARD_RE.search(f.basename)
        for f in files
    ]

    if all(m is not None for m in shard_matches):
        totals = {
            int(m.group(2))
            for m in shard_matches
        }

        indices = {
            int(m.group(1))
            for m in shard_matches
        }

        if len(totals) != 1:
            raise RuntimeError(
                "Inconsistent remote shard counts"
            )

        total = totals.pop()

        if indices != set(range(1, total + 1)):
            raise RuntimeError(
                "Incomplete remote shard sequence"
            )

    elif len(files) != 1 or any(shard_matches):
        raise RuntimeError(
            "Unexpected GGUF naming pattern"
        )

    return info.sha, files


def matching_local_file(target, remote, hashes):
    for candidate in target.local_files:
        if candidate.name != remote.basename:
            continue

        if hashes.matches(
            candidate,
            remote.size,
            remote.sha256,
        ):
            return candidate

    return None


def complete(directory, files, hashes):
    if not directory.is_dir():
        return False

    return all(
        hashes.matches(
            directory / f.basename,
            f.size,
            f.sha256,
        )
        for f in files
    )


def process(api, target, hashes, apply, workers):
    print(f"\n{'=' * 68}")
    print(f"{target.model} / {target.quant}")
    print(f"{'=' * 68}")
    print(f"Repository: {target.repo}")

    try:
        revision, files = remote_metadata(api, target)
    except RepositoryNotFoundError:
        print("SKIP: Unsloth repository not found")
        return "skipped"

    if not files:
        print("SKIP: quantization not published")
        return "skipped"

    print(f"Revision: {revision}")
    print(f"Files:    {len(files)}")
    print(
        "Total:    "
        + human_size(sum(f.size for f in files))
    )

    if hashes.fast:
        print(
            "FAST MODE: filenames and sizes only. "
            "Same-size changes cannot be detected."
        )
    elif hashes.rehash:
        print("REHASH MODE: bypassing cached hashes")
    else:
        print("CACHED MODE: reuse verified hashes")

    reuse = {}
    changed = []

    print("\nChecking local shards...", flush=True)

    for f in files:
        source = matching_local_file(
            target, f, hashes
        )

        if source is not None:
            reuse[f.basename] = source
        else:
            changed.append(f)

    if not changed:
        if hashes.fast:
            print(
                "SIZE MATCH: all installed shards "
                "have expected filenames and sizes. "
                "Contents not verified."
            )
        else:
            print(
                "UP TO DATE: all SHA-256 hashes match"
            )

        return "current"

    print(
        f"\nMatching: {len(reuse)}/{len(files)}"
    )
    print(
        "Changed or missing: "
        + human_size(sum(f.size for f in changed))
    )

    for f in changed:
        print(f"  UPDATE: {f.basename}")

    # Revision-specific sidecar; never overwrite the
    # original installed quantization.
    destination = (
        target.directory
        / f"{target.quant}-upstream-{revision[:12]}"
    )

    print(f"\nDestination: {destination}")

    if complete(destination, files, hashes):
        if hashes.fast:
            print(
                "ALREADY PRESENT: destination sizes "
                "match (not checksum-verified)"
            )
        else:
            print(
                "ALREADY INSTALLED: checksums match"
            )
        return "current"

    if destination.exists():
        raise RuntimeError(
            "Destination exists but does not match "
            f"upstream: {destination}"
        )

    if not apply:
        print("DRY RUN: no model files changed")
        return "available"

    # Unique per-repository staging name.
    repo_id = hashlib.sha256(
        target.repo.encode()
    ).hexdigest()[:8]

    stage_root = (
        target.directory
        / ".unsloth-update-stage"
        / f"{target.quant}-{repo_id}-{revision[:12]}"
    )

    staged = stage_root / target.quant

    staged.mkdir(
        parents=True,
        exist_ok=True,
    )

    needed = []

    for f in files:
        dst = staged / f.basename

        # Reuse previously verified staged files.
        if hashes.matches(
            dst, f.size, f.sha256
        ):
            continue

        if dst.exists():
            dst.unlink()

        source = reuse.get(f.basename)

        if source is not None:
            try:
                os.link(source, dst)
                print(f"LINK: {f.basename}")
            except OSError:
                shutil.copy2(source, dst)
                print(f"COPY: {f.basename}")
        else:
            needed.append(f)

    bytes_needed = sum(f.size for f in needed)

    if needed:
        free = shutil.disk_usage(stage_root).free

        if free < bytes_needed + 1024**3:
            raise RuntimeError(
                "Insufficient disk space: "
                f"need {human_size(bytes_needed + 1024**3)}, "
                f"free {human_size(free)}"
            )

        print(
            f"\nDownloading {len(needed)} shards "
            f"({human_size(bytes_needed)})...",
            flush=True,
        )

        command = [
            "hf",
            "download",
            target.repo,
            *[f.name for f in needed],
            "--revision",
            revision,
            "--local-dir",
            str(stage_root),
            "--max-workers",
            str(workers),
        ]

        subprocess.run(
            command,
            check=True,
        )

        # Normalize root-level repository GGUFs into
        # the staging quantization directory.
        for f in needed:
            src = stage_root / f.name
            dst = staged / f.basename

            if src != dst:
                if not src.is_file():
                    raise RuntimeError(
                        f"Downloaded file missing: {src}"
                    )

                os.replace(src, dst)

    print("\nVerifying complete staged model...")

    # All staged files must have valid SHA-256 hashes
    # before the directory becomes available.
    for f in files:
        path = staged / f.basename

        if not hashes.matches(
            path, f.size, f.sha256
        ):
            raise RuntimeError(
                f"Checksum mismatch: {path}"
            )

    manifest = {
        "repository": target.repo,
        "revision": revision,
        "quantization": target.quant,
        "verified": "sha256",
        "files": [
            {
                "filename": f.basename,
                "size": f.size,
                "sha256": f.sha256,
            }
            for f in files
        ],
    }

    (staged / ".unsloth-update.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    if destination.exists():
        raise RuntimeError(
            f"Destination appeared unexpectedly: "
            f"{destination}"
        )

    staged.rename(destination)

    print(f"\nINSTALLED: {destination}")

    shutil.rmtree(stage_root)

    try:
        stage_root.parent.rmdir()
    except OSError:
        pass

    print(
        "Original GGUFs preserved. "
        "Revalidate Maya preprocessing before use."
    )

    return "installed"


def main():
    parser = argparse.ArgumentParser(
        description="Generic Unsloth GGUF updater"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
        help="Model root (default: /models)",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Download verified updates",
    )

    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Filter models by name (repeatable)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    verification = parser.add_mutually_exclusive_group()

    verification.add_argument(
        "--fast",
        action="store_true",
        help="Compare filenames and sizes only",
    )

    verification.add_argument(
        "--rehash",
        action="store_true",
        help="Ignore cache and recompute SHA-256",
    )

    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Override persistent hash cache location",
    )

    args = parser.parse_args()

    if args.fast and args.apply:
        parser.error(
            "--fast cannot be combined with --apply "
            "because size-only checks are insufficient"
        )

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    if args.apply and shutil.which("hf") is None:
        parser.error("'hf' CLI not found in PATH")

    if args.cache_file:
        cache_file = args.cache_file.expanduser()
    else:
        root_id = hashlib.sha256(
            str(root).encode()
        ).hexdigest()[:16]

        cache_file = (
            Path.home()
            / ".cache"
            / "unsloth-updater"
            / f"sha256-{root_id}.json"
        )

    targets = discover(root)

    if args.model:
        targets = [
            t for t in targets
            if any(
                query.lower() in (
                    t.model + " " + t.directory.name
                ).lower()
                for query in args.model
            )
        ]

    if not targets:
        print("No matching UD quantizations")
        return 1

    print(f"Root: {root}")
    print(f"Discovered: {len(targets)} quantizations")

    mode = (
        "FAST" if args.fast
        else "REHASH" if args.rehash
        else "CACHED"
    )

    print(f"Verification: {mode}")
    print(f"Operation: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Hash cache: {cache_file}")

    hashes = HashCache(
        cache_file,
        fast=args.fast,
        rehash=args.rehash,
    )

    api = HfApi()

    results = {
        "current": 0,
        "available": 0,
        "installed": 0,
        "skipped": 0,
        "errors": 0,
    }

    for target in targets:
        try:
            status = process(
                api,
                target,
                hashes,
                args.apply,
                args.workers,
            )

            results[status] += 1

        except KeyboardInterrupt:
            print(
                "\nInterrupted. Completed SHA-256 "
                "hashes remain cached."
            )
            return 130

        except Exception as exc:
            results["errors"] += 1
            print(
                f"ERROR [{target.model}]: {exc}",
                file=sys.stderr,
            )

    print(f"\n{'=' * 68}")
    print("SUMMARY")
    print(f"{'=' * 68}")

    for name, count in results.items():
        print(f"{name:12s} {count}")

    return 1 if results["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

