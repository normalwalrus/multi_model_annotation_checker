"""
add_timestamps.py — Add `start` / `end` to each manifest entry from its filename.

Audio files are named  <name>_<name>_<starttime>-<endtime>.wav  (e.g.
`spk_clip_12.34-56.78.wav`): underscores separate the name parts, and a single
`-` separates start from end. This script reads the NeMo manifest, parses the
start and end times out of each `audio_filepath`, and writes them into the entry
as `start` and `end`.

It needs no model, so it can run on the host or inside any container:

    python code/add_timestamps.py                      # manifest from config.yml
    python code/add_timestamps.py outputs/manifest.json  # explicit manifest
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import DEFAULT_CONFIG, load_config, load_manifest, save_manifest

# Keys are written in this order at the front of each entry (when present).
_PREFERRED_ORDER = ("audio_filepath", "duration", "start", "end", "text")


def _num(token: str):
    """Parse a time token to int (whole) or float (decimal)."""
    if token.lstrip("-").isdigit():
        return int(token)
    return float(token)  # raises ValueError on non-numeric tokens


def parse_times(audio_filepath: str) -> tuple:
    """
    Extract (start, end) from a filename of the form
    <name>_<name>_<starttime>-<endtime>.wav.

    Name parts are joined by `_`; the start and end times are the final
    underscore-separated token, themselves joined by a single `-`. Leading name
    parts containing underscores are therefore handled fine.
    """
    stem = Path(audio_filepath).name
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]
    last = stem.split("_")[-1]              # "<starttime>-<endtime>"
    times = last.split("-")
    if len(times) != 2:
        raise ValueError(f"expected '<start>-<end>' in {stem!r}, got {last!r}")
    return _num(times[0]), _num(times[1])


def _reorder(entry: dict) -> dict:
    ordered = {k: entry[k] for k in _PREFERRED_ORDER if k in entry}
    for k, v in entry.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", nargs="?", default=None,
                        help="Manifest path. Default: <output_folder>/manifest.json "
                             "from config.yml.")
    parser.add_argument("--config", default=None,
                        help="Config path (default: <repo>/config.yml or $TRANSCRIBE_CONFIG).")
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        config_path = Path(args.config or os.environ.get("TRANSCRIBE_CONFIG", DEFAULT_CONFIG))
        if not config_path.exists():
            sys.exit(f"[FATAL] Config not found: {config_path}")
        cfg = load_config(config_path)
        manifest_path = Path(cfg.get("output_folder", "outputs")) / "manifest.json"

    if not manifest_path.exists():
        sys.exit(f"[FATAL] Manifest not found: {manifest_path}")

    entries = load_manifest(manifest_path)
    if not entries:
        sys.exit(f"[FATAL] No entries in {manifest_path}")

    updated = 0
    for key, entry in list(entries.items()):
        try:
            start, end = parse_times(entry["audio_filepath"])
        except (ValueError, KeyError) as e:
            print(f"  [WARN] {entry.get('audio_filepath', key)}: {e} — skipping")
            continue
        entry["start"] = start
        entry["end"] = end
        entries[key] = _reorder(entry)
        updated += 1

    save_manifest(manifest_path, entries)
    print(f"Added start/end to {updated}/{len(entries)} entries -> {manifest_path}")


if __name__ == "__main__":
    main()
