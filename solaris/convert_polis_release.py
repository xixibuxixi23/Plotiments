#!/usr/bin/env python3
"""Build a Solaris Duet view of a Polis two-player release.

Videos are symlinked. Only compact per-agent action JSON files are generated.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(
    "/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/"
    "polis_two_player_fixed_skins_complete_20260917_360p"
)
DEFAULT_OUTPUT = Path(
    "/public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/solaris/data"
)


def read_manifest(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bool_value(keys, index):
    return bool(keys[index])


def convert_agent(keys, mouse, key_index, agent):
    actions = []
    for step in range(keys.shape[0]):
        row = keys[step, agent]
        camera = mouse[step, agent]
        action = {
            "inventory": bool_value(row, key_index["inventory"]),
            "ESC": False,
            "forward": bool_value(row, key_index["forward"]),
            "back": bool_value(row, key_index["backward"]),
            "left": bool_value(row, key_index["left"]),
            "right": bool_value(row, key_index["right"]),
            "jump": bool_value(row, key_index["jump"]),
            "sneak": bool_value(row, key_index["sneak"]),
            "sprint": bool_value(row, key_index["aux1"]),
            "swapHands": False,
            "attack": bool_value(row, key_index["dig"]),
            "use": bool_value(row, key_index["place"]),
            "pickItem": False,
            "drop": bool_value(row, key_index["drop"]),
            "camera": [float(camera[0]), float(camera[1])],
        }
        for slot in range(1, 10):
            action[f"hotbar.{slot}"] = bool_value(row, key_index[f"slot_{slot}"])
        actions.append({"renderTime": step * 1000, "action": action})
    return actions


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    temporary.replace(path)


def ensure_symlink(source, destination):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite: {destination}")
    os.symlink(source.resolve(), destination)


def convert_split(records, destination, limit):
    destination.mkdir(parents=True, exist_ok=True)
    converted = []
    skipped_short = 0
    for record in records:
        if limit and len(converted) >= limit:
            break
        episode = Path(record["source_path"])
        with np.load(episode / "data.npz", allow_pickle=True) as payload:
            keys = payload["action_keys"]
            mouse = payload["action_mouse"]
        with (episode / "training_metadata.json").open() as handle:
            action_names = json.load(handle)["action_keys"]

        if keys.ndim != 3 or keys.shape[1] < 2 or keys.shape[2] != 21:
            raise ValueError(f"Unexpected action_keys shape {keys.shape}: {episode}")
        if (
            mouse.ndim != 3
            or mouse.shape[0] != keys.shape[0]
            or mouse.shape[1] < 2
            or mouse.shape[2] != 2
        ):
            raise ValueError(f"Unexpected action_mouse shape {mouse.shape}: {episode}")
        if keys.shape[0] < 33:
            skipped_short += 1
            continue

        key_index = {name: index for index, name in enumerate(action_names)}
        required = {
            "forward", "backward", "left", "right", "jump", "aux1", "sneak",
            "dig", "place", "drop", "inventory", *[f"slot_{i}" for i in range(1, 10)]
        }
        missing = required.difference(key_index)
        if missing:
            raise ValueError(f"Missing action keys {sorted(missing)}: {episode}")

        stem = f"polis_{len(converted):08d}"
        video_a = destination / f"{stem}_Alpha_instance_000.mp4"
        video_b = destination / f"{stem}_Bravo_instance_000.mp4"
        ensure_symlink(episode / "rgb_agent0.mp4", video_a)
        ensure_symlink(episode / "rgb_agent1.mp4", video_b)
        write_json(video_a.with_suffix(".json"), convert_agent(keys, mouse, key_index, 0))
        write_json(video_b.with_suffix(".json"), convert_agent(keys, mouse, key_index, 1))
        converted.append(
            {"index": len(converted), "episode_id": record["episode_id"], "steps": int(keys.shape[0])}
        )
    return converted, skipped_short


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    train_records = read_manifest(args.source / "balanced_v1_train.jsonl")
    test_records = read_manifest(args.source / "val_id.jsonl")
    train, train_short = convert_split(train_records, args.output / "duet/train", args.max_train)
    test, test_short = convert_split(test_records, args.output / "duet/test", args.max_test)
    summary = {
        "source": str(args.source.resolve()),
        "alignment": "action[t] -> video_frame[t+1]",
        "train_episodes": len(train),
        "test_episodes": len(test),
        "skipped_short": {"train": train_short, "test": test_short},
        "train": train,
        "test": test,
    }
    write_json(args.output / "conversion_manifest.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"train", "test"}}, indent=2))


if __name__ == "__main__":
    main()
