#!/usr/bin/env python
"""Decode one Polis window and verify the Gamma-facing batch contract."""

from __future__ import annotations

import argparse
import json

import torch

from Plotiments.gammaworld.polis_dataset import PolisAugmentationConfig, PolisGammaDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--frames", type=int, default=189)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    dataset = PolisGammaDataset(
        data_root=args.data_root,
        split=args.split,
        augmentation_config=PolisAugmentationConfig(
            resolution_hw=(args.height, args.width),
            num_video_frames=args.frames,
            random_sampling=False,
        ),
    )
    sample = dataset[args.sample]
    frames = args.frames
    assert sample["video"].shape == (3, 2 * frames, args.height, args.width)
    assert torch.equal(sample["view_indices"][:frames], torch.zeros(frames, dtype=torch.int64))
    assert torch.equal(sample["view_indices"][frames:], torch.ones(frames, dtype=torch.int64))
    for agent in range(2):
        keyboard = sample[f"action_{agent}_keyboard"]
        camera = sample[f"action_{agent}_camera"]
        assert keyboard.shape == (frames, 23)
        assert camera.shape == (frames, 2)
        assert not keyboard[0].any() and not camera[0].any(), "first frame must have zero-prefix action"
    report = {
        "episode": sample["__key__"],
        "video_shape": list(sample["video"].shape),
        "frame_range": [int(sample["frame_indices"][0]), int(sample["frame_indices"][-1])],
        "fps": float(sample["fps"]),
        "caption": sample["ai_caption"][0],
        "agents": {
            str(agent): {
                "keyboard_shape": list(sample[f"action_{agent}_keyboard"].shape),
                "keyboard_nonzero": int(torch.count_nonzero(sample[f"action_{agent}_keyboard"])),
                "camera_min": sample[f"action_{agent}_camera"].amin(0).tolist(),
                "camera_max": sample[f"action_{agent}_camera"].amax(0).tolist(),
            }
            for agent in range(2)
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

