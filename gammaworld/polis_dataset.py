"""Read the canonical Polis release in Gamma-World's training batch format.

Polis stores raw transitions as observation[t] -> action[t] -> observation[t+1].
Gamma-World consumes an action aligned to every video frame.  For a sampled
window this loader therefore emits a zero prefix for the first observation and
aligns action[t] to the following observation.  This is intentionally not the
same indexing used by Gamma-World's native Solaris tar loader.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import InterpolationMode, Resize

from gamma_world._src.imaginaire.utils import log


# Solaris/VPT keyboard order after cameraX/cameraY are split into `camera`.
SOLARIS_ACTION_KEYS = (
    "inventory",
    "escape",
    "slot_1",
    "slot_2",
    "slot_3",
    "slot_4",
    "slot_5",
    "slot_6",
    "slot_7",
    "slot_8",
    "slot_9",
    "forward",
    "backward",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "swap_hands",
    "attack",
    "use",
    "pick_item",
    "drop",
)
SOLARIS_KEY_TO_INDEX = {name: idx for idx, name in enumerate(SOLARIS_ACTION_KEYS)}

# Polis names that are semantically equivalent to Solaris/VPT controls.
POLIS_TO_SOLARIS = {
    "inventory": "inventory",
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "jump": "jump",
    "sneak": "sneak",
    "aux1": "sprint",
    "dig": "attack",
    "place": "use",
    "drop": "drop",
    **{f"slot_{idx}": f"slot_{idx}" for idx in range(1, 10)},
}

# Gamma's released examples apply CameraLinearConverterMatrixGame2.  The
# observed conversion is raw * rad2deg / 15 (e.g. .15 -> .5729578).
DEFAULT_MOUSE_SCALE = 180.0 / math.pi / 15.0


@attrs.define(slots=False)
class PolisAugmentationConfig:
    resolution_hw: tuple[int, int] = (320, 480)
    num_video_frames: int = 189
    fps_downsample_factor: int = 1
    random_sampling: bool = True
    mouse_scale: float = DEFAULT_MOUSE_SCALE
    load_action: bool = True
    # Kept only for compatibility with the bidirectional experiment overlay.
    # Polis already stores one MP4 per player, so half-width splitting is invalid.
    take_half_width: bool = False


def _episode_paths(data_root: Path, split: str, index_file: str | None) -> list[Path]:
    index_path = Path(index_file) if index_file else data_root / f"{split}.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Polis index does not exist: {index_path}")
    result: list[Path] = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate = row.get("source_path")
            if candidate:
                path = Path(candidate)
            else:
                episode_id = row.get("episode_id")
                if not episode_id:
                    raise ValueError(f"Index row has neither source_path nor episode_id: {row}")
                path = data_root / split / episode_id
            if not path.is_absolute():
                path = data_root / path
            result.append(path)
    return result


def _read_manifest_caption(episode: Path) -> str:
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    task = manifest.get("task_text")
    if task:
        return str(task)
    task_id = manifest.get("task_id", "cooperative Minecraft task")
    return f"Two players performing {str(task_id).replace('_', ' ')}."


def _polis_keyboard(action_keys: np.ndarray, source_names: list[str]) -> np.ndarray:
    """Map [..., 21] Polis keys by name into [..., 23] Solaris slots."""
    if action_keys.shape[-1] != len(source_names):
        raise ValueError(
            f"action_keys width {action_keys.shape[-1]} != metadata key count {len(source_names)}"
        )
    out = np.zeros((*action_keys.shape[:-1], 23), dtype=np.float32)
    for source_idx, source_name in enumerate(source_names):
        target_name = POLIS_TO_SOLARIS.get(source_name)
        if target_name is not None:
            out[..., SOLARIS_KEY_TO_INDEX[target_name]] = action_keys[..., source_idx]
    return out


def _incoming_actions(
    keys: np.ndarray,
    mouse: np.ndarray,
    frame_indices: list[int],
    source_names: list[str],
    mouse_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align transition actions to sampled observations, aggregating skipped ticks."""
    num_frames, num_agents = len(frame_indices), keys.shape[1]
    keyboard = np.zeros((num_frames, num_agents, 23), dtype=np.float32)
    camera = np.zeros((num_frames, num_agents, 2), dtype=np.float32)
    mapped = _polis_keyboard(keys, source_names)
    for dst in range(1, num_frames):
        previous_observation = frame_indices[dst - 1]
        observation = frame_indices[dst]
        if not 0 <= previous_observation < observation <= len(keys):
            raise ValueError(
                f"Invalid observation/action interval [{previous_observation}, {observation}) "
                f"for {len(keys)} transitions"
            )
        # A held/discrete control is active if it occurred during any skipped tick.
        keyboard[dst] = mapped[previous_observation:observation].max(axis=0)
        # Mouse deltas compose additively across skipped ticks.
        camera[dst] = mouse[previous_observation:observation].sum(axis=0) * mouse_scale
    return keyboard, camera


class PolisGammaDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        augmentation_config: PolisAugmentationConfig,
        index_file: str | None = None,
        seed: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.config = augmentation_config
        self.seed = int(seed)
        self.resize = Resize(
            tuple(augmentation_config.resolution_hw),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        required = (
            (augmentation_config.num_video_frames - 1)
            * augmentation_config.fps_downsample_factor
            + 1
        )
        self.episodes: list[Path] = []
        rejected = 0
        for episode in _episode_paths(self.data_root, split, index_file):
            metadata_path = episode / "training_metadata.json"
            if not metadata_path.is_file():
                rejected += 1
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != "textagent-action-training-v3"
                or int(metadata.get("num_agents", -1)) != 2
                or int(metadata.get("num_observations", 0)) < required
            ):
                rejected += 1
                continue
            if not all((episode / f"rgb_agent{agent}.mp4").is_file() for agent in range(2)):
                rejected += 1
                continue
            self.episodes.append(episode)
        if not self.episodes:
            raise RuntimeError(
                f"No usable two-player Polis episodes in {data_root} split={split}; "
                f"need at least {required} observations"
            )
        log.info(
            f"PolisGammaDataset split={split}: {len(self.episodes)} usable episodes, "
            f"{rejected} rejected, frames={augmentation_config.num_video_frames}, "
            f"stride={augmentation_config.fps_downsample_factor}"
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def _decode(self, path: Path, indices: list[int]) -> tuple[torch.Tensor, float, tuple[int, int]]:
        from decord import VideoReader

        reader = VideoReader(os.fspath(path))
        if indices[-1] >= len(reader):
            raise ValueError(f"{path} has {len(reader)} frames, requested {indices[-1]}")
        fps = float(reader.get_avg_fps())
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy())
        frames = rearrange(frames, "t h w c -> t c h w")
        original_hw = (int(frames.shape[-2]), int(frames.shape[-1]))
        return self.resize(frames), fps, original_hw

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode = self.episodes[idx]
        metadata = json.loads((episode / "training_metadata.json").read_text(encoding="utf-8"))
        observations = int(metadata["num_observations"])
        count = self.config.num_video_frames
        stride = self.config.fps_downsample_factor
        span = (count - 1) * stride + 1
        max_start = observations - span
        if self.config.random_sampling:
            start = random.randint(0, max_start)
        else:
            start = min((self.seed + idx * span) % (max_start + 1), max_start)
        frame_indices = [start + offset * stride for offset in range(count)]

        streams: list[torch.Tensor] = []
        fps_values: list[float] = []
        original_sizes: list[tuple[int, int]] = []
        for agent in range(2):
            frames, fps, original_hw = self._decode(episode / f"rgb_agent{agent}.mp4", frame_indices)
            streams.append(frames)
            fps_values.append(fps)
            original_sizes.append(original_hw)
        if not math.isclose(fps_values[0], fps_values[1], rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"Unsynchronized FPS in {episode}: {fps_values}")

        result: dict[str, Any] = {
            "__key__": episode.name,
            "__url__": os.fspath(episode),
            "video": rearrange(torch.cat(streams, dim=0), "t c h w -> c t h w"),
            "ai_caption": [_read_manifest_caption(episode)],
            "view_indices": torch.cat(
                [torch.full((count,), agent, dtype=torch.int64) for agent in range(2)]
            ),
            "fps": torch.tensor(fps_values[0] / stride, dtype=torch.float64),
            "chunk_index": torch.tensor(0, dtype=torch.int64),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(count, dtype=torch.int64),
            "view_indices_selection": torch.arange(2, dtype=torch.int64),
            "camera_keys_selection": ["agent0", "agent1"],
            "sample_n_views": torch.tensor(2, dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.config.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor(0, dtype=torch.int64),
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
        }

        if self.config.load_action:
            with np.load(episode / "data.npz", allow_pickle=False) as data:
                raw_keys = np.asarray(data["action_keys"], dtype=np.float32)
                raw_mouse = np.asarray(data["action_mouse"], dtype=np.float32)
            if len(raw_keys) + 1 != observations or raw_mouse.shape != (*raw_keys.shape[:2], 2):
                raise ValueError(
                    f"Bad T actions/T+1 observations contract in {episode}: "
                    f"keys={raw_keys.shape}, mouse={raw_mouse.shape}, observations={observations}"
                )
            keyboard, camera = _incoming_actions(
                raw_keys,
                raw_mouse,
                frame_indices,
                list(metadata["action_keys"]),
                self.config.mouse_scale,
            )
            for agent in range(2):
                result[f"action_{agent}_keyboard"] = torch.from_numpy(keyboard[:, agent])
                result[f"action_{agent}_camera"] = torch.from_numpy(camera[:, agent])
        return result


def _distributed_rank_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def get_polis_video_loader(
    *,
    augmentation_config: PolisAugmentationConfig = PolisAugmentationConfig(),
    data_root: str,
    is_train: bool = True,
    split: str | None = None,
    index_file: str | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int | None = 1,
    seed: int = 0,
    **_: Any,
) -> DataLoader:
    # Keep the heavy Cosmos/WebDataset dependency surface out of standalone
    # data-contract audits; it is only needed when constructing a train loader.
    from gamma_world._src.predict2_multiview.datasets.multiview import collate_fn

    split = split or ("train" if is_train else "val_id")
    dataset = PolisGammaDataset(
        data_root=data_root,
        split=split,
        index_file=index_file,
        augmentation_config=augmentation_config,
        seed=seed,
    )
    rank, world = _distributed_rank_world()
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=is_train, drop_last=is_train)
        if world > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=collate_fn,
    )
