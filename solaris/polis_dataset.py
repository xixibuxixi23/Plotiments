"""Solaris multiplayer dataset adapter for converted Polis episodes.

The release stores observations with length T+1 and actions with length T.
Solaris intentionally reads video frames [start+1, stop+1), so action t is
paired with the post-action observation t+1 without any additional shift.
"""

import json

import numpy as np
from decord import VideoReader, cpu

from src.data import minecraft
from src.data.dataset import DatasetMultiplayer, VideoReadError
from src.data.segment import Segment


_ACTION_INDEX = {name: i for i, name in enumerate(minecraft.ACTION_KEYS)}
_BOOLEAN_FIELDS = (
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
)


def convert_polis_actions(actions):
    """Convert normalized Polis JSON actions to Solaris' 25-D action vector."""
    result = np.zeros((len(actions), len(minecraft.ACTION_KEYS)), dtype=np.float32)
    for row_index, row in enumerate(actions):
        action = row["action"]
        for name in _BOOLEAN_FIELDS:
            if action.get(name, False):
                result[row_index, _ACTION_INDEX[name]] = 1.0

        camera = action.get("camera", (0.0, 0.0))
        result[row_index, _ACTION_INDEX["cameraX"]] = np.degrees(camera[0])
        result[row_index, _ACTION_INDEX["cameraY"]] = np.degrees(camera[1])
    return result


class DatasetMultiplayerPolis(DatasetMultiplayer):
    """DatasetMultiplayer with a lossless mapping for Polis keyboard actions."""

    def __getitem__(self, segment_id):
        episode_paths = self.get_episode_paths(segment_id.episode_id)
        video_path_bot1 = self.directory / episode_paths["bot1_video_path"]
        video_path_bot2 = self.directory / episode_paths["bot2_video_path"]
        actions_path_bot1 = self.directory / episode_paths["bot1_actions_path"]
        actions_path_bot2 = self.directory / episode_paths["bot2_actions_path"]

        try:
            video_bot1 = VideoReader(str(video_path_bot1), ctx=cpu(0))
            video_bot2 = VideoReader(str(video_path_bot2), ctx=cpu(0))
        except Exception as exc:
            raise ValueError(f"Error reading episode {segment_id.episode_id}: {exc}") from exc

        try:
            with actions_path_bot1.open() as handle:
                actions_bot1 = json.load(handle)[
                    segment_id.bot1_start : segment_id.bot1_stop
                ]
            with actions_path_bot2.open() as handle:
                actions_bot2 = json.load(handle)[
                    segment_id.bot2_start : segment_id.bot2_stop
                ]
        except Exception as exc:
            raise ValueError(
                f"Error reading actions for episode {segment_id.episode_id}: {exc}"
            ) from exc

        try:
            obs_bot1 = minecraft.read_obs_slice_decord(
                video_bot1,
                segment_id.bot1_start,
                segment_id.bot1_stop,
                self._obs_resize,
            )
            obs_bot2 = minecraft.read_obs_slice_decord(
                video_bot2,
                segment_id.bot2_start,
                segment_id.bot2_stop,
                self._obs_resize,
            )
        except Exception as exc:
            raise VideoReadError(f"Error reading segment slice: {exc}") from exc

        action_bot1 = convert_polis_actions(actions_bot1)
        action_bot2 = convert_polis_actions(actions_bot2)
        for converter in self._converters:
            action_bot1 = converter.convert(action_bot1)
            action_bot2 = converter.convert(action_bot2)

        bot_obs = [obs_bot1, obs_bot2]
        bot_act = [action_bot1, action_bot2]
        if self._shuffle_bots:
            indices = [0, 1]
            self._rng.shuffle(indices)
            bot_obs = [bot_obs[i] for i in indices]
            bot_act = [bot_act[i] for i in indices]

        return Segment(np.stack(bot_obs, axis=1), np.stack(bot_act, axis=1))
