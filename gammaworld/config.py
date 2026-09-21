"""Gamma-World base config plus registration of the native Polis loader."""

import copy

from hydra.core.config_store import ConfigStore

from gamma_world._src.gamma_world.configs.causal_cosmos2.config import make_config as make_base_config
from gamma_world._src.gamma_world.configs.causal_cosmos2.experiment.exp_mc import BIDIRECTIONAL
from gamma_world._src.gamma_world.configs.causal_cosmos2.experiment.exp_mc_causal import CAUSAL
from gamma_world._src.imaginaire.lazy_config import LazyCall as L

from Plotiments.gammaworld.polis_dataset import PolisAugmentationConfig, get_polis_video_loader

DEFAULT_DATA_ROOT = (
    "/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/"
    "polis_two_player_fixed_skins_complete_20260917_360p"
)


def _make_polis_experiment(base_experiment, name: str):
    experiment = copy.deepcopy(base_experiment)
    experiment["defaults"] = [
        {"override /data_train": "polis"}
        if item == {"override /data_train": "video_solaris_action"}
        else {"override /callbacks": ["basic", "cluster_speed"]}
        if item == {"override /callbacks": ["basic", "viz_online_sampling", "wandb", "cluster_speed"]}
        else item
        for item in experiment["defaults"]
    ]
    experiment["job"]["name"] = name
    # Online sampling and W&B are unnecessary for the smoke run, and the
    # released experiment's sampling callback uses a stale constructor field.
    experiment["trainer"]["callbacks"].pop("every_n_sample_reg", None)
    return experiment


def _register_polis() -> None:
    store = ConfigStore.instance()
    store.store(
        group="data_train",
        package="dataloader_train",
        name="polis",
        node=L(get_polis_video_loader)(
            data_root=DEFAULT_DATA_ROOT,
            is_train=True,
            split="train",
            augmentation_config=L(PolisAugmentationConfig)(
                resolution_hw=(320, 480),
                num_video_frames=189,
                fps_downsample_factor=1,
                random_sampling=True,
                load_action=True,
            ),
        ),
    )
    store.store(
        group="experiment",
        package="_global_",
        name="polis_causal",
        node=_make_polis_experiment(CAUSAL, "polis_causal"),
    )
    # Keep the teacher registration available for comparisons, but the launch
    # script defaults to the pre-DMD stage-2 causal student.
    store.store(
        group="experiment",
        package="_global_",
        name="polis_bidirectional",
        node=_make_polis_experiment(BIDIRECTIONAL, "polis_bidirectional"),
    )


def make_config():
    config = make_base_config()
    _register_polis()
    return config
