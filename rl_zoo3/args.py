from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class TrainArgs:
    algo: str
    game: str
    env: str
    tensorboard_log: str
    trained_agent: str
    truncate_last_trajectory: bool
    n_timesteps: int
    num_threads: int
    log_interval: int
    eval_freq: int
    optimization_log_path: Optional[str]
    eval_episodes: int
    n_eval_envs: int
    save_freq: int
    save_replay_buffer: bool
    log_folder: str
    seed: int
    vec_env: str
    device: str
    n_trials: int
    max_total_trials: int
    optimize_hyperparameters: bool
    no_optim_plots: bool
    n_jobs: int
    sampler: str
    pruner: str
    n_startup_trials: int
    n_evaluations: int
    storage: Optional[str]
    study_name: Optional[str]
    verbose: int
    gym_packages: List[str]
    env_kwargs: Dict[str, str]
    eval_env_kwargs: Dict[str, str]
    hyperparams: Dict[str, str]
    conf_file: str
    uuid_on: bool
    track: bool
    wandb_project_name: str
    wandb_entity: str
    progress: bool
    wandb_tags: List[str]
