import os
import json
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator
from jaxued.utils import max_mc, positive_value_loss
from jaxued.wrappers import AutoResetWrapper
from jaxued.level_history import LevelHistory
from jaxued.cyclic_level_sampler import CyclicLevelSampler
import jax.profiler

def main(config=None, project="JAXUED_TEST"):
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env_params = env.default_params

    key = jax.random.PRNGKey(config['seed'])
    level = sample_random_level(key)
    # render = env_renderer.render_level(level, env_params)

    BATCH_SIZE = 3

    level_history = LevelHistory(50, BATCH_SIZE)
    history = level_history.initialize(level)

    level_sampler = CyclicLevelSampler(
        sample_random_level,
        n_levels_per_step=BATCH_SIZE,
        n_levels_per_phase=1,
        n_steps_per_phase=5,
        n_phases=2
    )
    sampler = level_sampler.initialize(key)

    for i in range(15):
        sampler, (level_idxs, levels) = level_sampler.get_levels(sampler)
        history, idxs = level_history.insert_batch(history, levels)
        history = level_history.inc_step_idx(history)
        

    # equivalence on levels as: np.all(jax.tree.leaves(jax.tree.map(lambda x, y : jnp.all(x == y), l1, l2)))
    print(jnp.int8(level_history.get_sparse_visit_map(history)))
    breakpoint()

    # pholder_history = {update_idx : pholder_level_batch for update_idx in range(config['num_updates'])}
    pholder_history = jax.tree_map(lambda x : jnp.array([x]).repeat(config['num_updates'], axis=0), pholder_level_batch)

    def select_i(x, i):
        return jax.tree.map(lambda x : x[i] if x.ndim > 1 else x[i].item(), x)
    
    z = select_i(pholder_history, 0)
    print(jax.tree.map(lambda x : x.shape, z))
    z = select_i(z, 0)
    print(jax.tree.map(lambda x : x.shape if isinstance(x, jnp.ndarray) else x, z))

    breakpoint()

    @jax.jit
    def step(rng, state):
        rng, rng_levels = jax.random.split(rng)
        state['i'] = state['i'] + 1

        new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
        new_history = jax.tree.map(lambda x, y : x.at[state['i']].set(y), state['level_history'], new_levels)
        
        state['new_levels'] = new_levels
        state['level_history'] = new_history
        return rng, state
    
    state = {
        'i': -1,
        'new_levels': pholder_level_batch,
        'level_history': pholder_history
    }
    
    key, state = step(key, state)
    breakpoint()
    key, state = step(key, state)
    breakpoint()
    key, state = step(key, state)
    breakpoint()




if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs='+', default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group('Training params')
    # === PPO ===
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
