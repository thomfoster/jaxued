from typing import Literal, Optional, TypedDict, Tuple
import chex
import jax
import jax.numpy as jnp

from jaxued.environments.underspecified_env import Level

Prioritization = Literal["rank", "topk"]

class Sampler(TypedDict):
    levels:         chex.Array # shape (capacity, ...)
    step_idx:       int

class CyclicLevelSampler:
    """
    The `CyclicLevelSampler` handles level management for an MNIST switching style experiment over levels
    """
    def __init__(
        self,
        sample_random_level,
        n_levels_per_step,
        n_levels_per_phase,
        n_steps_per_phase,
        n_phases
    ):
        self.sample_random_level = sample_random_level
        self.n_levels_per_step = n_levels_per_step
        self.n_levels_per_phase = n_levels_per_phase
        self.n_steps_per_phase = n_steps_per_phase
        self.n_phases = n_phases
        self.n_levels = self.n_levels_per_phase * self.n_phases
        if self.n_levels_per_step * self.n_steps_per_phase < self.n_levels_per_phase:
            print(f'Warning: Only {self.n_levels_per_step} x {self.n_steps_per_phase} = {self.n_levels_per_step * self.n_steps_per_phase} can be sampled per phase, but you have n_levels_per_phase set to {self.n_levels_per_phase}')

    def initialize(self, rng) -> Sampler:
        """
        Returns the `sampler` object as a dictionary.

        Returns:
            Sampler: The initialized sampler object
        """
        def _get_levels_for_phase(rng):
            return jax.vmap(self.sample_random_level)(jax.random.split(rng, self.n_levels_per_phase))
        
        levels = jax.vmap(_get_levels_for_phase)(jax.random.split(rng, self.n_phases))
        return {
            "levels": levels,
            "step_idx": 0
        }

    # Move get_levels to .sample and add a get levels for score based tracking
    # Add size, episode_count, scores
    # Maybe .level_weights and just return 1's
    
    def get_levels(self, sampler: Sampler) -> Tuple[Sampler, Tuple[int, Level]]:
        """
        This deterministically selects n_levels_per_step levels from the buffer according to the current step_idx.
        
        Args:
            sampler (Sampler): The sampler object
            num (int): The number of levels to sample

        Returns:
            Tuple[Sampler, Tuple[int, Level]]: The updated sampler object, the sampled level's index and the level itself.
        """
        phase_idx = (sampler['step_idx'] // self.n_steps_per_phase) % self.n_phases
        step_within_phase = sampler['step_idx'] % self.n_steps_per_phase
        level_idxs = (self.n_levels_per_step*step_within_phase + jnp.arange(self.n_levels_per_step)) % self.n_levels_per_phase
        levels = jax.tree.map(lambda x : x[phase_idx, level_idxs], sampler["levels"])
        sampler = {
            **sampler,
            "step_idx": sampler['step_idx'] + 1
        }
        return sampler, (level_idxs, levels)
    
    def find(self, sampler: Sampler, level: Level) -> int:
        """
        Returns the index of level in the level buffer. If level is not present, -1 is returned.

        Args:
            sampler (Sampler): The sampler object
            level (Level): The level to find

        Returns:
            int: index or -1 if not found.
        """
        eq_tree = jax.tree_map(lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1), sampler["levels"], level)
        eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
        eq_mask = jnp.array(eq_tree_flat).all(axis=0) & (jnp.arange(self.capacity) < sampler["size"])
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
    
    def get_levels_for_phase(self, sampler: Sampler, phase_idx: int) -> Level:
        """
        Returns all the levels for a particular phase

        Args:
            sampler (Sampler): The sampler object
            phase_idx (int): The index to return

        Returns:
            Levels: 
        """
        return jax.tree_map(lambda x: x[phase_idx], sampler["levels"])
    
    def get_all_levels(self, sampler: Sampler) -> Level:
        return jax.tree.map(lambda x : x.reshape((-1, *x.shape[2:])), sampler['levels'])
