from typing import Literal, Optional, TypedDict, Tuple, Mapping
import chex
import jax
import jax.numpy as jnp

from jaxued.environments.underspecified_env import Level

class History(TypedDict):
    levels:         chex.Array # shape (n_updates * batch_size, ...)
    visits:         chex.Array # shape (n_updates * batch_size, n_updates)
    step_idx:  int
    size:           int # number of unique levels in levels

class LevelHistory:
    """
    The `LevelHistory` provides stores a library of levels and when they were visited. In the standard Jax style, the level sampler class does not store any data itself, and accepts a `sampler` object for most operations.
    """
    def __init__(
        self,
        n_updates: int,
        batch_size: int,
    ):
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.capacity = n_updates * batch_size
        
    def initialize(self, pholder_level: Level) -> History:
        """
        Returns the `history` object as a dictionary.

        Args:
            pholder_level (Level): A placeholder level that will be used to initialize the level buffer.
        Returns:
            History: The initialized history object
        """
        history = {
            "levels": jax.tree_map(lambda x: jnp.array([x],).repeat(self.capacity, axis=0), pholder_level),
            "visits": jnp.zeros((self.capacity, self.n_updates), dtype=jnp.bool),
            "size": 0,
            "step_idx": 0,
        }
        return history
    
    def insert(self, history: History, level: Level) -> Tuple[History, int]:
        """
        Attempt to insert level into the level buffer.

        If step_idx >= self.n_updates, we won't insert, 
        as we have run out of room in the visits buffer to record the insertion
        
        If the level to be inserted already exists in the level
        buffer, the corresponding buffer entry will be updated instead.

        Args:
            sampler (Sampler): The sampler object
            level (Level): Level to insert

        Returns:
            Tuple[Sampler, int]: The updated sampler, and the level's index in the buffer (-1 if it was not inserted)
        """
        visit_buffer_has_capacity = history['step_idx'] < self.n_updates

        def _insert():
            idx = self.find(history, level)
            return jax.lax.cond(
                idx == -1,
                lambda: self._insert_new(history, level),
                lambda: ({
                    **history,
                    "visits": history["visits"].at[idx, history['step_idx']].set(1),
                }, idx),
            )
        
        return jax.lax.cond(visit_buffer_has_capacity, _insert, lambda : (history, -1))
    
    def insert_batch(self, history: History, levels: Level) -> Tuple[History, chex.Array]:
        """
        Inserts a batch of levels.

        Args:
            history (History): The history object
            levels (_type_): The levels to insert. This must be a `batched` level, in that it has an extra dimension at the front.
        """
        def _insert(history, level):
            return self.insert(history, level)
        return jax.lax.scan(_insert, history, levels)
    
    def find(self, history: History, level: Level) -> int:
        """
        Returns the index of level in the level buffer. If level is not present, -1 is returned.

        Args:
            history (History): The history object
            level (Level): The level to find

        Returns:
            int: index or -1 if not found.
        """
        eq_tree = jax.tree_map(lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1), history["levels"], level)
        eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
        eq_mask = jnp.array(eq_tree_flat).all(axis=0) & (jnp.arange(self.capacity) < history["size"])
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
    
    def get_levels(self, history: History, level_idx: int) -> Level:
        """
        Returns the level at a particular index.

        Args:
            history (History): The history object
            level_idx (int): The index to return

        Returns:
            Level: 
        """
        return jax.tree_map(lambda x: x[level_idx], history["levels"])
    
    def inc_step_idx(self, history: History):
        return {**history, 'step_idx': history['step_idx'] + 1}
    
    def _insert_new(self, history: History, level: Level) -> Tuple[History, int]:
        idx = self._get_next_idx(history)

        _insert = lambda : {
            **history,
            "levels": jax.tree_map(lambda x, y: x.at[idx].set(y), history["levels"], level),
            "visits": history["visits"].at[idx, history['step_idx']].set(1),
            "size": history["size"] + 1
        }

        new_history = jax.lax.cond(idx >= 0, _insert, lambda : history)
        return new_history, idx
    
    def _proportion_filled(self, history: History) -> float:
        return history["size"] / self.capacity
    
    def _get_next_idx(self, history: History) -> int:
        return jax.lax.select(
            history['size'] < self.capacity,
            history['size'],
            -1
        )
    
    def get_sparse_visit_map(self, history):
        return history['visits'][:history['size'], :history['step_idx'] + 1]
    
    def get_visit_counts(self, history: History) -> chex.Array:
        visit_map = self.get_sparse_visit_map(history)
        return visit_map.sum(axis=1)

    def print_stats(self, history: History) -> None:
        trimmed_visits = history['visits'][:history['size'], :history['step_idx']+1]
        visit_counts = trimmed_visits.sum(axis=1)
        min_visit_count = jnp.min(visit_counts)
        n_with_min = jnp.sum(visit_counts == min_visit_count)
        max_visit_count = jnp.max(visit_counts)
        n_with_max = jnp.sum(visit_counts == max_visit_count)
        mean_visit_count = jnp.mean(visit_counts)
        q25 = jnp.quantile(visit_counts, 0.25)
        q75 = jnp.quantile(visit_counts, 0.75)

        print(f"[Step: {history['step_idx']}] n unique levels: {history['size']}")
        print(f"[Step: {history['step_idx']}] {mean_visit_count} mean visits")
        print(f"[Step: {history['step_idx']}] 25% <= {q25}")
        print(f"[Step: {history['step_idx']}] 25% > {q75}")
        print(f"[Step: {history['step_idx']}] {n_with_min} with min of {min_visit_count}")
        print(f"[Step: {history['step_idx']}] {n_with_max} with max of {max_visit_count}")