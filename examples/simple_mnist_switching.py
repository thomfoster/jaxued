import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax import jit, vmap, pmap, grad, value_and_grad
import optax

import time
from functools import partial

import wandb
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch


mnist_img_size = (28, 28)

# init mlp 
def init_mlp(layer_widths, parent_key, scale=0.01):
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)
    params = []
    for key, in_width, out_width in zip(keys, layer_widths[:-1], layer_widths[1:]):
        weight_key, bias_key = jax.random.split(key)
        weight = scale * jax.random.normal(key=weight_key, shape=(in_width, out_width))
        bias = scale * jax.random.normal(key=bias_key, shape=(out_width,))
        params.append([weight, bias])
    return params

# predict function
@partial(jax.jit, static_argnames='return_activations')
def apply(params, x, return_activations=False):
    activations = []
    # apply params, do activations for each layer, apply softmax on last
    for layer_idx, (weight, bias) in enumerate(params):
        x = x @ weight
        x += bias
        if layer_idx == len(params) - 1:
            x = x - logsumexp(x)
        else:
            x = jax.nn.relu(x)
            if return_activations:
                activations.append(x)

    if return_activations:
        return x, activations
    return x

def get_dataloaders(config):
    # dataloading with torchvision datasets
    def custom_transform(x):
        return np.ravel(np.array(x, dtype=np.float32))

    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))

        labels = np.array(transposed_data[1])
        imgs = np.stack(transposed_data[0])

        return imgs, labels

    train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=custom_transform)
    test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=custom_transform)

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

    return train_loader, test_loader

# @partial(jax.jit, static_argnames='prefix')
def compute_metrics(logits, gt_labels, params, gradients, activations, prefix=''):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    weight_norm = param_norm(params, ord=2, weighted=True)
    active_weights = frac_gt_thresh(jax.tree.map(lambda x : jnp.abs(x), params), 1e-3)
    grad_norm = param_norm(gradients, ord=2, weighted=True)
    active_activations = frac_gt_thresh(activations, 0.0)

    return {
        prefix+'loss': loss,
        prefix+'accuracy': accuracy,
        prefix+'weight_norm': weight_norm,
        prefix+'active_weights': active_weights,
        prefix+'grad_norm': grad_norm,
        prefix+'active_activations': active_activations
    }

def aggregate_metrics(metrics):
    batch_metrics_np = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    aggregated_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return aggregated_metrics

def param_norm(params, ord=1, weighted=True):
    norms = jax.tree.map(lambda x : jnp.linalg.norm(x, ord=ord), params)
    if weighted:
        n_params = jax.tree.map(lambda x : jnp.prod(jnp.array(x.shape)), params)
        total_params = sum(jax.tree.leaves(n_params))
        norms = jax.tree.map(lambda norm, n : norm * n / total_params, norms, n_params)
    return jnp.mean(jnp.array(jax.tree.leaves(norms)))

def frac_gt_thresh(pytree, thresh):
    nel = jax.tree.map(lambda x : jnp.prod(jnp.array(x.shape)), pytree)
    nel_gt_0 = jax.tree.map(lambda x : jnp.sum(x > thresh), pytree)
    frac_gt_0 = jax.tree.map(lambda x, y : x / y , nel_gt_0, nel)
    return jnp.mean(jnp.array(jax.tree.leaves(frac_gt_0)))

def get_new_permutation(key):
    key, perm_key = jax.random.split(key)
    label_perm = jax.random.permutation(perm_key, 10)
    return label_perm, key

@jit
def apply_permutation(perm, labels):
    return jnp.array([perm[i] for i in labels])


# def train(key, params, train_loader, test_loader, num_epochs):
#     accs = []
#     
#     n_params = np.sum(jax.tree.leaves(jax.tree.map(lambda x : jnp.prod(jnp.array(x.shape)), params)))
#     start = time.time()
#     step = 0
#     for epoch in range(num_epochs):
#         for imgs, lbls in train_loader:
#             lbls = apply_permutation(label_perm, lbls)
#             old_params = params
#             loss, params = update(params, imgs, lbls)

#             if step % 100 == 0:
#                 norm = param_norm(params, ord=2, weighted=True)

#                 delta = jax.tree.map(lambda x, y : y - x, old_params, params)
#                 delta_norm = param_norm(delta, ord=2, weighted=True)

#                 _, activations = batched_predict(old_params, imgs, True)
#                 n_dead_units = np.sum(jax.tree.map(lambda x : jnp.sum(x > 0), activations))

#                 print(f"[+{time.time() - start :.0f}s] Step: {step}, Loss: {loss:.2}, Norm: {norm:.3}, Delta: {delta_norm:.3}, Dead units: {n_dead_units/n_params:.3}")

#             if step % 250 == 0:
#                 acc = run_eval(params, test_loader, label_perm)
#                 accs.append((step, acc))
#                 print(f"[+{time.time() - start :.0f}s] Step: {step}, Acc: {acc:.2f}")

#             if step != 0 and step % 4000 == 0:
#                 print('\nSWITCHING LABELS!!!!!!!!')
#                 key, perm_key = jax.random.split(key)
#                 label_perm = jax.random.permutation(perm_key, 10)

#                 acc = run_eval(params, test_loader, label_perm)
#                 accs.append((step, acc))
#                 print(f"[+{time.time() - start :.0f}s] Step: {step}, Acc: {acc:.2f}")

#             step += 1
            
#     plt.scatter(
#         [s for s,a in accs],
#         [a for s,a in accs]
#     )
#     plt.savefig('l curves.png')
#     plt.plot(
#         [s for s,a in accs],
#         [a for s,a in accs]
#     )
#     plt.savefig('l curves plot.png')


def main(config=None, project="mnist plasticity"):
    tags = []
    # can add tags based on config here
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("loss/*", step_metric="num_updates")
    wandb.define_metric("accuracy/*", step_metric="num_updates")
    wandb.define_metric("dormancy/*", step_metric="num_updates")

    # initialisations
    key = jax.random.PRNGKey(config.seed)
    torch.manual_seed(config.seed) # for the dataloaders
    params = init_mlp([784, 512, 256, 10], key)
    print("initialised network as:", jax.tree.map(lambda x: x.shape, params))
    match config.optimiser:
        case 'sgd':
            optim = optax.sgd(learning_rate=config.lr)
        case 'adam':
            optim = optax.adam(learning_rate=config.lr)

    opt_state = optim.init(params)
    print("initialised optim:", opt_state)
    label_perm, perm_key = get_new_permutation(key)

    @jax.jit
    def train_step(params, opt_state, imgs, gt_labels):
        def loss_fn(params):
            logits, activations = apply(params, imgs, return_activations=True)
            one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
            loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
            return loss, (logits, activations)
    
        (loss, (logits, activations)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        metrics = compute_metrics(
            logits=logits, 
            gt_labels=gt_labels,
            params=params,
            gradients=grads,
            activations=activations,
            prefix='train/'
        )  # duplicating loss calculation (not forward pass) but it's a bit cleaner

        return params, opt_state, metrics
    
    def evaluate(params, dataloader, label_perm):
        """Evaluate over dataloader."""
        
        @jax.jit
        def eval_step(params, imgs, gt_labels):
            logits, activations = apply(params, imgs, return_activations=True)
            # TODO: Analyse activations
            return compute_metrics(logits=logits, gt_labels=gt_labels, prefix='eval/')
        
        batch_metrics = []
        for cnt, (imgs, labels) in enumerate(dataloader):
            labels = apply_permutation(label_perm, labels)
            metrics = eval_step(params, imgs, labels)
            batch_metrics.append(metrics)
        return aggregate_metrics(batch_metrics)
    
    # actual running of loops
    train_loader, test_loader = get_dataloaders(config)

    num_updates = 0
    epoch = 0
    while num_updates < config.max_num_updates:

        batch_metrics = []
        for cnt, (imgs, labels) in enumerate(train_loader):
            epoch += 1
            
            if num_updates % config.switch_every == 0:
                # evaluate for a final time before switching the labels
                # eval_metrics = evaluate(params, test_loader, label_perm)
                # wandb.log(eval_metrics)
                label_perm, perm_key = get_new_permutation(perm_key)

            labels = apply_permutation(label_perm, labels)

            if (num_updates % config.log_every == 0) and (len(batch_metrics) >= 1):
                metrics = aggregate_metrics(batch_metrics)
                print(f"n_updates: {num_updates}, loss: {metrics['train/loss']}")
                wandb.log(metrics)
                batch_metrics = []

            # if num_updates % config.eval_every == 0:
            #     eval_metrics = evaluate(params, test_loader, label_perm)
            #     wandb.log(eval_metrics)

            params, opt_state, metrics = train_step(params, opt_state, imgs, labels)
            num_updates += 1
            batch_metrics.append(metrics)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="mnist plasticity")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--switch_every", type=int, default=50000000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_num_updates", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimiser", type=str, default='sgd')

    config = vars(parser.parse_args())

    wandb.login()
    main(config, project=config['project'])
