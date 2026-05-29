import copy

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


_SCAFFOLD_TENSOR_ATTRS = [
    '_anchor', '_anchor_index', '_offset', '_anchor_feat',
    '_scaling', '_rotation', '_opacity', '_level',
    'opacity_accum', 'max_radii2D', 'offset_gradient_accum',
    'offset_denom', 'anchor_demon',
]

_MLP_ATTRS = ['mlp_opacity', 'mlp_cov', 'mlp_color', 'mlp_feature_bank', 'embedding_appearance']

def clone_obj(obj):
    # Save MLP state_dicts from original BEFORE any copy
    mlp_state_dicts = {}
    for attr in _MLP_ATTRS:
        module = getattr(obj, attr, None)
        if isinstance(module, torch.nn.Module):
            mlp_state_dicts[attr] = {k: v.cpu() for k, v in module.state_dict().items()}

    # Use deepcopy to avoid sharing __dict__ with original
    clone = copy.deepcopy(obj)

    # Null out optimizers and MLP modules on clone only
    for attr in _MLP_ATTRS:
        if hasattr(clone, attr):
            setattr(clone, attr, None)
    for attr, val in list(vars(clone).items()):
        if isinstance(val, torch.optim.Optimizer):
            setattr(clone, attr, None)
        elif isinstance(val, torch.nn.Module):
            setattr(clone, attr, None)

    # Move all tensors to CPU
    for attr in _SCAFFOLD_TENSOR_ATTRS + ['unique_kfIDs', 'n_obs', 'intrinsics']:
        val = getattr(clone, attr, None)
        if isinstance(val, torch.Tensor):
            setattr(clone, attr, val.detach().cpu().clone())
    for attr, val in list(vars(clone).items()):
        if isinstance(val, torch.Tensor) and val.is_cuda:
            setattr(clone, attr, val.detach().cpu().clone())
        elif isinstance(val, dict):
            setattr(clone, attr, {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) and v.is_cuda else v
                for k, v in val.items()
            })

    clone._mlp_state_dicts = mlp_state_dicts
    return clone
