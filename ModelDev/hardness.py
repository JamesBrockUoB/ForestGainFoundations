"""
Pixel-hardness learning (Not All Pixels Are Equal: Learning Pixel Hardness for Semantic Segmentation, arXiv 2305.08462),
adapted to a binary gain model on image time series.

Paper (Sec. 3)
  - An auxiliary hardness branch has the SAME ARCHITECTURE AS THE SEGMENTATION HEAD but one output channel. It reads the
    backbone features DETACHED, so it never changes the backbone, and it is dropped at inference.
  - Hardness map   H_i = (sigmoid(D_i) + c) / sum_j (sigmoid(D_j) + c)   over the pixels of an image: sums to 1, pixels
    compete. c > 0 bounds the hardest/easiest pixel weight ratio to (1 + c) / c.
  - Segmentation loss  L^s = sum_i H_i.detach() * L_i           (hard pixels count more; gradient only to the seg head)
  - Hardness loss      L^h = - sum_i H_i * L_i.detach()         (hardness rises where the loss is large)
  - Total              L = L^s + alpha * L^h
  The "historical loss" of the abstract is implicit: the branch is trained over iterations to track each pixel's loss.

Adaptation to time series
  - The gain label is ONE mask per tile (change over the whole sequence), so the hardness map lives on the output grid,
    not per frame. H is normalised over each tile's valid pixels, and the valid mask is already taken over every year.
  - The branch mirrors whatever the segmentation head receives. It captures the head's detached input (a (B, C, h, w)
    map after temporal aggregation, a (B, T, C, h, w) sequence, or a list of multi-scale maps) and runs a copy of the head
    on it, so temporal and multi-scale structure is handled by the head itself.
  - If the head is just a classifier (a lone conv), there is nothing to copy; a small default head is used instead
    that also accepts (B, T, C, h, w) features (shared per-frame trunk, learned softmax attention over time).
"""

import copy

import torch
import torch.nn as nn


def hardness_map(d_logits, valid, c):
    """(B, H, W) hardness H, summing to 1 over the valid pixels of each tile (0 on invalid pixels).
    c > 0 bounds the ratio between the hardest and the easiest pixel to (1 + c) / c."""
    z = (torch.sigmoid(d_logits) + c) * valid
    return z / z.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)


def _detach(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, (list, tuple)):
        items = [_detach(o) for o in obj]
        return items if isinstance(obj, list) else tuple(items)
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    return obj


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


class HardnessHead(nn.Module):
    """Default head for a lone-classifier segmentation head: Conv3x3 - BN - ReLU - Conv1x1 -> one logit per pixel.
    temporal=True also accepts (B, T, C, h, w): the trunk is shared across frames, a learned softmax over T pools them.
    """

    def __init__(
        self, in_channels: int, hidden: int | None = None, temporal: bool = False
    ):
        super().__init__()
        hidden = hidden or max(8, in_channels // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.frame_score = nn.Conv2d(hidden, 1, 1) if temporal else None
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, feats):
        if feats.ndim == 5:
            if self.frame_score is None:
                raise ValueError(
                    "got (B, T, C, h, w) features; build the head with temporal=True"
                )
            b, t = feats.shape[:2]
            f = self.trunk(feats.flatten(0, 1))  # (B*T, hidden, h, w)
            a = self.frame_score(f).view(b, t, 1, *f.shape[-2:]).softmax(dim=1)
            f = (f.view(b, t, *f.shape[1:]) * a).sum(dim=1)
        else:
            f = self.trunk(feats)
        return self.out(f)


def mirror_segmentation_head(head: nn.Module) -> nn.Module:
    """Fresh copy of the segmentation head with a single-channel output (the paper's hardness branch).
    The last Conv2d is replaced by a one-channel conv and every module that has reset_parameters() is re-initialised;
    modules without one keep the copied values."""
    head = copy.deepcopy(head)  # copy before any hook is registered on the original
    convs = [(n, m) for n, m in head.named_modules() if isinstance(m, nn.Conv2d)]
    if not convs or convs[-1][0] == "":
        raise ValueError(
            "the segmentation head needs at least one Conv2d inside it to mirror"
        )
    name, last = convs[-1]
    new = nn.Conv2d(
        last.in_channels,
        1,
        last.kernel_size,
        last.stride,
        last.padding,
        last.dilation,
        groups=1,
        bias=True,
    )
    parent = head
    *path, leaf = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)
    for m in head.modules():
        if m is not new and hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return head


class HardnessBranch(nn.Module):
    """The hardness branch. `layer` names the model's segmentation head (its final decoder / classifier module).

        with branch.capture():          # keep the head's detached inputs during this forward only
            logits = model(x)
        d = branch()                    # (B, 1, h, w) raw hardness logits D, at the head's output resolution

    head: 'copy' mirrors the segmentation head, 'default' uses HardnessHead, 'auto' copies when the head holds at least
    two convolutions (otherwise it is just a classifier and the default head is used).
    """

    def __init__(
        self,
        model: nn.Module,
        layer: str,
        example_input: torch.Tensor,
        head: str = "auto",
    ):
        super().__init__()
        target = dict(model.named_modules()).get(layer)
        if target is None:
            raise ValueError(f"hardness.layer '{layer}' not found in the model.")
        self._target = [
            target
        ]  # in a list: the model's module must not become a submodule of the branch
        self._captured = None
        self._on = False

        n_conv = sum(isinstance(m, nn.Conv2d) for m in target.modules())
        kind = (
            head
            if head != "auto"
            else (
                "copy"
                if n_conv >= 2 and not isinstance(target, nn.Conv2d)
                else "default"
            )
        )
        if kind == "copy" and (isinstance(target, nn.Conv2d) or n_conv == 0):
            raise ValueError(
                f"'{layer}' has no convolutions to mirror; use hardness.head=default"
            )
        self.kind = kind

        # one dry forward to see what the head receives
        was_training = model.training
        model.eval()
        hook = target.register_forward_pre_hook(self._capture, with_kwargs=True)
        try:
            with torch.no_grad(), self.capture():
                model(example_input)
        finally:
            hook.remove()
            model.train(was_training)
        if self._captured is None:
            raise RuntimeError(
                f"'{layer}' was not called during the model's forward pass."
            )
        args, kwargs = self._captured
        self._captured = None

        if kind == "copy":
            self.head = mirror_segmentation_head(target)
            with torch.no_grad():
                out = self.head.eval()(*args, **kwargs)
            self.head.train(was_training)
            if not (torch.is_tensor(out) and out.ndim == 4 and out.shape[1] == 1):
                raise ValueError(
                    f"the mirrored head returned {tuple(getattr(out, 'shape', ()))}, not (B, 1, h, w); "
                    "set hardness.head=default."
                )
        else:
            feats = list(_tensors(args))
            if len(feats) != 1 or feats[0].ndim not in (4, 5):
                raise ValueError(
                    f"hardness.head=default needs the head to take one (B, C, h, w) or (B, T, C, h, w) tensor, got "
                    f"{[tuple(t.shape) for t in feats]}; use hardness.head=copy."
                )
            self.head = HardnessHead(feats[0].shape[-3], temporal=feats[0].ndim == 5)
        self._hook = target.register_forward_pre_hook(self._capture, with_kwargs=True)
        self._on = False

    def _capture(self, module, args, kwargs):
        if self._on:
            self._captured = (_detach(args), _detach(kwargs))

    def capture(self):
        branch = self

        class _Ctx:
            def __enter__(self_):
                branch._on = True

            def __exit__(self_, *exc):
                branch._on = False

        return _Ctx()

    def forward(self):
        if self._captured is None:
            raise RuntimeError("call the model inside `with branch.capture():` first")
        args, kwargs = self._captured
        self._captured = None
        if self.kind == "copy":
            return self.head(*args, **kwargs)
        return self.head(next(_tensors(args)))
