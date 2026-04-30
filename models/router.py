import torch
import torch.nn as nn


class SoftPlanningRouter(nn.Module):
    def __init__(
        self,
        layer,
        critic,
        skip_rate=0.5,
        threshold_refresh_every=8,
        threshold_sample_size=64,
        threshold_momentum=0.9,
    ):
        super().__init__()
        self.layer = layer
        self.critic = critic
        self.skip_rate = float(skip_rate)
        self.threshold_refresh_every = max(1, int(threshold_refresh_every))
        self.threshold_sample_size = max(1, int(threshold_sample_size))
        self.threshold_momentum = float(threshold_momentum)
        self._cached_threshold = None
        self._cached_signature = None
        self._forward_calls = 0

    def _estimate_threshold(self, probs):
        if self.skip_rate <= 0.0:
            return probs.new_tensor(float("-inf"))
        if self.skip_rate >= 1.0:
            return probs.new_tensor(float("inf"))

        flat = probs.float().reshape(-1)
        if flat.numel() == 0:
            return probs.new_tensor(float("inf"))

        if flat.numel() <= self.threshold_sample_size:
            sample = flat
        else:
            stride = max(1, flat.numel() // self.threshold_sample_size)
            sample = flat[::stride][: self.threshold_sample_size]

        k = max(1, min(sample.numel(), int(self.skip_rate * sample.numel())))
        return sample.kthvalue(k).values

    def _get_threshold(self, probs):
        signature = tuple(probs.shape)
        self._forward_calls += 1

        should_refresh = (
            self._cached_threshold is None
            or self._cached_signature != signature
            or self._forward_calls % self.threshold_refresh_every == 0
        )
        if should_refresh:
            current = self._estimate_threshold(probs)
            if self._cached_threshold is None or self._cached_signature != signature:
                cached = current
            else:
                cached = self._cached_threshold.to(device=current.device, dtype=current.dtype)
                cached = cached * self.threshold_momentum + current * (1.0 - self.threshold_momentum)
            self._cached_threshold = cached.detach()
            self._cached_signature = signature

        return self._cached_threshold.to(device=probs.device, dtype=probs.dtype)

    def forward(self, hidden_states, *args, **kwargs):
        # hidden_states: [batch, seq, hidden]
        with torch.no_grad():
            critic = self.critic.module if isinstance(self.critic, nn.DataParallel) else self.critic
            critic_dtype = next(critic.parameters()).dtype
            probs = critic(hidden_states.detach().to(dtype=critic_dtype))  # [batch, seq]
            thresh = self._get_threshold(probs)

            # Full-batch skip fast path: when every token falls below the threshold,
            # return the residual branch without paying the layer forward cost.
            if (probs.max() < thresh).item():
                return (hidden_states,)

            # Boolean mask: True = token is important enough to process
            run_mask = probs >= thresh  # [batch, seq]

        layer_out = self.layer(hidden_states, *args, **kwargs)
        layer_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Blend: keep layer output for important tokens, residual for skipped tokens
        float_mask = run_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)  # [batch, seq, 1]
        mixed_hidden = layer_hidden * float_mask + hidden_states * (1 - float_mask)

        if isinstance(layer_out, tuple):
            return (mixed_hidden,) + layer_out[1:]
        return (mixed_hidden,)
