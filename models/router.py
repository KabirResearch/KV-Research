import torch
import torch.nn as nn


class SoftPlanningRouter(nn.Module):
    def __init__(self, layer, critic, skip_rate=0.5):
        super().__init__()
        self.layer = layer
        self.critic = critic
        self.skip_rate = skip_rate

    def forward(self, hidden_states, *args, **kwargs):
        # hidden_states: [batch, seq, hidden]
        with torch.no_grad():
            critic = self.critic.module if isinstance(self.critic, nn.DataParallel) else self.critic
            critic_dtype = next(critic.parameters()).dtype
            probs = critic(hidden_states.detach().to(dtype=critic_dtype))  # [batch, seq]
            thresh = torch.quantile(probs, self.skip_rate)
            # Boolean mask: True = token is important enough to process
            run_mask = probs >= thresh  # [batch, seq]

        # True skip: if every token is below the threshold, skip the layer entirely
        if not run_mask.any():
            return (hidden_states,) + ((None,) * len(kwargs.get("past_key_value", ())) if False else ())

        layer_out = self.layer(hidden_states, *args, **kwargs)
        layer_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Blend: keep layer output for important tokens, residual for skipped tokens
        float_mask = run_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)  # [batch, seq, 1]
        mixed_hidden = layer_hidden * float_mask + hidden_states * (1 - float_mask)

        if isinstance(layer_out, tuple):
            return (mixed_hidden,) + layer_out[1:]
        return (mixed_hidden,)
