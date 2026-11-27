import sys

import torch


class SigmaBlock:
    def __init__(self, model, prefix, embeddings):
        self.model = model
        self.device = prefix.device
        recipe = model.generation_config
        self.speculative = getattr(recipe, "decoding", None) == "self_speculative"
        self.block = model.config.block_size
        steps = 32
        if self.speculative:
            self.block = recipe.draft_width
            steps = recipe.diffusion_calls_per_round
        self.grid = model._log_snr_gamma_grid(steps, *model.gamma_bounds(), self.device)
        self.embeddings = embeddings
        self.cache = sys.modules[type(model).__module__].KVCache(model.config.num_hidden_layers)
        self.valid = torch.ones_like(prefix, dtype=torch.bool)
        self.offsets = torch.arange(self.block, device=self.device)[None, :]
        positions = torch.arange(prefix.shape[1], device=self.device)[None, :]
        hidden = model._encode_clean(prefix, 0, positions, self.valid, self.cache)
        if self.speculative:
            self.next_ar = model.ar_head(hidden[:, -1]).argmax(-1)

    def noise(self):
        return torch.randn(
            1, self.block, self.model.config.embed_dim, dtype=torch.float64, device=self.device
        )

    def draft(self):
        z = self.noise()
        condition = None
        positions = self.cache.seq_length() + self.offsets
        calls = len(self.grid) - int(self.speculative)
        guided = not self.speculative
        for i in range(calls):
            gamma = self.grid[i : i + 1]
            logits, clean, mean, _ = self.model._denoise_block(
                z,
                gamma,
                self.cache,
                self.valid,
                positions,
                self.embeddings,
                condition,
                3.0 if guided else 1.0,
            )
            if i + 1 == calls:
                return logits.argmax(-1)
            del logits
            following = self.grid[i + 1 : i + 2]
            clean = clean.double()
            if guided:
                alpha = (-gamma).sigmoid().sqrt()
                sigma = gamma.sigmoid().sqrt()
                noise = (z - alpha * clean) / sigma.clamp_min(1e-8) / 0.3
                clean = (z - sigma * noise) / alpha.clamp_min(1e-8)
            else:
                noise = self.noise()
            z = (-following).sigmoid().sqrt() * clean + following.sigmoid().sqrt() * noise
            condition = None if self.speculative else mean.float()
            yield None

    def verify(self, draft, remaining):
        start = self.cache.seq_length()
        valid = torch.cat((self.valid, torch.ones_like(draft, dtype=torch.bool)), 1)
        hidden = self.model._encode_clean(draft, start, start + self.offsets, valid, self.cache)
        predictions = self.model.ar_head(hidden).argmax(-1)
        expected = torch.cat((self.next_ar[:, None], predictions[:, :-1]), 1)
        agreed = int((draft == expected).long().cumprod(1).sum().item())
        bonus = self.next_ar if agreed == 0 else predictions[:, agreed - 1]
        tokens = torch.cat((draft[:, :agreed], bonus[:, None]), 1)[:, :remaining]
        if tokens.shape[1] < remaining:
            start += agreed
            self.cache.trim(start)
            self.valid = valid[:, :start]
            self.valid = torch.cat(
                (self.valid, torch.ones_like(bonus[:, None], dtype=torch.bool)), 1
            )
            hidden = self.model._encode_clean(
                bonus[:, None], start, self.offsets[:, :1] + start, self.valid, self.cache
            )
            self.next_ar = self.model.ar_head(hidden[:, -1]).argmax(-1)
        return tokens

    def run(self):
        if not self.speculative:
            tokens = yield from self.draft()
            yield tokens[0]
            return
        parts = []
        remaining = self.model.config.block_size
        while remaining:
            draft = yield from self.draft()
            yield None
            tokens = self.verify(draft, remaining)
            parts.append(tokens)
            remaining -= tokens.shape[1]
            if remaining:
                yield None
        yield torch.cat(parts, 1)[0]
