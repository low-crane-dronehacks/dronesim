"""Success Guided Sampling (SGS) over a fixed bank of start configurations.

Zhang et al., "A Balanced Data Diet" (arXiv 2610.12465). Each bank cell keeps its last H binary outcomes;
``p_i`` is their mean and resets draw cells i.i.d. from

    w_i  = (p_i + eps)^(k t) * (1 - p_i + eps)^(k (1 - t))
    P(i) = softmax(log(max(w_i, eps) + eps) / max(T, 1))

which peaks at ``p = t`` (default 0.5). Unvisited cells read as ``p = t``, so the bank gets explored.
"""
from __future__ import annotations

from typing import Optional

import torch


class SGSSampler:
    """Per-configuration success ring buffers and the SGS draw."""

    def __init__(self, bank_size: int, history: int = 100, target: float = 0.5,
                 concentration: float = 1.0, eps: float = 1e-4,
                 temperature: float = 2.0, min_count: int = 1,
                 device='cpu'):
        if bank_size <= 0 or history <= 0:
            raise ValueError('bank_size and history must be > 0')
        if not 0.0 <= target <= 1.0:
            raise ValueError('target must be in [0, 1]')
        if min_count < 1:
            raise ValueError('min_count must be >= 1')
        self.n = int(bank_size)
        self.h = int(history)
        self.t = float(target)
        self.k = float(concentration)
        self.eps = float(eps)
        self.temperature = float(temperature)
        self.min_count = int(min_count)
        self.device = device
        self.buf = torch.zeros(self.n, self.h, dtype=torch.int8, device=device)
        #: Outcomes ever recorded per cell (the ring's write head is count % H).
        self.count = torch.zeros(self.n, dtype=torch.long, device=device)
        self.total_records = 0
        self._probs: Optional[torch.Tensor] = None

    # ── estimates ───────────────────────────────────────────────────────────

    def success_rate(self) -> torch.Tensor:
        """(N,) mean of the recorded outcomes; ``t`` for under-visited cells."""
        filled = self.count.clamp(max=self.h)
        p = self.buf.sum(dim=1, dtype=torch.float32) / filled.clamp_min(1)
        return torch.where(self.count >= self.min_count, p,
                           torch.full_like(p, self.t))

    def weights(self, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = self.success_rate() if p is None else p
        e = self.eps
        return ((p + e) ** (self.k * self.t)
                * (1.0 - p + e) ** (self.k * (1.0 - self.t)))

    def probs(self) -> torch.Tensor:
        """(N,) sampling distribution. Cached until the next ``record``."""
        if self._probs is None:
            w = self.weights()
            logit = torch.log(w.clamp_min(self.eps) + self.eps)
            self._probs = torch.softmax(logit / max(self.temperature, 1.0),
                                        dim=0)
        return self._probs

    # ── draw / record ───────────────────────────────────────────────────────

    def sample(self, k: int, generator: Optional[torch.Generator] = None
               ) -> torch.Tensor:
        """(k,) long cell indices drawn i.i.d. from ``probs()``."""
        if k <= 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.multinomial(self.probs(), int(k), replacement=True,
                                 generator=generator)

    def record(self, idx: torch.Tensor, success: torch.Tensor) -> None:
        """Push one outcome per entry of ``idx`` into that cell's ring.

        Duplicate indices in one call are written to consecutive ring slots
        (in the order given), so no outcome is lost to a scatter collision
        unless a single call carries more than H outcomes for one cell.
        """
        idx = idx.to(device=self.device, dtype=torch.long).flatten()
        if idx.numel() == 0:
            return
        succ = success.to(device=self.device).flatten().to(torch.int8)
        order = torch.argsort(idx, stable=True)
        s_idx, s_succ = idx[order], succ[order]
        first = torch.searchsorted(s_idx, s_idx)        # first slot of each run
        rank = torch.arange(s_idx.numel(), device=self.device) - first
        slot = (self.count[s_idx] + rank) % self.h
        self.buf[s_idx, slot] = s_succ
        self.count.index_add_(0, s_idx, torch.ones_like(s_idx))
        self.total_records += int(idx.numel())
        self._probs = None

    # ── diagnostics ─────────────────────────────────────────────────────────

    def stats(self, bins: int = 10) -> dict:
        """Host-side snapshot. Syncs; call at metrics time only."""
        visited = self.count >= 1
        filled = self.count.clamp(max=self.h)
        p_raw = self.buf.sum(dim=1, dtype=torch.float32) / filled.clamp_min(1)
        pv = p_raw[visited]
        P = self.probs()
        ess = float(1.0 / (P * P).sum())
        hist = (torch.histc(pv, bins=bins, min=0.0, max=1.0)
                if pv.numel() else torch.zeros(bins))
        return {
            'sgs_mean_p': float(pv.mean()) if pv.numel() else float('nan'),
            'sgs_ess': ess,
            'sgs_ess_frac': ess / self.n,
            'sgs_visited_frac': float(visited.float().mean()),
            'sgs_mean_count': float(self.count.float().mean()),
            'sgs_records': int(self.total_records),
            'sgs_p_hist': [int(x) for x in hist.tolist()],
        }
