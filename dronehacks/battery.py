"""LiPo battery model: voltage sag under load and with discharge, and the thrust it leaves the motors.

    pack = LiPoPack.from_json('assets/battery_4s5000.json')
    v = pack.voltage(soc=0.8, current_a=18.0)          # loaded terminal voltage
    f = pack.thrust_factor(v)                          # delivered / nominal thrust at full throttle authority
    soc = pack.step(soc, current_a, dt)                # coulomb counting

Model: terminal voltage = cells x OCV(SoC) - I x R_pack. A brushless motor's speed at a given throttle scales with
the voltage, and prop thrust with speed squared, so thrust capability scales with (V / V_ref)^2, V_ref being the
voltage the nominal plant is tuned at. Current comes from the electrical power the rotors need for their thrust,
P = P_hover x (T / T_hover)^1.5 (momentum theory), I = P / V.
Numbers are typical for a 4S 5000 mAh pack until measured ones replace them (assets/battery_4s5000.json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np

# per-cell open-circuit voltage vs state of charge, typical LiPo (rested)
_SOC = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_OCV = [3.27, 3.50, 3.61, 3.69, 3.73, 3.77, 3.80, 3.84, 3.89, 3.96, 4.06, 4.20]


@dataclass
class LiPoPack:
    cells: int = 4
    capacity_mah: float = 5000.0
    r_pack_ohm: float = 0.020                   # whole-pack internal resistance (wires + cells)
    hover_power_w: float = 200.0                # electrical power at hover (1.5 kg drone)
    hover_thrust_n: float = 1.5 * 9.81
    v_ref: float = 16.0                         # voltage the nominal plant (motor constants) corresponds to
    soc_curve: List[float] = field(default_factory=lambda: list(_SOC))
    ocv_curve: List[float] = field(default_factory=lambda: list(_OCV))

    @classmethod
    def from_json(cls, path: str) -> 'LiPoPack':
        with open(path) as f:
            return cls(**{k: v for k, v in json.load(f).items() if not k.startswith('_')})

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)

    def ocv(self, soc):
        return self.cells * np.interp(np.clip(soc, 0.0, 1.0), self.soc_curve, self.ocv_curve)

    def current(self, thrust_n, voltage):
        p = self.hover_power_w * (np.maximum(thrust_n, 0.0) / self.hover_thrust_n) ** 1.5
        return p / np.maximum(voltage, 1.0)

    def voltage(self, soc, current_a):
        return self.ocv(soc) - np.asarray(current_a) * self.r_pack_ohm

    def solve(self, soc, thrust_n, iters: int = 3):
        """(voltage, current) for a given total thrust: the current depends on the voltage it sags to."""
        v = self.ocv(soc)
        for _ in range(iters):
            i = self.current(thrust_n, v)
            v = self.voltage(soc, i)
        return v, i

    def thrust_factor(self, voltage):
        return np.clip((np.asarray(voltage) / self.v_ref) ** 2, 0.3, 1.15)

    def step(self, soc, current_a, dt):
        return np.clip(soc - np.asarray(current_a) * dt / (self.capacity_mah * 3.6), 0.0, 1.0)


def summary(pack: LiPoPack) -> str:
    rows = []
    for soc in (1.0, 0.8, 0.6, 0.4, 0.2):
        v, i = pack.solve(soc, pack.hover_thrust_n)
        rows.append(f'SoC {soc:.0%}: hover {v:.2f} V {i:.1f} A, thrust factor {float(pack.thrust_factor(v)):.3f}')
    flight = pack.hover_thrust_n * 1.3                   # a manoeuvring average
    v, i = pack.solve(1.0, flight)
    mins = pack.capacity_mah * 3.6 * 0.8 / i / 60
    rows.append(f'~{mins:.0f} min to 20% SoC at 1.3x hover thrust ({i:.0f} A)')
    return '\n'.join(rows)


if __name__ == '__main__':
    import sys
    p = LiPoPack.from_json(sys.argv[1]) if len(sys.argv) > 1 else LiPoPack()
    print(summary(p))
