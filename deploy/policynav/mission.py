"""policynav mission logic: when the policy may fly, and what ends the flight. PURE: no ROS, no clock reads.

  wait     until PX4 has a valid local position + heading, the estimator is healthy, depth is live, attitude
           is known; and the OPERATOR has passed arm=true (nothing arms without it)
  arm      stream zero-velocity setpoints for stream_lead_s, request Offboard, arm
  fly      every control period: newest depth + state -> policy -> velocity setpoint
  land     PX4 AUTO_LAND (stop streaming; PX4 owns the landing)
  done     landed + disarmed, yielded, or refused
Ends the flight:
  * completion: a 5-action (handoff) policy's 'complete' output while landed, or (4-action policy) a touchdown
    after having been airborne -> disarm (PX4 refuses an in-air disarm)
  * depth older than depth_stale_s -> hold (zero velocity); older than depth_lost_land_s -> land
  * estimator unhealthy/silent for health_bad_land_s while airborne -> land
  * PX4 leaves Offboard while airborne (pilot / failsafe) -> stop commanding entirely (yield)
  * flight time > max_flight_s, or farther than geofence_m from the takeoff point -> land
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NAV_OFFBOARD, NAV_AUTO_LAND = 14, 18


@dataclass
class MissionParams:
    arm: bool = False                 # operator authorisation; False = run everything except arming
    control_dt: float = 0.1           # the policy's control period (training: 10 Hz)
    stream_lead_s: float = 1.0
    arm_wait_s: float = 15.0
    depth_stale_s: float = 0.5
    depth_lost_land_s: float = 3.0
    health_bad_land_s: float = 1.0
    max_flight_s: float = 180.0
    geofence_m: float = 8.0
    airborne_alt_m: float = 0.3       # above this the flight counts as airborne (touchdown = landing after it)


@dataclass
class Vehicle:
    pos: tuple = None                 # NED local (m)
    vel: tuple = None
    q: tuple = None                   # (w, x, y, z) FRD -> NED
    pos_ok: bool = False
    armed: bool = False
    nav_state: int = -1
    landed: bool = True


class Mission:
    def __init__(self, p: MissionParams):
        self.p = p
        self.phase, self.why, self.result = 'wait', 'starting', None
        self.t_phase = None
        self.home = None
        self.depth_t = None
        self.healthy, self.health_t = False, None
        self.airborne = False
        self.events = []

    # inputs
    def on_depth(self, t):
        self.depth_t = t

    def on_health(self, t, ok):
        self.healthy, self.health_t = bool(ok), t

    def _set(self, t, phase, why):
        if phase != self.phase:
            self.events.append(f'{self.phase} -> {phase}: {why}')
        self.phase, self.why, self.t_phase = phase, why, t

    def step(self, t, v: Vehicle, policy_complete: bool = False):
        """-> (stream_velocity: bool, commands: list[str] of 'offboard'|'arm'|'land'|'disarm'). The node fills the
        velocity from the policy while phase == 'fly' (zero while holding)."""
        p, cmds = self.p, []
        if self.t_phase is None:
            self.t_phase = t
        depth_age = math.inf if self.depth_t is None else t - self.depth_t
        health_ok = self.healthy and self.health_t is not None and t - self.health_t <= 1.0
        if self.phase == 'wait':
            missing = [n for n, ok in (('local position', v.pos_ok), ('attitude', v.q is not None),
                                       ('estimator health', health_ok), ('depth', depth_age <= p.depth_stale_s)) if not ok]
            if missing:
                self.why = 'waiting for ' + ', '.join(missing)
                return False, cmds
            if not p.arm:
                self.why = 'ready; not arming (arm:=false)'
                return False, cmds
            self.home = np.array(v.pos, float)
            self._set(t, 'arm', 'ready')
        if self.phase == 'arm':
            if v.armed and v.nav_state == NAV_OFFBOARD:
                self._set(t, 'fly', 'armed in Offboard')
            elif t - self.t_phase > p.arm_wait_s:
                self._set(t, 'done', 'PX4 did not arm / enter Offboard')
                self.result = 'refused'
                return False, ['disarm'] if v.armed else []
            else:
                if t - self.t_phase >= p.stream_lead_s:
                    cmds = ['offboard'] if v.nav_state != NAV_OFFBOARD else []
                    cmds += [] if v.armed else ['arm']
                return True, cmds
        if self.phase == 'fly':
            alt = -(v.pos[2] - self.home[2])
            self.airborne |= alt > p.airborne_alt_m
            if v.armed and v.nav_state != NAV_OFFBOARD and v.nav_state != NAV_AUTO_LAND:
                self._set(t, 'done', f'PX4 left Offboard (nav_state {v.nav_state}): yielded')
                self.result = 'yielded'
                return False, []
            if not v.armed:
                self._set(t, 'done', 'disarmed')
                self.result = 'complete' if self.airborne else 'disarmed'
                return False, []
            touchdown = v.landed and self.airborne
            if policy_complete or touchdown:
                if v.landed:
                    self._set(t, 'done', 'complete: landed' + (' + policy complete' if policy_complete else ''))
                    self.result = 'complete'
                    return True, ['disarm']
            land_why = None
            if depth_age > p.depth_lost_land_s:
                land_why = f'no depth for {depth_age:.1f} s'
            elif self.airborne and not health_ok and (self.health_t is None or t - self.health_t > p.health_bad_land_s
                                                      or not self.healthy):
                land_why = 'estimator unhealthy'
            elif t - self.t_phase > p.max_flight_s:
                land_why = f'flight time > {p.max_flight_s:.0f} s'
            elif np.linalg.norm(np.asarray(v.pos, float) - self.home) > p.geofence_m:
                land_why = f'beyond the {p.geofence_m:.0f} m geofence'
            if land_why:
                self._set(t, 'land', land_why)
                return False, ['land']
            self.why = 'holding (stale depth)' if depth_age > p.depth_stale_s else 'policy'
            return True, cmds
        if self.phase == 'land':
            if v.landed and not v.armed:
                self._set(t, 'done', 'landed')
                self.result = self.result or 'landed'
            elif v.nav_state != NAV_AUTO_LAND and v.armed:
                cmds = ['land']
            return False, cmds
        return False, cmds

    def holding(self, t) -> bool:
        return self.depth_t is None or t - self.depth_t > self.p.depth_stale_s
