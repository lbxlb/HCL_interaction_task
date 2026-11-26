"""Pure task logic for a slider button; inputs must refer to the same simulation state."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PressCriteria:
    # Demonstration settings for the temporary slider, not hardware specifications.
    press_position_m: float = 0.00015
    release_position_m: float = 0.00003
    press_hold_s: float = 0.020
    release_hold_s: float = 0.020

    def __post_init__(self):
        values = (self.press_position_m, self.release_position_m,
                  self.press_hold_s, self.release_hold_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Criteria must be finite')
        if not 0 <= self.release_position_m < self.press_position_m:
            raise ValueError('Require 0 <= release position < press position')
        if min(self.press_hold_s, self.release_hold_s) <= 0:
            raise ValueError('Hold durations must be positive')


class ButtonPressState:
    """Debounce sampled conditions using simulation seconds, with a latched press.

    Call once per physics observation. Persistence means all observed samples
    satisfy the condition; this cannot detect behavior between missing samples.
    contact_loaded must describe the matching finger/button pair with dist <= 0
    and positive normal force. A mere proximity record is insufficient.
    """

    def __init__(self, criteria=None):
        self.criteria = criteria if criteria is not None else PressCriteria()
        self.reset()

    def reset(self):
        self.phase = 'ready'
        self.press_count = 0
        self.release_count = 0
        self._since = None
        self._last_time = None

    @property
    def pending_elapsed_s(self):
        return 0.0 if self._since is None else self._last_time - self._since

    def update(self, time_s, position_m, contact_loaded):
        if not math.isfinite(time_s) or not math.isfinite(position_m):
            raise ValueError('Time and position must be finite')
        if self._last_time is not None and time_s <= self._last_time:
            raise ValueError('Simulation time must increase; reset between episodes')
        if type(contact_loaded) is not bool:
            raise ValueError('contact_loaded must be a bool')
        self._last_time = time_s
        event = None
        c = self.criteria
        if self.phase in ('ready', 'press_pending'):
            if position_m >= c.press_position_m and contact_loaded:
                if self.phase == 'ready':
                    self.phase, self._since = 'press_pending', time_s
                if time_s - self._since >= c.press_hold_s - 1e-12:
                    self.phase, self._since = 'pressed', None
                    self.press_count += 1
                    event = 'pressed'
            else:
                self.phase, self._since = 'ready', None
        else:
            if position_m <= c.release_position_m and not contact_loaded:
                if self.phase == 'pressed':
                    self.phase, self._since = 'release_pending', time_s
                if time_s - self._since >= c.release_hold_s - 1e-12:
                    self.phase, self._since = 'ready', None
                    self.release_count += 1
                    event = 'released'
            else:
                self.phase, self._since = 'pressed', None
        return dict(phase=self.phase, event=event, press_count=self.press_count,
                    release_count=self.release_count)
