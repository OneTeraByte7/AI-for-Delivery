from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import random

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

def _sign(x: int) -> int:
    return (x > 0) - (x < 0)

def _move_towards(src: Tuple[int, int], dist: Tuple[int, int]) -> int:
    (r0, c0), (r1, c1) = src, dist
    dr, dc = r1 - r0, c1 - c0
    if abs(dr) >= abs(dc):
        return DOWN if dr > 0 else (UP if dr < 0 else (RIGHT if dc > 0 else (LEFT if dc < 0 else STAY)))
    else:
        return RIGHT if dc > 0 else (LEFT if dc < 0 else (DOWN if dr > 0 else (UP if dr < 0 else STAY)))

def _closest(src: Tuple[int, int], targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not targets:
        return None
    return min(targets, key=lambda t: abs(t[0]-src[0]) + abs(t[1]-src[1]))