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