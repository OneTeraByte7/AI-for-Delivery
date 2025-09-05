import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "data/replay_greedy_coordinated.npz"
    data = np.load(path, allow_pickle=True)