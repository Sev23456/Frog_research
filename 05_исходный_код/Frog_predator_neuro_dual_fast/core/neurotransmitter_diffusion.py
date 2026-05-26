"""Neuromodulator diffusion models adapted from Frog-main."""

import numpy as np


class NeurotransmitterDiffusion:
    def __init__(self, space_size=(400, 400), diffusion_rate=0.1, grid_resolution=20):
        self.space_size = space_size
        self.diffusion_rate = diffusion_rate
        self.grid_resolution = grid_resolution
        self.grid_size = max(8, int(np.sqrt(space_size[0] * space_size[1] / (grid_resolution ** 2))))
        self.dopamine_map = np.zeros((self.grid_size, self.grid_size))
        self.serotonin_map = np.zeros((self.grid_size, self.grid_size))
        self.acetylcholine_map = np.zeros((self.grid_size, self.grid_size))
        self.dopamine_baseline = 0.3
        self.serotonin_baseline = 0.3
        self.acetylcholine_baseline = 0.2
        self.decay_tau = 500.0
        self.diffusion_kernel = self._create_diffusion_kernel()

    def _create_diffusion_kernel(self):
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
        kernel /= kernel.sum()
        return kernel

    def _position_to_grid(self, position):
        x = int(position[0] / self.space_size[0] * self.grid_size)
        y = int(position[1] / self.space_size[1] * self.grid_size)
        return (np.clip(x, 0, self.grid_size - 1), np.clip(y, 0, self.grid_size - 1))

    def release(self, position, amount, transmitter_type="dopamine"):
        grid_pos = self._position_to_grid(position)
        target_map = self.dopamine_map
        if transmitter_type == "serotonin":
            target_map = self.serotonin_map
        elif transmitter_type == "acetylcholine":
            target_map = self.acetylcholine_map
        target_map[grid_pos] = min(1.0, target_map[grid_pos] + amount)

    def _apply_diffusion(self, concentration_map, dt, baseline):
        # Prefer scipy's fast convolution if available, otherwise fallback to numpy loop
        try:
            from scipy.signal import convolve2d

            diffused = convolve2d(concentration_map, self.diffusion_kernel, mode="same", boundary="symm")
        except Exception:
            padded = np.pad(concentration_map, 1, mode="edge")
            diffused = np.zeros_like(concentration_map)
            for y in range(concentration_map.shape[0]):
                for x in range(concentration_map.shape[1]):
                    window = padded[y:y + 3, x:x + 3]
                    diffused[y, x] = float(np.sum(window * self.diffusion_kernel))
        decay = np.exp(-dt / self.decay_tau)
        result = baseline + (diffused - baseline) * decay
        return np.clip(result, 0.0, 1.0)

    def diffuse(self, dt):
        self.dopamine_map = self._apply_diffusion(self.dopamine_map, dt, self.dopamine_baseline)
        self.serotonin_map = self._apply_diffusion(self.serotonin_map, dt, self.serotonin_baseline)
        self.acetylcholine_map = self._apply_diffusion(self.acetylcholine_map, dt, self.acetylcholine_baseline)

    def get_concentration(self, position, transmitter_type="dopamine"):
        grid_pos = self._position_to_grid(position)
        target_map = self.dopamine_map
        if transmitter_type == "serotonin":
            target_map = self.serotonin_map
        elif transmitter_type == "acetylcholine":
            target_map = self.acetylcholine_map
        return float(target_map[grid_pos])

    def get_concentration_vector(self, position):
        return {
            "dopamine": self.get_concentration(position, "dopamine"),
            "serotonin": self.get_concentration(position, "serotonin"),
            "acetylcholine": self.get_concentration(position, "acetylcholine"),
        }

