import numpy as np
import os
import matplotlib.pyplot as plt

class AdaptiveSampler:
    """Adaptive sampler with dynamic probability weighting"""

    def __init__(self, initial_weights: np.ndarray, smoothing_factor: float = 0.5):
        """Initialize sampler with initial weights and smoothing factor"""
        self.probability_weights = initial_weights
        self.update_counts = np.zeros_like(initial_weights, dtype=np.int32)
        self.smoothing_factor = smoothing_factor

    def sample_with_power_weighting(self, num_samples: int = 1, exponent: float = 3.0) -> np.ndarray:
        """Sample indices using power-weighted probabilities"""
        weighted_probs = np.power(self.probability_weights, exponent)

        if np.sum(weighted_probs) == 0:
            weighted_probs = np.ones_like(weighted_probs)

        normalized_probs = weighted_probs / np.sum(weighted_probs)

        sampled_indices = np.random.choice(
            len(self.probability_weights),
            size=num_samples,
            replace=True,
            p=normalized_probs
        )

        return sampled_indices

    def update_distribution(self, new_observations) -> None:
        """Update probability weights based on new observations"""
        new_observations = new_observations.cpu().numpy()

        # Create update mask (only positions with positive observations)
        update_mask = new_observations > 0

        # Smooth weight updates
        self.probability_weights[update_mask] = (
                self.probability_weights[update_mask] * self.smoothing_factor +
                (1 - self.smoothing_factor) * new_observations[update_mask]
        )

        # Update counts
        self.update_counts[update_mask] += 1

    def export_weights(self, file_path: str) -> None:
        """Export current weights to file"""
        np.save(file_path, self.probability_weights)

    def export_statistics(self, file_path: str) -> None:
        """Export update statistics to file"""
        np.save(file_path, self.update_counts)

    def get_current_weights(self) -> np.ndarray:
        """Get current probability weights"""
        return self.probability_weights.copy()

    def get_update_statistics(self) -> np.ndarray:
        """Get update counts for each index"""
        return self.update_counts.copy()


def visualize_sampling_pattern(weights_data: np.ndarray,
                               counts_data: np.ndarray,
                               title: str = "Sampling Pattern Analysis") -> None:
    """Visualize weight distribution and update frequency"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(weights_data, 'b-', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Weight Values', fontsize=12)
    ax1.set_title(f'{title} - Weight Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(counts_data, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Sampling Index', fontsize=12)
    ax2.set_ylabel('Update Counts', fontsize=12)
    ax2.set_title('Update Frequency', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


# Usage example
if __name__ == "__main__":
    analysis_dir = 'experiment_2025_09_11'
    base_log_dir = "../experiment_logs"
    experiment_id = 'exp_7000'

    weights_file = os.path.join(base_log_dir, analysis_dir, f'{experiment_id}_sampling_weights.npy')
    stats_file = os.path.join(base_log_dir, analysis_dir, f'{experiment_id}_sampling_stats.npy')

    weights_data = np.load(weights_file)
    stats_data = np.load(stats_file)

    fig = visualize_sampling_pattern(weights_data, stats_data, "Adaptive Sampling Analysis")
    plt.show()