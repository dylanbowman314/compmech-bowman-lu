from typing import List

import torch
from torch import nn


class LinearProbe(nn.Module):
    """
    Basic linear probe.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, dtype=torch.float32)

    def forward(self, residual_stream: torch.Tensor):
        return self.linear(residual_stream)


class LinearProbeWithSoftmax(nn.Module):
    """
    Linear layer -> softmax for inputs of shape (batch_size, seq_len, num_states]
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(LinearProbeWithSoftmax, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, dtype=torch.float32)

    def forward(self, residual_stream: torch.Tensor):
        x = torch.softmax(self.linear(residual_stream), dim=2)
        return x


class CorrelationDimensionLoss(nn.Module):
    def __init__(self, r_values, k=10, device=torch.device("cuda")):
        super(CorrelationDimensionLoss, self).__init__()
        self.r_values = torch.tensor(r_values, dtype=torch.float32, device=device)
        self.k = k

    def forward(self, points):
        distances = torch.cdist(points, points, p=2)
        distances = distances.triu(diagonal=1)
        distances = distances[distances > 0]

        correlation_sums = []
        for r in self.r_values:
            smooth_heaviside = torch.sigmoid(self.k * (r - distances))
            correlation_sum = smooth_heaviside.mean()
            correlation_sums.append(correlation_sum)

        correlation_sums = torch.stack(correlation_sums)
        log_r = torch.log(self.r_values)
        log_c = torch.log(correlation_sums)

        # Perform linear regression on log-log values
        A = torch.stack([log_r, torch.ones_like(log_r)], dim=1)
        slope, intercept = torch.linalg.lstsq(A, log_c).solution

        correlation_dimension = slope  # .item()

        return -correlation_dimension


class BoxCountingDimensionLoss(nn.Module):
    def __init__(
        self,
        epsilons,
        sigma=0.1,
        spread_weight=0.1,
        p=2,
        ltz_weight=0.1,
        ato_weight=0.1,
        device=torch.device("cuda"),
    ):
        super(BoxCountingDimensionLoss, self).__init__()
        self.epsilons = torch.tensor(epsilons, dtype=torch.float32, device=device)
        self.sigma = sigma
        self.spread_weight = spread_weight
        self.p = p  # norm
        self.ltz_weight = ltz_weight
        self.ato_weight = ato_weight

    def forward(self, points):
        batch_size, num_points, dim = points.size()
        counts = []

        for epsilon in self.epsilons:
            # Scale points to fit in unit cube
            scaled_points = points / epsilon

            # Create a Gaussian kernel to approximate box occupancy
            diff = scaled_points.unsqueeze(2) - scaled_points.unsqueeze(1)
            dist = torch.exp(-torch.sum(diff**2, dim=-1) / (2 * self.sigma**2))
            occupancy = dist.mean(dim=2)

            # Approximate number of occupied boxes
            count = torch.sum(occupancy, dim=1).mean()
            counts.append(count)

        counts = torch.stack(counts)
        log_epsilon = torch.log(self.epsilons)
        log_counts = torch.log(counts)

        # Perform linear regression on log-log values
        A = torch.stack([log_epsilon, torch.ones_like(log_epsilon)], dim=1)
        slope, intercept = torch.linalg.lstsq(A, log_counts).solution

        fractal_dimension = slope

        # Spread/Diversity term to avoid point collapse
        pairwise_distances = torch.cdist(points, points, p=self.p)
        spread_term = torch.mean(pairwise_distances)

        # Convexity regularization
        less_than_zero_term = torch.mean(torch.square(torch.relu(-points)))
        add_to_one_term = torch.mean(torch.square(torch.sum(points, dim=2) - 1))

        # Total loss
        total_loss = (
            fractal_dimension
            - (self.spread_weight * spread_term)
            + (self.ltz_weight * less_than_zero_term)
            + (self.ato_weight * add_to_one_term)
        )

        return total_loss
