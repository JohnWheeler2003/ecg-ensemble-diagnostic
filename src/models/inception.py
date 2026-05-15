import torch
import torch.nn as nn
import numpy as np


class InceptionModule(nn.Module):
    """Inception block for time-series data. Applies multiple kernel sizes in parallel to capture both short-term (high frequency) and long-term (low frequency) temporal patterns simultaneously"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels=32,
        kernel_sizes=[10, 20, 40],
    ):
        """Initializes Inception module"""
        super().__init__()
        # Bottleneck reduces dimensionality to save compute, unless input channels are already small
        self.bottleneck = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=False,
            )
            if in_channels > 1
            else nn.Identity()
        )

        # Input to the parallel convs depends on whether a bottleneck was used
        conv_in_channels = bottleneck_channels if in_channels > 1 else in_channels

        # Parallel 1D Convolutions with varying receptive fields
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conv_in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding="same",  # Ensures temporal length remains constant
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        # Max-Pooling branch
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )

        # Batch Normalization and Activation applied after concatenation
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes) + out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        """Forward pass of the Inception module.

        Input: x (torch.Tensor) which is a tensor of shape (Batch, Channels, Sequence_Length).

        Output: torch.Tensor that has been concatenated, normalized, and activated output"""
        x_bottleneck = self.bottleneck(x)

        # Process parallel branches
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]
        pool_output = self.maxpool(x)

        # Concatenate along the cahnnel dimension (dim=1)
        x_out = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.silu(self.bn(x_out))


class ECGInceptionTime(nn.Module):
    """Macro-architecture for ECG classification. Stacks multiple InceptionModules with redisual skip connections, followed by Global Average Pooling and a linear classification head"""

    def __init__(self, in_channels=12, num_classes=5, num_blocks=3):
        """Initializes full ECG InceptionTime network"""
        super().__init__()

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        current_channels = in_channels

        # Stack multiple Inception blocks with Residual Connections
        for i in range(num_blocks):
            out_channels = 32  # Base channel count per parallel conv
            inception_out_channels = (
                out_channels * 3
            ) + out_channels  # 3 convs + 1 pool branch

            self.blocks.append(
                InceptionModule(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    bottleneck_channels=32,
                    kernel_sizes=[10, 20, 40],
                )
            )

            # If the input channels don't match the output channels, project them using 1x1 Conv
            if current_channels != inception_out_channels:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(
                            current_channels,
                            inception_out_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.BatchNorm1d(inception_out_channels),
                    )
                )
            else:
                # If dimensions already match, pass data straight through
                self.shortcuts.append(nn.Identity())

            current_channels = inception_out_channels

        # Global Average Pooling to flatten temporal dimension
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.fc = nn.Linear(current_channels, num_classes)

        # Initialize weights immediately upon instantiation
        self._initialize_weights()

    def forward(self, x):
        """Forward pass for the full network

        Input: x (torch.Tensor) of shape (Batch, 12, 1000)

        Output: torch.Tensor that is the raw prediction logits of shape (Batch, 5)"""
        # x shape : (Batch, 12, 1000)
        for i, block in enumerate(self.blocks):
            # Save original input and push it through the shortcut
            residual = self.shortcuts[i](x)

            # Push input through inception block
            x = block(x)

            # Add them together (skip connection)
            x = x + residual

        x = self.gap(x)
        x = x.squeeze(-1)
        logits = self.fc(x)
        return logits

    def _initialize_weights(self):
        """Applies Kaiming Normalization Initialization to Convolutional layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_prior_bias(self, class_priors):
        """Initializes linear layer based on training set class distributions. prevents massive loss spikes when dealing with the imbalanced data"""
        assert len(class_priors) == self.fc.out_features, (
            "Priors must match number of classes"
        )

        class_priors = np.clip(class_priors, 1e-7, 1 - 1e-7)
        init_biases = np.log(class_priors / (1.0 - class_priors))
        self.fc.bias.data = torch.tensor(init_biases, dtype=torch.float32)
