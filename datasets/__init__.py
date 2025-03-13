# datasets/__init__.py
from datasets.dataset import TrajectoryDataset
from datasets.eth_ucy import ETHUCYDataset

__all__ = ['TrajectoryDataset', 'ETHUCYDataset']