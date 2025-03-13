"""ETH-UCY dataset loader."""

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.dataset import TrajectoryDataset

class ETHUCYDataset(TrajectoryDataset):
    """ETH-UCY dataset for pedestrian trajectory prediction."""
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        delim: str = ' '
    ):
        """
        Initialize the ETH-UCY dataset.

        Args:
            data_dir: Directory containing dataset files
            dataset_name: Dataset name (eth, hotel, univ, zara1, zara2)
            obs_len: Observed trajectory length
            pred_len: Prediction trajectory length
            skip: Number of frames to skip
            min_ped: Minimum number of pedestrians in a sequence
            delim: Delimiter used in the dataset files
        """
        super(ETHUCYDataset, self).__init__(
            data_dir=data_dir,
            obs_len=obs_len,
            pred_len=pred_len,
            skip=skip,
            min_ped=min_ped,
            delim=delim
        )

        self.dataset_name = dataset_name

        # Map dataset names to files
        self.dataset_files = {
            'eth': ['biwi_hotel.txt'],
            'hotel': ['biwi_eth.txt'],
            'univ': ['students001.txt', 'students003.txt'],
            'zara1': ['crowds_zara01.txt'],
            'zara2': ['crowds_zara02.txt'],
            'all': ['biwi_hotel.txt', 'biwi_eth.txt', 'students001.txt',
                   'students003.txt', 'crowds_zara01.txt', 'crowds_zara02.txt']
        }

        # Load data
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset files and process them."""
        files = self.dataset_files.get(self.dataset_name, [])
        if not files:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        all_sequences = []
        all_seq_start_end = []
        num_peds_so_far = 0  # 跟踪到目前为止处理的行人数量

        for file_path in files:
            full_path = os.path.join(self.data_dir, file_path)
            sequences, seq_start_end = self._load_data(full_path)

            # 调整序列开始和结束索引，考虑到前面文件中已有的行人
            if all_sequences:
                seq_start_end = [(s + num_peds_so_far, e + num_peds_so_far)
                                  for s, e in seq_start_end]

            all_sequences.extend(sequences)
            all_seq_start_end.extend(seq_start_end)

            # 更新到目前为止处理的行人总数
            # 每个序列包含一组行人轨迹，将它们全部加到计数器中
            for seq in sequences:
                num_peds_so_far += seq.shape[1]  # 第二维是行人数量

        self.sequences = all_sequences
        self.seq_start_end = all_seq_start_end

        print(f"Loaded {self.dataset_name} dataset with {len(self.sequences)} sequences")
        print(f"Total pedestrians: {num_peds_so_far}")

    def _load_data(self, file_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Load and process data from a file.

        Args:
            file_path: Path to the data file

        Returns:
            Tuple of (sequences, sequence_start_end)
        """
        data = []
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            for line in f:
                # 检测并处理不同类型的分隔符
                if '\t' in line:
                    line = line.strip().split('\t')
                else:
                    line = line.strip().split(self.delim)

                # 确保去除任何空字符串
                line = [i for i in line if i]
                line = [float(i) for i in line]

                frame_id = int(line[0])
                ped_id = int(line[1])
                x, y = line[2], line[3]

                data.append([frame_id, ped_id, x, y])

        data = np.array(data)

        # Get all unique pedestrians and frames
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []

        # Group data by frame
        for frame in frames:
            frame_data.append(data[data[:, 0] == frame, :])

        # Create sequences of consecutive frames
        num_sequences = len(frames) - self.seq_len + 1
        sequences = []
        seq_start_end = []

        # 跟踪所有序列中的行人总数
        total_peds = 0

        for idx in range(0, num_sequences, self.skip):
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0
            )

            # Get all pedestrians in this sequence
            peds_in_seq = np.unique(curr_seq_data[:, 1])

            # Filter pedestrians that appear throughout the sequence
            valid_peds = []
            for ped in peds_in_seq:
                ped_data = curr_seq_data[curr_seq_data[:, 1] == ped]
                if len(ped_data) == self.seq_len:
                    valid_peds.append(ped)

            # Skip if not enough pedestrians
            if len(valid_peds) < self.min_ped:
                continue

            # 记录此序列中行人的开始和结束索引
            start_ped_idx = total_peds
            end_ped_idx = start_ped_idx + len(valid_peds)

            # 添加到seq_start_end中
            seq_start_end.append((start_ped_idx, end_ped_idx))

            # 更新行人总数
            total_peds += len(valid_peds)

            # Create a new sequence
            curr_seq = np.zeros((self.seq_len, len(valid_peds), 2))

            # Fill the sequence with data
            for i, ped in enumerate(valid_peds):
                ped_data = curr_seq_data[curr_seq_data[:, 1] == ped]
                curr_seq[:, i, 0] = ped_data[:, 2]  # x-coord
                curr_seq[:, i, 1] = ped_data[:, 3]  # y-coord

            # Add the sequence
            sequences.append(curr_seq)

        return sequences, seq_start_end