# Dataset Information

This directory should contain the ETH-UCY dataset files. The ETH-UCY dataset is a collection of pedestrian trajectories recorded in various public spaces.

## Dataset Structure

The ETH-UCY dataset consists of 5 scenes:
- ETH
- Hotel
- University (Univ)
- Zara1
- Zara2

## File Format

Each dataset file contains trajectory data in the following format:
```
frame_id pedestrian_id pos_x pos_y
```

Where:
- `frame_id`: Frame number
- `pedestrian_id`: Unique pedestrian ID
- `pos_x`, `pos_y`: Position coordinates in meters

## Download Instructions

To download the ETH-UCY dataset:

1. Download the raw data from the original sources:
   - ETH and Hotel datasets: https://icu.ee.ethz.ch/research/datsets.html
   - UCY datasets (Univ, Zara1, Zara2): https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data

2. Preprocess the data to match the required format.

3. Place the processed files in this directory with the following names:
   - `biwi_eth.txt`
   - `biwi_hotel.txt`
   - `students001.txt`
   - `students003.txt`
   - `crowds_zara01.txt`
   - `crowds_zara02.txt`

## Data Statistics

| Dataset | Num Pedestrians | Num Frames | Avg Trajectory Length |
|---------|-----------------|------------|------------------------|
| ETH     | ~360            | ~7000      | ~57                    |
| Hotel   | ~390            | ~7000      | ~42                    |
| Univ    | ~430            | ~5000      | ~52                    |
| Zara1   | ~150            | ~8000      | ~41                    |
| Zara2   | ~200            | ~10000     | ~35                    |

## References

- Pellegrini et al., "You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking", ICCV 2009
- Lerner et al., "Crowds by Example", Computer Graphics Forum, 2007
