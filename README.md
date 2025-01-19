# KITTI Odometry

Scenes and characteristics of the dataset, according to sequences.

| Sequence | Frames | Scenes and characteristic       |
| -------- | ------ | ------------------------------- |
| 00       | 4541   | Town.                           |
| 01       | 1101   | High way.                       |
| 02       |        |                                 |
| 03       |        |                                 |
| 04       |        |                                 |
| 05       | 2761   | Town.                           |
| 06       | 1101   | Straight line.                  |
| 07       | 1101   | Town.                           |
| 08       | 4071   | Town, with lots of trees aside. |
| 09       | 1591   | Long way, around the town.      |
| 10       | 1201   | Long way.                       |



## Scheme

Sequences `00`, `05` and `07` are to be adopted in our approach as the data to train.



## Our Dataset

| Sequence | Frames | Scenes and characteristic    |
| -------- | ------ | ---------------------------- |
| 00       | 1407   | 5th floor of gml by mid360   |
| 01       | 1405   | 5th floor of gml by xt16     |
| 02       | 3739   | underground of gml by mid360 |
| 03       | None   | underground of gml by xt16   |
| 04       | 3506   | outside of gml by mid360     |
| 05       | None   | outside of gml by xt16       |



## Network and Training

Before training, ensure that the `workspace name` and input data `cropping resolution` are correctly set.