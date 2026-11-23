# Leakage audit: DF40 items removed

`excluded_videos.csv` lists the 723 DF40 videos (29,328 frames of the sampled training pool, 196 FaceForensics++ identifiers) whose identifiers occur in our FaceForensics++ validation split and were therefore excluded before retraining. Columns: video name, number of frames in the pool, DF40 methods containing it. The items are marked `excluded_ffval` in `data/benchmarks/df40_x/index.csv`.
