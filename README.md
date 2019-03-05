# AVSnap

This repository contains demo code for the paper <a href = "https://arxiv.org/abs/1808.06250">Dynamic Temporal Alignment of Speech to Lips </a>
The repository forks the demo for the audio-to-video synchronisation network (<a href = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/">SyncNet</a>).

The model and code can be used for research purposes under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</a>.
Please cite the papers below if you make use of the software.

## Prerequisites
The following packages are required:
```
pytorch (0.4.0)
numpy (1.14.3)
scipy (1.0.1)
python_speech_features (0.6)
cuda (8.0)
ffmpeg (3.4.2)
tensorflow (1.2, 1.4)
scikit-image (0.14,1)
imageio (2.2.0)
```

The demo has been tested with the package versions shown above, but may also work on other versions.

## Demo

First, download face detection and SyncNet models
```
sh download_model.sh
```

To run the demo, aligning audio from video2.avi to video from video1.avi
```
python run_pipeline.py data/video1.avi
python run_pipeline.py data/video2.avi
python align_audio.py data/video1.avi data/video2.avi --modality=all
```

The alignment will be computed based on both video and audio of the input videos.
You can use only one of the modalities by changing the flag ```--modality``` to one of ```va/aa/av/vv```, the first letter indicates which modality is used from the first input, and the last letter indicates which is used from the second.

Outputs:
```
data/out/video1_video2/modality_all.mp4
```

## Publications

```
@inproceedings{halperin19dynamic,
  title     =   {Dynamic Temporal Alignment of Speech to Lips‏},
  author    =   {Halperin, Tavi and Ephrat, Ariel and Peleg, Shmuel},
  booktitle =   {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      =   {2019},
}
```
SyncNet is taken from:
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
