# Depth Based Semantic Scene Completion with Position Importance Aware Loss

By [Yu Liu*](https://sites.google.com/site/yuliuunilau/home), Jie Li*, Xia Yuan, Chunxia Zhao, [Roland Siegwart](https://scholar.google.com/citations?user=MDIyLnwAAAAJ&hl=en), [Ian Reid](https://cs.adelaide.edu.au/~ianr/) and [Cesar Cadena](http://n.ethz.ch/~cesarc/) (* indicates equal contribution)


ICRA2020 In Conjunction of RAL

## Video Demo: 
https://youtu.be/j-LAMcMh0yg

## Requirements:
python 2.7

pytorch 0.4.1

CUDA 8


## Testing
python ./test.py \
--data_test=/path/to/NYUCADtest \
--batch_size=1 \
--workers=4 \
--resume='PALNet_weights.pth.tar'

## Weights
[Model trained on NYUCAD](https://drive.google.com/open?id=1BRNliQmEaPIphZvbzhR55fEHeOh9U9Ix)

## Datasets 
The original dataset is from [SSCNet](https://github.com/shurans/sscnet)

Here is the [NYUCAD](https://drive.google.com/open?id=10Iz7lkJf8kbtUf1OyL-Z1xW6eZRoF3d8) data reproduced from SSCNet for a quick demo.


## Adelaide AI Group
more work from Adelaide can be found in: https://github.com/Adelaide-AI-Group/Adelaide-AI-Group.github.io
