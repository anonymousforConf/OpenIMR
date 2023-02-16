# OpenIMR
The PyTorch implementation of OpenIMR.

OpenIMR is proposed for open-world semi-supervised learning on graph data.

# Overview
You can enter the folder `run/` and then run run_ours.sh for open-world semi-supervised learning on the Coauthor CS dataset.

Specifically, the repository is organized as follows:

* `losses/` contains the implementation of supervised contrastive loss, which can be used for implementing the proposed PLCL loss.

* `networks/` contains the implementation of a GAT backbone.
 
* `util.py` is used for loading and pre-processing the dataset.

* `train_ours.py` is used for implementing the pipeline of OpenIMR.
 
