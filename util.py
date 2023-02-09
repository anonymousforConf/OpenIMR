from typing import AnyStr, List, Tuple, Union
import random
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

import torch
import dgl
import dgl.data as dgl_data

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def getDataset(dataset_name: AnyStr):
    dataset = None
    if dataset_name == "citeseer":
        dataset = dgl_data.CiteseerGraphDataset(verbose=False)[0]
    elif dataset_name == "coauthor_cs":
        dataset = dgl_data.CoauthorCSDataset(verbose=False)[0]
    elif dataset_name == "coauthor_phy":
        dataset = dgl_data.CoauthorPhysicsDataset(verbose=False)[0]
    elif dataset_name == "amazon_computers":
        dataset = dgl_data.AmazonCoBuyComputerDataset(verbose=False)[0]
    elif dataset_name == "amazon_photos":
        dataset = dgl_data.AmazonCoBuyPhotoDataset(verbose=False)[0]
    else:
        raise ValueError
    return dataset


def split_node_dataset(dataset: dgl.DGLGraph, seed: int, num_node_per_class: int):
    """
    Split train, validation, and test sets
    For each class, we sample num_node_per_class labeled instances to form the train set.
    """
    g: dgl.DGLGraph = dataset
    num_nodes = g.num_nodes()
    development_labels = g.ndata["label"].cpu().numpy()
    n_class = torch.max(g.ndata["label"]).item() + 1
    class_list = list(range(n_class))
    random.shuffle(class_list)
    n_train_class = round(n_class / 2)
    # print(class_list, n_train_class)
    dic_temp = {}
    new_id = 0
    for cls in class_list:
        dic_temp[cls] = new_id
        new_id += 1

    train_idx, val_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    for cls_id in class_list[: n_train_class]:
        candidate = np.where(development_labels == cls_id)[0]
        random.shuffle(candidate)
        train_idx = np.append(train_idx, candidate[: num_node_per_class])
        val_idx = np.append(val_idx, candidate[num_node_per_class: 2*num_node_per_class])
        test_idx = np.append(test_idx, candidate[2*num_node_per_class:])

    for cls_id in class_list[n_train_class: n_class]:
        candidate = np.where(development_labels == cls_id)[0]
        test_idx = np.append(test_idx, candidate)

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    train_mask = get_mask(train_idx)
    val_mask = get_mask(val_idx)
    test_mask = get_mask(test_idx)

    g.ndata['train_mask'] = train_mask.to(g.device)
    g.ndata['val_mask'] = val_mask.to(g.device)
    g.ndata['test_mask'] = test_mask.to(g.device)

    mask_lab = np.append(mask_lab, g.ndata['train_mask'].cpu().bool().numpy()+g.ndata['val_mask'].cpu().bool().numpy())
    mask_cls = np.append(mask_cls, np.array([True if x.item() in class_list[: n_train_class]
                                         else False for x in development_labels]))
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)
    
    for i in range(g.num_nodes()):
        g.ndata["label"][i] = dic_temp[g.ndata["label"][i].item()]

    return mask_lab, mask_cls, n_class, n_train_class
    

def load_data(dataset_str: str, seed: int, num_node_per_class: int):
    g = getDataset(dataset_str)
    g = dgl.add_self_loop(g)
    mask_lab, mask_cls, n_class, n_train_class = split_node_dataset(g, seed, num_node_per_class)

    input_dim = g.ndata["feat"].shape[1]
    n_train_nodes = len(np.where(g.ndata["train_mask"] == True)[0])
    n_val_nodes = len(np.where(g.ndata["val_mask"] == True)[0])
    n_test_nodes = len(np.where(g.ndata["test_mask"] == True)[0])

    print('Number of nodes: ', g.num_nodes())
    print('Number of edges: ', g.num_edges())
    print('Initial feature dimension: ', input_dim)
    print('Number of classes: ', n_class)
    print('Number of training nodes: ', n_train_nodes)
    print('Number of validation nodes: ', n_val_nodes)
    print('Number of testing nodes: ', n_test_nodes)

    return g, input_dim, n_class, n_train_class, mask_lab, mask_cls


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    if total_new_instances !=0:
        new_acc /= total_new_instances  

    return total_acc, old_acc, new_acc


def compute_var_mean(emb, labels):
    centers = []
    for cls_id in set(labels):
        idxs = np.where(labels == cls_id)[0]
        centers.append(np.reshape(np.mean(emb[idxs], 0), (1, -1)))
    centers = np.concatenate(centers, 0)
    distances = (np.sum((emb - centers[labels])**2, 1))
    variances = []
    for cls_id in set(labels):
        idxs = np.where(labels == cls_id)[0]
        variances.append(np.mean(distances[idxs])**0.5)

    return variances, centers