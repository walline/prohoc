import argparse
import os
from collections import defaultdict
import importlib

import torch

from lib.hierarchy import Hierarchy
from lib import hierarchy_metrics as hm

from lib.utils.hierarchy_utils import (get_avg_hdist,
                                       get_path_indices,
                                       get_hdist_matrix,
                                       expected_hdist,
                                       gen_flat2node_map,
                                       get_children_and_group_maps,
                                       get_multidepth_classes)
from lib.utils.dataset_util import get_id_classes

parser = argparse.ArgumentParser()

parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--basedir", type=str, required=True)
parser.add_argument("--id_split", type=str, required=True)
parser.add_argument("--uncertainty_methods", nargs="+", required=True)
parser.add_argument("--betas", nargs="+", default=[0])
parser.add_argument("--farood", nargs="+", default=[])


def fuse_predictions(softmax,
                     hierarchy,
                     multi_classes,
                     children_maps,
                     group_sizes,
                     path_indices,
                     flat2node_map,
                     uncertainty_method,
                     uncertainty_args,
                     ):

    max_height = len(softmax)
    n_samples = softmax[0].size(0)

    device = softmax[0].device

    comp_sums = []

    for depth in range(max_height - 1):

        height = max_height - depth - 1

        n_parents = len(multi_classes[depth])

        single_element_mask = group_sizes[depth] == 1
        children_map = children_maps[depth]

        p = softmax[height-1]

        result, p_comp = uncertainty_method(p,
                                            children_map,
                                            group_sizes[depth],
                                            n_samples,
                                            n_parents,
                                            device=device,
                                            **uncertainty_args)

        mapped_single_mask = single_element_mask[children_map]

        p_comp[:, single_element_mask] = 0.0
        result[:, mapped_single_mask] = 1.0

        comp_sums.append(p_comp)
        softmax[height-1].copy_(result)

    expanded_probs = [p[:, path_indices[i]] for i, p in enumerate(reversed(softmax))]
    stacked_probs = torch.stack(expanded_probs, dim=-1)
    cumulative_probs = torch.cumprod(stacked_probs, dim=-1)
    
    results = []

    for height in range(max_height):

        n_classes = len(multi_classes[-height-1])
        depth = max_height - height - 1

        intermediate_prod = torch.zeros(n_samples, n_classes, device=device)
        intermediate_prod[:, path_indices[depth]] = cumulative_probs[:, :, depth]

        if height > 0:
            intermediate_prod = intermediate_prod * comp_sums[depth]

        results.append(intermediate_prod)

    results = torch.cat(results, dim=1)

    results_merged = torch.zeros(n_samples, len(hierarchy.id_node_list), device=device)
    
    results_merged.scatter_add_(1, flat2node_map.expand(n_samples, -1), results)

    psums = torch.sum(results_merged, dim=-1)

    assert torch.allclose(psums, torch.ones_like(psums), atol=1e-4)

    return results_merged


def get_root_probability(softmax_probs, beta):

    top_probs = softmax_probs[-1]
    entropy_probs = softmax_probs[0]

    eps = 1e-9
    entropy = -torch.sum(entropy_probs * torch.log(entropy_probs + eps), dim=1)

    p_ = torch.cat([top_probs, beta * entropy.unsqueeze(-1)], dim=1)
    new_sums = torch.sum(p_, dim=-1, keepdim=True)
    top_probs_rescaled = p_ / new_sums

    root_probs = top_probs_rescaled[:, -1]

    return root_probs


def predict_classes(val_logits,
                    ood_logits,
                    heights_val,
                    heights_ood,
                    id_hierarchy,
                    multi_classes):

    ood_preds = []

    for i, h in enumerate(heights_ood):
        _, pred = ood_logits[h][i, :].max(dim=0)
        c = multi_classes[-(h+1)][int(pred)]
        full_class = id_hierarchy.id_node_list.index(c)
        ood_preds.append(full_class)

    val_preds = []

    for i, h in enumerate(heights_val):
        _, pred = val_logits[h][i, :].max(dim=0)
        c = multi_classes[-(h+1)][int(pred)]
        full_class = id_hierarchy.id_node_list.index(c)
        val_preds.append(full_class)

    ood_preds = torch.tensor(ood_preds)
    val_preds = torch.tensor(val_preds)

    return val_preds, ood_preds


def get_results(preds,
                node_labels,
                id_hierarchy,
                dists_mats=None,
                ):

    hmet = hm.HierarchicalPredAccuracy(id_hierarchy, track_hdist=True)

    hmet.update_state(preds.long(),
                      node_labels,
                      dists_mats=dists_mats)

    hd = hmet.result_hierarchy_distances()
    balanced_acc = hmet.result_balanced_accuracy()
    balanced_hdist = hmet.result_balanced_hierarchy_distance()
    class_hdists = hmet.result_class_hdists()

    return {"acc": hmet.result(),
            "balanced_acc": balanced_acc,
            "hdist": hd,
            "avg_hdist": get_avg_hdist(hd),
            "balanced_hdist": balanced_hdist,
            "class_hdists": class_hdists,
            }


class HInferenceEvaluator():

    def __init__(self, args):

        id_classes = get_id_classes(args.id_split)
        id_hierarchy = Hierarchy(id_classes, args.hierarchy)
        ood_classes = id_hierarchy.ood_train_classes

        self.max_depth = id_hierarchy._max_depth
        print("Depth: ", self.max_depth)

        print("Gather scores...")

        data = defaultdict(list)
        logits = defaultdict(list)
        softmax = defaultdict(list)
        scores = defaultdict(list)
        nr_samples = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"        

        dsets = ["val", "ood"] + args.farood

        for dset in dsets:
            for h in range(self.max_depth):
                height_dir = os.path.join(args.basedir, f"H{h}")
                fname = f"{dset}-preds.out"
                data[dset].append(torch.load(os.path.join(height_dir, fname)))
                logits[dset].append(data[dset][h]["logits"].to(self.device))
                p = torch.softmax(data[dset][h]["logits"], dim=-1).to(self.device)
                softmax[dset].append(p)

            nr_samples[dset] = logits[dset][0].size(0)

        node_labels = {}

        for dset in dsets:
            if dset == "val":
                val2node_map = id_hierarchy.gen_ds2node_map(id_classes)
                node_labels[dset] = val2node_map[data["val"][0]["targets"]]

            elif dset == "ood":
                ood2node_map = id_hierarchy.gen_ds2node_map(ood_classes)
                node_labels[dset] = ood2node_map[data["ood"][0]["targets"]]

            else:
                root_idx = id_hierarchy.id_node_list.index("root")
                root_labels = torch.full((nr_samples[dset],), root_idx, dtype=torch.long)
                node_labels[dset] = root_labels

        multi_classes = get_multidepth_classes(id_hierarchy, id_classes)

        flattened_classes = [item for sublist in reversed(multi_classes) for item in sublist]
        self.flattened_classes = flattened_classes
        self.flat2node = gen_flat2node_map(id_hierarchy, self.flattened_classes, self.device)

        children_maps, group_sizes = get_children_and_group_maps(id_hierarchy,
                                                                 multi_classes,
                                                                 device=self.device)

        self.children_maps = children_maps
        self.group_sizes = group_sizes

        leaf_height = 0
        path_indices = get_path_indices(id_hierarchy, multi_classes, leaf_height)
        path_indices = [torch.tensor(x, dtype=torch.long, device=self.device) for x in path_indices]
        self.path_indices = path_indices


        gt_dists_mat, pred_dists_mat = get_hdist_matrix(id_hierarchy,
                                                        range(len(id_hierarchy.id_node_list)),
                                                        return_pair=True)
        
        hdist_mat = gt_dists_mat + pred_dists_mat
        self.hdist_mat = hdist_mat.to(self.device)
        
        self.hierarchy = id_hierarchy

        self.gt_dists_mat = gt_dists_mat.long()
        self.pred_dists_mat = pred_dists_mat.long()

        self.multi_classes = multi_classes

        self.dsets = dsets
        self.data = data
        self.logits = logits
        self.softmax = softmax
        self.scores = scores

        self.nr_samples = nr_samples

        self.node_labels = node_labels


    def multi_predict(self,
                      logits,
                      softmax,
                      u_method=None,
                      u_args={},
                      min_hdist=False,
                      beta=1.0,
                      ):

        logits = [x.detach().clone() for x in logits]
        softmax = [x.detach().clone() for x in softmax]

        fused_p = fuse_predictions(softmax,
                                   self.hierarchy,
                                   self.multi_classes,
                                   self.children_maps,
                                   self.group_sizes,
                                   self.path_indices,
                                   self.flat2node,
                                   u_method,
                                   u_args,
                                   )
        
        root_probs = get_root_probability(softmax, beta)
        
        root_idx = self.hierarchy.id_node_list.index("root")

        assert torch.allclose(fused_p[:, root_idx], torch.zeros_like(fused_p[:, root_idx]))

        fused_p = fused_p * (1 - root_probs.view(-1, 1))
        fused_p[:, root_idx] = root_probs

        psums = torch.sum(fused_p, dim=-1)

        assert torch.allclose(psums, torch.ones_like(psums), atol=1e-4)

        if min_hdist:
            neg_dists = -1.0 * expected_hdist(fused_p, self.hdist_mat)
            _, preds = neg_dists.max(dim=-1)
        else:
            _, preds = fused_p.max(dim=-1)

        return preds

    def predict_and_eval(self, **kwargs):

        res = {}

        for dset in self.dsets:
            preds = self.multi_predict(self.logits[dset],
                                       self.softmax[dset],
                                       **kwargs)

            results = get_results(preds.to("cpu"),
                                  self.node_labels[dset],
                                  self.hierarchy,
                                  dists_mats=(self.gt_dists_mat, self.pred_dists_mat))
            res[dset] = results

        return res
    

    def predict_oracle(self):

        ood_depths = torch.empty(self.nr_samples["ood"], dtype=torch.int)
        
        # get ood depths:
        for i, ood_label in enumerate(self.node_labels["ood"]):
            class_name = self.hierarchy.id_node_list[ood_label]
            ancestors = self.hierarchy.node_ancestors[class_name]
            ood_depths[i] = len(ancestors)

        ood_heights_oracle = self.max_depth - ood_depths
        val_heights_oracle = torch.zeros(self.nr_samples["val"], dtype=torch.long)

        print("Evaluating oracle...")
        val_preds, ood_preds = predict_classes(self.logits["val"],
                                               self.logits["ood"],
                                               val_heights_oracle,
                                               ood_heights_oracle,
                                               self.hierarchy,
                                               self.multi_classes)

        res_val = get_results(val_preds,
                              self.node_labels["val"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))                              
        
        res_ood = get_results(ood_preds,
                              self.node_labels["ood"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))                              

        return {"val": res_val, "ood": res_ood}

    def predict_leafmodel(self):

        print("Evaluating leaf model...")

        val_heights_leafs = torch.zeros(self.nr_samples["val"], dtype=torch.long)
        ood_heights_leafs = torch.zeros(self.nr_samples["ood"], dtype=torch.long)

        val_preds, ood_preds = predict_classes(self.logits["val"],
                                               self.logits["ood"],
                                               val_heights_leafs,
                                               ood_heights_leafs,
                                               self.hierarchy,
                                               self.multi_classes)

        res_val = get_results(val_preds,
                              self.node_labels["val"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))                              

        res_ood = get_results(ood_preds,
                              self.node_labels["ood"],
                              self.hierarchy,
                              dists_mats=(self.gt_dists_mat, self.pred_dists_mat))                              

        return {"val": res_val, "ood": res_ood}


def main(args):

    hinf = HInferenceEvaluator(args)

    allres = {}

    allres["leafmodel"] = hinf.predict_leafmodel()
    allres["oracle"] = hinf.predict_oracle()

    betas = [float(x) for x in args.betas]

    score_module = importlib.import_module("lib.utils.score_util")

    for method in args.uncertainty_methods:
        print(f"Evaluating {method}")

        method_fun = getattr(score_module, method)

        for beta in betas:
            res = hinf.predict_and_eval(u_method=method_fun, u_args={}, min_hdist=False, beta=beta)
            name = f"{method}_beta{beta}"
            allres[name] = res

            res = hinf.predict_and_eval(u_method=method_fun, u_args={}, min_hdist=True, beta=beta)
            name = f"{method}_minhdist_beta{beta}"
            allres[name] = res

    hierarchy_name = (args.hierarchy).split("/")[-1]
    hierarchy_name = hierarchy_name.split(".")[0]

    torch.save(allres,
               os.path.join(os.path.dirname(args.basedir),
                            f"hinference-{hierarchy_name}.result") )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
