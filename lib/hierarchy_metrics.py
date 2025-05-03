import torch

from sklearn.metrics import balanced_accuracy_score


class HierarchicalPredAccuracy:

    def __init__(self, hierarchy, track_hdist=False):
        super().__init__()
        self._hierarchy = hierarchy
        self._counts = 0.
        self._running_scores = 0.
        self._track_hdist = track_hdist
        if self._track_hdist:
            self._dist_matrix = torch.zeros((self._hierarchy.max_depth+1,
                                             self._hierarchy.max_depth+1),
                                            dtype=int)
            

    def update_state(self, preds, targets, dists_mats=None):
        with torch.no_grad():

            n_samples = targets.size(0)

            self._unique_classes, self._class_counts = torch.unique(targets, return_counts=True)

            self._tp_gts = targets
            self._preds = preds
            
            tp_correct = preds.eq(targets)
            self._hierarchy_distances = torch.zeros(n_samples)
            self._running_scores += torch.sum(tp_correct).float()
            self._counts += tp_correct.size(0)

            # Track hierarchy distance
            if self._track_hdist:

                if not dists_mats:
                    gt_dists, pred_dists = self.calc_hdists(targets, preds)
                else:
                    gt_dists_mat = dists_mats[0]
                    pred_dists_mat = dists_mats[1]
                    gt_dists = gt_dists_mat[targets, preds]
                    pred_dists = pred_dists_mat[targets, preds]

                for i, (g, p) in enumerate(zip(gt_dists, pred_dists)):
                    self._dist_matrix[g, p] += 1
                    self._hierarchy_distances[i] = g + p

    def calc_hdists(self, gts, preds):
        return calc_hdists(gts, preds, self._hierarchy)

    def reset_state(self,):
        self._running_scores = 0.
        self._counts = 0.
        self._dist_matrix = torch.zeros((self._hierarchy.max_depth+1,
                                         self._hierarchy.max_depth+1),
                                        dtype=int)

    def result(self,):
        return self._running_scores / self._counts

    def result_hierarchy_distances(self,):
        return self._dist_matrix.cpu().numpy()

    def result_balanced_hierarchy_distance(self,):
        hierdists_class = torch.zeros(self._unique_classes.size(0))

        for i, class_ in enumerate(self._unique_classes):
            mask = self._tp_gts == class_
            mask = mask.cpu()
            hierdists_class[i] = torch.mean(self._hierarchy_distances[mask])

        return torch.mean(hierdists_class)

    def result_class_hdists(self,):

        hierdists_class = torch.zeros(self._unique_classes.size(0))

        for i, class_ in enumerate(self._unique_classes):
            mask = self._tp_gts == class_
            mask = mask.cpu()
            hierdists_class[i] = torch.mean(self._hierarchy_distances[mask])

        results = {}
            
        for i, cls in enumerate(self._unique_classes):
            class_name = self._hierarchy.id_node_list[cls]
            results[class_name] = hierdists_class[i]

        return results       
        

    def result_balanced_accuracy(self,):
        return balanced_accuracy_score(self._tp_gts.cpu().numpy(),
                                       self._preds.cpu().numpy())

    def result_class_recalls(self,):
        return get_class_recalls(self._preds.cpu(), self._tp_gts.cpu(), self._hierarchy)


def calc_hdists(gts, preds, hierarchy):

    # Initialize trackers
    gt_dists = torch.ones(gts.size(0), dtype=int) * -1
    pred_dists = torch.ones(gts.size(0), dtype=int) * -1

    def find_lca_distance(gt_ancestors, pred_ancestors):
        # Find the index where the ancestors lists diverge
         
        i = 0
        while (i < min(len(gt_ancestors), len(pred_ancestors))
               and gt_ancestors[i] == pred_ancestors[i]):
            i += 1

        gt_dist = len(gt_ancestors) - i
        pred_dist = len(pred_ancestors) - i
        
        return gt_dist, pred_dist
    
    for i, (gt, pred) in enumerate(zip(gts, preds)):

        # prediction is correct
        if gt == pred:
            gt_dists[i] = 0
            pred_dists[i] = 0
            continue

        gt_class = hierarchy.id_node_list[int(gt)]
        pred_class = hierarchy.id_node_list[int(pred)]

        gt_parents = hierarchy.node_ancestors[gt_class].copy()
        pred_parents = hierarchy.node_ancestors[pred_class].copy()
        
        # append predictions
        gt_parents = gt_parents + [int(gt)]
        pred_parents = pred_parents + [int(pred)]

        # if there is no prediction (prediction is root node)
        if pred == -1:
            pred_dists[i] = 0
            gt_dists[i] = len(gt_parents)
            continue
        
        gt_dist, pred_dist = find_lca_distance(gt_parents, pred_parents)

        gt_dists[i] = gt_dist
        pred_dists[i] = pred_dist

    return gt_dists, pred_dists


def get_class_recalls(preds, gts, hierarchy):

    classes = torch.unique(gts)
    n_classes = classes.size(0)

    true_positives = torch.zeros(n_classes)
    false_negatives = torch.zeros(n_classes)

    for i, cls in enumerate(classes):
        true_positives[i] = ((preds == cls) & (gts == cls)).sum().item()
        false_negatives[i] = ((preds != cls) &
                                (gts == cls)).sum().item()

    # Compute recall for each class
    # add epsilon to avoid division by zero
    recalls = true_positives / (true_positives + false_negatives + 1e-6)

    results = {}

    # Print out the recall for each class
    for i, cls in enumerate(classes):
        class_name = hierarchy.id_node_list[cls]
        results[class_name] = recalls[i]

    return results
