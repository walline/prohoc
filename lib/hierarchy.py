import torch
from collections import defaultdict
import json


class Hierarchy:
    
    def __init__(self,
                 train_classes,
                 hierarchy_fn,
                 ):

        # check if train_classes is sorted
        assert train_classes == sorted(train_classes)
        
        full_node_list, node_description, child2parent = self._read_json_hierarchy(hierarchy_fn)

        self.node_description = node_description

        assert len(full_node_list) == len(set(full_node_list))
        self._assert_train_classes(train_classes, full_node_list, child2parent)

        self.full_node_list = full_node_list
        self.train_classes = train_classes

        id_node_list = self._trim_hierarchy(train_classes, child2parent)

        self.ood_train_classes = self._get_ood_train_classes(full_node_list,
                                                             id_node_list,
                                                             child2parent)

        id_node_list, child2parent = self._reduce_hierarchy(id_node_list, child2parent)

        self.id_node_list = id_node_list
        self.child2parent = child2parent
        
        self.node_ancestors = self._gen_node_ancestors()
        
        self._num_classes = len(self.id_node_list)
        self._max_depth = self._gen_maxdepth()

        self.parent2children = self._gen_parent2children(self.id_node_list, self.child2parent)
        self.parents_list = sorted(list(self.parent2children.keys()))

        self.print_hierarchy_info()


    def print_hierarchy_info(self):

        print(f"Number of nodes in ID hierarchy: {self._num_classes}")
        print(f"Number of nodes in full hierarchy: {len(self.full_node_list)}")
        print(f"Number of ID leaf classes: {len(self.train_classes)}")
        print(f"Max depth of hierarchy: {self._max_depth}")


    def _assert_train_classes(self, train_classes, node_list, child2parent):

        parent_nodes = set(child2parent.values())

        for train_class in train_classes:
            if train_class in parent_nodes:
                raise AssertionError(f"Train class {train_class} is not a leaf node")

            if train_class not in node_list:
                raise AssertionError(f"Train class {train_class} does not exist in the hierarchy")

    def _reduce_hierarchy(self, id_node_list, child2parent):

        parent2children = self._gen_parent2children(id_node_list, child2parent)

        redundant_nodes = []

        for k, val in parent2children.items():

            assert len(val) > 0

            if len(val) == 1:
                redundant_nodes.append(k)

        print(f"Pruning {len(redundant_nodes)} redundant nodes...")

        node_list = id_node_list.copy()
        child2parent = child2parent.copy()
        
        for node in redundant_nodes:
            parent = child2parent[node]

            for k, val in child2parent.items():
                if val == node:
                    child2parent[k] = parent

            child2parent.pop(node)
            node_list.remove(node)

        return sorted(node_list), child2parent


    def _read_json_hierarchy(self, hierarchy_fn):

        node_list = []
        class_description = {}
        child2parent = {}        

        def _process_node(node, parent=None):
            name = node['name']
            description = node['description']

            node_list.append(name)

            class_description[name] = description

            if parent:
                child2parent[name] = parent

            for child in node.get('children', []):
                _process_node(child, parent=name)

        with open(hierarchy_fn, 'r') as f:
            hierarchy_data = json.load(f)

        _process_node(hierarchy_data)

        node_list = sorted(node_list)

        return node_list, class_description, child2parent
            
        
    def _trim_hierarchy(self, train_classes, child2parent):

        keep_classes = set()

        for train_class in train_classes:
            current_node = train_class
            while current_node:
                keep_classes.add(current_node)
                current_node = child2parent.get(current_node)

        trimmed_node_list = sorted(list(keep_classes))

        return trimmed_node_list

    def _get_ood_train_classes(self, full_node_list, id_node_list, child2parent):
        
        parent_nodes = set(child2parent.values())
        all_leaves = [node for node in full_node_list if node not in parent_nodes]
        ood_leaves = [node for node in all_leaves if node not in id_node_list]
        return sorted(ood_leaves)
    
    def _gen_node_ancestors(self):

        node_to_index = {node: idx for idx, node in enumerate(self.id_node_list)}

        node_ancestors = {}

        for node in self.id_node_list:

            parents = []
            current_node = node

            while current_node:
                parent = self.child2parent.get(current_node)
                if parent:
                    parents.append(node_to_index[parent])
                current_node = parent

            parents.reverse()

            node_ancestors[node] = parents

        return node_ancestors

    def _gen_maxdepth(self,):
        """Get the maximum depth of tree"""
        max_depth = 0
        for pars in self.node_ancestors.values():
            if len(pars) > max_depth:
                max_depth = len(pars)
        return max_depth


    def _gen_parent2children(self, node_list, child2parent):

        parent2children = defaultdict(list)

        for c in node_list:

            if c not in child2parent:
                continue

            parent = child2parent[c]
            parent2children[parent].append(c)

        return dict(parent2children)


    def get_descendants(self, node):

        descendants = set()

        def dfs(n):
            if n in self.parent2children:
                for child in self.parent2children[n]:
                    if child not in descendants:
                        descendants.add(child)
                        dfs(child)

        dfs(node)

        return list(descendants)


    def get_leaf_descendants(self, node):
        leaf_descendants = set()

        def dfs(n):
            # If the node has no children, it's a leaf
            if n not in self.parent2children or not self.parent2children[n]:
                leaf_descendants.add(n)
                return

            # Otherwise, continue DFS
            for child in self.parent2children[n]:
                dfs(child)

        dfs(node)

        return list(leaf_descendants)


    @property
    def num_classes(self):
        """Get the number of classes in the hierarchy"""
        return self._num_classes


    @property
    def max_depth(self):
        """Get the maximum depth of the hierarchy"""
        return self._max_depth

    
    def gen_ds2node_map(self, class_list):

        n_classes = len(class_list)
        map_tensor = torch.empty(n_classes, dtype=torch.long)

        for i, class_name in enumerate(class_list):

            c = class_name

            while c not in self.id_node_list:
                c = self.child2parent[c]

            map_tensor[i] = self.id_node_list.index(c)

        return map_tensor

    def gen_depth_map(self, class_list):

        n_classes = len(class_list)
        map_tensor = torch.empty(n_classes, dtype=torch.long)

        for i, class_name in enumerate(class_list):

            depth = len(self.node_ancestors[class_name])
            map_tensor[i] = depth

        return map_tensor

    def gen_nchild_map(self, class_list):

        n_classes = len(class_list)
        map_tensor = torch.empty(n_classes, dtype=torch.long)

        for i, class_name in enumerate(class_list):
            nchild = len(self.parent2children.get(class_name, []))
            map_tensor[i] = nchild

        return map_tensor

    def gen_descendant_mask(self, class_list):

        n_classes = len(class_list)
        mask = torch.zeros((n_classes, n_classes), dtype=torch.float32)

        for i, c in enumerate(class_list):
            
            descendants = self.get_descendants(c)

            for descendant in descendants:
                idx = class_list.index(descendant)
                mask[i, idx] = 1.0

            mask[i, i] = 1.0

        return mask
