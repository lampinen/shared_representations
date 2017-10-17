import numpy as np
from scipy.linalg import block_diag
import os


def IO_from_tree(tree, top=True):
    """Takes a tree of nested list form, where each element of the list is
    either another list or is a 1 to indicate a leaf. Produces a dataset
    corresponding to that tree"""
    if tree == 1:
	return [1]
    if tree == []:
	raise ValueError("lists cannot be empty! use 1 to indicate leaf nodes.")
    
    subtrees = [IO_from_tree(subtree, top=False) for subtree in tree]
    IO = block_diag(*subtrees)
    if not top: # add indicator to group these subtrees
	IO = np.concatenate((np.ones((len(IO), 1)), IO), axis=1)
    return IO 



print IO_from_tree([[1,1],[1,1]])
print IO_from_tree([
[[1,1],[1,1,[1,1]]],
[[1,1],[1,1,[1,1]]],
])



