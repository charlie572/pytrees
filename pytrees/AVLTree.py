"""
AVL Tree. 

Balanced Binary Search Tree. Gurantee for balance.

Convention: 

- "key" and "val" are almost the same in this implementation. use term "key" for search and delete a particular node. use term "val" for other cases

API: 

- insert(self, val)
- delete(self, key)
- search(self, key)
- getDepth(self)
- preOrder(self)
- inOrder(self)
- postOrder(self)
- countNodes(self)
- buildFromList(cls, l)

Author: Yi Zhou
Date: May 19, 2018 
Reference: https://en.wikipedia.org/wiki/AVL_tree
Reference: https://github.com/pgrafov/python-avl-tree/blob/master/pyavltree.py
"""

from collections import deque
import random


class AVLNode:
    def __init__(self, val, op_count):
        op_count[0] += 2
        self.val = val
        op_count[0] += 2
        self.parent = None
        op_count[0] += 2
        self.left = None
        op_count[0] += 2
        self.right = None
        op_count[0] += 2
        self.height = 0

    def isLeaf(self, op_count):
        op_count[0] += 3
        return (self.height == 0)
    
    def maxChildrenHeight(self, op_count):
        op_count[0] += 3
        if self.left and self.right:
            op_count[0] += 4
            return max(self.left.height, self.right.height)
        elif self.left and not self.right:
            op_count[0] += 4
            op_count[0] += 1
            return self.left.height
        elif not self.left and  self.right:
            op_count[0] += 4
            op_count[0] += 4
            op_count[0] += 1
            return self.right.height
        else:
            op_count[0] += 4
            op_count[0] += 4
            op_count[0] += 1
            return -1
    
    def balanceFactor(self, op_count):
        op_count[0] += 7
        return (self.left.height if self.left else -1) - (self.right.height if self.right else -1)
    
    def __str__(self):
        return "AVLNode("+ str(self.val)+ ", Height: %d )" % self.height

class AVLTree:
    def __init__(self):
        self.root = None
        self.rebalance_count = 0
        self.nodes_count = 0
        self.op_count = [0]
    
    def setRoot(self, val):
        """
        Set the root value
        """
        self.op_count[0] += 2
        self.root = AVLNode(val, self.op_count)
    
    def countNodes(self):
        self.op_count[0] += 1
        return self.nodes_count
    
    def getDepth(self):
        """
        Get the max depth of the BST
        """
        self.op_count[0] += 1
        if self.root:
            self.op_count[0] += 1
            return self.root.height
        else:
            self.op_count[0] += 1
            return -1

    def get_min(self):
        self.op_count[0] += 3
        if self.root is None:
            self.op_count[0] += 1
            return None

        self.op_count[0] += 2
        node = self._findSmallest(self.root)
        self.op_count[0] += 1
        return node.val

    def get_max(self):
        self.op_count[0] += 3
        if self.root is None:
            self.op_count[0] += 1
            return None

        self.op_count[0] += 2
        node = self._findBiggest(self.root)
        self.op_count[0] += 1
        return node.val

    def _findSmallest(self, start_node):
        assert (not start_node is None)
        self.op_count[0] += 2
        node = start_node
        while node.left:
            self.op_count[0] += 1
            self.op_count[0] += 2
            node = node.left
        self.op_count[0] += 1
        self.op_count[0] += 1
        return node
    
    def _findBiggest(self, start_node):
        assert (not start_node is None)
        self.op_count[0] += 2
        node = start_node
        while node.right:
            self.op_count[0] += 1
            self.op_count[0] += 2
            node = node.right
        self.op_count[0] += 1
        self.op_count[0] += 1
        return node

    def insert(self, val):
        """
        insert a val into AVLTree
        """
        self.op_count[0] += 3
        if self.root is None:
            self.op_count[0] += 1
            self.setRoot(val)
        else:
            self.op_count[0] += 2
            self._insertNode(self.root, val)
        self.op_count[0] += 4
        self.nodes_count += 1
    
    def _insertNode(self, currentNode, val):
        """
        Helper function to insert a value into AVLTree.
        """
        self.op_count[0] += 2
        node_to_rebalance = None
        self.op_count[0] += 3
        if currentNode.val > val:
            self.op_count[0] += 1
            if(currentNode.left):
                self.op_count[0] += 2
                self._insertNode(currentNode.left, val)
            else:
                self.op_count[0] += 2
                child_node = AVLNode(val, self.op_count)
                self.op_count[0] += 2
                currentNode.left = child_node
                self.op_count[0] += 2
                child_node.parent = currentNode
                self.op_count[0] += 3
                if currentNode.height == 0:
                    self.op_count[0] += 1
                    self._recomputeHeights(currentNode)
                    self.op_count[0] += 2
                    node = currentNode
                    while node:
                        self.op_count[0] += 1
                        self.op_count[0] += 1
                        self.op_count[0] += 6
                        if node.balanceFactor(self.op_count) in [-2 , 2]:
                            self.op_count[0] += 2
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        self.op_count[0] += 2
                        node = node.parent
                    self.op_count[0] += 1
        else:
            self.op_count[0] += 1
            if(currentNode.right):
                self.op_count[0] += 2
                self._insertNode(currentNode.right, val)
            else:
                self.op_count[0] += 3
                child_node = AVLNode(val, self.op_count)
                self.op_count[0] += 2
                currentNode.right = child_node
                self.op_count[0] += 2
                child_node.parent = currentNode
                self.op_count[0] += 3
                if currentNode.height == 0:
                    self.op_count[0] += 1
                    self._recomputeHeights(currentNode)
                    self.op_count[0] += 2
                    node = currentNode
                    while node:
                        self.op_count[0] += 1
                        self.op_count[0] += 6
                        if node.balanceFactor(self.op_count) in [-2 , 2]:
                            self.op_count[0] += 2
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        self.op_count[0] += 2
                        node = node.parent
                    self.op_count[0] += 1
        self.op_count[0] += 1
        if node_to_rebalance:
            self.op_count[0] += 1
            self._rebalance(node_to_rebalance)

    def _rebalance(self, node_to_rebalance):
        self.op_count[0] += 2
        A = node_to_rebalance
        self.op_count[0] += 2
        F = A.parent #allowed to be NULL
        self.op_count[0] += 3
        if A.balanceFactor(self.op_count) == -2:
            self.op_count[0] += 3
            if A.right.balanceFactor(self.op_count) <= 0:
                """Rebalance, case RRC 
                [Original]:                   
                        F                         
                      /  \
                 SubTree  A
                           \                
                            B
                             \
                              C

                [After Rotation]:
                        F                         
                      /  \
                 SubTree  B
                         / \  
                        A   C
                """
                self.op_count[0] += 2
                B = A.right
                self.op_count[0] += 2
                C = B.right
                assert (not A is None and not B is None and not C is None)
                self.op_count[0] += 2
                A.right = B.left
                self.op_count[0] += 1
                if A.right:
                    self.op_count[0] += 2
                    A.right.parent = A
                self.op_count[0] += 2
                B.left = A
                self.op_count[0] += 2
                A.parent = B
                self.op_count[0] += 3
                if F is None:
                    self.op_count[0] += 2
                    self.root = B
                    self.op_count[0] += 2
                    self.root.parent = None
                else:
                    self.op_count[0] += 3
                    if F.right == A:
                        self.op_count[0] += 2
                        F.right = B
                    else:
                        self.op_count[0] += 2
                        F.left = B
                    self.op_count[0] += 2
                    B.parent = F
                self.op_count[0] += 1
                self._recomputeHeights(A)
                self.op_count[0] += 1
                self._recomputeHeights(B.parent)
            else:
                """Rebalance, case RLC 
                [Original]:                   
                        F                         
                      /  \
                 SubTree  A
                           \                
                            B
                           /
                          C

                [After Rotation]:
                        F                         
                      /  \
                 SubTree  C
                         / \  
                        A   B
                """
                self.op_count[0] += 2
                B = A.right
                self.op_count[0] += 2
                C = B.left
                assert (not A is None and not B is None and not C is None)
                self.op_count[0] += 2
                B.left = C.right
                self.op_count[0] += 1
                if B.left:
                    self.op_count[0] += 2
                    B.left.parent = B
                self.op_count[0] += 2
                A.right = C.left
                self.op_count[0] += 1
                if A.right:
                    self.op_count[0] += 2
                    A.right.parent = A
                self.op_count[0] += 2
                C.right = B
                self.op_count[0] += 2
                B.parent = C
                self.op_count[0] += 2
                C.left = A
                self.op_count[0] += 2
                A.parent = C
                self.op_count[0] += 3
                if F is None:
                    self.op_count[0] += 2
                    self.root = C
                    self.op_count[0] += 2
                    self.root.parent = None
                else:
                    self.op_count[0] += 3
                    if F.right == A:
                        self.op_count[0] += 2
                        F.right = C
                    else:
                        self.op_count[0] += 2
                        F.left = C
                    self.op_count[0] += 2
                    C.parent = F
                self.op_count[0] += 1
                self._recomputeHeights(A)
                self.op_count[0] += 1
                self._recomputeHeights(B)
        else:
            assert(node_to_rebalance.balanceFactor([0]) == +2)
            self.op_count[0] += 3
            if node_to_rebalance.left.balanceFactor(self.op_count) >= 0:
                """Rebalance, case LLC 
                [Original]:                   
                        F                         
                      /  \
                     A   SubTree
                    /
                   B
                  /
                 C   

                [After Rotation]:
                        F                         
                       / \  
                      B  SubTree
                     / \  
                    C   A
                """
                self.op_count[0] += 2
                B = A.left
                self.op_count[0] += 2
                C = B.left
                assert (not A is None and not B is None and not C is None)
                self.op_count[0] += 2
                A.left = B.right
                self.op_count[0] += 1
                if A.left:
                    self.op_count[0] += 2
                    A.left.parent = A
                self.op_count[0] += 2
                B.right = A
                self.op_count[0] += 2
                A.parent = B
                self.op_count[0] += 3
                if F is None:
                    self.op_count[0] += 2
                    self.root = B
                    self.op_count[0] += 2
                    self.root.parent = None
                else:
                    self.op_count[0] += 3
                    if F.right == A:
                        self.op_count[0] += 2
                        F.right = B
                    else:
                        self.op_count[0] += 2
                        F.left = B
                    self.op_count[0] += 2
                    B.parent = F
                self.op_count[0] += 1
                self._recomputeHeights(A)
                self.op_count[0] += 1
                self._recomputeHeights(B.parent)
            else:
                """Rebalance, case LRC 
                [Original]:                   
                        F                         
                      /  \
                     A   SubTree
                    /
                   B
                    \
                     C
                   

                [After Rotation]:
                        F                         
                       / \  
                      C  SubTree
                     / \  
                    B   A
                """
                self.op_count[0] += 2
                B = A.left
                self.op_count[0] += 2
                C = B.right
                assert (not A is None and not B is None and not C is None)
                self.op_count[0] += 2
                A.left = C.right
                self.op_count[0] += 1
                if A.left:
                    self.op_count[0] += 2
                    A.left.parent = A
                self.op_count[0] += 2
                B.right = C.left
                self.op_count[0] += 1
                if B.right:
                    self.op_count[0] += 2
                    B.right.parent = B
                self.op_count[0] += 2
                C.left = B
                self.op_count[0] += 2
                B.parent = C
                self.op_count[0] += 2
                C.right = A
                self.op_count[0] += 2
                A.parent = C
                self.op_count[0] += 3
                if F is None:
                    self.op_count[0] += 2
                    self.root = C
                    self.op_count[0] += 2
                    self.root.parent = None
                else:
                    self.op_count[0] += 3
                    if (F.right == A):
                        self.op_count[0] += 2
                        F.right = C
                    else:
                        self.op_count[0] += 2
                        F.left = C
                    self.op_count[0] += 2
                    C.parent = F
                self.op_count[0] += 1
                self._recomputeHeights(A)
                self.op_count[0] += 1
                self._recomputeHeights(B)
        self.op_count[0] += 4
        self.rebalance_count += 1

    def _recomputeHeights(self, start_from_node):
        self.op_count[0] += 2
        changed = True
        self.op_count[0] += 2
        node = start_from_node
        self.op_count[0] += 3
        while node and changed:
            self.op_count[0] += 2
            old_height = node.height
            self.op_count[0] += 10
            node.height = (node.maxChildrenHeight(self.op_count) + 1 if (node.right or node.left) else 0)
            self.op_count[0] += 4
            changed = node.height != old_height
            self.op_count[0] += 2
            node = node.parent
    
    def search(self, key):
        """
        Search a AVLNode satisfies AVLNode.val = key.
        if found return AVLNode, else return None.
        """
        self.op_count[0] += 2
        return self._dfsSearch(self.root, key)
    
    def _dfsSearch(self, currentNode, key):
        """
        Helper function to search a key in AVLTree.
        """
        self.op_count[0] += 3
        if currentNode is None:
            self.op_count[0] += 1
            return None
        elif currentNode.val == key:
            self.op_count[0] += 3
            self.op_count[0] += 1
            return currentNode
        elif currentNode.val > key:
            self.op_count[0] += 3
            self.op_count[0] += 3
            self.op_count[0] += 2
            return self._dfsSearch(currentNode.left, key)
        else:
            self.op_count[0] += 3
            self.op_count[0] += 3
            self.op_count[0] += 2
            return self._dfsSearch(currentNode.right, key)

    def delete(self, key):
        """
        Delete a key from AVLTree
        """
        # first find
        self.op_count[0] += 2
        node = self.search(key)

        self.op_count[0] += 4
        if not node is None:
            self.op_count[0] += 4
            self.nodes_count -= 1
            #     There are three cases:
            # 
            #     1) The node is a leaf.  Remove it and return.
            # 
            #     2) The node is a branch (has only 1 child). Make the pointer to this node 
            #        point to the child of this node.
            # 
            #     3) The node has two children. Swap items with the successor
            #        of the node (the smallest item in its right subtree) and
            #        delete the successor from the right subtree of the node.
            if node.isLeaf(self.op_count):
                self.op_count[0] += 1
                self._removeLeaf(node)
            elif (bool(node.left)) ^ (bool(node.right)):
                self.op_count[0] += 7
                self.op_count[0] += 1
                self._removeBranch(node)
            else:
                self.op_count[0] += 7
                assert (node.left) and (node.right)
                self.op_count[0] += 1
                self._swapWithSuccessorAndRemove(node)

    def _removeLeaf(self, node):
        self.op_count[0] += 2
        parent = node.parent
        self.op_count[0] += 1
        if (parent):
            self.op_count[0] += 3
            if parent.left == node:
                self.op_count[0] += 2
                parent.left = None
            else:
                assert (parent.right == node)
                self.op_count[0] += 2
                parent.right = None
            self.op_count[0] += 1
            self._recomputeHeights(parent)
        else:
            self.op_count[0] += 2
            self.root = None
        self.op_count[0] += 2
        del node
        # rebalance
        self.op_count[0] += 2
        node = parent
        while (node):
            self.op_count[0] += 1
            self.op_count[0] += 11
            if not node.balanceFactor(self.op_count) in [-1, 0, 1]:
                self.op_count[0] += 1
                self._rebalance(node)
            self.op_count[0] += 2
            node = node.parent
        self.op_count[0] += 1

    def _removeBranch(self, node):
        self.op_count[0] += 2
        parent = node.parent
        self.op_count[0] += 1
        if (parent):
            self.op_count[0] += 3
            if parent.left == node:
                self.op_count[0] += 3
                parent.left = node.right if node.right else node.left
            else:
                assert (parent.right == node)
                self.op_count[0] += 3
                parent.right = node.right if node.right else node.left
            self.op_count[0] += 1
            if node.left:
                self.op_count[0] += 2
                node.left.parent = parent
            else:
                assert (node.right)
                self.op_count[0] += 2
                node.right.parent = parent
            self.op_count[0] += 1
            self._recomputeHeights(parent)
        else:
            self.op_count[0] += 3
            if node.left is None:
                self.op_count[0] += 2
                self.root = node.right
            else:
                self.op_count[0] += 2
                self.root = node.left

            self.op_count[0] += 2
            self.root.parent = None

        self.op_count[0] += 2
        del node
        # rebalance
        self.op_count[0] += 2
        node = parent
        self.op_count[0] += 1
        while (node):
            self.op_count[0] += 11
            if not node.balanceFactor(self.op_count) in [-1, 0, 1]:
                self.op_count[0] += 1
                self._rebalance(node)
            self.op_count[0] += 2
            node = node.parent
    
    def _swapWithSuccessorAndRemove(self, node):
        self.op_count[0] += 2
        successor = self._findSmallest(node.right)
        self.op_count[0] += 2
        self._swapNodes(node, successor)
        assert (node.left is None)
        self.op_count[0] += 3
        if node.height == 0:
            self.op_count[0] += 1
            self._removeLeaf(node)
        else:
            self.op_count[0] += 1
            self._removeBranch(node)
    
    def _swapNodes(self, node1, node2):
        assert (node1.height > node2.height)
        self.op_count[0] += 2
        parent1 = node1.parent
        self.op_count[0] += 2
        leftChild1 = node1.left
        self.op_count[0] += 2
        rightChild1 = node1.right
        self.op_count[0] += 2
        parent2 = node2.parent
        assert (not parent2 is None)
        assert (parent2.left == node2 or parent2 == node1)
        self.op_count[0] += 2
        leftChild2 = node2.left
        assert (leftChild2 is None)
        self.op_count[0] += 2
        rightChild2 = node2.right
        
        # swap heights
        self.op_count[0] += 2
        tmp = node1.height
        self.op_count[0] += 2
        node1.height = node2.height
        self.op_count[0] += 2
        node2.height = tmp

        self.op_count[0] += 1
        if parent1:
            self.op_count[0] += 3
            if parent1.left == node1:
                self.op_count[0] += 2
                parent1.left = node2
            else:
                assert (parent1.right == node1)
                self.op_count[0] += 2
                parent1.right = node2
            self.op_count[0] += 2
            node2.parent = parent1
        else:
            self.op_count[0] += 2
            self.root = node2
            self.op_count[0] += 2
            node2.parent = None

        self.op_count[0] += 2
        node2.left = leftChild1
        self.op_count[0] += 2
        leftChild1.parent = node2
        self.op_count[0] += 2
        node1.left = leftChild2 # None
        self.op_count[0] += 2
        node1.right = rightChild2
        self.op_count[0] += 1
        if rightChild2:
            self.op_count[0] += 2
            rightChild2.parent = node1
        self.op_count[0] += 4
        if not (parent2 == node1):
            self.op_count[0] += 2
            node2.right = rightChild1
            self.op_count[0] += 2
            rightChild1.parent = node2
            self.op_count[0] += 2
            parent2.left = node1
            self.op_count[0] += 2
            node1.parent = parent2
        else:
            self.op_count[0] += 2
            node2.right = node1
            self.op_count[0] += 2
            node1.parent = node2

    def inOrder(self):
        self.op_count[0] += 2
        res = []
        def _dfs_in_order(node, res):
            self.op_count[0] += 2
            if not node:
                self.op_count[0] += 1
                return
            self.op_count[0] += 2
            _dfs_in_order(node.left,res)
            self.op_count[0] += 2
            res.append(node.val)
            self.op_count[0] += 2
            _dfs_in_order(node.right,res)

        self.op_count[0] += 2
        _dfs_in_order(self.root, res)
        self.op_count[0] += 1
        return res
    
    def preOrder(self):
        self.op_count[0] += 2
        res = []
        def _dfs_pre_order(node, res):
            self.op_count[0] += 2
            if not node:
                return
            self.op_count[0] += 1
            res.append(node.val)
            self.op_count[0] += 2
            _dfs_pre_order(node.left,res)
            self.op_count[0] += 2
            _dfs_pre_order(node.right,res)

        self.op_count[0] += 2
        _dfs_pre_order(self.root, res)
        self.op_count[0] += 1
        return res
    
    def postOrder(self):
        self.op_count[0] += 2
        res = []
        def _dfs_post_order(node, res):
            self.op_count[0] += 2
            if not node:
                return
            self.op_count[0] += 2
            _dfs_post_order(node.left,res)
            self.op_count[0] += 2
            _dfs_post_order(node.right,res)
            self.op_count[0] += 2
            res.append(node.val)

        self.op_count[0] += 2
        _dfs_post_order(self.root, res)
        self.op_count[0] += 1
        return res
    
    @classmethod
    def buildFromList(cls, l, shuffle = True):
        """
        return a AVLTree object from l.
        suffle the list first for better balance.
        """
        if shuffle:
            random.seed()
            random.shuffle(l)
        AVL = AVLTree()
        for item in l:
            AVL.insert(item)
        return AVL
    
    def visulize(self):
        """
        Naive Visulization. 
        Warn: Only for simple test usage.
        """
        if self.root is None:
            print("EMPTY TREE.")
        else:
            print("-----------------Visualize Tree----------------------")
            layer = deque([self.root])
            layer_count = self.getDepth()
            while len( list(filter(lambda x:x is not None, layer) )):
                new_layer = deque([])
                val_list = []
                while len(layer):
                    node = layer.popleft()
                    if node is not None:
                        val_list.append(node.val)
                    else:
                        val_list.append(" ")
                    if node is None:
                        new_layer.append(None)
                        new_layer.append(None)
                    else:
                        new_layer.append(node.left)
                        new_layer.append(node.right)
                val_list = [" "] * layer_count + val_list
                print(*val_list, sep="  ", end="\n")
                layer = new_layer
                layer_count -= 1
            print("-----------------End Visualization-------------------")


if __name__ == "__main__":
    print("[BEGIN]Test Implementation of AVLTree.")
    # Simple Insert Test
    AVL = AVLTree()
    AVL.insert(0)
    AVL.insert(1)
    AVL.insert(2)
    AVL.insert(3)
    AVL.insert(4)
    AVL.insert(5)
    AVL.insert(6)
    AVL.insert(7)
    AVL.insert(8)
    AVL.insert(9)
    AVL.insert(10)
    AVL.insert(11)
    AVL.insert(12)
    AVL.insert(13)
    AVL.insert(14)
    print("Total nodes: ",AVL.nodes_count)
    print("Total rebalance: ",AVL.rebalance_count)
    # Simple Delete Test
    AVL.visulize()
    AVL.delete(2)
    AVL.visulize()
    AVL.delete(5)
    AVL.visulize()
    AVL.delete(6)
    AVL.visulize()
    AVL.delete(4)
    AVL.visulize()
    AVL.delete(0)
    AVL.visulize()
    AVL.delete(3)
    AVL.visulize()
    AVL.delete(1)
    AVL.delete(7)
    AVL.delete(8)
    AVL.delete(9)
    AVL.visulize()
    AVL.delete(10)
    AVL.delete(12)
    AVL.visulize()
    print("Total nodes: ",AVL.nodes_count)
    print("Total rebalance: ",AVL.rebalance_count)
    print("----------------------------------------")
    input_list = list(range(2**16))
    new_AVL = AVLTree.buildFromList(input_list,shuffle = False)
    print("Total Nodes:",new_AVL.countNodes())
    print("Total Depth:",new_AVL.getDepth())
    print("Total rebalance: ",new_AVL.rebalance_count)
    new_AVL = AVLTree.buildFromList(input_list,shuffle = True)
    print("Total Nodes:",new_AVL.countNodes())
    print("Total Depth:",new_AVL.getDepth())
    print("Total rebalance: ",new_AVL.rebalance_count)
    print("Test inOrder:",  new_AVL.inOrder()==list(range(2**16)))
    print("[END]Test Implementation of AVLTree.")
