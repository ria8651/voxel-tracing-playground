# Data Structures

## Octree

The octree is made with nodes and leafs. The octree always has the first 8 children leafs. A node is made of two parts. A u8 containg 0 for node, 1 for empty leaf and 2 for solid leaf and a u24 containg the index of the node's first child. The other children can be found by adding the child index to the index of the first child. A leaf contains the same first u8 with the type and then a u8 for red a u8 for green and a u8 for blue.

## Framebuffers

0. Colour in xyz
1. Normals in xyz
2. Depth in xy and shadow map in zw