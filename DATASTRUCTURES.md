# Data Structures

## Octree

### SVO Buffer

Each node uses 2 ints. The bytes are laid out like this:

    00000000 00000000 00000000 01101010 01101001 01101001 01101001 01101001
     ^-----empty :(-----^ child mask-^   ^---------child pointer---------^


Each voxel uses 1 int. The bytes are laid out like this:

    01101001 01101001 01101001 00000000
    ^-red    ^-green  ^-blue   ^-empty

The octree is made with nodes and voxels. A node is made of two parts. A u8 with each bit refering to a child of the node, and a u24 containg the index of the node's first child. Becuase all the children of a node are grouped together other children of a node can be found by adding to the child index. A voxel currently only contains a rgb value for its colour. Multiple nodes *can* point to the same voxel.

## Framebuffers

0. Colour in rgb and depth in a
1. Normals in rgb and shadow map in a