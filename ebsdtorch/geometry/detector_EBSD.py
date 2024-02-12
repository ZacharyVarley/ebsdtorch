"""

Detector EBSD geometry:

Some notes...

I have implemented logic to accept an average pattern center and define 3D
homogeneous coordinates transformation matrices (T) that maps 2D detector pixel
coordinates in the detector reference frame to 3D coordinates in the sample
frame. This is a 4x4 matrix in SE(3), the Lie group of rigid body
transformations in R^3 (3D Euclidean space). If you are doing some population of
geometries (to try to fit the geometry), you can easily define a batch of these
matrices of shape (..., 4, 4), but optimization directly on the matrices
(constrained 16 dimensional space) is not adivsable. Instead, you should use the
6 dimensional tangent space of SE(3) which is the Lie algebra se(3). This is an
axis-angle 3D vector (its norm is the angle in radians) and a 3D translation
vector. The file "lie_algebra_se3" implements the exponential map, logarithm
map, in terms of both rotation matrices and quaternions.

The gradient can be backpropagated through the projection of patterns to either
average pattern centers or 6D Lie algebra vectors. The gradient can be used to
optimize the average pattern center or the 6D Lie algebra vectors, and a staged
optimization approach probably makes the most sense. This framework lets us fit
the geometry for all of the samples at once, but this has to be done on the
sphere and not on the plane to be computationally tractable. Stay tuned...

"""
