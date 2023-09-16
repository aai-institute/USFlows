# Veriflow: Generative Flow Based Density Models for Neuro-Symbolic Verification
VeriFow provides a stable and convenient library of flow based density models with flexibile base distributions, 
which are specifically tailored towards the use in neuro-symbolic verification procedures. The major goal is to
provide models that can represent reference distribution which are suitable for verification, 
abstract interpretation. and hypothesis testing.
The implemented layer are carefully chosen to guarntee the following properties:

- Efficient computation of exact densities as well as efficient sampling.
-  A piece-wise log-density function for all models with ReLU nonlinearity and Laplacian base distribution.

Additionally, UDL preserving layers map the upper density level sets of the data distribution to the upper density level sets
of the base Distribution.





