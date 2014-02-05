Fermi-Hubbard on the GPU with Cuda
----------------------------------

This program can calculate the groundstate energy of 1 dimensional and
2 dimension Fermi-Hubbard model on the GPU (and the CPU). It is based
on [Exact diagonalization of the Hubbard model on graphics processing units](http://arxiv.org/abs/1204.3425)
by Siro and Harju.

There are 2 programs: main and main2D. The former is for 1D Hubbard,
the latter is for 2D Hubbard. Everything is split up in several classes
to make reusing code easy:  The Hamiltonian and HubHam2D classes build the
full (dense) Hamiltonian Matrix. SparseHamiltonian stores the Hubbard
Hamiltonian in parts: an spin up and a spin down part. The matrix
themselves are storred in the ELL format.  The SparseHamiltonian2D does
the same but for 2D Hubbard. However, here we make a detour: we first
store the matrices in the CRS format (the SparseHamiltonian2DCSR class)
and then convert it in the ELL format. The reason is that for ELL,
we need to know the maximum number of non-zero elements (nnz) of a row.

There are several branches in the git repo: the master contains
only the CPU version. The branch 'GPU' constains the GPU version 
and the branch 'PRIMME' used the PRIMME library to find the 
eigenvalues and eigenvectors. You can find PRIMME
at [http://www.cs.wm.edu/~andreas/software/](http://www.cs.wm.edu/~andreas/software/)

All code is under the [GPLv3](https://www.gnu.org/licenses/gpl.txt).

Documentation
-------------
All code is documented with doxygen. The full docs can be 
generate or [read online](http://wpoely86.github.io/Hubbard-GPU/).
