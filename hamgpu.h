#ifndef HAMGPU_H
#define HAMGPU_H

#include "hamsparse.h"

/**
 * Use the GPU (Cuda) for the Lanczos calculation
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
template<class T>
class GPUHamiltonian: public T
{
    public:
	GPUHamiltonian(int Ns, int Nu, int Nd, double J, double U);
	GPUHamiltonian(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~GPUHamiltonian();

        void mvprod(double *x, double *y, double alpha);

        double LanczosDiagonalize(int m=0);

    protected:

	//! The diagonal of the full hamiltonian (onsite interaction)
        double *Umat_gpu;

	//! The array to hold the data (ELL format) for the up hamiltonian
	double *Up_data_gpu;
	//! The array to hold the data (ELL format) for the down hamiltonian
	double *Down_data_gpu;

	//! The array to hold the indices for the up hamiltonian
	unsigned int *Up_ind_gpu;
	//! The array to hold the indices for the down hamiltonian
	unsigned int *Down_ind_gpu;

};

#endif /* HAMGPU_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
