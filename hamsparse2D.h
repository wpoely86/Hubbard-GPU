#ifndef HAMSPARSE2D_H
#define HAMSPARSE2D_H

#include "hamsparse2D_CSR.h"

/**
 * Store the 2D Hubbard Hamiltonian in the ELL format
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class SparseHamiltonian2D: public SparseHamiltonian2D_CSR
{
    public:
	SparseHamiltonian2D(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~SparseHamiltonian2D();

	void BuildSparseHam();

	void PrintSparse() const;

	void PrintRawELL() const;

        void mvprod(double *, double *, double) const;

    protected:

	//! The array to hold the data (ELL format) for the up hamiltonian
	double *Up_datas;
	//! The array to hold the data (ELL format) for the down hamiltonian
	double *Down_datas;

	//! The array to hold the indices for the up hamiltonian
	unsigned int *Up_ind;
	//! The array to hold the indices for the down hamiltonian
	unsigned int *Down_ind;

	//! Maximum number of non zero elements in a row for the up hamiltonian
	int size_Up;
	//! Maximum number of non zero elements in a row for the down hamiltonian
	int size_Down;

};

#endif /* HAMSPARSE2D_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
