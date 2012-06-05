#ifndef HAMSPARSE_H
#define HAMSPARSE_H

#include "ham.h"

/**
 * Store the Hamiltonian in the ELL format
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class SparseHamiltonian: public Hamiltonian
{
    public:
	SparseHamiltonian(int Ns, int Nu, int Nd, double J, double U);
	virtual ~SparseHamiltonian();

	void BuildSparseHam();

	void PrintSparse() const;

	void PrintRawELL() const;

        void mvprod(double *, double *, double) const;

        double LanczosDiagonalize(int m=0);

        double* Umatrix() const;

    protected:

	//! The array to hold the data (ELL format) for the up hamiltonian
	double *Up_data;
	//! The array to hold the data (ELL format) for the down hamiltonian
	double *Down_data;

	//! The array to hold the indices for the up hamiltonian
	unsigned int *Up_ind;
	//! The array to hold the indices for the down hamiltonian
	unsigned int *Down_ind;

	//! Maximum number of non zero elements in a row for the up hamiltonian
	int size_Up;
	//! Maximum number of non zero elements in a row for the down hamiltonian
	int size_Down;

};

#endif /* HAMSPARSE_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
