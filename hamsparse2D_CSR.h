#ifndef HAMSPARSE2D_CSR_H
#define HAMSPARSE2D_CSR_H

#include "ham2D.h"
#include <vector>

/**
 * Store the Hamiltonian for 2D Hubbard in the CSR format
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class SparseHamiltonian2D_CSR: public HubHam2D
{
    public:
	SparseHamiltonian2D_CSR(int L,int D, int Nu, int Nd, double J, double U);
	virtual ~SparseHamiltonian2D_CSR();

	void BuildSparseHam();

	void PrintSparse() const;

	void PrintRawCSR() const;

        virtual void mvprod(double *, double *, double) const;

        double* Umatrix() const;

    protected:

	//! The array to hold the data (CSR format) for the up hamiltonian
        std::vector<double> Up_data_CSR;
	//! The array to hold the data (CSR format) for the down hamiltonian
	std::vector<double> Down_data_CSR;

	//! The array to hold the column indices for the up hamiltonian
	std::vector<unsigned int> Up_col;
	//! The array to hold the column indices for the down hamiltonian
	std::vector<unsigned int> Down_col;

	//! The array to hold the row indices for the up hamiltonian
	std::vector<unsigned int> Up_row;
	//! The array to hold the row indices for the down hamiltonian
	std::vector<unsigned int> Down_row;

};

#endif /* HAMSPARSE2D_CSR_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
