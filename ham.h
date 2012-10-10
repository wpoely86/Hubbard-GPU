/* Copyright (C) 2012  Ward Poelmans

This file is part of Hubbard-GPU.

Hubbard-GPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hubbard-GPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Hubbard-GPU.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HAM_H
#define HAM_H

#include <vector>
#include <string>

//! type to store the basis set vectors in
typedef unsigned int myint;

extern "C" {
    void dsyevd_( char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
    double ddot_(int *n,double *x,int *incx,double *y,int *incy);
    void dscal_(int *n,double *alpha,double *x,int *incx);
    void dsymv_(char *uplo, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    void daxpy_(int *n,const double *alpha,double *x,int *incx,double *y,int *incy);
    void dstev_( const char* jobz, const int* n, double* d, double* e, double* z, const int* ldz, double* work, int* info );
}

/**
 * This is the main (base) class. It calculates the full Hamiltonian matrix for 1D Hubbard. It can both exact diagonlize
 * or use a Lanczos algorithm to calculate the groundstate energy.
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class Hamiltonian
{
    public:
	Hamiltonian(int Ns, int Nu, int Nd, double J, double U);
	virtual ~Hamiltonian();

	int CalcDim(int Ns, int N) const;

	int CountBits(myint bits) const;

	std::string print_bin(myint num,int bitcount) const;

	void BuildBase();
	void BuildFullHam();

	int getNs() const;
	int getNu() const;
	int getNd() const;
	int getDim() const;

	double getJ() const;
	double getU() const;

	myint getBaseUp(unsigned int i) const;
	myint getBaseDown(unsigned int i) const;

	double ExactDiagonalizeFull() const;
	double LanczosDiagonalize(int m=0);
	double arpackDiagonalize();

	void Print() const;

        void PrintBase() const;

        virtual void mvprod(double *x, double *y, double alpha) const;

        double MemoryNeededFull() const;

        double MemoryNeededLanczos() const;

        double MemoryNeededArpack() const;

    protected:
	int hopping(myint a, myint b, int jumpsign) const;

	//! Number of sites
	int Ns;
	//! Number of up electrons
	int Nu;
	//! Number of down electrons
	int Nd;
	//! Hopping strength
	double J;
	//! On site interaction strength
	double U;
	//! Storage for the hamiltonian matrix
	double *ham;
	//! Dimension of the hamiltonian matrix
	int dim;

	//! Hightest bit used
	myint Hb;

	//! vector to hold all bases ket's for up electrons
	std::vector<myint> baseUp;
	//! vector to hold all bases ket's for down electrons
	std::vector<myint> baseDown;
};

#endif /* HAM_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
