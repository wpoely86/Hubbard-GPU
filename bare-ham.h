/* Copyright (C) 2014  Ward Poelmans

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

#ifndef BARE_HAM_H
#define BARE_HAM_H

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

    void dsaupd_(int *ido, char *bmat, int *n, char *which,
            int *nev, double *tol, double *resid, int *ncv,
            double *v, int *ldv, int *iparam, int *ipntr,
            double *workd, double *workl, int *lworkl, int *info);

    void dseupd_(int *rvec, char *All, int *select, double *d,
            double *z, int *ldz, double *sigma,
            char *bmat, int *n, char *which, int *nev,
            double *tol, double *resid, int *ncv, double *v,
            int *ldv, int *iparam, int *ipntr, double *workd,
            double *workl, int *lworkl, int *info);
}

/**
 * The is a pure virtual basic hamiltonian class. It's the base class for both the sites basis and the momentum basis
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class BareHamiltonian
{
    public:
	BareHamiltonian(int Ns, int Nu, int Nd, double J, double U);
	virtual ~BareHamiltonian();

	int CalcDim(int Ns, int N) const;

	int CountBits(myint bits) const;

	std::string print_bin(myint num,int bitcount=0) const;

	virtual void BuildBase();

	virtual void BuildFullHam() = 0;

        virtual void BuildHam();

	int getL() const;
	int getNu() const;
	int getNd() const;
	int getDim() const;

	double getJ() const;
	double getU() const;

        void setU(double);

	myint getBaseUp(unsigned int i) const;
	myint getBaseDown(unsigned int i) const;

        virtual std::vector<double> ExactDiagonalizeFull(bool calc_eigenvectors=false);
	virtual double LanczosDiagonalize(int m=0);
	virtual double arpackDiagonalize();

	void Print(bool list=false) const;

        void PrintBase() const;

        void PrintGroundstateVector() const;

        virtual void mvprod(double *x, double *y, double alpha) const = 0;

        double MemoryNeededFull() const;

        double MemoryNeededLanczos() const;

        double MemoryNeededArpack() const;

    protected:
        int CalcSign(int i,int j,myint a) const;

        static void Diagonalize(int dim, double *mat, double *eigs, bool calc_eigenvectors);

	//! Number of sites
	int L;
	//! Number of up electrons
	int Nu;
	//! Number of down electrons
	int Nd;
	//! Hopping strength
	double J;
	//! On site interaction strength
	double U;
	//! Storage for the BareHamiltonian matrix
	double *ham;
	//! Dimension of the BareHamiltonian matrix
	int dim;

	//! Hightest bit used
	myint Hb;
        //! the location of the highest bit
        int Hbc;

	//! vector to hold all bases ket's for up electrons
	std::vector<myint> baseUp;
	//! vector to hold all bases ket's for down electrons
	std::vector<myint> baseDown;
};

#endif /* BARE_HAM_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
