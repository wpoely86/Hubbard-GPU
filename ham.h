#ifndef HAM_H
#define HAM_H

#include <vector>
#include <string>

typedef unsigned int myint;

extern "C" {
    void dsyevd_( char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
    double ddot_(int *n,double *x,int *incx,double *y,int *incy);
    void dscal_(int *n,double *alpha,double *x,int *incx);
    void dsymv_(char *uplo, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    void daxpy_(int *n,double *alpha,double *x,int *incx,double *y,int *incy);
    void dstev_( const char* jobz, const int* n, double* d, double* e, double* z, const int* ldz, double* work, int* info );
}

/**
 * This is the main class where all the magic happens
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
	double LanczosDiagonalizeFull(int m);

	void Print() const;

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
