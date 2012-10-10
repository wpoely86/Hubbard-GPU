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

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ham.h"

// arpack
extern "C"
{
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
 * Constructor of the Hamiltonian class
 * @param Ns Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
Hamiltonian::Hamiltonian(int Ns, int Nu, int Nd, double J, double U)
{
    this->Ns = Ns;
    this->Nu = Nu;
    this->Nd = Nd;
    this->J = J;
    this->U = U;

    if(Ns > 30)
	std::cerr << "We cannot do more then 30 sites" << std::endl;

    if(Nu > Ns || Nd > Ns)
	std::cerr << "Too many electrons for this lattice" << std::endl;

    dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);

    ham = 0;

    Hb = 1; // highest power of 2 used
    for(int i=0;i<Ns;i++)
	Hb <<= 1;
}

/**
 * Destructor of the Hamiltonian class
 */
Hamiltonian::~Hamiltonian()
{
    if(ham)
	delete [] ham;
}

/**
 * Calculates the dimension for the up or down electron space.
 * Basically, it calculates the N-combination out of Ns. This algorithm
 * is not perfect, overflow can occur but should be no problem for our
 * small problems.
 * @param Ns number of lattice sites
 * @param N number of up or down electrons
 * @return the dimension of the up or down electron space
 */
int Hamiltonian::CalcDim(int Ns, int N) const
{
    int result = 1;

    for(int i=1;i<=N;i++)
    {
	result *= Ns--;
	result /= i;
    }

    return result;
}

/**
 * Counts the number of bits set. It uses the builtin gcc function
 * __buildtin_popcount. You must compile with -march=native and it
 * will then generate a POPCNT instruction on platforms that support
 * it. Otherwise it will be slow. Consider using a different method
 * here if you don't have the POPCNT instruction: there are fast SSSE3
 * and SSE2 ways to do this(google is your friend).
 * @param bits the myint of which to count the number of bits set
 * @return the number of bits set
 */
int Hamiltonian::CountBits(myint bits) const
{
    return __builtin_popcount(bits);
}

/**
 * Print a int in binary form to a string. It only prints the bitcount
 * least significant bits
 * @param num the myint to print in binary form
 * @param bitcount the number of bits to print (starting from the LSB)
 * @return a string with the binary representation of num
 */
std::string Hamiltonian::print_bin(myint num,int bitcount) const
{
    std::string output = "";
    output.reserve(bitcount);

    for(int i=bitcount-1;i>=0;i--)
	if( (num>>i) & 0x1 )
	    output += "1";
	else
	    output += "0";

    return output;
}

/**
 * Builds all the up and down base kets
 */
void Hamiltonian::BuildBase()
{
    baseUp.reserve(CalcDim(Ns,Nu));
    baseDown.reserve(CalcDim(Ns,Nd));

    for(myint i=0;i<Hb;i++)
    {
	if(CountBits(i) == Nd)
	    baseDown.push_back(i);

	if(CountBits(i) == Nu)
	    baseUp.push_back(i);
    }
}

/**
  * Builds the full Hamiltonian matrix
  */
void Hamiltonian::BuildFullHam()
{
    if( !baseUp.size() || !baseDown.size() )
    {
	std::cerr << "Build base before building Hamiltonian" << std::endl;
	return;
    }

    ham = new double[dim*dim];

    int NumDown = CalcDim(Ns,Nd);

    int upjumpsign, downjumpsign;

    if( Nu % 2 == 0)
	upjumpsign = -1;
    else
	upjumpsign = 1;

    if( Nd % 2 == 0)
	downjumpsign = -1;
    else
	downjumpsign = 1;

    for(unsigned int a=0;a<baseUp.size();a++)
	for(unsigned int b=0;b<baseDown.size();b++)
	{
	    int i = a * NumDown + b;

	    for(unsigned int c=a;c<baseUp.size();c++)
		for(unsigned int d=0;d<baseDown.size();d++)
		{
		    int j = c * NumDown + d;

		    ham[j+dim*i] = 0;

		    if(b == d)
			ham[j+dim*i] += J * hopping(baseUp[a], baseUp[c],upjumpsign);

		    if(a == c)
			ham[j+dim*i] += J * hopping(baseDown[b], baseDown[d],downjumpsign);

		    ham[i+dim*j] = ham[j+dim*i];
		}

	    // count number of double occupied states
	    ham[i+dim*i] = U * CountBits(baseUp[a] & baseDown[b]);
	}
}

/**
  * private method used to see if a hopping between state a and b is
  * possible and with which sign. Only for 1D Hubbard.
  * @param a the bra to use
  * @param b the ket to use
  * @param jumpsign the sign to use when a jump over the periodic boundary occurs
  * @returns matrix element of the hopping term between the ket and the bra. You still
  * have to multiply this with the hopping strength J
  */
int Hamiltonian::hopping(myint a, myint b,int jumpsign) const
{
    int result = 0;
    int sign;
    myint cur = a;
    // move all electrons one site to the right
    cur <<= 1;

    // periodic boundary condition
    if( cur & Hb )
	cur ^= Hb + 0x1; // flip highest bit and lowest bit

    // find places where a electron can jump into
    cur &= ~a;

    while(cur)
    {
	// isolate the rightmost 1 bit
	myint hop = cur & (~cur + 1);

	cur ^= hop;

	sign = 1;

	if(hop & 0x1)
	{
	    hop += Hb>>1;
	    sign = jumpsign;
	}
	else
	    hop += hop>>1;

	if( (a ^ hop) == b )
	{
	    result -= sign;
	    break;
	}
    }

    cur = a;
    // move all electrons one site to the left
    cur >>= 1;

    // periodic boundary condition
    if( a & 0x1 )
	cur ^= Hb>>1; // flip highest bit

    // find places where a electron can jump into
    cur &= ~a;

    while(cur)
    {
	// isolate the rightmost 1 bit
	myint hop = cur & (~cur + 1);

	cur ^= hop;

	sign = 1;

	if(hop & Hb>>1)
	{
	    hop += 0x1;
	    sign = jumpsign;
	}
	else
	    hop += hop<<1;

	if( (a ^ hop) == b )
	{
	    result -= sign;
	    break;
	}
    }

    return result;
}

/**
 * Getter for the number of lattice sites
 * @return the number of lattice sites
 */
int Hamiltonian::getNs() const
{
    return Ns;
}

/**
 * Getter for the number of up electrons
 * @return the number of up electrons
 */
int Hamiltonian::getNu() const
{
    return Nu;
}

/**
 * Getter for the number of down electrons
 * @return the number of down electrons
 */
int Hamiltonian::getNd() const
{
    return Nd;
}

/**
 * Getter for the dimension of the hamiltonian matrix
 * @return dimension of the hamiltonian matrix
 */
int Hamiltonian::getDim() const
{
    return dim;
}

/**
 * Getter for the hopping strength
 * @return the hopping strength
 */
double Hamiltonian::getJ() const
{
    return J;
}

/**
 * Getter for the onsite interaction strength
 * @return the onsite interaction strength
 */
double Hamiltonian::getU() const
{
    return U;
}

/**
 * Getter for the i base up ket
 * @param i the number of the base ket
 * @return the base ket
 */
myint Hamiltonian::getBaseUp(unsigned int i) const
{
    return baseUp[i];
}

/**
 * Getter for the i base down ket
 * @param i the number of the base ket
 * @returns the base ket
 */
myint Hamiltonian::getBaseDown(unsigned int i) const
{
    return baseDown[i];
}

/**
 * Exactly calculates the eigenvalues of the hamiltonian matrix.
 * Needs lapack.
 * @return the lowest eigenvalue
 */
double Hamiltonian::ExactDiagonalizeFull() const
{
    if(!ham)
	return 0;

    char jobz = 'N';
    char uplo = 'U';

    int dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);

    double eigenvalues[dim];

    int lwork = 2*dim+1;

    double *work = new double[lwork];

    int liwork = 1;

    int *iwork = new int[liwork];

    int info;

    dsyevd_(&jobz, &uplo, &dim, ham, &dim, &eigenvalues[0], work, &lwork,iwork,&liwork,&info);

    if(info != 0)
	std::cerr << "Calculating eigenvalues failed..." << std::endl;

    delete [] work;
    delete [] iwork;

    return eigenvalues[0];
}

/**
 * Calculates the lowest eigenvalue of the hamiltonian matrix using
 * the lanczos algorithm. Needs lapack.
 * @param m an optional estimate for the lanczos space size
 * @return the lowest eigenvalue
 */
double Hamiltonian::LanczosDiagonalize(int m)
{
    if(!m)
        m = 10;

    std::vector<double> a(m,0);
    std::vector<double> b(m,0);

    double *qa = new double [dim];
    double *qb = new double [dim];

    double E = 1;

    int i;

    srand(time(0));

    for(i=0;i<dim;i++)
    {
	qa[i] = 0;
	qb[i] = rand()*1.0/RAND_MAX;
    }

    int incx = 1;

    double norm = 1.0/sqrt(ddot_(&dim,qb,&incx,qb,&incx));

    dscal_(&dim,&norm,qb,&incx);

    norm = 1;

    double *f1 = qa;
    double *f2 = qb;
    double *tmp;

    double alpha;

    std::vector<double> acopy(a);
    std::vector<double> bcopy(b);

    i = 1;

    while(fabs(E-acopy[0]) > 1e-4)
    {
        E = acopy[0];

        for(;i<m;i++)
        {
            alpha = -b[i-1];
            dscal_(&dim,&alpha,f1,&incx);

            mvprod(f2,f1,norm);

            a[i-1] = ddot_(&dim,f1,&incx,f2,&incx);

            alpha = -a[i-1];
            daxpy_(&dim,&alpha,f2,&incx,f1,&incx);

            b[i] = sqrt(ddot_(&dim,f1,&incx,f1,&incx));

            if( fabs(b[i]) < 1e-10 )
                break;

            alpha = 1.0/b[i];

            dscal_(&dim,&alpha,f1,&incx);

            tmp = f2;
            f2 = f1;
            f1 = tmp;
        }

        acopy = a;
        bcopy = b;

        char jobz = 'N';
        int info;

        dstev_(&jobz,&m,acopy.data(),&bcopy.data()[1],&alpha,&m,&alpha,&info);

        if(info != 0)
            std::cerr << "Error in Lanczos" << std::endl;

        m += 10;
        a.resize(m);
        b.resize(m);
    }

    delete [] qa;
    delete [] qb;

    return acopy[0];
}

/**
  * Prints the full hamiltonian matrix to stdout
  */
void Hamiltonian::Print() const
{
    for(int i=0;i<dim;i++)
    {
	for(int j=0;j<dim;j++)
	    std::cout << ham[j+i*dim] << "\t";
	std::cout << std::endl;
    }
}

/**
 * Print the basis set used.
 */
void Hamiltonian::PrintBase() const
{
    for(unsigned int a=0;a<baseUp.size();a++)
        for(unsigned int b=0;b<baseDown.size();b++)
            std::cout << a*baseDown.size()+b << "\t" << print_bin(baseUp[a],Ns) << "\t" << print_bin(baseDown[b],Ns) << std::endl;
}

/**
 * Matrix vector product with hamiltonian: y = ham*x + alpha*y
 * @param x the input vector
 * @param y the output vector
 * @param alpha the scaling value
 */
void Hamiltonian::mvprod(double *x, double *y, double alpha) const
{
    double beta = 1;
    int incx = 1;
    char uplo = 'U';

    dsymv_(&uplo,&dim,&beta,ham,&dim,x,&incx,&alpha,y,&incx);
}

double Hamiltonian::arpackDiagonalize()
{
    // dimension of the matrix
    int n = dim;

    // number of eigenvalues to calculate
    int nev = 1;

    // reverse communication parameter, must be zero on first iteration
    int ido = 0;
    // standard eigenvalue problem A*x=lambda*x
    char bmat = 'I';
    // calculate the smallest algebraic eigenvalue
    char which[] = {'S','A'};
    // calculate until machine precision
    double tol = 0;

    // the residual vector
    double *resid = new double[n];

    // the number of columns in v: the number of lanczos vector
    // generated at each iteration, ncv <= n
    // We use the answer to life, the universe and everything, if possible
    int ncv = 5;

    if( n < ncv )
        ncv = n;

    // v containts the lanczos basis vectors
    int ldv = n;
    double *v = new double[ldv*ncv];

    int *iparam = new int[11];
    iparam[0] = 1;   // Specifies the shift strategy (1->exact)
    iparam[2] = 3*n; // Maximum number of iterations
    iparam[6] = 1;   /* Sets the mode of dsaupd.
                        1 is exact shifting,
                        2 is user-supplied shifts,
                        3 is shift-invert mode,
                        4 is buckling mode,
                        5 is Cayley mode. */

    int *ipntr = new int[11]; /* Indicates the locations in the work array workd
                                 where the input and output vectors in the
                                 callback routine are located. */

    // array used for reverse communication
    double *workd = new double[3*n];

    int lworkl = ncv*(ncv+8); /* Length of the workl array */
    double *workl = new double[lworkl];

    // info = 0: random start vector is used
    int info = 0; /* Passes convergence information out of the iteration
                     routine. */

    // rvec == 0 : calculate only eigenvalue
    // rvec > 0 : calculate eigenvalue and eigenvector
    int rvec = 0;

    // how many eigenvectors to calculate: 'A' => nev eigenvectors
    char howmny = 'A';

    int *select;
    // when howmny == 'A', this is used as workspace to reorder the eigenvectors
    if( howmny == 'A' )
        select = new int[ncv];

    // This vector will return the eigenvalues from the second routine, dseupd.
    double *d = new double[nev];

    double *z = 0;

    if( rvec )
        z = new double[n*nev];

    // not used if iparam[6] == 1
    double sigma;

    // first iteration
    dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

    while( ido != 99 )
    {
        // matrix-vector multiplication
        mvprod(workd+ipntr[0]-1, workd+ipntr[1]-1,0);

        dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
    }

    if( info < 0 )
        std::cerr << "Error with dsaupd, info = " << info << std::endl;
    else if ( info == 1 )
        std::cerr << "Maximum number of Lanczos iterations reached." << std::endl;
    else if ( info == 3 )
        std::cerr << "No shifts could be applied during implicit Arnoldi update, try increasing NCV." << std::endl;

    dseupd_(&rvec, &howmny, select, d, z, &ldv, &sigma, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

    if ( info != 0 )
        std::cerr << "Error with dseupd, info = " << info << std::endl;

    // use something to store the result before deleting...
    sigma = d[0];

    delete [] resid;
    delete [] v;
    delete [] iparam;
    delete [] ipntr;
    delete [] workd;
    delete [] workl;
    delete [] d;

    if( rvec )
        delete [] z;

    if( howmny == 'A' )
        delete [] select;

    return sigma;
}

/**
 * Calculates the amount of memory needed for the calculation
 * of all eigenvalues using the exact method.
 * It doesn't include the program code and stuff.
 * @return the amount of memory needed in bytes
 */
double Hamiltonian::MemoryNeededFull() const
{
    unsigned int dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);
    double result;

    // base kets
    result = (CalcDim(Ns,Nu) + CalcDim(Ns,Nd)) * sizeof(myint);

    // hamiltonian matrix
    result += dim * dim * sizeof(double);

    // eigenvalues of ham matrix
    result += dim * sizeof(double);

    // work array for dsyev
    result += (2*dim+1) * sizeof(double);

    return result;
}

/**
 * Calculates the amount of memory needed for the calculation
 * if we use the lanczos method.
 * It doesn't include the program code and stuff.
 * @return the amount of memory needed in bytes
 */
double Hamiltonian::MemoryNeededLanczos() const
{
    unsigned int dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);
    double result;

    // base kets
    result = (CalcDim(Ns,Nu) + CalcDim(Ns,Nd)) * sizeof(myint);

    // hamup sparse matrix
    result += CalcDim(Ns,Nu) * (((Ns-Nu)>Nu) ? 2*Nu : 2*(Ns-Nu)) * sizeof(double);

    // hamdown sparse matrix
    result += CalcDim(Ns,Nd) * (((Ns-Nd)>Nd) ? 2*Nd : 2*(Ns-Nd)) * sizeof(double);

    // we store 2 vectors
    result += 2 * dim * sizeof(double);

    return result;
}

/**
 * Calculates the amount of memory needed for the calculation
 * if we use the arpack.
 * It doesn't include the program code and stuff.
 * @return the amount of memory needed in bytes
 */
double Hamiltonian::MemoryNeededArpack() const
{
    unsigned int dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);
    double result;

    // base kets
    result = (CalcDim(Ns,Nu) + CalcDim(Ns,Nu)) * sizeof(myint);

    // hamup sparse matrix
    result += CalcDim(Ns,Nu) * (((Ns-Nu)>Nu) ? 2*Nu : 2*(Ns-Nu)) * sizeof(double);

    // hamdown sparse matrix
    result += CalcDim(Ns,Nd) * (((Ns-Nd)>Nd) ? 2*Nd : 2*(Ns-Nd)) * sizeof(double);

    // arpack stuff:
    // resid
    result += 1 * dim * sizeof(double);

    // v
    result += 5 * dim * sizeof(double);

    // workd
    result += 3 * dim * sizeof(double);

    // lworkd
    result += 5 * (5+8) * sizeof(double);

    return result;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
