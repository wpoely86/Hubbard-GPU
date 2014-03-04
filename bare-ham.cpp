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

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <assert.h>
#include <hdf5.h>
#include "bare-ham.h"

// macro to help check return status of HDF5 functions
#define HDF5_STATUS_CHECK(status) {                 \
    if(status < 0)                                  \
    std::cerr << __FILE__ << ":" << __LINE__ <<     \
    ": Problem with writing to file. Status code="  \
    << status << std::endl;                         \
}

/**
 * Constructor of the BareHamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
BareHamiltonian::BareHamiltonian(int L, int Nu, int Nd, double J, double U)
{
    this->L = L;
    this->Nu = Nu;
    this->Nd = Nd;
    this->J = J;
    this->U = U;

    if(L > 30)
	std::cerr << "We cannot do more then 30 sites" << std::endl;

    if(Nu > L || Nd > L)
	std::cerr << "Too many electrons for this lattice" << std::endl;

    dim = CalcDim(L,Nu) * CalcDim(L,Nd);

    ham = 0;

    Hb = 1<<L; // highest power of 2 used
    Hbc = CountBits(Hb-1);
}

/**
 * Destructor of the BareHamiltonian class
 */
BareHamiltonian::~BareHamiltonian()
{
    if(ham)
	delete [] ham;
}

/**
 * Calculates the dimension for the up or down electron space.
 * Basically, it calculates the N-combination out of L. This algorithm
 * is not perfect, overflow can occur but should be no problem for our
 * small problems.
 * @param L number of lattice sites
 * @param N number of up or down electrons
 * @return the dimension of the up or down electron space
 */
int BareHamiltonian::CalcDim(int L, int N)
{
    int result = 1;

    for(int i=1;i<=N;i++)
    {
	result *= L--;
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
int BareHamiltonian::CountBits(myint bits)
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
std::string BareHamiltonian::print_bin(myint num,int bitcount)
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
 * Print a int in binary form to a string. Prints a much as needed
 * to represent the current system.
 * @param num the myint to print in binary form
 * @return a string with the binary representation of num
 */
std::string BareHamiltonian::print_bin(myint num) const
{
    return print_bin(num, Hbc);
}

/**
 * Builds all the up and down base kets
 */
void BareHamiltonian::BuildBase()
{
    baseUp.reserve(CalcDim(L,Nu));
    baseDown.reserve(CalcDim(L,Nd));

    for(myint i=0;i<Hb;i++)
    {
	if(CountBits(i) == Nd)
	    baseDown.push_back(i);

	if(CountBits(i) == Nu)
	    baseUp.push_back(i);
    }
}

/**
 * This method is always present and should build
 * the Hamiltonian, either full or sparse
 */
void BareHamiltonian::BuildHam()
{
    BuildFullHam();
}

/**
 * Calculates the sign of the hop between site i and site j on ket a. Make sure
 * that i < j!
 * @param i the first sp
 * @param j the second sp
 * @param a the ket to use
 * @return the sign of the hop
 */
int BareHamiltonian::CalcSign(int i,int j,myint a) const
{
    int sign;

    // count the number of set bits between i and j in ket a
    sign = CountBits(( ((1<<j) - 1) ^ ((1<<(i+1)) - 1) ) & a);

    // when uneven, we get a minus sign
    if( sign & 0x1 )
	return -1;
    else
	return +1;
}

/**
 * Getter for the number of lattice sites
 * @return the number of lattice sites
 */
int BareHamiltonian::getL() const
{
    return L;
}

/**
 * Getter for the number of up electrons
 * @return the number of up electrons
 */
int BareHamiltonian::getNu() const
{
    return Nu;
}

/**
 * Getter for the number of down electrons
 * @return the number of down electrons
 */
int BareHamiltonian::getNd() const
{
    return Nd;
}

/**
 * Getter for the dimension of the BareHamiltonian matrix
 * @return dimension of the BareHamiltonian matrix
 */
int BareHamiltonian::getDim() const
{
    return dim;
}

/**
 * Getter for the hopping strength
 * @return the hopping strength
 */
double BareHamiltonian::getJ() const
{
    return J;
}

/**
 * Getter for the onsite interaction strength
 * @return the onsite interaction strength
 */
double BareHamiltonian::getU() const
{
    return U;
}

/**
 * Set the value of U
 * @param myU the value of U
 */
void BareHamiltonian::setU(double myU)
{
    U = myU;
}

/**
 * Getter for the i base up ket
 * @param i the number of the base ket
 * @return the base ket
 */
myint BareHamiltonian::getBaseUp(unsigned int i) const
{
    return baseUp[i];
}

/**
 * Getter for the i base down ket
 * @param i the number of the base ket
 * @returns the base ket
 */
myint BareHamiltonian::getBaseDown(unsigned int i) const
{
    return baseDown[i];
}

/**
 * Exactly calculates the eigenvalues of the BareHamiltonian matrix.
 * Needs lapack.
 * @return the lowest eigenvalue
 */
std::vector<double> BareHamiltonian::ExactDiagonalizeFull(bool calc_eigenvectors)
{
    assert(ham);

    std::vector<double> eigenvalues(dim);

    Diagonalize(dim, ham, eigenvalues.data(), calc_eigenvectors);

    return eigenvalues;
}

/**
 * Prints the groundstate in function of the basis vectors
 * Only gives meanfull output after running BareHamiltonian::ExactDiagonalizeFull(true)
 */
void BareHamiltonian::PrintGroundstateVector() const
{
    int NumDown = CalcDim(L,Nd);

    for(int i=0;i<dim;i++)
    {
        int b = i % NumDown;
        int a = (i - b)/NumDown;
        std::cout << ham[i] << "\t |" << print_bin(getBaseUp(a)) << " " << print_bin(getBaseDown(b)) << ">" << std::endl;
    }
}

/**
 * Calculates the lowest eigenvalue of the BareHamiltonian matrix using
 * the lanczos algorithm. Needs lapack.
 * @param m an optional estimate for the lanczos space size
 * @return the lowest eigenvalue
 */
double BareHamiltonian::LanczosDiagonalize(int m)
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
  * Prints the full BareHamiltonian matrix to stdout
  */
void BareHamiltonian::Print(bool list) const
{
    for(int i=0;i<dim;i++)
    {
        if(list)
            for(int j=0;j<dim;j++)
                std::cout << i << "\t" << j << "\t\t" << ham[i+j*dim] << std::endl;
        else
        {
            for(int j=0;j<dim;j++)
                std::cout << ham[i+j*dim] << "\t";
            std::cout << std::endl;
        }
    }
}

/**
 * Print the basis set used.
 */
void BareHamiltonian::PrintBase() const
{
    for(unsigned int a=0;a<baseUp.size();a++)
        for(unsigned int b=0;b<baseDown.size();b++)
            std::cout << a*baseDown.size()+b << "\t" << print_bin(baseUp[a]) << "\t" << print_bin(baseDown[b]) << std::endl;
}

double BareHamiltonian::arpackDiagonalize()
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
    int ncv = 42;

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
    int rvec = 1;

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

    if( rvec )
    {
        std::stringstream name;
        name << "results-" << L << "-" << Nu << "-" << Nd << "-" << U << ".h5";
        SaveToFile(name.str(), z, n*nev);

/*         int NumDown = CalcDim(L,Nd);
 *         double max = 0;
 * 
 *         for(int i=0;i<dim;i++)
 *         {
 *             int b = i % NumDown;
 *             int a = (i - b)/NumDown;
 *             if( ! (getBaseUp(a) & getBaseDown(b)) )
 *                 {
 * //            if( fabs(z[i]) > 1e-3 )
 *                 std::cout << z[i] << "\t |" << print_bin(getBaseUp(a)) << " " << print_bin(getBaseDown(b)) << ">\t" << (getBaseUp(a) & getBaseDown(b)) << std::endl;
 * 
 *                 if(fabs(z[i]) > max)
 *                     max = fabs(z[i]);
 *                 }
 *         }
 * 
 *         std::cout << "max: " << max << std::endl;
 */
    }

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
double BareHamiltonian::MemoryNeededFull() const
{
    unsigned int dim = CalcDim(L,Nu) * CalcDim(L,Nd);
    double result;

    // base kets
    result = (CalcDim(L,Nu) + CalcDim(L,Nd)) * sizeof(myint);

    // BareHamiltonian matrix
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
double BareHamiltonian::MemoryNeededLanczos() const
{
    unsigned int dim = CalcDim(L,Nu) * CalcDim(L,Nd);
    double result;

    // base kets
    result = (CalcDim(L,Nu) + CalcDim(L,Nd)) * sizeof(myint);

    // hamup sparse matrix
    result += CalcDim(L,Nu) * (((L-Nu)>Nu) ? 2*Nu : 2*(L-Nu)) * sizeof(double);

    // hamdown sparse matrix
    result += CalcDim(L,Nd) * (((L-Nd)>Nd) ? 2*Nd : 2*(L-Nd)) * sizeof(double);

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
double BareHamiltonian::MemoryNeededArpack() const
{
    unsigned int dim = CalcDim(L,Nu) * CalcDim(L,Nd);
    double result;

    // base kets
    result = (CalcDim(L,Nu) + CalcDim(L,Nu)) * sizeof(myint);

    // hamup sparse matrix
    result += CalcDim(L,Nu) * (((L-Nu)>Nu) ? 2*Nu : 2*(L-Nu)) * sizeof(double);

    // hamdown sparse matrix
    result += CalcDim(L,Nd) * (((L-Nd)>Nd) ? 2*Nd : 2*(L-Nd)) * sizeof(double);

    // arpack stuff:
    // resid
    result += 1 * dim * sizeof(double);

    // v
    result += 16 * dim * sizeof(double);

    // workd
    result += 3 * dim * sizeof(double);

    // lworkd
    result += 16 * (16+8) * sizeof(double);

    return result;
}

/**
 * Helper for exact diagonalization: diagonalize the matrix given in mat with dimension dim, stores
 * the eigenvalues in eigs and optionally the eigenvectors in mat
 * @param dim the dimension of the matrix
 * @param mat the actual matrix (size: dim*dim)
 * @param eigs array of size dim to store eigenvalues
 * @param calc_eigenvectors whether or not the calculate the eigenvectors
 */
void BareHamiltonian::Diagonalize(int dim, double *mat, double *eigs, bool calc_eigenvectors)
{
    assert(mat && "mat must be allocated");
    assert(eigs && "eigs must be allocated");

    char jobz;

    if(calc_eigenvectors)
        jobz = 'V';
    else
        jobz = 'N';

    char uplo = 'U';

    int lwork, liwork;

    if(calc_eigenvectors)
    {
        lwork = 6*dim+1+2*dim*dim;
        liwork = 3+5*dim;
    } else
    {
        lwork = 2*dim+1;
        liwork = 1;
    }

    double *work = new double[lwork];

    int *iwork = new int[liwork];

    int info;

    dsyevd_(&jobz, &uplo, &dim, mat, &dim, eigs, work, &lwork, iwork, &liwork, &info);

    if(info != 0)
        std::cerr << "Calculating eigenvalues failed..." << std::endl;

    delete [] work;
    delete [] iwork;
}


/**
 * Save the hamiltonian matrix to a file in the HDF5 format
 * @param filename the name of the file to write to
 */
void BareHamiltonian::SaveToFile(const std::string filename) const
{
    if(!ham)
        return;

    SaveToFile(filename, ham, dim*dim);
}

/**
 * Save a array to a file in the HDF5 format
 * @param filename the name of the file to write to
 * @param data a pointer to the array to write to a file
 * @param dim the size of the array
 */
void BareHamiltonian::SaveToFile(const std::string filename, double *data, int dim) const
{
    hid_t       file_id, dataset_id, dataspace_id, attribute_id;
    herr_t      status;

    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_STATUS_CHECK(file_id);

    hsize_t dimarr = dim;

    dataspace_id = H5Screate_simple(1, &dimarr, NULL);

    dataset_id = H5Dcreate(file_id, "ham", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data );
    HDF5_STATUS_CHECK(status);

    status = H5Sclose(dataspace_id);
    HDF5_STATUS_CHECK(status);

    dataspace_id = H5Screate(H5S_SCALAR);

    attribute_id = H5Acreate (dataset_id, "L", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &L );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (dataset_id, "Nu", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &Nu );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (dataset_id, "Nd", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &Nd );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (dataset_id, "J", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &J );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (dataset_id, "U", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &U );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (dataset_id, "dim", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &dim );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    status = H5Sclose(dataspace_id);
    HDF5_STATUS_CHECK(status);

    status = H5Dclose(dataset_id);
    HDF5_STATUS_CHECK(status);

    status = H5Fclose(file_id);
    HDF5_STATUS_CHECK(status);
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
