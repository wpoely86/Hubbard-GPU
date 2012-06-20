#include <iostream>
#include <cstdlib>
#include <cmath>
#include "hamsparse2D_CSR.h"

/**
 * Constructor of the SparseHamiltonian2D_CSR class
 * @param Ns Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
SparseHamiltonian2D_CSR::SparseHamiltonian2D_CSR(int L, int D, int Nu, int Nd, double J, double U)
    : HubHam2D(L,D,Nu,Nd,J,U)
{
}

/**
 * Destructor of the SparseHamiltonian2D_CSR class
 */
SparseHamiltonian2D_CSR::~SparseHamiltonian2D_CSR()
{
}

/**
 * Builds and fills the sparse hamiltonian
 */
void SparseHamiltonian2D_CSR::BuildSparseHam()
{
    if( !baseUp.size() || !baseDown.size() )
    {
	std::cerr << "Build base before building Hamiltonian" << std::endl;
	return;
    }

    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    Up_row.reserve(NumUp+1);
    Down_row.reserve(NumDown+1);
    Up_row.push_back(0);
    Down_row.push_back(0);
    int count = 0;

    for(unsigned int a=0;a<NumUp;a++)
    {
	for(unsigned int b=0;b<NumUp;b++)
	{
	    int result = hopping(baseUp[a], baseUp[b]);

	    if(result != 0)
	    {
                Up_data.push_back(J*result);
                Up_col.push_back(b);
                count++;
	    }
	}
        Up_row.push_back(count);
    }

    Up_row.push_back(count+1);

    count = 0;

    for(unsigned int a=0;a<NumDown;a++)
    {
	for(unsigned int b=0;b<NumDown;b++)
	{
	    int result = hopping(baseDown[a], baseDown[b]);

	    if(result != 0)
	    {
                Down_data.push_back(J*result);
                Down_col.push_back(b);
                count++;
	    }
	}
        Down_row.push_back(count);
    }

    Down_row.push_back(count+1);
}

/**
 * Prints the Sparse Hamiltonian
 */
void SparseHamiltonian2D_CSR::PrintSparse() const
{
    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    std::cout << "Up:" << std::endl;

    int count = 0;

    for(unsigned int i=0;i<NumUp;i++)
    {
	for(unsigned int j=0;j<NumUp;j++)
            if( Up_col[count] == j )
                std::cout << Up_data[count++] << "\t";
            else
		std::cout << "0\t";

	std::cout << std::endl;
    }


    std::cout << "Down:" << std::endl;
    count = 0;

    for(unsigned int i=0;i<NumDown;i++)
    {
	for(unsigned int j=0;j<NumDown;j++)
            if( Down_col[count] == j )
                std::cout << Down_data[count++] << "\t";
            else
		std::cout << "0\t";

	std::cout << std::endl;
    }
}

void SparseHamiltonian2D_CSR::PrintRawCSR() const
{
    std::cout << "Up:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<Up_data.size();i++)
        std::cout << Up_data[i] << " ";
    std::cout << std::endl;

    std::cout << "Col indices:" << std::endl;
    for(unsigned int i=0;i<Up_col.size();i++)
        std::cout << Up_col[i] << " ";
    std::cout << std::endl;

    std::cout << "Row indices:" << std::endl;
    for(unsigned int i=0;i<Up_row.size();i++)
        std::cout << Up_row[i] << " ";
    std::cout << std::endl;

    std::cout << "Down:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<Down_data.size();i++)
        std::cout << Down_data[i] << " ";
    std::cout << std::endl;

    std::cout << "Col indices:" << std::endl;
    for(unsigned int i=0;i<Down_col.size();i++)
        std::cout << Down_col[i] << " ";
    std::cout << std::endl;

    std::cout << "Row indices:" << std::endl;
    for(unsigned int i=0;i<Down_row.size();i++)
        std::cout << Down_row[i] << " ";
    std::cout << std::endl;
}

/**
 * Matrix vector product with (sparse) hamiltonian: y = ham*x + alpha*y
 * @param x the x vector
 * @param y the y vector
 * @param alpha the scaling value
 */
void SparseHamiltonian2D_CSR::mvprod(double *x, double *y, double alpha) const
{
    int NumUp = baseUp.size();
    int NumDown = baseDown.size();

    int incx = 1;

    for(int i=0;i<NumUp;i++)
    {
#pragma omp parallel for
        for(int k=0;k<NumDown;k++)
        {
            // the interaction part
            y[i*NumDown+k] = alpha*y[i*NumDown+k] + U * CountBits(baseUp[i] & baseDown[k]) * x[i*NumDown+k];

            // the Hop_down part
            for(unsigned int l=Down_row[k];l<Down_row[k+1];l++)
                y[i*NumDown+k] += Down_data[l] * x[i*NumDown + Down_col[l]];
        }

        // the Hop_Up part
        for(unsigned int l=Up_row[i];l<Up_row[i+1];l++)
            daxpy_(&NumDown,&Up_data[l],&x[Up_col[l]*NumDown],&incx,&y[i*NumDown],&incx);
    }
}

/**
 * Calculates the lowest eigenvalue of the hamiltonian matrix using
 * the lanczos algorithm. Needs lapack.
 * @param m an optional estimate for the lanczos space size
 * @return the lowest eigenvalue
 */
double SparseHamiltonian2D_CSR::LanczosDiagonalize(int m)
{
    if(!m)
        m = dim/1000 > 10 ? dim/1000 : 10;

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

    std::cout << "Done in " << m-10 << " Iterations" << std::endl;

    delete [] qa;
    delete [] qb;

    return acopy[0];
}

/**
 * Builds the interaction diagonal
 * @return pointer to interaction vector. You have to free this yourself
 */
double* SparseHamiltonian2D_CSR::Umatrix() const
{
    double *Umat = new double[getDim()];

    int NumDown = baseDown.size();

    for(int i=0;i<getDim();i++)
    {
	int a = i / NumDown;
	int b = i % NumDown;
	Umat[i] = U * CountBits(baseUp[a] & baseDown[b]);
    }

    return Umat;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
