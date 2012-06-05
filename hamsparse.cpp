#include <iostream>
#include <cstdlib>
#include <cmath>
#include "hamsparse.h"

/**
 * Constructor of the SparseHamiltonian class
 * @param Ns Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
SparseHamiltonian::SparseHamiltonian(int Ns, int Nu, int Nd, double J, double U)
    : Hamiltonian(Ns,Nu,Nd,J,U)
{
    Up_data = 0;
    Down_data = 0;

    Up_ind = 0;
    Down_ind = 0;

    size_Up = 0;
    size_Down = 0;
}

/**
 * Destructor of the SparseHamiltonian class
 */
SparseHamiltonian::~SparseHamiltonian()
{
    if(Up_data)
	delete [] Up_data;

    if(Down_data)
	delete [] Down_data;

    if(Up_ind)
	delete [] Up_ind;

    if(Down_ind)
	delete [] Down_ind;
}

/**
 * Builds and fills the sparse hamiltonian
 */
void SparseHamiltonian::BuildSparseHam()
{
    if( !baseUp.size() || !baseDown.size() )
    {
	std::cerr << "Build base before building Hamiltonian" << std::endl;
	return;
    }

    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    int upjumpsign, downjumpsign;

    if( Nu % 2 == 0)
	upjumpsign = -1;
    else
	upjumpsign = 1;

    if( Nd % 2 == 0)
	downjumpsign = -1;
    else
	downjumpsign = 1;

    size_Up = ((Ns-Nu)>Nu) ? 2*Nu : 2*(Ns-Nu);
    size_Down = ((Ns-Nd)>Nd) ? 2*Nd : 2*(Ns-Nd);

    Up_data = new double [NumUp*size_Up];
    Up_ind = new unsigned int [NumUp*size_Up];

    Down_data = new double [NumDown*size_Down];
    Down_ind = new unsigned int [NumDown*size_Down];

    for(unsigned int a=0;a<NumUp;a++)
    {
	int count = 0;

	for(unsigned int b=0;b<NumUp;b++)
	{
	    int result = hopping(baseUp[a], baseUp[b],upjumpsign);

	    if(result != 0)
	    {
		Up_data[a+count*NumUp] = J*result;
		Up_ind[a+count*NumUp] = b;
		count++;
	    }
	}

	if(count < size_Up)
	    for(int i=count;i<size_Up;i++)
	    {
		Up_data[a+i*NumUp] = 0;
		Up_ind[a+i*NumUp] = 0;
	    }
	else if(count > size_Up)
	    std::cerr << "Something went terribly wrong!" << std::endl;
    }

    for(unsigned int a=0;a<NumDown;a++)
    {
	int count = 0;

	for(unsigned int b=0;b<NumDown;b++)
	{
	    int result = hopping(baseDown[a], baseDown[b],downjumpsign);

	    if(result != 0)
	    {
		Down_data[a+count*NumDown] = J*result;
		Down_ind[a+count*NumDown] = b;
		count++;
	    }
	}

	if(count < size_Down)
	    for(int i=count;i<size_Down;i++)
	    {
		Down_data[a+i*NumDown] = 0;
		Down_ind[a+i*NumDown] = 0;
	    }
	else if(count > size_Down)
	    std::cerr << "Something went terribly wrong!" << std::endl;
    }
}

/**
 * Prints the Sparse Hamiltonian
 */
void SparseHamiltonian::PrintSparse() const
{
    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    std::cout << "Up:" << std::endl;

    for(unsigned int i=0;i<NumUp;i++)
    {
	int count = 0;

	for(unsigned int j=0;j<NumUp;j++)
	    if(count < size_Up && Up_ind[i+count*NumUp] == j)
		std::cout << Up_data[i + NumUp*count++] << "\t";
	    else
		std::cout << "0\t";

	std::cout << std::endl;
    }

    std::cout << "Down:" << std::endl;

    for(unsigned int i=0;i<NumDown;i++)
    {
	int count = 0;

	for(unsigned int j=0;j<NumDown;j++)
	    if(count < size_Down && Down_ind[i + count*NumDown] == j)
		std::cout << Down_data[i + NumDown*count++] << "\t";
	    else
		std::cout << "0\t";

	std::cout << std::endl;
    }
}

void SparseHamiltonian::PrintRawELL() const
{
    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    std::cout << "Up:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<NumUp;i++)
    {
	for(int j=0;j<size_Up;j++)
	    std::cout << Up_data[i+j*NumUp] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Indices:" << std::endl;
    for(unsigned int i=0;i<NumUp;i++)
    {
	for(int j=0;j<size_Up;j++)
	    std::cout << Up_ind[i+j*NumUp] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Down:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<NumDown;i++)
    {
	for(int j=0;j<size_Down;j++)
	    std::cout << Down_data[i+j*NumDown] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Indices:" << std::endl;
    for(unsigned int i=0;i<NumDown;i++)
    {
	for(int j=0;j<size_Down;j++)
	    std::cout << Down_ind[i+j*NumDown] << "\t";

	std::cout << std::endl;
    }
}

/**
 * Matrix vector product with (sparse) hamiltonian: y = ham*x + alpha*y
 * @param x the x vector
 * @param y the y vector
 * @param alpha the scaling value
 */
void SparseHamiltonian::mvprod(double *x, double *y, double alpha) const
{
    int NumUp = baseUp.size();
    int NumDown = baseDown.size();

    int incx = 1;

    for(int i=0;i<NumUp;i++)
    {
        for(int k=0;k<NumDown;k++)
        {
            // the interaction part
            y[i*NumDown+k] = alpha*y[i*NumDown+k] + U * CountBits(baseUp[i] & baseDown[k]) * x[i*NumDown+k];

            // the Hop_down part
            for(int l=0;l<size_Down;l++)
                y[i*NumDown+k] += Down_data[k+l*NumDown] * x[i*NumDown + Down_ind[k+l*NumDown]];
        }

        // the Hop_Up part
        for(int l=0;l<size_Up;l++)
            daxpy_(&NumDown,&Up_data[i+l*NumUp],&x[Up_ind[i+l*NumUp]*NumDown],&incx,&y[i*NumDown],&incx);
    }
}

/**
 * Calculates the lowest eigenvalue of the hamiltonian matrix using
 * the lanczos algorithm. Needs lapack.
 * @param m an optional estimate for the lanczos space size
 * @return the lowest eigenvalue
 */
double SparseHamiltonian::LanczosDiagonalize(int m)
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

    delete [] qa;
    delete [] qb;

    return acopy[0];
}

/**
 * Builds the interaction diagonal
 * @return pointer to interaction vector. You have to free this yourself
 */
double* SparseHamiltonian::Umatrix() const
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
