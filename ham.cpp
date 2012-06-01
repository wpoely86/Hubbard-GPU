#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ham.h"

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
  * possible and with which sign
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
double Hamiltonian::LanczosDiagonalizeFull(int m)
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

    char uplo = 'U';
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

            dsymv_(&uplo,&dim,&norm,ham,&dim,f2,&incx,&norm,f1,&incx);

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

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
