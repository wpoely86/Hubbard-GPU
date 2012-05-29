/*
 * =====================================================================================
 *
 *       Filename:  ham.cpp
 *
 *    Description:  Make Hubbard ham
 *
 *        Version:  1.0
 *        Created:  14-05-12 18:43:20
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ward Poelmans (), wpoely86@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdint.h>

using namespace std;

typedef unsigned int myint;

int CountBits(myint bytes);
string print_bin(myint num,int bitcount);
myint CalcDim(int Ns,int N);
double hopping(myint a, myint b,myint Nb,int jumpsign);

extern "C" {
   void dsyev_(char *jobz,char *uplo,int *n,double *A,int *lda,double *W,double *work,int *lwork,int *info);
}

int main(void)
{
    int Ns = 4; // number of sites
    int Nu = 2; // number of up electrons
    int Nd = 3; // number of down electrons
    double J = 1.0; // hopping term
    double U = 0.0; // on-site interaction strength

    cout.precision(10);

    if(sizeof(myint) != 4)
    {
	cerr << "I expect 32 bit int's!" << endl;
	return 1;
    }

    myint Nb = 1; // highest power of 2 used
    for(int i=0;i<Ns;i++)
	Nb <<= 1;

    vector<myint> baseUp;
    vector<myint> baseDown;
    baseUp.reserve(CalcDim(Ns,Nu));
    baseDown.reserve(CalcDim(Ns,Nd));

    for(myint i=0;i<Nb;i++)
    {
//	cout << i << "\t" << CountBits(i) << endl;
//	printf("%d\t%d\t%x\n",i,CountBits(i),i);

	if(CountBits(i) == Nd)
	    baseDown.push_back(i);

	if(CountBits(i) == Nu)
	    baseUp.push_back(i);
    }

    int tel = 0;
    for(unsigned int i=0;i<baseUp.size();i++)
	for(unsigned int j=0;j<baseDown.size();j++)
	    cout << tel++ << "\t" << print_bin(baseUp[i],Ns) << "\t" << print_bin(baseDown[j],Ns) << endl;
//	    printf("%d\t%x\t%x\n",tel++,baseUp[i],baseDown[j]);


    int dim = CalcDim(Ns,Nu) * CalcDim(Ns,Nd);

    cout << "Dim: " << dim << endl;

    double *ham = new double[dim*dim];

//    memset(ham,0,dim*dim);

    int NumUp = CalcDim(Ns,Nu);
    int NumDown = CalcDim(Ns,Nd);
//    myint highestbit = Nb>>1;

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
		for(unsigned int d=b;d<baseDown.size();d++)
		{
		    int j = c * NumDown + d;
//		    cout << i << "\t" << print_bin(baseUp[a],Ns) << "\t" << print_bin(baseDown[b],Ns) << "\t";
//		    cout << j << "\t" << print_bin(baseUp[c],Ns) << "\t" << print_bin(baseDown[d],Ns) << endl;

		    ham[j+dim*i] = 0;

		    if(b == d)
			ham[j+dim*i] += J * hopping(baseUp[a], baseUp[c],Nb,upjumpsign);

		    if(a == c)
			ham[j+dim*i] += J * hopping(baseDown[b], baseDown[d],Nb,downjumpsign);

		    ham[i+dim*j] = ham[j+dim*i];
		}

	    // count number of double occupied states
	    ham[i+dim*i] += U * CountBits(baseUp[a] & baseDown[b]);
	}

    double *hamup = new double[NumUp*NumUp];
    double *hamdown = new double[NumDown*NumDown];

    for(unsigned int a=0;a<baseUp.size();a++)
	for(unsigned int c=a;c<baseUp.size();c++)
	    hamup[c+NumUp*a] = hamup[a+NumUp*c] = J * hopping(baseUp[a], baseUp[c],Nb,upjumpsign);

    for(unsigned int b=0;b<baseDown.size();b++)
	for(unsigned int d=b;d<baseDown.size();d++)
	    hamdown[d+NumDown*b] = hamdown[b+NumDown*d] = J * hopping(baseDown[b], baseDown[d],Nb,downjumpsign);

    for(int i=0;i<dim;i++)
    {
	for(int j=0;j<dim;j++)
	    std::cout << ham[j+i*dim] << "\t";
	std::cout << std::endl;
    }

    std::cout << std::endl;

    for(int i=0;i<NumUp;i++)
    {
	for(int j=0;j<NumUp;j++)
	    std::cout << hamup[j+i*NumUp] << "\t";
	std::cout << std::endl;
    }

    std::cout << std::endl;

    for(int i=0;i<NumDown;i++)
    {
	for(int j=0;j<NumDown;j++)
	    std::cout << hamdown[j+i*NumDown] << "\t";
	std::cout << std::endl;
    }

    double *eigenvalues = new double [dim];

    char jobz = 'N';
    char uplo = 'U';

    int lwork = 3*dim - 1;

    double *work = new double [lwork];

    int info;

    dsyev_(&jobz,&uplo,&dim,ham,&dim,eigenvalues,work,&lwork,&info);

    delete [] work;

    cout << "Groundstate Energy: " << eigenvalues[0] << endl;

    delete [] eigenvalues;

    delete [] ham;

    delete [] hamup;

    delete [] hamdown;

    return 0;
}

// you have to compile with -march=native or this will be quite slow...
inline int CountBits(myint bytes)
{
    return __builtin_popcount(bytes);
}

//unsigned int v; // count the number of bits set in v
//unsigned int c; // c accumulates the total bits set in v
//for (c = 0; v; c++)
//{
//  v &= v - 1; // clear the least significant bit set
//}

string print_bin(myint num,int bitcount)
{
    string output = "";
    output.reserve(bitcount);

    for(int i=bitcount-1;i>=0;i--)
	if( (num>>i) & 0x1 )
	    output += "1";
	else
	    output += "0";

    return output;
}

myint CalcDim(int Ns,int N)
{
    double result = 1;

    for(int i=Ns;i>N;i--)
	result *= i*1.0/(i-N);

    return (myint)result;
}

double hopping(myint a, myint b,myint Nb,int jumpsign)
{
    double result = 0;
    int sign;
    myint cur = a;
    // move all electrons one site to the right
    cur <<= 1;

    // periodic boundary condition
    if( cur & Nb )
	cur ^= Nb + 0x1; // flip highest bit and lowest bit

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
	    hop += Nb>>1;
	    sign = jumpsign;
	}
	else
	    hop += hop>>1;

	if( (a ^ hop) == b )
	    result -= sign;
    }

    cur = a;
    // move all electrons one site to the left
    cur >>= 1;

    // periodic boundary condition
    if( a & 0x1 )
	cur ^= Nb>>1; // flip highest bit

    // find places where a electron can jump into
    cur &= ~a;

    while(cur)
    {
	// isolate the rightmost 1 bit
	myint hop = cur & (~cur + 1);

	cur ^= hop;

	sign = 1;

	if(hop & Nb>>1)
	{
	    hop += 0x1;
	    sign = jumpsign;
	}
	else
	    hop += hop<<1;

	if( (a ^ hop) == b )
	    result -= sign;
    }

    return result;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
