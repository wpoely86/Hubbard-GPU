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

/**
 * Constructor of the Hamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
Hamiltonian::Hamiltonian(int L, int Nu, int Nd, double J, double U) : BareHamiltonian(L,Nu,Nd,J,U)
{
}

/**
 * Destructor of the Hamiltonian class
 */
Hamiltonian::~Hamiltonian()
{
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

    int NumDown = CalcDim(L,Nd);

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

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
