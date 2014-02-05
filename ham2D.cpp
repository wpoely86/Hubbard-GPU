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
#include "ham2D.h"

/**
 * Constructor of the HubHam2D class
 * @param L The Length of the 2D grid
 * @param D The depth of the 2D grid
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
HubHam2D::HubHam2D(int L, int D, int Nu, int Nd, double J, double U)
    : Hamiltonian(L*D,Nu,Nd,J,U)
{
    this->L = L;
    this->D = D;
}

/**
 * Destructor of the HubHam2D class
 */
HubHam2D::~HubHam2D()
{
}

/**
  * private method used to see if a hopping between state a and b is
  * possible and with which sign: this is for 2D hubbard.
  * @param a the bra to use
  * @param b the ket to use
  * @returns matrix element of the hopping term between the ket and the bra. You still
  * have to multiply this with the hopping strength J
  */
int HubHam2D::hopping(myint a, myint b) const
{
    int result(0), sign;

    for(int i=0;i<L;i++)
        if( a & 1<<i ) // is the i'th bit set?
        {
	    int j;

	    // jump up
	    j = (i + L) % L;
	    if( (~a & 1<<j) && ((a ^ ((1<<i)+(1<<j)) ) == b ) )
	    {
		if(j>i)
		    sign = CalcSign(i,j,a);
		else
		    sign = CalcSign(j,i,a);

                // a minus sign in the hamiltonian ( -J *)
		result = -1 * sign;
		break;
	    }

	    // jump down
	    j = (i - L + L) % L;
	    if( (~a & 1<<j) && ((a ^ ((1<<i)+(1<<j)) ) == b ) )
	    {
		if(j>i)
		    sign = CalcSign(i,j,a);
		else
		    sign = CalcSign(j,i,a);

		result = -1 * sign;
		break;
	    }

	    // jump right
	    j = L * (i/L) + (i + 1) % L;
	    if( (~a & 1<<j) && ((a ^ ((1<<i)+(1<<j)) ) == b ) )
	    {
		if(j>i)
		    sign = CalcSign(i,j,a);
		else
		    sign = CalcSign(j,i,a);

		result = -1 * sign;
		break;
	    }

	    // jump left
	    j = L * (i/L) + (i - 1 + L) % L;
	    if( (~a & 1<<j) && ((a ^ ((1<<i)+(1<<j)) ) == b ) )
	    {
		if(j>i)
		    sign = CalcSign(i,j,a);
		else
		    sign = CalcSign(j,i,a);

		result = -1 * sign;
		break;
	    }
        }

    return result;
}

/**
  * Builds the full 2D Hubbard Hamiltonian matrix
  */
void HubHam2D::BuildFullHam()
{
    if( !baseUp.size() || !baseDown.size() )
    {
	std::cerr << "Build base before building Hamiltonian" << std::endl;
	return;
    }

    ham = new double[dim*dim];

    int NumDown = CalcDim(L,Nd);

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
			ham[j+dim*i] += J * hopping(baseUp[a], baseUp[c]);

		    if(a == c)
			ham[j+dim*i] += J * hopping(baseDown[b], baseDown[d]);

		    ham[i+dim*j] = ham[j+dim*i];
		}

	    // count number of double occupied states
	    ham[i+dim*i] = U * CountBits(baseUp[a] & baseDown[b]);
	}
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
