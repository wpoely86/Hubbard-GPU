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
#include "nonp-ham.h"

/**
 * Constructor of the NonPeriodicHamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
NonPeriodicHamiltonian::NonPeriodicHamiltonian(int L, int Nu, int Nd, double J, double U):
    Hamiltonian(L,Nu,Nd,J,U)
{

}

NonPeriodicHamiltonian::~NonPeriodicHamiltonian()
{
}

/**
 * Calculates the hopping for the non periodic boundary conditions
 * @param a the bra to use
 * @param b the ket to use
 * @param jumpsign ignored, has no meaning here, set to zero by default
 * @returns matrix element of the hopping term between the ket and the bra. You still
 * have to multiply this with the hopping strength J
 */
int NonPeriodicHamiltonian::hopping(myint a, myint b, int jumpsign) const
{
    int result = 0;
    // find all the holes in the ket
    myint cur = ~b;

    while(cur)
    {
        // isolate the rightmost 1 bit
        myint hop = cur & (~cur + 1);
        cur ^= hop;

        myint tryjump;
        // you cannot jump if rightmost site
        if(! (hop & 0x1))
        {
            tryjump = hop >> 1;

            if(tryjump & b)
            {
                myint bra_test = b ^ (tryjump + hop);

                if(bra_test == a)
                    result -= 1;
            }
        }

        // you cannot jump if leftmost site
        if(! (hop & Hb))
        {
            tryjump = hop << 1;

            if(tryjump & b)
            {
                myint bra_test = b ^ (tryjump + hop);

                if(bra_test == a)
                    result -= 1;
            }
        }
    }

    return result;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
