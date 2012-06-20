#ifndef HAM2D_H
#define HAM2D_H

#include "ham.h"

/**
 * This is the main class where all the magic happens:
 * this class is for 2D Hubbard. It makes a grid of length L
 * and depth D. For example a grid of L=4 and D=2 is:
 * x  x  x  x
 * x  x  x  x
 * The periodic boundary condition is used.
 *
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class HubHam2D : public Hamiltonian
{
    public:
	HubHam2D(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~HubHam2D();

	void BuildFullHam();

        void mvprod(double *x, double *y, double alpha) const;

    protected:
	int hopping(myint a, myint b) const;
        int CalcSign(int i,int j,myint a) const;

	//! The length of the 2D grid
        int L;
	//! The depth of the 2D grid
        int D;
};

#endif /* HAM2D_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
