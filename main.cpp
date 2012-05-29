/**
 * @mainpage
 * This is an implementation of http://arxiv.org/abs/1204.3425
 * "Exact diagonalization of the Hubbard model on graphics processing units"
 * @author Ward Poelmans <wpoely86@gmail.com>
 * @date 29-05-2012
 */

#include <iostream>
#include "ham.h"
#include "hamsparse.h"

using namespace std;

int main(int argc, char **argv)
{
    int Ns = 4; // number of sites
    int Nu = 2; // number of up electrons
    int Nd = 3; // number of down electrons
    double J = 1.0; // hopping term
    double U = 1.0; // on-site interaction strength

    cout.precision(10);

    Hamiltonian ham(Ns,Nu,Nd,J,U);

    ham.BuildBase();
    ham.BuildFullHam();

    cout << "Dim: " << ham.getDim() << endl;
    ham.Print();

    cout << "E = " << ham.LanczosDiagonalizeFull(10) << endl;
    cout << "E = " << ham.ExactDiagonalizeFull() << endl;

    SparseHamiltonian sham(Ns,Nu,Nd,J,U);

    sham.BuildBase();
    sham.BuildSparseHam();

    sham.PrintSparse();
    cout << endl << endl;
    sham.PrintRawELL();

    return 0;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
