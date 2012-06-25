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

/**
 * @mainpage
 * This is an implementation of http://arxiv.org/abs/1204.3425
 * "Exact diagonalization of the Hubbard model on graphics processing units" by Siro and Harju\n\n\n
 * There are 2 programs: main and main2D. The former is for 1D Hubbard, the latter is for
 * 2D Hubbard. Everything is split up in several classes to make reusing code easy:\n
 * The Hamiltonian and HubHam2D classes build the full (dense) Hamiltonian Matrix. SparseHamiltonian
 * stores the Hubbard Hamiltonian in parts: an spin up and a spin down part. The matrix themselves are
 * storred in the ELL format.\n
 * The SparseHamiltonian2D does the same but for 2D Hubbard. However, here we make a detour: we first store
 * the matrices in the CRS format (the SparseHamiltonian2D_CSR class) and then convert it in the ELL format.
 * The reason is that for ELL, we need to know the maximum number of non-zero elements (nnz) of a row.\n\n
 *
 * There are several branches in the git repo: the master contains only the CPU version. The branch 'GPU'
 * constains the GPU version and the branch 'PRIMME' used the PRIMME library to find the eigenvalues and
 * eigenvectors. You can find PRIMME at http://www.cs.wm.edu/~andreas/software/
 *
 * @author Ward Poelmans <wpoely86@gmail.com>
 * @date 29-05-2012
 * @section LICENSE
 *
 * Hubbard-GPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Hubbard-GPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Hubbard-GPU.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <boost/timer.hpp>
#include <getopt.h>
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

    bool exact = false;
    bool lanczos = false;

    struct option long_options[] =
    {
        {"up",  required_argument, 0, 'u'},
        {"down",  required_argument, 0, 'd'},
        {"sites",  required_argument, 0, 's'},
        {"interaction",  required_argument, 0, 'U'},
        {"hopping",  required_argument, 0, 'J'},
        {"exact",  no_argument, 0, 'e'},
        {"lanczos",  no_argument, 0, 'l'},
        {"help",  no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int i,j;
    while( (j = getopt_long (argc, argv, "hu:d:s:U:J:el", long_options, &i)) != -1)
        switch(j)
        {
            case 'h':
            case '?':
                cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                    "\n"
                    "    -s  --sites=Ns               The number of sites\n"
                    "    -u  --up=Nu                  The number of up electrons\n"
                    "    -d  --down=Nd                The number of down electrons\n"
                    "    -U  --interaction=U          The onsite interaction strength\n"
                    "    -J  --hopping=J              The hopping strength\n"
                    "    -e  --exact                  Solve with exact diagonalisation\n"
                    "    -l  --lanczos                Solve with Lanczos algorithm\n"
                    "    -h, --help                   Display this help\n"
                    "\n";
                return 0;
                break;
            case 'u':
                Nu = atoi(optarg);
                break;
            case 'd':
                Nd = atoi(optarg);
                break;
            case 's':
                Ns = atoi(optarg);
                break;
            case 'U':
                U = atof(optarg);
                break;
            case 'J':
                J = atof(optarg);
                break;
            case 'l':
                lanczos = true;
                break;
            case 'e':
                exact = true;
                break;
        }

    cout << "Ns = " << Ns << "; Nu = " << Nu << "; Nd = " << Nd << "; J = " << J << "; U = " << U << ";" << endl;

    cout.precision(10);

    boost::timer tijd;

    if(exact)
    {
        tijd.restart();

        Hamiltonian ham(Ns,Nu,Nd,J,U);

        ham.BuildBase();
        ham.BuildFullHam();

        cout << "Dim: " << ham.getDim() << endl;

        double E = ham.ExactDiagonalizeFull();
        cout << "E = " << E << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    if(lanczos)
    {
        tijd.restart();

        SparseHamiltonian sham(Ns,Nu,Nd,J,U);

        sham.BuildBase();
        sham.BuildSparseHam();

        cout << "Dim: " << sham.getDim() << endl;

        double E = sham.LanczosDiagonalize();
        cout << "E = " << E << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    return 0;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
