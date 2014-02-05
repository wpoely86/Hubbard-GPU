/* Copyright (C) 2012-2014  Ward Poelmans

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
#include <algorithm>
#include <memory>
#include <boost/timer.hpp>
#include <getopt.h>
#include "ham.h"
#include "hamsparse.h"
#include "ham-mom.h"

using namespace std;

int main(int argc, char **argv)
{
    int L = 4; // number of sites
    int Nu = 2; // number of up electrons
    int Nd = 3; // number of down electrons
    double J = 1.0; // hopping term
    double U = 1.0; // on-site interaction strength

    bool exact = false;
    bool lanczos = false;
    bool momentum = false;

    struct option long_options[] =
    {
        {"up",  required_argument, 0, 'u'},
        {"down",  required_argument, 0, 'd'},
        {"sites",  required_argument, 0, 's'},
        {"interaction",  required_argument, 0, 'U'},
        {"hopping",  required_argument, 0, 'J'},
        {"exact",  no_argument, 0, 'e'},
        {"lanczos",  no_argument, 0, 'l'},
        {"momentum",  no_argument, 0, 'p'},
        {"help",  no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int i,j;
    while( (j = getopt_long (argc, argv, "hu:d:s:U:J:elp", long_options, &i)) != -1)
        switch(j)
        {
            case 'h':
            case '?':
                cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                    "\n"
                    "    -s  --sites=L               The number of sites\n"
                    "    -u  --up=Nu                  The number of up electrons\n"
                    "    -d  --down=Nd                The number of down electrons\n"
                    "    -U  --interaction=U          The onsite interaction strength\n"
                    "    -J  --hopping=J              The hopping strength\n"
                    "    -e  --exact                  Solve with exact diagonalisation\n"
                    "    -l  --lanczos                Solve with Lanczos algorithm\n"
                    "    -p  --momentum               Solve in the momentum basis\n"
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
                L = atoi(optarg);
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
            case 'p':
                momentum = true;
                break;
            case 'e':
                exact = true;
                break;
        }

    cout << "L = " << L << "; Nu = " << Nu << "; Nd = " << Nd << "; J = " << J << "; U = " << U << ";" << endl;

    cout.precision(10);

    boost::timer tijd;

    if(exact)
    {
        tijd.restart();

        std::unique_ptr<BareHamiltonian> ham;

        if(momentum)
            ham.reset(new MomHamiltonian(L,Nu,Nd,J,U));
        else
            ham.reset(new Hamiltonian(L,Nu,Nd,J,U));

        cout << "Memory needed: " << ham->MemoryNeededFull()*1.0/1024*1.0/1024 << " MB" << endl;

        ham->BuildBase();

        ham->BuildFullHam();

        cout << "Dim: " << ham->getDim() << endl;

        double Egroundstate = 0;

        if(momentum)
        {
            auto E = (static_cast<MomHamiltonian *>(ham.get()))->ExactMomDiagonalizeFull(false);

            cout << "Energy levels: " << endl;
            for(auto p : E)
                cout << p.second << "\t" << p.first << endl;

            Egroundstate = E[0].second;

//            (static_cast<MomHamiltonian *>(ham.get()))->GenerateData(0,10,1,"data-test.h5");
        }
        else
        {
            auto E = ham->ExactDiagonalizeFull(true);

            cout << "Energy levels: " << endl;
            for(auto p : E)
                cout << p << endl;

            Egroundstate = E[0];
        }

        cout << endl;
        cout << "lowest E = " << Egroundstate << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    if(lanczos)
    {
        tijd.restart();

        SparseHamiltonian sham(L,Nu,Nd,J,U);

        cout << "Memory needed: " << sham.MemoryNeededArpack()*1.0/1024*1.0/1024 << " MB" << endl;

        sham.BuildBase();
        sham.BuildSparseHam();

        cout << "Dim: " << sham.getDim() << endl;

        double E = sham.arpackDiagonalize();
        cout << "E = " << E << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    return 0;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
