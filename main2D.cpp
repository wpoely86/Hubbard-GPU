/**
 * This program is for 2D Hubbard.
 * @author Ward Poelmans <wpoely86@gmail.com>
 * @date 20-06-2012
 */

#include <iostream>
#include <boost/timer.hpp>
#include <getopt.h>
#include "ham2D.h"
#include "hamsparse2D_CSR.h"
#include "hamsparse2D.h"

using namespace std;

int main(int argc, char **argv)
{
    int L = 3; // The Length of the 2D grid
    int D = 3; // The Depth of the 2D grid
    int Nu = 4; // number of up electrons
    int Nd = 5; // number of down electrons
    double J = 1.0; // hopping term
    double U = 1.0; // on-site interaction strength

    bool exact = false;
    bool lanczos = false;
    bool PRIMME = false;

    struct option long_options[] =
    {
        {"up",  required_argument, 0, 'u'},
        {"down",  required_argument, 0, 'd'},
        {"length",  required_argument, 0, 'L'},
        {"depth",  required_argument, 0, 'D'},
        {"interaction",  required_argument, 0, 'U'},
        {"hopping",  required_argument, 0, 'J'},
        {"exact",  no_argument, 0, 'e'},
        {"lanczos",  no_argument, 0, 'l'},
        {"primme",  no_argument, 0, 'p'},
        {"help",  no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int i,j;
    while( (j = getopt_long (argc, argv, "hu:d:L:D:U:J:elp", long_options, &i)) != -1)
        switch(j)
        {
            case 'h':
            case '?':
                cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                    "\n"
                    "    -L  --length=l               The length of the 2D grid\n"
                    "    -D  --depth=d                The depth of the 2D grid\n"
                    "    -u  --up=Nu                  The number of up electrons\n"
                    "    -d  --down=Nd                The number of down electrons\n"
                    "    -U  --interaction=U          The onsite interaction strength\n"
                    "    -J  --hopping=J              The hopping strength\n"
                    "    -e  --exact                  Solve with exact diagonalisation\n"
                    "    -l  --lanczos                Solve with Lanczos algorithm\n"
                    "    -p  --primme                 Solve with PRIMME library\n"
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
            case 'L':
                L = atoi(optarg);
                break;
            case 'D':
                D = atoi(optarg);
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
            case 'p':
                PRIMME = true;
                break;
        }

    cout << "L = " << L << "; D = " << D << "; Nu = " << Nu << "; Nd = " << Nd << "; J = " << J << "; U = " << U << ";" << endl;

    cout.precision(10);

    boost::timer tijd;

    if(exact)
    {
        tijd.restart();

        HubHam2D ham(L,D,Nu,Nd,J,U);

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

        SparseHamiltonian2D sham(L,D,Nu,Nd,J,U);

        sham.BuildBase();
        sham.BuildSparseHam();

        cout << "Dim: " << sham.getDim() << endl;

        double E = sham.LanczosDiagonalize();
        cout << "E = " << E << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    if(PRIMME)
    {
        tijd.restart();

        SparseHamiltonian2D sham(L,D,Nu,Nd,J,U);

        sham.BuildBase();
        sham.BuildSparseHam();

        cout << "Dim: " << sham.getDim() << endl;

        double E = sham.PRIMMEDiagonlize();
        cout << "E = " << E << endl;

        cout << "Time: " << tijd.elapsed() << " s" << endl;
    }

    return 0;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
