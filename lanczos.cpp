/*
 * =====================================================================================
 *
 *       Filename:  lanczos.cpp
 *
 *    Description:  Lanczos algorithm
 *
 *        Version:  1.0
 *        Created:  11-05-12 18:33:27
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ward Poelmans (), wpoely86@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;

int main(void)
{
    int size = 1000;
    int m = 5*size;

    mat matrix(size,size);
    matrix.zeros();

    srand(time(0));

    for(int i=0;i<size;i++)
	for(int j=i;j<size;j++)
	    matrix(i,j) = matrix(j,i) = rand()*1.0/RAND_MAX;

    vector<double> a(m,0);
    vector<double> b(m,0);

    int i;

    vec qa(size);
    vec qb(size);

    qa.zeros();

    qb = randu(size);
    double norm = sqrt(dot(qb,qb));
    qb *= 1.0/norm;

    vec *f1 = &qa;
    vec *f2 = &qb;
    vec *tmp;

    for(i=1;i<m;i++)
    {
	(*f1) *= -b[i-1];
	(*f1) += matrix*(*f2);

	a[i-1] = dot(*f1,*f2);

	(*f1) -= a[i-1]*(*f2);

	b[i] = sqrt(dot(*f1,*f1));

	if( fabs(b[i]) < 1e-10 )
	    break;

	(*f1) /= b[i];
	tmp = f2;
	f2 = f1;
	f1 = tmp;
    }

    mat T(i,i);

    T.zeros();
    T(0,0) = a[0];

    for(int j=1;j<i;j++)
    {
	T(j,j) = a[j];
	T(j,j-1) = T(j-1,j) = b[j];
    }

    vec eigs1 = eig_sym(T);
    vec eigs2 = eig_sym(matrix);

    cout << eigs1[0] << endl;
    cout << endl;
    cout << eigs2[0] << endl;

    return 0;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
