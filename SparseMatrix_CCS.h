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
#ifndef SPARSEMATRIX_CCS_H
#define SPARSEMATRIX_CCS_H

#include <iostream>
#include <vector>

class matrix;

using std::ostream;

/**
 * @author Ward Poelmans
 * @date 09-08-2013\n\n
 * This is a class written for sparse n x m matrices to use on the gpu.
 * It uses the CCS format to store a matrix.
 */

class SparseMatrix_CCS
{
   /**
    * Output stream operator overloaded, the usage is simple, if you want to print to a file, make an
    * ifstream object and type:\n\n
    * object << matrix << endl;\n\n
    * For output onto the screen type: \n\n
    * cout << matrix << endl;\n\n
    * @param output The stream to which you are writing (e.g. cout)
    * @param matrix_p de SparseMatrix_CCS you want to print
    */
   friend ostream &operator<<(ostream &output,SparseMatrix_CCS &matrix_p);

   public:

      //constructor
      SparseMatrix_CCS(unsigned int n, unsigned int m);

      //copy constructor
      SparseMatrix_CCS(const SparseMatrix_CCS &);

      //destructor
      virtual ~SparseMatrix_CCS();

      //overload equality operator
      SparseMatrix_CCS &operator=(const SparseMatrix_CCS &);

      //easy to change the numbers
   //   double &operator()(unsigned int i,unsigned int j);

      //easy to access the numbers
      double operator()(unsigned int i,unsigned int j) const;

      unsigned int gn() const;

      unsigned int gm() const;

      void ConvertFromMatrix(const matrix &dense);

      void ConvertToMatrix(matrix &dense) const;

      void PrintRaw() const;

      void PushToCol(unsigned int j, double value);

      void NewCol();

      void mvprod(const matrix &, matrix &) const;

      unsigned int NumOfElInCol(unsigned int idx) const;

      double GetElementInCol(unsigned int col_index, unsigned int element_index) const;

      unsigned int GetElementRowIndexInCol(unsigned int col_index, unsigned int element_index) const;

      void prod(const matrix &, const SparseMatrix_CCS &);

   private:

      //! Array that holds the non zero values
      std::vector<double> data;
      //! Array that holds the column indexes
      std::vector<unsigned int> col;
      //! Array that holds the row index of data
      std::vector<unsigned int> row;

      //!dimension of the matrix (number of rows)
      unsigned int n;
      //!dimension of the matrix (number of columns)
      unsigned int m;
};

#endif /* SPARSEMATRIX_CCS_H */

/* vim: set ts=3 sw=3 expandtab :*/
