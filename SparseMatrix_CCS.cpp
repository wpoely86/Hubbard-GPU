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
#include <cmath>
#include "helpers.h"
#include "SparseMatrix_CCS.h"

#define MY_ZEROLIMIT 1e-10

/**
 * Construct SparseMatrix_CCS object for n x m matrix
 * @param n the number of rows
 * @param m the number of columns
 */
SparseMatrix_CCS::SparseMatrix_CCS(unsigned int n, unsigned int m)
{
   this->n = n;
   this->m = m;
}

/**
 * Copy constructor
 * @param origin the object to copy
 */
SparseMatrix_CCS::SparseMatrix_CCS(const SparseMatrix_CCS &origin)
{
   this->n = origin.n;
   this->m = origin.m;
   this->data = origin.data;
   this->col = origin.col;
   this->row = origin.row;
}

SparseMatrix_CCS::~SparseMatrix_CCS()
{
}

/**
 * Read only access operator
 * @param i the row number
 * @param j the column number
 * @return the matrix element
 */
double SparseMatrix_CCS::operator()(unsigned int i,unsigned int j) const
{
   for(unsigned int k=col[j];k<col[j+1];k++)
      if( row[k] == i )
         return data[k];

   return 0;
}

/**
 * @return the number of rows
 */
unsigned int SparseMatrix_CCS::gn() const
{
   return n;
}

/**
 * @return the number of columns
 */
unsigned int SparseMatrix_CCS::gm() const
{
   return m;
}

/**
 * Copy operator
 * @param origin the object to copy
 * @return this object
 */
SparseMatrix_CCS &SparseMatrix_CCS::operator=(const SparseMatrix_CCS &origin)
{
   this->n = origin.n;
   this->m = origin.m;
   this->data = origin.data;
   this->col = origin.col;
   this->row = origin.row;

   return *this;
}

/**
 * Convert a dense matrix to CCS format
 * @param dense the matrix to convert
 */
void SparseMatrix_CCS::ConvertFromMatrix(const matrix &dense)
{
   this->n = dense.getn();
   this->m = dense.getm();
   col.resize(m+1);

   data.clear();
   row.clear();

   col[0] = 0;

   for(unsigned int j=0;j<m;j++)
   {
      for(unsigned int i=0;i<n;i++)
         if( fabs(dense(i,j)) > MY_ZEROLIMIT )
         {
            data.push_back(dense(i,j));
            row.push_back(i);
         }

      col[j+1] = row.size();
   }

   col.back() = row.size();
}

/**
 * Convert this CCS matrix to a dense matrix (only works for square matrices)
 * @param dense the matrix to fill
 */
void SparseMatrix_CCS::ConvertToMatrix(matrix &dense) const
{
   dense = 0;
   unsigned int dn = dense.getn();
   unsigned int dm = dense.getm();

   assert(dn == n && dm == m);

   for(unsigned int i=0;i<m;i++)
      for(unsigned int k=col[i];k<col[i+1];k++)
         dense(row[k],i) = data[k];
}

/**
 * Print the raw CCS data to stdout
 */
void SparseMatrix_CCS::PrintRaw() const
{
   std::cout << n << " x " << m << " matrix" << std::endl;
   std::cout << "Data(" << data.size() << "):" << std::endl;
   for(unsigned int i=0;i<data.size();i++)
      std::cout << data[i] << " ";
   std::cout << std::endl;

   std::cout << "Row indices:" << std::endl;
   for(unsigned int i=0;i<row.size();i++)
      std::cout << row[i] << " ";
   std::cout << std::endl;

   std::cout << "Col indices:" << std::endl;
   for(unsigned int i=0;i<col.size();i++)
      std::cout << col[i] << " ";
   std::cout << std::endl;
}

/**
 * Print sparse matrix to output
 * @param output the ostream to print to
 * @param matrix_p the matrix to print
 * @return the filled ostream (with the matrix)
 */
ostream &operator<<(ostream &output,SparseMatrix_CCS &matrix_p)
{
   for(unsigned int i=0;i<matrix_p.m;i++)
      for(unsigned int k=matrix_p.row[i];k<matrix_p.row[i+1];k++)
         output << matrix_p.row[k] << "\t" << i << "\t" << matrix_p.data[k] << std::endl;

   return output;
}

/**
 * Adds a new row element to the current column.
 * To use this, first call NewCol() to start a column and then
 * use PushToCol() to add elements to that col. Always end
 * with calling NewCol() again.
 * @param j row
 * @param value the matrix element value
 */
void SparseMatrix_CCS::PushToCol(unsigned int j, double value)
{
   // thirth condition is for a new row begins
   if(row.empty() || row.back() < j || col.back() == row.size())
   {
      data.push_back(value);
      row.push_back(j);
   }
   else
   {
      unsigned int begin = col.back();
      for(unsigned int i=begin;i<row.size();i++)
      {
         if( row[i] > j )
         {
            row.insert(row.begin() + i,j);
            data.insert(data.begin() + i,value);
            break;
         } else if (row[i] == j)
         {
            data[i] += value;

            if(fabs(data[i]) < MY_ZEROLIMIT)
            {
               data.erase(data.begin() + i);
               row.erase(row.begin() + i);
            }
            break;
         }
      }
   }
}

/**
 * Adds the next col to the sparsematrix
 */
void SparseMatrix_CCS::NewCol()
{
   if(col.size() == (m+1))
      return;

   col.push_back(data.size());
}

/**
 * Number of elements in a column
 * @idx the number of the column
 * @return the number of elements in column idx
 */
unsigned int SparseMatrix_CCS::NumOfElInCol(unsigned int idx) const
{
   assert(idx<m);
   return (col[idx+1]-col[idx]);
}

/**
 * Gives an element in from a column
 * @param col_index the column from whiche to select an element
 * @param element_index the index of the element in column col_index
 * @return the value of the element element_index in column col_index
 */
double SparseMatrix_CCS::GetElementInCol(unsigned int col_index, unsigned int element_index) const
{
   assert(col_index < m);
   assert((col[col_index]+element_index) < data.size());
   return data[col[col_index]+element_index];
}

/**
 * Get the row index of a element in a certain column
 * @param col_index the column index to use
 * @param element_index the index of the element
 * @return the row number of the element
 */
unsigned int SparseMatrix_CCS::GetElementRowIndexInCol(unsigned int col_index, unsigned int element_index) const
{
   assert((col[col_index]+element_index) < row.size());
   return row[col[col_index]+element_index];
}


/**
 * Matrix-Matrix product of A and B: AB
 * Stores the result in this object, destroying any
 * matrix in it. The size is adjusted to the result of AB.
 * @param A dense matrix
 * @param B sparse matrix
 */
void SparseMatrix_CCS::prod(const matrix &A, const SparseMatrix_CCS &B)
{
   assert(A.getm() == B.gn());
   this->n = A.getn();
   this->m = B.gm();
   col.resize(m+1);

   data.clear();
   row.clear();

   NewCol();

   for(int j=0;j<B.gm();j++)
   {
      for(int k=0;k<B.NumOfElInCol(j);k++)
         for(int i=0;i<A.getn();i++)
         {
            double val = A(i, B.GetElementRowIndexInCol(j,k)) * B.GetElementInCol(j,k);

            if(fabs(val) > MY_ZEROLIMIT)
               PushToCol(i, val);
         }

      NewCol();
   }

   NewCol();
}

/**
 * Do the matrix vector product y = A * x
 * @param xmat a m component vector
 * @param ymat a n component vector
 */
//void SparseMatrix_CCS::mvprod(const matrix &xmat, matrix &ymat) const
//{
//   double *x = const_cast<Matrix &>(xmat).gMatrix()[0];
//   double *y = ymat.gMatrix()[0];
//
//
//   // first run to initialize all values
//   for(unsigned int i=0;i<n;i++)
//      y[i] = 0;
//
//   for(unsigned int i=0;i<m;i++)
//   {
//      for(unsigned int k=col[i];k<col[i+1];k++)
//         y[row[k]] += data[k] * x[i];
//   }
//}

/* vim: set ts=3 sw=3 expandtab :*/
