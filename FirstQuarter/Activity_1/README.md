# Matrix Multiplication
This repository contains C code for multiplying two matrices. Matrix multiplication is a fundamental operation in linear algebra and is often used in various mathematical and scientific applications.

## What is a Matrix?
In mathematics, a matrix is a two-dimensional array of numbers, symbols, or expressions arranged in rows and columns. The number of rows and columns in a matrix defines its dimensions. For example, a matrix with 3 rows and 4 columns is referred to as a "3x4 matrix."

## Matrix Multiplication Example
Matrix multiplication involves taking the dot product of rows from the first matrix and columns from the second matrix. The result is a new matrix where each element is computed as the sum of the products of the corresponding elements from the row and column.

![Example](https://imgs.search.brave.com/32Qpo169ORno-kL4LjdgYlbgMfWDLvvKD0qOagfuIVo/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZWVrc2Zvcmdl/ZWtzLm9yZy93cC1j/b250ZW50L3VwbG9h/ZHMvMjAyMzA1MzEx/MDU4NDgvTWF0cml4/LU11bHRpcGxpY2F0/aW9uLlBORw)

# Code
```
#include <stdio.h>

//multiply 2 matrix 10,10
int main() {

    int N[10];
//Result
    int C[10][10];
//Matrix 1
    int A[10][10] = {{1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }, 
                     {3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 }, 
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }, 
                     {3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 } , 
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }, 
                     {3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 }, 
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }, 
                     {3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 }, 
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }, 
                     {3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 , 3 , 4 }};
//Matrix 2
    int B[10][10] = {{2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 },
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 },
                     {2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 },
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 },
                     {2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 },
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 },
                     {2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 },
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 },
                     {2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 , 2 , 0 },
                     {1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 , 1 , 2 }};

//Function to calculate and print the resulting matrix
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){
            C[i][j] = 0;
            for(int k = 0; k < 10; k++){
                N[k] = A[i][k] * B[k][j];
                C[i][j] += N[k];
            }
            printf("%d ,", C[i][j]);
        }
        printf("\n");
    }


    return 0;
}
```

## Explanation

in my code i call 4 variables. 
- N to add in the operation of the matrix
- B and A as the 2 matixes i will be multiplying
- C as the result of the two matrixes

I then use a function with 3 for's to do the multiplication
- for [i] is used for what number column its using
- for [j] is used for what number row its using
- for [k] is used to declare and multiply what the result is going to be

Then we use a printf to write out the result as its calculating it