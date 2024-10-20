#include <iostream>
#include <math.h>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include <algorithm>

enum Sizes
{
    MAT_A_SIZE = 6250000,
    VECTOR_SIZE = 2500
};

using namespace std;

vector<float> MinimumResidualMethod(vector<float> &xPrev, int size, vector<float> &vecB, float *recvbuf, int rank, int *sendCounts, int *displs);

vector<float> loadData(const string filePath, size_t size);

void writeData(vector<float> x);
