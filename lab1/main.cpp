#include <iostream>
#include <math.h>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <istream>

vector<float> loadData(const string filePath, size_t size)
{
    ifstream file(filePath.data(), ios::binary);
    if (!file.is_open())
    {
        printf("Failed to open file \"%s\"!\n", filePath.data());
        return vector<float>{};
    }

    vector<float> buffer(size);
    file.seekg(0, ios::end);
    const auto fileSize = file.tellg() / sizeof(float);
    file.seekg(0, ios::beg);
    buffer.resize(fileSize);

    file.read(reinterpret_cast<char*>(buffer.data()), fileSize * sizeof(float));

    file.close();
    return buffer;
}


void writeData(vector<float> x)
{
    ofstream file("myVec.bin", std::ios::binary);
    if (file.is_open()) 
    {
        file.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
        
        file.close();
        std::cout << "Вектор был успешно записан в файл." << std::endl;
    } else {
        std::cout << "Не удалось открыть файл для записи." << std::endl;
    }
}

vector<float> MinimumResidualMethod(vector<float> &xPrev, int size, vector<float> &vecB, float *recv, int rank, int *sendCounts, int *displs)
{
    float e = 0.00001;
    float criterion, critSum = 1;
    float normY, normB = 0;
    float tau = 0;
    float numerator, denominator;

    int errorCode;
    char errorStr[MPI_MAX_ERROR_STRING];
    int errorLen;

    int chunkSize = sendCounts[rank] / size; // количество переданных в поток строк
    int offset = displs[rank] / size; // смещение в строках
    cout << "rank: " << rank << " offset: " << offset << " chunkSize: " << chunkSize  <<  endl; 


    for (int i = offset; i < offset + chunkSize; i++)
    {
        normB += vecB[i] * vecB[i];
    }
    vector<float> y(chunkSize, 0.0);
    vector<float> mult(chunkSize, 0.0);
    float xLocal[chunkSize];
    for(int i = 0; i < chunkSize; i++)
    {
	xLocal[i] = xPrev[i + offset]; 
    }

    cout << "begin...." << endl;

    double startTime = MPI_Wtime();

    while (critSum > e * e)
    {
        mult.assign(chunkSize, 0);

        normY = 0;
        numerator = 0, denominator = 0;

        for (int i = 0; i < chunkSize; i++) // A*x_n
        {
            for (int j = 0; j < size; j++)
            {
                mult[i] += recv[i * size + j] * xLocal[j];
            }
        }
		
	//cout << "y = mult - b" << endl;
        for (int i = 0; i <  chunkSize; i++) //  y_n = A*x_n - b
        {
            y[i] = mult[i] - vecB[i + offset];
	    //cout <<  mult[i] << endl;
	    //cout << y [i] << endl;
	    //cout << mult[i] << " - " << vecB[i+offset] << " = " << y[i] << endl;
        }

        // criterion

        for (int i = 0; i < chunkSize; i++)
        {
            normY += y[i] * y[i];
        }

        criterion = normY / normB;
	//cout << "поток " << rank << " критерий: " << criterion << " норма Y: " << normY  << endl;

        errorCode = MPI_Allreduce(&criterion, &critSum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (errorCode != MPI_SUCCESS)
        {
            MPI_Error_string(errorCode, errorStr, &errorLen);
            cout << "Ошибка MPI_Allreduce: " << errorStr << endl;
            break;
        }
	//cout << "rank: " << rank << " critSum: " << critSum << endl;
        ///

        mult.assign(chunkSize, 0);

        for (int i = 0; i <  chunkSize; i++) // A*y_n
        {
            for (int j = 0; j < size; j++)
            {
                mult[i] += recv[i * size + j] * y[j];
            }
        }

        for (int i = 0; i < chunkSize; i++)
        {
            numerator += y[i] * mult[i]; // (y_n, A*y_n)
        }

        for (int i = 0; i < chunkSize; i++)
        {
            denominator += mult[i] * mult[i]; //(a*y_n, A*y_n)
        }

        tau = numerator / denominator;
	//cout << "tau: " << tau << endl;
	//cout << "num and den: " << numerator << " " << denominator << endl;

	//cout << "---" << endl;
        for (int i = 0; i < chunkSize; i++) // x_n+1 = x_n - t*y_n
        {
            xLocal[i] = xLocal[i] - tau * y[i];
	    //cout << xLocal[i] << endl;
        }
    }

    double endTime = MPI_Wtime();
    cout << "Поток: " << rank << ", время: " << endTime - startTime << endl;

    vector<float> xNext(size, 0);

    int localCounts[size];
    for(int i = 0; i < size; i++)
    {
	    localCounts[i] = sendCounts[i] / size;
    }
    int sum = 0;
    for(int i = 0; i < size; i++)
    {
	sum += localCounts[i];
    }
    cout << "sum: " << sum << endl;

    int localDispls[size];
    localDispls[0] = 0;
    for(int i = 1; i < size; i++)
    {
	localDispls[i] = localDispls[i - 1] + localCounts[i - 1];
    }

    errorCode = MPI_Allgatherv(xLocal, chunkSize, MPI_FLOAT, xNext.data(), localCounts, localDispls, MPI_FLOAT, MPI_COMM_WORLD);

    // cout << "iterations: " << count << endl;

    return xNext;
}


int main(int argc, char **argv)
{

    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение ранга текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего количества процессов

    int lineNum;
    int *sendCounts = new int[size];

    lineNum = VECTOR_SIZE / size;
    int rest = VECTOR_SIZE - lineNum * size;
    for (int i = 0; i < size; i++) // заполнение кол-ва элементов для каждого потока
    {
        sendCounts[i] = lineNum * VECTOR_SIZE;
        if (rest != 0)
        {
            sendCounts[i] += VECTOR_SIZE;
            rest--;
        }
    }	

    int *displs = new int[size];
    displs[0];
    for (int i = 0; i < size; i++)
    {
        displs[i] = displs[i - 1] + sendCounts[i - 1];
    }

    float *recvbuf = new float[sendCounts[rank]];
    int recvcount = sendCounts[rank];

    int errorCode;
    char errorStr[MPI_MAX_ERROR_STRING];
    int errorLen;

    vector<float> matA;
    //float *matA = new float [100];
    if (rank == 0)
    {
        matA = loadData("matA.bin", MAT_A_SIZE);
        if (matA.size() != MAT_A_SIZE)
        {
            cout << "failed to fill matrix A" << endl;
            return 0;
        }
        /*for(int i = 0; i < 10; i++)
        {
            for(int j = 0; j < 10; j++)
            {
                if(i == j) matA[i * 10 + j] = 2.0;
                else  matA[i * 10 + j] = 1.0;
            }
        }*/
    }

    errorCode = MPI_Scatterv(matA.data(), sendCounts, displs, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //errorCode = MPI_Scatterv(matA, sendCounts, displs, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);


    if (errorCode != MPI_SUCCESS)
    {
        MPI_Error_string(errorCode, errorStr, &errorLen);
        cout << "Ошибка MPI_Scatterv: " << errorStr << endl;
        return 0;
    }

    vector<float> vecB = loadData("vecB.bin", VECTOR_SIZE);
    if (vecB.size() != VECTOR_SIZE)
    {
        cout << "failed to fill vector B" << endl;
        return 0;
    }

    //float *vecB = new float [10];
    //fill(vecB, vecB + 10,11);
    vector<float> x(VECTOR_SIZE, 0); // x0


    x = MinimumResidualMethod(x, VECTOR_SIZE, vecB, recvbuf, rank, sendCounts, displs);

    delete[] sendCounts;
    delete[] displs;
    delete[] recvbuf;
    //delete[] matA;
    //delete[] vecB;
    MPI_Finalize();

    for(int i = 0; i < 10; i++)
    {
	cout << x[i] << endl;
    }
    writeData(x);
    return 0;
}
