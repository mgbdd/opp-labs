#include "sec_met.hpp"

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
