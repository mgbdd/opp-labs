#include "sec_met.hpp"
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




