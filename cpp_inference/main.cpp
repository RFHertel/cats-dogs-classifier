#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <NvInfer.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

const float MEAN[] = {0.485f, 0.456f, 0.406f};
const float STD[] = {0.229f, 0.224f, 0.225f};

bool preprocessImage(const char* path, float* output, int size = 224) {
    int w, h, c;
    unsigned char* img = stbi_load(path, &w, &h, &c, 3);
    if (!img) {
        std::cerr << "Failed to load: " << path << std::endl;
        return false;
    }
    
    for (int ch = 0; ch < 3; ch++) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int srcX = x * w / size;
                int srcY = y * h / size;
                float pixel = img[(srcY * w + srcX) * 3 + ch] / 255.0f;
                pixel = (pixel - MEAN[ch]) / STD[ch];
                output[ch * size * size + y * size + x] = pixel;
            }
        }
    }
    
    stbi_image_free(img);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <engine> <image>" << std::endl;
        return 1;
    }
    
    const char* engineFile = argv[1];
    const char* imageFile = argv[2];
    
    std::cout << "================================" << std::endl;
    std::cout << "TensorRT C++ Inference" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Engine: " << engineFile << std::endl;
    std::cout << "Image: " << imageFile << std::endl;
    
    Logger logger;
    
    // Load engine
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open engine file" << std::endl;
        return 1;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    // Create runtime and engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    
    // Allocate buffers
    const int inputSize = 1 * 3 * 224 * 224;
    const int outputSize = 2;
    
    float* hostInput = new float[inputSize];
    float* hostOutput = new float[outputSize];
    
    void* deviceInput;
    void* deviceOutput;
    cudaMalloc(&deviceInput, inputSize * sizeof(float));
    cudaMalloc(&deviceOutput, outputSize * sizeof(float));
    
    // Preprocess
    std::cout << "\nPreprocessing..." << std::endl;
    if (!preprocessImage(imageFile, hostInput)) {
        return 1;
    }
    
    // Copy to GPU
    cudaMemcpy(deviceInput, hostInput, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set tensor addresses
    context->setTensorAddress(engine->getIOTensorName(0), deviceInput);
    context->setTensorAddress(engine->getIOTensorName(1), deviceOutput);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        context->enqueueV3(0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        context->enqueueV3(0);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double avgMs = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Copy output
    cudaMemcpy(hostOutput, deviceOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Softmax
    float maxVal = std::max(hostOutput[0], hostOutput[1]);
    float exp0 = exp(hostOutput[0] - maxVal);
    float exp1 = exp(hostOutput[1] - maxVal);
    float sum = exp0 + exp1;
    float probCat = exp0 / sum;
    float probDog = exp1 / sum;
    
    // Result
    std::cout << "\n================================" << std::endl;
    std::cout << "Result" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Prediction: " << (probCat > probDog ? "Cat" : "Dog") << std::endl;
    std::cout << "Confidence: " << (std::max(probCat, probDog) * 100) << "%" << std::endl;
    std::cout << "  Cat: " << (probCat * 100) << "%" << std::endl;
    std::cout << "  Dog: " << (probDog * 100) << "%" << std::endl;
    
    std::cout << "\n================================" << std::endl;
    std::cout << "Performance" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Inference: " << avgMs << " ms" << std::endl;
    std::cout << "Throughput: " << (1000.0 / avgMs) << " FPS" << std::endl;
    
    // Cleanup
    delete[] hostInput;
    delete[] hostOutput;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}