//Project 1

//Libraries
#include <vector>
#include <immintrin.h> //this is AVX SIMD instruction (we will only use AVX2)
#include <random>//from gemini, prompt "how to quickly generate random value vectors in c/c++"
#include <iostream>
#include <chrono> //from copilot, prompt "...first will probably be timing"
//#include <windows.h> //from copilot, "first off, we should do thread pinning", to pin to core 0 to avoid scheduler overhead and resource contention
#include <algorithm> //from bing copilot "std::generate in c++ (for gcc)"
//copilot says to put these 3 lines down for thread pinning in linux
//#define _GNU_SOURCE
#include <sched.h>//linux scheduler
#include <pthread.h>//thread stuff


//Helper functions:
//from gemini, prompt "how to quickly generate random value vectors in c/c++". this code function is loosely based on Gemini's code
std::vector<float> generateRandomFloatVector(uint32_t vectorLength){
    // Standard fast PRNG engine and distribution
    std::mt19937 rng(std::random_device{}()); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vec(vectorLength);
    
    // Fill vector sequentially using lambda call
    std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });

    return vec;
}

//from copilot, i prompted it for a timing benchmark mechanism. I added some more comments for better understanding
template<typename Func>
uint64_t benchmark(Func&& f, int iterations)
{
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        f();

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < iterations; i++)//run the function for a set number of iterations
        f();

    auto end = std::chrono::steady_clock::now();//end time

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    return ns;//return the total time elapsed in nanoseconds for the number of iterations
}

//[Baseline] SAXPY / AXPY: y = a * x + y

void AXPY(float a, std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){//SAXPY
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        #pragma loop(no_vector)
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            ypointer[i] += a * xpointer[i];
        }
    }//no epilogue/tail handling?
}

//[SIMD AVX2, Auto-vectoriztion] SAXPY / AXPY: y = a * x + y

void AXPY_SIMD(float a, std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){//SAXPY
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            ypointer[i] += a * xpointer[i];
        }
    }//no epilogue/tail handling?
}

//[Baseline] Dot product / reduction: s = sum(x_i * y_i)

float DotProductReduction(std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        float s = 0.0f;
        #pragma loop(no_vector)
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            s += ypointer[i] * xpointer[i];
        }
        return s;//Epilogue/tail handling?
    }
    return 0.0f;//Epilogue/tail handling?
}

//[SIMD(AXV2), auto-vectorized] Dot product / reduction: s = sum(x_i * y_i)

float DotProductReduction_SIMD(std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        float s = 0.0f;
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            s += ypointer[i] * xpointer[i];
        }
        return s;//Epilogue/tail handling?
    }
    return 0.0f;//Epilogue/tail handling?
}

//[Baseline] Elementwise multiply: z_i = x_i * y_i

void ElementWiseMultiply(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    size_t zsize = z.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    float* __restrict zpointer = z.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (ysize == zsize) && (alignmentOffset < zsize)){
        #pragma loop(no_vector)
        for(size_t i = 0; i < zsize - alignmentOffset; i += strides){//Main Vector Loop?
            zpointer[i] = xpointer[i] * ypointer[i];
        }
    }//No Epilogue/tail handling?
}

//[SIMD(AVX2), auto-vectorized] Elementwise multiply: z_i = x_i * y_i

void ElementWiseMultiply_SIMD(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    size_t zsize = z.size();
    float* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    float* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    float* __restrict zpointer = z.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (ysize == zsize) && (alignmentOffset < zsize)){
        for(size_t i = 0; i < zsize - alignmentOffset; i += strides){//Main Vector Loop?
            zpointer[i] = xpointer[i] * ypointer[i];
        }
    }//No Epilogue/tail handling?
}

//I will make my own version based off of this:
struct results{
    size_t elapsed_time; //average time for all runs
    size_t average_time; //average time per run in nanoseconds
    size_t runs; //number of runs elapsed
    size_t flop; //how many floating point operations total (across all runs)
    float gflops; //GigaFLOPS per second (averaged across runs)
    double arithmeticIntensity; //FLOPS/Byte accessed (averaged across all runs)
};

void printResults(results res){
    std::cout << "Elapsed Time (in nanoseconds) across all runs: " << res.elapsed_time << "\nAverage time (in nanoseconds, per run): " << res.average_time << "\nNumber of Runs: " << res.runs << "\nTotal floating point operations (all runs): " << res.flop << "\nGFLOPS (avg per run): " << res.gflops << "\nHW Arithmetic Intensity (FLOPS/Byte): " << res.arithmeticIntensity << std::endl;
}

results BenchmarkAXPY(float a, std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset, size_t runs, bool isVectorized){
    //
    if(isVectorized){
        // Warm-up to stabilize caches and branch predictors
        for (int i = 0; i < 3; i++)
            AXPY_SIMD(a, x, y, strides, alignmentOffset);
    }
    else{
        // Warm-up to stabilize caches and branch predictors
        for (int i = 0; i < 3; i++)
            AXPY(a, x, y, strides, alignmentOffset);
    }
    auto start = std::chrono::steady_clock::now();//start time
    
    if(isVectorized){
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            AXPY_SIMD(a, x, y, strides, alignmentOffset);
    }
    else{
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            AXPY(a, x, y, strides, alignmentOffset);
    }
    auto end = std::chrono::steady_clock::now();//end time
    volatile float sink = y[0];

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 4; //floats are 4 bytes, doubles are 8 bytes
    size_t flopsPerInstruction = 2; //AXPY: 2 floating point operations per instruction (1 mult + 1 add)
    size_t accessesPerInstruction = 3; //AXPY: 3 accesses per instruction (1 for read x, 1 for read y, another for write y)
    //timing
    size_t timePerRun = ns / runs;
    size_t timeAllRuns = ns;
    //number of instructions/accesses
    size_t instructionsPerRun = (y.size() - alignmentOffset)/strides;
    size_t instructionsAllRuns = instructionsPerRun * runs;
    //number of floating point operations across all runs
    //size_t flopsPerRun = instructionsPerRun * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    size_t flopsAllRuns = instructionsAllRuns * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    //number of bytes accessed
    //size_t bytesPerRun = instructionsPerRun * accessesPerInstruction * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    size_t bytesAllRuns = instructionsAllRuns * accessesPerInstruction * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    //GFLOPs/Second
    double seconds = timeAllRuns * 1e-9;
    float GFLOPs = float(flopsAllRuns / seconds / 1e9); // original expression (total GFLOP) / (seconds) //I understand this, it is becuase we need it in terms of GFLOPs/s rather than FLOPs/s. Copilot gave feedback on my original expression
    //HW Arithmetic Intensity (FLOP/Byte)
    double HWArithmeticIntensity = double(flopsAllRuns) / double(bytesAllRuns);

    results res = {timeAllRuns, timePerRun, runs, flopsAllRuns, GFLOPs, HWArithmeticIntensity};
    //results res = {ns, ns/runs, runs, 2*((y.size()-alignmentOffset)/strides)*runs, float(((2*((y.size()-alignmentOffset)/strides)*runs)/1000000000)/(ns/1000000000)), (2*((y.size()-alignmentOffset)/strides)*runs)/(((y.size() - alignmentOffset)/strides)*3)};
    return res;
}

results BenchmarkDotProductReduction(std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset, size_t runs, bool isVectorized){
    //
    float s = 0.0f;
    if(isVectorized){
        // Warm-up to stabilize caches and branch predictors
        for (int i = 0; i < 3; i++)
            s = DotProductReduction_SIMD(x, y, strides, alignmentOffset);
    }
    else{
        for(int i = 0; i < 3; i++){
            s = DotProductReduction(x, y, strides, alignmentOffset);
        }
    }
    auto start = std::chrono::steady_clock::now();//start time

    if(isVectorized){
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            s = DotProductReduction_SIMD(x, y, strides, alignmentOffset);
    }
    else{
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            s = DotProductReduction(x, y, strides, alignmentOffset);
    }
    auto end = std::chrono::steady_clock::now();//end time
    volatile float sink3 = s;

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 4; //floats are 4 bytes, doubles are 8 bytes
    size_t flopsPerInstruction = 2; //DotProductReduction: 2 floating point operations per instruction (1 mult + 1 add)
    size_t accessesPerInstruction = 2; //DotProductReduction: 2 accesses per instruction (1 for read x, 1 for read y)(there is 1 access at the end for accumulation into s)
    //timing
    size_t timePerRun = ns / runs;
    size_t timeAllRuns = ns;
    //number of instructions/accesses
    size_t instructionsPerRun = (y.size() - alignmentOffset)/strides;
    size_t instructionsAllRuns = instructionsPerRun * runs;
    //number of floating point operations across all runs
    //size_t flopsPerRun = instructionsPerRun * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    size_t flopsAllRuns = instructionsAllRuns * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    //number of bytes accessed
    //size_t bytesPerRun = instructionsPerRun * accessesPerInstruction * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    size_t bytesAllRuns = (instructionsAllRuns * accessesPerInstruction + 1) * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    //GFLOPs/Second
    double seconds = timeAllRuns * 1e-9;
    float GFLOPs = float(flopsAllRuns / seconds / 1e9); // original expression (total GFLOP) / (seconds) //I understand this, it is becuase we need it in terms of GFLOPs/s rather than FLOPs/s. Copilot gave feedback on my original expression
    //HW Arithmetic Intensity (FLOP/Byte)
    double HWArithmeticIntensity = double(flopsAllRuns) / double(bytesAllRuns);

    results res = {timeAllRuns, timePerRun, runs, flopsAllRuns, GFLOPs, HWArithmeticIntensity};
    //results res = {ns, ns/runs, runs, 2*((y.size()-alignmentOffset)/strides)*runs, float(((2*((y.size()-alignmentOffset)/strides)*runs)/1000000000)/(ns/1000000000)), (2*((y.size()-alignmentOffset)/strides)*runs)/(((y.size() - alignmentOffset)/strides)*3)};
    return res;
}

results BenchmarkElementWiseMultiply(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z, size_t strides, size_t alignmentOffset, size_t runs, bool isVectorized){
    //
    if(isVectorized){
        // Warm-up to stabilize caches and branch predictors
        for (int i = 0; i < 3; i++)
            ElementWiseMultiply_SIMD(x, y, z, strides, alignmentOffset);
    }
    else{
        // Warm-up to stabilize caches and branch predictors
        for (int i = 0; i < 3; i++)
            ElementWiseMultiply(x, y, z, strides, alignmentOffset);
    }
    auto start = std::chrono::steady_clock::now();//start time

    if(isVectorized){
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            ElementWiseMultiply_SIMD(x, y, z, strides, alignmentOffset);
    }
    else{
        for (int i = 0; i < runs; i++)//run the function for a set number of iterations
            ElementWiseMultiply(x, y, z, strides, alignmentOffset);
    }
    auto end = std::chrono::steady_clock::now();//end time
    volatile float sink = y[0];

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 4; //floats are 4 bytes, doubles are 8 bytes
    size_t flopsPerInstruction = 1; //ElementWiseMultiply: 1 floating point operation per instruction (x * y)
    size_t accessesPerInstruction = 3; //ElementWiseMultiply: 3 accesses per instruction (read x, read y, write z)
    //timing
    size_t timePerRun = ns / runs;
    size_t timeAllRuns = ns;
    //number of instructions/accesses
    size_t instructionsPerRun = (y.size() - alignmentOffset)/strides;
    size_t instructionsAllRuns = instructionsPerRun * runs;
    //number of floating point operations across all runs
    //size_t flopsPerRun = instructionsPerRun * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    size_t flopsAllRuns = instructionsAllRuns * flopsPerInstruction; //2 floating point operations per instruction (1 mult + 1 add)
    //number of bytes accessed
    //size_t bytesPerRun = instructionsPerRun * accessesPerInstruction * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    size_t bytesAllRuns = instructionsAllRuns * accessesPerInstruction * bytesPerElement; //3 accesses per instruction, 4 bytes per access (becuase floats are 32-bit)
    //GFLOPs/Second
    double seconds = timeAllRuns * 1e-9;
    float GFLOPs = float(flopsAllRuns / seconds / 1e9); // original expression (total GFLOP) / (seconds) //I understand this, it is becuase we need it in terms of GFLOPs/s rather than FLOPs/s. Copilot gave feedback on my original expression
    //HW Arithmetic Intensity (FLOP/Byte)
    double HWArithmeticIntensity = double(flopsAllRuns) / double(bytesAllRuns);

    results res = {timeAllRuns, timePerRun, runs, flopsAllRuns, GFLOPs, HWArithmeticIntensity};
    //results res = {ns, ns/runs, runs, 2*((y.size()-alignmentOffset)/strides)*runs, float(((2*((y.size()-alignmentOffset)/strides)*runs)/1000000000)/(ns/1000000000)), (2*((y.size()-alignmentOffset)/strides)*runs)/(((y.size() - alignmentOffset)/strides)*3)};
    return res;
}

//NOTE: the length of the input vectors for these baseline kernels should be much larger than the 256 bits, 
//and should probably be some multiple of the stride * number of floats or doubles

int main(){
    //some settings:
    size_t stride = 1; //1 is default. other options: 2,4,8, gather-like index patterns if needed
    size_t workingSetSize = 0; //Sweep N across memory regimes, this is the amount of memory allocated
    size_t workingSetMemMode = 0;// 0 - DRAM (default), 1 - LLC(L3 Cache), 2 - L2 Cache, 3 - L1 Cache
    size_t alignmentOffset = 0;// 0 (default), how many elements of offset. floats [0-7 max], doubles [0-3 max]
    size_t vectorLength = 1000;//1000 (default), Try not to make it too big, as it becomes CPU limited
    bool datatypeMode = true;//true - float(FP32) (default), false - double(FP64)

    //internal variables (don't touch)
    //uint32_t ;
    

    //timing logic/benchmark data collection stuff
    
    //stride sweep
    std::vector<size_t> strideValues = {1,2,4,8};

    /**
    //Note: this group of code below is for thread pinning. From Copilot, it pins to thread0, "first off, we should do thread pinning"
    //pin to core/thread 0 (windows only)
    HANDLE hThread = GetCurrentThread();
    //1 = use CPU 0 only (bitmask)
    DWORD_PTR mask = 1;
    SetThreadAffinityMask(hThread, mask);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    **/

    //this chunk is from copilot "I changed the platform to linux debian, can you help me adjust it for this new platform? (the old one only works on windows)"
    //"I also want to keep the frequency constant, disable turbo boost, and disable Intel speedstep"
    //new core/thread pinning for linux
    //core isolation requires cset and cpuset
    //sudo apt install cpuset <-- this should just install the cset with it
    //first, requires you to run the program like this: 
    //sudo cset shield --cpu 0 --kthread=on
    //sudo cset shield --exec -- ./your_program
    //then, to disable that shield: sudo cset shield --reset OR sudo cset shield -r
    int core_id = 0;//pins to core 0 by default
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    //some default values
    size_t STRIDE = 1;
    size_t VECTORLENGTH = 32768;
    size_t ALIGNMENTOFFSET = 0;
    bool DATATYPEMODE = true;//FP32
    size_t RUNS = 20;

    //we need to just do experimental plot stuff now:
    //1.) Scalar baseline vs auto-vectorized build for each selected kernel across working-set sizes. 
    //Report speedup and achieved throughput.

    std::vector<results> AXPY_Scalar;
    std::vector<results> AXPY_Vector;
    std::vector<results> DotProductReduction_Scalar;
    std::vector<results> DotProductReduction_Vector;
    std::vector<results> ElementWiseMultiply_Scalar;
    std::vector<results> ElementWiseMultiply_Vector;

    std::cout << "\nScalar-Only: Locality sweep: \n" << std::endl;

    //working set sweep
    std::vector<size_t> workingSetSizes = {//for FP32 only
        4'096, //~16KB (L1 I, per core)
        9'216, //~36KB (L1 D, per core)
        262'144, //~1MB (L2, per core)
        1'572'864, //~6MB (L3 shared)
        16'777'216 //~64MB (DRAM)
    };

    //locality sweep. modified from Copilot, "sure, working set sweep is next", "sure, stride sweep next"
    for(size_t N : workingSetSizes){
        //std::cout << "\nWorking set size (x4 gets your kb(thousands), mb(millions) count): " << N << std::endl;
        std::vector<float> x1 = generateRandomFloatVector(N);
        std::vector<float> y1 = generateRandomFloatVector(N);
        std::vector<float> z1(N, 0.0f);
        //from bing copilot, prompt: "generate random float in c++"
        std::random_device rd1;//non-deterministic seed
        std::mt19937 gen1(rd1());//Mersenne Twister engine
        std::uniform_real_distribution<float> dist1(0.0f, 1.0f);//Create a uniform real distribution in the given range
        float a1 = dist1(gen1);//generate random float

        //benchmark each with default settings
        results axpyRESFloatAligned1 = BenchmarkAXPY(a1, x1, y1, STRIDE, ALIGNMENTOFFSET, RUNS, false);
        AXPY_Scalar.push_back(axpyRESFloatAligned1);
        //printResults(axpyRESFloatAligned1);
        results dotRESFloatAligned1 = BenchmarkDotProductReduction(x1, y1, STRIDE, ALIGNMENTOFFSET, RUNS, false);
        DotProductReduction_Scalar.push_back(dotRESFloatAligned1);
        //printResults(dotRESFloatAligned1);
        results elementwisemultRESFloatAligned1 = BenchmarkElementWiseMultiply(x1, y1, z1, STRIDE, ALIGNMENTOFFSET, RUNS, false);
        ElementWiseMultiply_Scalar.push_back(elementwisemultRESFloatAligned1);
        //printResults(elementwisemultRESFloatAligned1);
    }



    //2.) Locality sweep for one kernel crossing cache levels. Annotate where SIMD gains compress
    //as the workload becomes memory-bound
    /**
    std::cout << "\nAuto-Vectorization: Locality sweep: \n" << std::endl;

    //locality sweep. modified from Copilot, "sure, working set sweep is next", "sure, stride sweep next"
    for(size_t N : workingSetSizes){
        std::cout << "\nWorking set size (x4 gets your kb(thousands), mb(millions) count): " << N << std::endl;
        std::vector<float> x2 = generateRandomFloatVector(N);
        std::vector<float> y2 = generateRandomFloatVector(N);
        std::vector<float> z2(N, 0.0f);
        //from bing copilot, prompt: "generate random float in c++"
        std::random_device rd2;//non-deterministic seed
        std::mt19937 gen2(rd2());//Mersenne Twister engine
        std::uniform_real_distribution<float> dist2(0.0f, 1.0f);//Create a uniform real distribution in the given range
        float a2 = dist2(gen2);//generate random float

        //benchmark each with default settings
        results axpyRESFloatAligned2 = BenchmarkAXPY(a2, x2, y2, STRIDE, ALIGNMENTOFFSET, RUNS, true);
        AXPY_Vector.push_back(axpyRESFloatAligned2);
        //printResults(axpyRESFloatAligned2);
        results dotRESFloatAligned2 = BenchmarkDotProductReduction(x2, y2, STRIDE, ALIGNMENTOFFSET, RUNS, true);
        DotProductReduction_Vector.push_back(dotRESFloatAligned2);
        //printResults(dotRESFloatAligned2);
        results elementwisemultRESFloatAligned2 = BenchmarkElementWiseMultiply(x2, y2, z2, STRIDE, ALIGNMENTOFFSET, RUNS, true);
        ElementWiseMultiply_Vector.push_back(elementwisemultRESFloatAligned2);
        //printResults(elementwisemultRESFloatAligned2);
    }
    **/

    //std::cout << "Comparison between Scalar and Auto-Vectorization: " << std::endl;
    std::cout << "Scalar results: " << std::endl;

    int r = 0;
    for(size_t i : workingSetSizes){
        std::cout << "at working set size: " << i << ", " << std::endl;

        float AS_P = AXPY_Scalar[r].gflops;
        //float AV_P = AXPY_Vector[r].gflops;

        size_t AS_T = AXPY_Scalar[r].average_time;
        //size_t AV_T = AXPY_Vector[r].average_time;

        std::cout << "AXPY Scalar Throughput (GFLOPS): " << AS_P << std::endl;
        //std::cout << "AXPY Auto-Vectorization Throughput (GFLOPS): " << AV_P << std::endl;

        //double AXPY_Speedup = (double(AV_P/AS_P) - 1.0) * 100.0;

        //std::cout << "AXPY Vectorization vs Scalar Speedup (%): " << AXPY_Speedup << std::endl;
        
        //AXPY_Scalar;
        //AXPY_Vector;

        float DS_P = DotProductReduction_Scalar[r].gflops;
        //float DV_P = DotProductReduction_Vector[r].gflops;

        size_t DS_T = DotProductReduction_Scalar[r].average_time;
        //size_t DV_T = DotProductReduction_Vector[r].average_time;

        std::cout << "Dot Product Reduction Scalar Throughput (GFLOPS): " << DS_P << std::endl;
        //std::cout << "Dot Product Reduction Auto-Vectorization Throughput (GFLOPS): " << DV_P << std::endl;

        //double DotProductReduction_Speedup = (double(DV_P/DS_P) - 1.0) * 100.0;

        //std::cout << "Dot Product Reduction Vectorization vs Scalar Speedup (%): " << DotProductReduction_Speedup << std::endl;

        //DotProductReduction_Scalar;
        //DotProductReduction_Vector;

        float ES_P = ElementWiseMultiply_Scalar[r].gflops;
        //float EV_P = ElementWiseMultiply_Vector[r].gflops;

        size_t ES_T = ElementWiseMultiply_Scalar[r].average_time;
        //size_t EV_T = ElementWiseMultiply_Vector[r].average_time;

        std::cout << "Element-Wise Multiply Scalar Throughput (GFLOPS): " << ES_P << std::endl;
        //std::cout << "Element-Wise Multiply Auto-Vectorization Throughput (GFLOPS): " << EV_P << std::endl;

        //double ElementWiseMultiply_Speedup = (double(EV_P/ES_P) - 1.0) * 100.0;

        //std::cout << "Element-Wise Multiply Vectorization vs Scalar Speedup (%): " << ElementWiseMultiply_Speedup << std::endl;

        //ElementWiseMultiply_Scalar;
        //ElementWiseMultiply_Vector;
        //what results we want: .gflops, .average_time
        r += 1;
    }

    //6.) Vectorization verification. Include concise compiler-report or disassembly evidence; do not paste full dumps

    //try: /Qvec-report:2 in your compiler arguments, as it tells you which ones are vectorized and not vectorized

    //7.) Roofline interpretation for at least one kernel using measured memory bandwidth and estimated CPU peak FLOP rate.

    //note: I can probably use #2 results for this one, by sweeping the working set size for a single kernel, I can get the bandwidth and throughput and plot it

    std::cout << "ended Scalar program" << std::endl;

    //end timing and calculations and other stuff
    return 1;
}

//so the build arguments is like this:
//cl /O2 /EHsc /Qvec-report:2 project1ScalarVsVector.cpp -o project1ScalarVsVector.exe
//fastest one (vectorization and scalar): cl /O2 /EHsc /arch:AVX2 project1ScalarVsVector.cpp -o project1Vector.exe
//slowest one (don't use): cl /O1 /EHsc project1ScalarVsVector.cpp -o project1Scalar.exe
//optionally add: /Qvec-report:2

//NVM, LETS JUST SWITCH TO LINUX AND USE CLANG/GCC