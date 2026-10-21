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

//from gemini, prompt "how to quickly generate random value vectors in c/c++". this code function is loosely based on Gemini's code
std::vector<double> generateRandomDoubleVector(uint32_t vectorLength){
    // Standard fast PRNG engine and distribution
    std::mt19937 rng(std::random_device{}()); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> vec(vectorLength);
    
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

//We will choose these 3 kernels
//1.) SAXPY / AXPY: y = a * x + y
//2.) Dot product / reduction: s = sum(x_i * y_i)
//3.) Elementwise multiply: z_i = x_i * y_i

//Experimental Knobs:
//Datatype: FP32 vs FP64, optionally int32_t
//Alginment and tail: aligned arrays vs. arrays whose starting address is intentionally shifted from a vector/cache-line boundary; sizes with and without vector-width remainders
//Access pattern: unit stride vs. stride 2,4,8, or gather-like index patterns where meaningful
//working-set size: sweep N accross L1, L2, LLC, and DRAM-resident regimes
//compiler/ISA flags: scalar-only, default auto-vectorized, and CPU-targeted widest available vectors

//Required Experiments and Plots:
//Scalar baseline vs. auto-vectorized build for each selected kernel across working-set sizes. Report speedup and achieved throughput. 
//Locality sweep for one kernel crossing cache levels. Annotate where SIMD gains compress as the workload becomes memory-bound. 
//Alignment and tail-handling study. Quantify prologue/epilogue, unaligned load, or masking effects. 
//Stride/gather study. Show how non-unit stride reduces effective bandwidth and SIMD efficiency. 
//Data-type comparison. Explain results using lane count, arithmetic intensity, and operation cost. 
//Vectorization verification. Include concise compiler-report or disassembly evidence; do not paste full dumps. 
//Roofline interpretation for at least one kernel using measured memory bandwidth and estimated CPU peak FLOP rate. 
//Include one negative example: either a loop that appears SIMD-friendly but is not actually vectorized, or a loop that is vectorized but shows little speedup because memory bandwidth dominates. Verify and explain the result. 

//Deliverables:
// GitHub repository with code, build/run scripts, raw measurements, processed data, and plotting scripts. 
//Methodology: CPU model and ISA, compiler and flags, OS, frequency policy, SMT state, timing method, repetitions, and data initialization. 
//Plots/tables with units, legends, and variability/error bars when feasible. 
//Short report connecting results to cache locality, vector width, arithmetic intensity, memory bandwidth, and anomalies. 


//Some notes:
// - You need 3 builds: (you also need kernel versions for each of these)
// -- Build A: Scalar only
// ---- in the build/compile arguments, use: cl.exe /EHsc /O2 /arch:IA32 /Qvec- /fp:precise kernel.cpp
//-03 -fno-tree-vectorize
// -- Build B: Auto vectorized
// ---- in the build/compile arguments, use: cl.exe /EHsc /O2 /arch:AVX2 /Qvec+ /fp:fast kernel.cpp
//-03
// -- Build C: Targeted SIMD
// ---- in the build/compile arguments, use: -03 -mavx2

//[Baseline] SAXPY / AXPY: y = a * x + y

void AXPY(float a, std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){//SAXPY
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
    /** 
    if(x.size() == y.size()){
        for(int i = 0; i < y.size(); i++){
            y[i] += a * x[i];
        }
    }
    **/
}

void AXPY(double a, std::vector<double>& x, std::vector<double>& y, size_t strides, size_t alignmentOffset){//AXPY
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    double* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    double* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            ypointer[i] += a * xpointer[i];
        }
    }//No Epilogue/tail handling?
    /**
    if(x.size() == y.size()){
        for(int i = 0; i < y.size(); i++){
            y[i] += a * x[i];
        }
    }
    **/
}

//[SIMD(AVX2), auto-vectorized] SAXPY / AXPY: y = a * x + y



//[Baseline] Dot product / reduction: s = sum(x_i * y_i)

float DotProductReduction(std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset){
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
    /**
    if(x.size() == y.size()){
        s = 0.0f;
        for(int i = 0; i < x.size(); i++){
            s += x[i] * y[i];
        }
    }
    **/
}

double DotProductReduction(std::vector<double>& x, std::vector<double>& y, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    double* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    double* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (alignmentOffset < ysize)){
        double s = 0.0;
        for(size_t i = 0; i < ysize - alignmentOffset; i += strides){//Main Vector Loop?
            s += ypointer[i] * xpointer[i];
        }
        return s;//Epilogue/tail handling?
    }
    return 0.0;//Epilogue/tail handling?
    /**
    if(x.size() == y.size()){
        s = 0.0f;
        for(int i = 0; i < x.size(); i++){
            s += x[i] * y[i];
        }
    }
    **/
}

//[SIMD(AXV2), auto-vectorized] Dot product / reduction: s = sum(x_i * y_i)

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
        for(size_t i = 0; i < zsize - alignmentOffset; i += strides){//Main Vector Loop?
            zpointer[i] = xpointer[i] * ypointer[i];
        }
    }//No Epilogue/tail handling?
    /**
    if((x.size() == y.size()) && (y.size() == z.size())){
        for(int i = 0; i < z.size(); i++){
            z[i] = x[i] * y[i];
        }
    }
    **/
}

void ElementWiseMultiply(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, size_t strides, size_t alignmentOffset){
    //pointer version, allows for lower overhead and custom strides
    size_t xsize = x.size();
    size_t ysize = y.size();
    size_t zsize = z.size();
    double* __restrict xpointer = x.data() + alignmentOffset;//Prologue?
    double* __restrict ypointer = y.data() + alignmentOffset;//Prologue?
    double* __restrict zpointer = z.data() + alignmentOffset;//Prologue?
    if((xsize == ysize) && (ysize == zsize) && (alignmentOffset < zsize)){
        for(size_t i = 0; i < zsize - alignmentOffset; i += strides){//Main Vector Loop?
            zpointer[i] = xpointer[i] * ypointer[i];
        }
    }//No Epilogue/tail handling?
    /**
    if((x.size() == y.size()) && (y.size() == z.size())){
        for(int i = 0; i < z.size(); i++){
            z[i] = x[i] * y[i];
        }
    }
    **/
}

//[SIMD(AVX2), auto-vectorized] Elementwise multiply: z_i = x_i * y_i

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

results BenchmarkAXPY(float a, std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset, size_t runs){
    //
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        AXPY(a, x, y, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        AXPY(a, x, y, strides, alignmentOffset);

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

results BenchmarkAXPY(double a, std::vector<double>& x, std::vector<double>& y, size_t strides, size_t alignmentOffset, size_t runs){
    //
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        AXPY(a, x, y, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        AXPY(a, x, y, strides, alignmentOffset);

    auto end = std::chrono::steady_clock::now();//end time
    volatile double sink = y[0];

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 8; //floats are 4 bytes, doubles are 8 bytes
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

results BenchmarkDotProductReduction(std::vector<float>& x, std::vector<float>& y, size_t strides, size_t alignmentOffset, size_t runs){
    //
    float s = 0.0f;
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        s = DotProductReduction(x, y, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        s = DotProductReduction(x, y, strides, alignmentOffset);

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

results BenchmarkDotProductReduction(std::vector<double>& x, std::vector<double>& y, size_t strides, size_t alignmentOffset, size_t runs){
    //
    double s = 0.0;
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        s = DotProductReduction(x, y, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        s = DotProductReduction(x, y, strides, alignmentOffset);

    auto end = std::chrono::steady_clock::now();//end time

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    volatile double sink3 = s;

    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 8; //floats are 4 bytes, doubles are 8 bytes
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

results BenchmarkElementWiseMultiply(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z, size_t strides, size_t alignmentOffset, size_t runs){
    //
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        ElementWiseMultiply(x, y, z, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        ElementWiseMultiply(x, y, z, strides, alignmentOffset);

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

results BenchmarkElementWiseMultiply(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, size_t strides, size_t alignmentOffset, size_t runs){
    //
    // Warm-up to stabilize caches and branch predictors
    for (int i = 0; i < 3; i++)
        ElementWiseMultiply(x, y, z, strides, alignmentOffset);

    auto start = std::chrono::steady_clock::now();//start time

    for (int i = 0; i < runs; i++)//run the function for a set number of iterations
        ElementWiseMultiply(x, y, z, strides, alignmentOffset);

    auto end = std::chrono::steady_clock::now();//end time
    volatile float sink = y[0];

    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();//calculate the nanosecond time difference between the start and end times
    //preset stuff, good for tweaking between variants
    size_t bytesPerElement = 8; //floats are 4 bytes, doubles are 8 bytes
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

    //set governor to performance (locks at highest non-turbo frequency)
    //sudo apt install cpufrequtils
    //sudo cpufreq-set -c 0 -g performance //this sets core 0 to performance mode, which locks it at its highest non-turbo frequency
    //to disable it: just reboot your computer (as it isn't persistent across reboots)

    //set governor to performance (locks at highest non-turbo frequency)(the previous method is deprecated)
    //sudo apt install linux-cpupower
    //sudo cpupower -c 0 frequency-set -g performance //this sets core 0 to performance mode, which locks it at its highest non-turbo frequency
    //to undo this: restart the computer


    //disable turbo boost:
    //echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo //this disables turbo boost on all cores
    //to re-enable turbo boost: echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

    //lock the frequency to a fixed value
    //to find available frequencies: cpufreq-info
    //setting a fixed frequency
    //sudo cpufreq-set -c 0 -f 2400000 //this sets core 0 to 2.4GHz

    //lock the frequency to a fixed value (the previous method is deprecated)
    //sudo cpupower -c 0 frequency-set -f 2400000 //this sets core 0 to 2.4GHz (unit is in KHz by default)
    //to undo this: restart the computer



    //some default values
    size_t STRIDE = 1;
    size_t VECTORLENGTH = 32768;
    size_t ALIGNMENTOFFSET = 0;
    bool DATATYPEMODE = true;//FP32
    size_t RUNS = 20;

    //we need to just do experimental plot stuff now:
    //1.) Scalar baseline vs auto-vectorized build for each selected kernel across working-set sizes. 
    //Report speedup and achieved throughput.



    //2.) Locality sweep for one kernel crossing cache levels. Annotate where SIMD gains compress
    //as the workload becomes memory-bound

    std::cout << "\nLocality sweep: \n" << std::endl;

    //working set sweep
    /**
    std::vector<size_t> workingSetSizes = {//for FP32 only
        8'192, //~32KB (L0 D)
        12288, //~56KB (L0 I)
        32'768, //~128KB (L1)
        524288, //~2MB (L2, Pcore)
        1572864, //~6MB (L3 shared)
        16'777'216 //~64MB (DRAM)
    };
    **/
    //32KB: L0 D
    //56KB: L0 I
    //128KB: L1 (P core)
    //2MB: L2 (P core)
    //6MB: L3 (shared, P cores)
    //64MB: DRAM

    std::vector<size_t> workingSetSizes = {//for FP32 only
        4'096, //~16KB (L1 I, per core)
        9'216, //~36KB (L1 D, per core)
        262'144, //~1MB (L2, per core)
        1'572'864, //~6MB (L3 shared)
        16'777'216 //~64MB (DRAM)
    };

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
        results axpyRESFloatAligned2 = BenchmarkAXPY(a2, x2, y2, STRIDE, ALIGNMENTOFFSET, RUNS);
        std::cout << "AXPY: " << std::endl;
        printResults(axpyRESFloatAligned2);
        results dotRESFloatAligned2 = BenchmarkDotProductReduction(x2, y2, STRIDE, ALIGNMENTOFFSET, RUNS);
        std::cout << "Dot Product Reduction: " << std::endl;
        printResults(dotRESFloatAligned2);
        results elementwisemultRESFloatAligned2 = BenchmarkElementWiseMultiply(x2, y2, z2, STRIDE, ALIGNMENTOFFSET, RUNS);
        std::cout << "Element-wise Multiply: " << std::endl;
        printResults(elementwisemultRESFloatAligned2);
    }

    //3.) Alignment and tail-handling study. Quantify prologue/epilogue, unaligned load, or masking effects.

    std::cout << "\nAligned vs not aligned: \n" << std::endl;

    std::cout << "\nAligned: \n" << std::endl;
    std::vector<float> x3 = generateRandomFloatVector(VECTORLENGTH);
    std::vector<float> y3 = generateRandomFloatVector(VECTORLENGTH);
    std::vector<float> z3(VECTORLENGTH, 0.0f);
    //from bing copilot, prompt: "generate random float in c++"
    std::random_device rd3;//non-deterministic seed
    std::mt19937 gen3(rd3());//Mersenne Twister engine
    std::uniform_real_distribution<float> dist3(0.0f, 1.0f);//Create a uniform real distribution in the given range
    float a3 = dist3(gen3);//generate random float

    //benchmark each with default settings
    results axpyRESFloatAligned3 = BenchmarkAXPY(a3, x3, y3, STRIDE, 0, RUNS);
    std::cout << "AXPY: " << std::endl;
    printResults(axpyRESFloatAligned3);
    results dotRESFloatAligned3 = BenchmarkDotProductReduction(x3, y3, STRIDE, 0, RUNS);
    std::cout << "Dot Product Reduction: " << std::endl;
    printResults(dotRESFloatAligned3);
    results elementwisemultRESFloatAligned3 = BenchmarkElementWiseMultiply(x3, y3, z3, STRIDE, 0, RUNS);
    std::cout << "Element-wise Multiply: " << std::endl;
    printResults(elementwisemultRESFloatAligned3);

    std::cout << "\nMisaligned: 7\n" << std::endl;
    x3 = generateRandomFloatVector(VECTORLENGTH);
    y3 = generateRandomFloatVector(VECTORLENGTH);
    std::vector<float> z35(VECTORLENGTH, 0.0f);
    //from bing copilot, prompt: "generate random float in c++"
    std::random_device rd35;//non-deterministic seed
    std::mt19937 gen35(rd35());//Mersenne Twister engine
    std::uniform_real_distribution<float> dist35(0.0f, 1.0f);//Create a uniform real distribution in the given range
    a3 = dist35(gen35);//generate random float

    //benchmark each with default settings
    results axpyRESFloatOffset35 = BenchmarkAXPY(a3, x3, y3, STRIDE, 7, RUNS);
    std::cout << "AXPY: " << std::endl;
    printResults(axpyRESFloatOffset35);
    results dotRESFloatOffset35 = BenchmarkDotProductReduction(x3, y3, STRIDE, 7, RUNS);
    std::cout << "Dot Product Reduction: " << std::endl;
    printResults(dotRESFloatOffset35);
    results elementwisemultRESFloatOffset35 = BenchmarkElementWiseMultiply(x3, y3, z35, STRIDE, 7, RUNS);
    std::cout << "Element-wise Multiply: " << std::endl;
    printResults(elementwisemultRESFloatOffset35);

    //alignment and tail handling recommended explanation from copilot "Misaligned starting addresses force the compiler to emit scalar prologue code until the pointer reaches a 32‑byte boundary. Non‑multiple vector lengths force scalar or masked epilogue code to handle leftover elements. Both effects reduce SIMD efficiency without changing arithmetic intensity."
    //-->from prompt "what is prologue and epilogue with regards to alignment and tail-handling?"

    //4.) Stride/gather study. Show how non-unit stride reduces effective bandwidth and SIMD efficiency

    std::cout << "\nStride Sweep: \n" << std::endl;

    for(size_t S : strideValues){
        std::cout << "Stride Size: " << S << std::endl;
        std::vector<float> x4 = generateRandomFloatVector(VECTORLENGTH);
        std::vector<float> y4 = generateRandomFloatVector(VECTORLENGTH);
        std::vector<float> z4(VECTORLENGTH, 0.0f);
        //from bing copilot, prompt: "generate random float in c++"
        std::random_device rd4;//non-deterministic seed
        std::mt19937 gen4(rd4());//Mersenne Twister engine
        std::uniform_real_distribution<float> dist4(0.0f, 1.0f);//Create a uniform real distribution in the given range
        float a4 = dist4(gen4);//generate random float

        //benchmark each with default settings
        results axpyRESFloatAligned4 = BenchmarkAXPY(a4, x4, y4, S, ALIGNMENTOFFSET, RUNS);
        std::cout << "AXPY: " << std::endl;
        printResults(axpyRESFloatAligned4);
        results dotRESFloatAligned4 = BenchmarkDotProductReduction(x4, y4, S, ALIGNMENTOFFSET, RUNS);
        std::cout << "Dot Product Reduction: " << std::endl;
        printResults(dotRESFloatAligned4);
        results elementwisemultRESFloatAligned4 = BenchmarkElementWiseMultiply(x4, y4, z4, S, ALIGNMENTOFFSET, RUNS);
        std::cout << "Element-wise Multiply: " << std::endl;
        printResults(elementwisemultRESFloatAligned4);
    }

    //5.) Data-type comparison. Explain results using lane count, arithmetic intensity, and operation cost.
    std::cout << "\nData-type Comparison (FP32 vs FP64): \n" << std::endl;

    size_t FLOATVECTORSIZE = 16384; //64KB
    std::cout << "FP32 results: " << std::endl;
    std::vector<float> x5 = generateRandomFloatVector(FLOATVECTORSIZE);
    std::vector<float> y5 = generateRandomFloatVector(FLOATVECTORSIZE);
    std::vector<float> z5(FLOATVECTORSIZE, 0.0f);
    //from bing copilot, prompt: "generate random float in c++"
    std::random_device rd5;//non-deterministic seed
    std::mt19937 gen5(rd5());//Mersenne Twister engine
    std::uniform_real_distribution<float> dist5(0.0f, 1.0f);//Create a uniform real distribution in the given range
    float a5 = dist5(gen5);//generate random float

    //benchmark each with default settings
    results axpyRESFloatAligned5 = BenchmarkAXPY(a5, x5, y5, 1, 0, RUNS);
    std::cout << "AXPY: " << std::endl;
    printResults(axpyRESFloatAligned5);
    results dotRESFloatAligned5 = BenchmarkDotProductReduction(x5, y5, 1, 0, RUNS);
    std::cout << "Dot Product Reduction: " << std::endl;
    printResults(dotRESFloatAligned5);
    results elementwisemultRESFloatAligned5 = BenchmarkElementWiseMultiply(x5, y5, z5, 1, 0, RUNS);
    std::cout << "Element-wise Multiply: " << std::endl;
    printResults(elementwisemultRESFloatAligned5);

    size_t DOUBLEVECTORSIZE = 16384; //128KB
    std::cout << "FP64 results: " << std::endl;
    std::vector<double> x55 = generateRandomDoubleVector(DOUBLEVECTORSIZE);
    std::vector<double> y55 = generateRandomDoubleVector(DOUBLEVECTORSIZE);
    std::vector<double> z55(DOUBLEVECTORSIZE, 0.0);
    //from bing copilot, prompt: "generate random float in c++"
    std::random_device rd55;//non-deterministic seed
    std::mt19937 gen55(rd55());//Mersenne Twister engine
    std::uniform_real_distribution<double> dist55(0.0, 1.0);//Create a uniform real distribution in the given range
    double a55 = dist55(gen55);//generate random double

    //benchmark each with default settings
    results axpyRESDoubleAligned55 = BenchmarkAXPY(a55, x55, y55, 1, 0, RUNS);
    std::cout << "AXPY: " << std::endl;
    printResults(axpyRESDoubleAligned55);
    results dotRESDoubleAligned55 = BenchmarkDotProductReduction(x55, y55, 1, 0, RUNS);
    std::cout << "Dot Product Reduction: " << std::endl;
    printResults(dotRESDoubleAligned55);
    results elementwisemultRESDoubleAligned55 = BenchmarkElementWiseMultiply(x55, y55, z55, 1, 0, RUNS);
    std::cout << "Element-wise Multiply: " << std::endl;
    printResults(elementwisemultRESDoubleAligned55);

    //6.) Vectorization verification. Include concise compiler-report or disassembly evidence; do not paste full dumps



    //7.) Roofline interpretation for at least one kernel using measured memory bandwidth and estimated CPU peak FLOP rate.

    //note: I can probably use #2 results for this one, by sweeping the working set size for a single kernel, I can get the bandwidth and throughput and plot it

    //8.) Include one negative example: either a loop that appears SIMD-friendly but is not actually 
    //vectorized, or a loop that is vectorized but shows little speedup becuase memory bandwidth dominates.
    //Verify and explain the result.

    std::cout << "ended program" << std::endl;

    //end timing and calculations and other stuff
    return 1;
}
//on windows, to compile and run properly:
//0.) open vscode, go to file, then open folder, then find the folder that holds your cpp file
//1.) click your cpp file on the explorer tab on the left side of the screen
//2.) press the play button on the top right. it will run once, but it will also open a new terminal
//3.) in this new terminal, you can type the proper commands:
// -- cl project1.cpp -o project1.exe
// -- ./project1.exe

//for core ultra 7 255h,:
//-L0 D/I: 48KB/64KB per core
//-L1: 192KB per core
//-L2 3MB per core
//-L3: 24MB shared (1.5MB per core)
//-ECore L1: 96KB per core
//-Ecore L2: 4MB per module (1 module?)
//-LPE core L2: 2MB shared (1MB per LPE core)

//32KB: L0 D
//56KB: L0 I
//128KB: L1 (P core)
//2MB: L2 (P core)
//6MB: L3 (shared, P cores)
//64MB: DRAM

//For next time, 9/14/26:
//-- get GiB/s metric (add it to your results struct and your benchmark functions)
//-- roofline stuff
//-- Working set stuff and sizes, etc
//-- lane count

//Highest speed compiler argument combo:
//cl /EHsc /O2 /arch:AVX2 project1.cpp -o project1Fast.exe
//slow speed SSE2
//slowish speed?:
//cl /EHsc /O1 /arch:SSE2 project1.cpp -o project1Slow.exe