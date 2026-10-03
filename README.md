# Advanced-Computer-Systems-Class-Wide-Projects---Project-1-SIMD-Advantage-Profiling

Measurement Discipline (System Configuration):
- System Specs:
- CPU: Intel Core Ultra 7 255h (no SMT)
- --> Turbo boost disabled via BIOS, hence max clockspeed (pinned to Pcore 0) is 2GHz
- --> Cache working sets were based on cache size information pulled from intel Ark website, see further below
- RAM: 32GB DDR5 5600MHz (Dual Channel)
- OS: Windows 11 Home OEM (25H2, 26200.9445)
- Compiler/Tools:
- GCC/CLANG
- TBD

Pre-experiment Hypothesis:
- I expect that throughput will be best when memory access is aligned and using sequential strides (stride of 1). I also suspect that effective bandwidth will sharply drop once we have strides beyond 1 and misaligned memory. Also, sequential access will probably be faster than random access too.

AI Use Note:
- Tools used: Copilot, Gemini
- Copilot was used for:
- --> assisting with benchmark scripting methodology and function calls
- --> discussion around core pinning, testing methodology
- --> discussion around prologue and tail with regards to alignment and tail-handling study
- --> double checking my benchmarking calculations and formulas
- --> second opinion on results
- --> getting compiler flags for scalar vs auto-vectorization vs full targeted SIMD instructions
- Gemini was used for:
- --> setting up my Windows 11 C/C++ Compiler environment (it was broken before)
- --> helping with random number generation code

Testing Methodology:
- Core Pinning was pinned to Core 0 (P core)
- SMT is not supported on this system anyways
- Turbo boost is disabled via bios, along with Intel speedstep to ensure that clockspeed stays at constant 2GHz for P cores
- Each kernel/test run was run 20 times to warm up, with results and timing averaged across those 20 runs (skipping the first 3 runs in this calculation due to warm ups)

- Benchmarking information collected during benchmarking each kernel run:
- --> Total floating point operations (all runs)
- --> GFLOPS (avg per run)
- --> HW Arithmetic Intensity (FLOPS/Byte)
- --> Elapsed Time (in nanoseconds) across all runs
- --> Average Time (in nanoseconds, per run)

For scalar vs auto-vectorization, I used these compiler flags:
- Scalar: -03 -fno-tree-vectorize
- Auto-vectorization: -03
- (Optionally commented) Targeted SIMD: -03 -mavx2

Cache working set sizes: (these sizes fit within these cache hiarchies)
- For FP32 only:
- --> 8,192 Floats : ~32KB (L0 D)
- --> 12,288 Floats : ~56KB (L0 I)
- --> 32,768 Floats : ~128KB (L1)
- --> 524,288 Floats : ~2MB (L2, Pcore)
- --> 1,572,864 Floats : ~6MB (L3, Shared)
- --> 1,777,216 Floats : ~64MB (DRAM)
- For FP64: removed implementation

Intel has odd cache hierchies for the Intel Core Ultra 7 255H:
- L0 D/I: 48KB/64KB per core
- L1: 192KB per core
- L2: 3MB per core
- L3: 24MB shared (~1.5MB per core equivalent)
- ECore L1: 96KB per core
- ECore L2: 4MB per module (1 module simply isn't documented well for Arrow Lake H)
- LPE Core L2: 2MB Shared (1MB per LPE core equivalent)

Alignment options chosen for comparison:
- 0: Aligned
- 7: Misaligned
- This is the starting element offset from the original base address. so misaligned is 7 elements off from the original base address.

Strides chosen for sweep:
- 1: default
- 2,4,8
- This is how many elements in the array/vector that it runs over in granularity. so 1 means it strides 1 element at a time (which means it doesn't skip over elements). so 2 means it strides every other element (so element 1 to element 3 to ...)

Datatypes chosen:
- FP32 (float) : default
- FP64 (double) : chosen as the comparison






