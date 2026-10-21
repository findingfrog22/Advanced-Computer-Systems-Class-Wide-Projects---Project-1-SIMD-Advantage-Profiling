# Advanced-Computer-Systems-Class-Wide-Projects---Project-1-SIMD-Advantage-Profiling

Measurement Discipline (System Configuration):
- System Specs:
- CPU: Intel Core i5 1135G7, 4c/8t, 400MHz-4.20GHz
- --> Turbo boost disabled via Intel P_State driver modifications
- --> Pinned to Core 0
- --> Hyperthreading (SMT) is Disabled through BIOS prior to benchmarking
- --> Cache working sets were based on cache size information pulled from intel Ark website, see further below
- RAM: 8GB LPDDR4X 4267MHz (Dual Channel)
- OS: Linux Debian Trixie 13.7.0
- Compiler/Tools:
- GCC 14.2.0 (default with Debian 13)
- Governor: intel cpu_freq
- VSCode IDE

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
- --> bouncing ideas off eachother for possible explanations
- --> doing last minute polishing of my report explanations (I wrote the original explanations, but they were rough to read)
- Gemini was used for:
- --> setting up my Windows 11 C/C++ Compiler environment (it was broken before)
- --> helping with random number generation code
- --> Assisting with Linux benchmarking libraries support

Testing Methodology:
- Core Pinning was pinned to Core 0
- Hyperthreading was disabled in BIOS
- Turbo boost is disabled via modifications to the intel cpu_freq governor in Debian, along with linux-powercpu to ensure that clockspeed stays at constant 1.40GHz for core 0. In practice, it stayed exactly at 1.40GHz.
- --> Also all low-power C states were disabled in the script, so it is at C0 100% of the time
- Each kernel/test run was run 20 times to warm up, with results and timing averaged across those 20 runs (runs 3 runs before the 20-run loop in this calculation due to warm ups)

- Benchmarking information collected during benchmarking each kernel run:
- --> Total floating point operations (all runs)
- --> GFLOPS (avg per run)
- --> HW Arithmetic Intensity (FLOPS/Byte)
- --> Elapsed Time (in nanoseconds) across all runs
- --> Average Time (in nanoseconds, per run)

For scalar vs auto-vectorization, I used these compiler flags:
- Scalar: -03 -fno-tree-vectorize
- Auto-vectorization: -03
- (Used for plots #2-5) Targeted SIMD: -03 -mavx512vl (this adds vector length support too)

Cache working set sizes: (these sizes fit within these cache hiarchies)
- For FP32 only:
- --> 4,096 Floats : ~16KB (L1 I, per core)
- --> 9,216 Floats : ~36KB (L1 D, per core)
- --> 262,144 Floats : ~1MB (L2, per core)
- --> 1,572,864 Floats : ~6MB (L3, Shared)
- --> 16,777,216 Floats : ~64MB (DRAM)
- For FP64: removed implementation

According to Intel's website for the I5 1135G7 + some linux commands, I was able to derive the cache hierarchy for this CPU:
- L1-I: 128KiB (32KiB per core)
- L1-D: 192KiB (48KiB per core)
- L2 cache: 5MiB (1.25MiB per core)
- L3 "Smart" cache: 8MiB (shared)

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






