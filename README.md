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

Datatypes chosen:
- FP32 (float) : default
- FP64 (double) : chosen as the comparison

Strides chosen for sweep:
- 1: default
- 2,4,8
- This is how many elements in the array/vector that it runs over in granularity. so 1 means it strides 1 element at a time (which means it doesn't skip over elements). so 2 means it strides every other element (so element 1 to element 3 to ...)

Alignment options chosen for comparison:
- 0: Aligned
- 7: Misaligned
- This is the starting element offset from the original base address. so misaligned is 7 elements off from the original base address.

Pre-experiment Hypothesis:
- TBD
