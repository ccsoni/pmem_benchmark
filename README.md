# pmem_benchmark
Benchmark tests for Optane persistent memory in Pegasus supercomputer

- Usage
    
    After loading the module for intel compiler by issuing

    ```% module load intel/2023.0.0```

    one can measure the IO bandwidth for PMEM and DRAM with a command 

    ```% pmem_io_single 30 4```

    where the data size is 2^30 bytes (8 GiB) and the number of OpenMP threads is 4. 
