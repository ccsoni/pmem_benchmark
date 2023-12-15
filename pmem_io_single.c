#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <libpmem.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

#define ALIGN_BYTE (8)
#define NMEASURE (16)

#define WRITE_IMM_TO_DRAM (0)
#define WRITE_IMM_TO_PMEM (0)
#define COPY_DRAM_TO_DRAM_MEMCPY (1)
#define COPY_DRAM_TO_DRAM_LOOP (0)
#define COPY_DRAM_TO_PMEM_MEMCPY (1)
#define COPY_DRAM_TO_PMEM_LOOP (1)
#define COPY_PMEM_TO_PMEM_MEMCPY (1)
#define COPY_PMEM_TO_PMEM_LOOP (1)

#define VERBOSE (0)

double get_realtime(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (double)((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

double get_cputime(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

double get_average(double *input, int32_t n)
{
  double ave=0.0;
  for(int32_t i=0;i<n;i++) ave += input[i];

  return (ave/(double)n);
}

//typedef float REAL;
typedef double REAL;

int main(int argc, char **argv)
{
  size_t mapped_len;
  int is_pmem;
  REAL *pmesh, *dmesh;
  double bandwidth;

  if(argc != 3) {
    fprintf(stderr, "Usage: %s nshift openmp_nthread(Size of the data is 2^nshift bytes)\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  int32_t nshift = atoi(argv[1]);
  int32_t nthread = atoi(argv[2]);
  int64_t nsize = 1L<<nshift;
  double data_size_in_GiB = (float)(sizeof(REAL)*nsize)/(float)(1024*1024*1024);
  printf("# data size : %12.4e MiB \n",(float)(sizeof(REAL)*nsize)/(float)(1024*1024));

  int64_t stride = nsize/nthread;
  assert(nsize % nthread == 0);

  struct timespec ts;
  clock_getres(CLOCK_REALTIME, &ts);

  pmesh = pmem_map_file("/dev/dax0.0", 0, 0, 0, &mapped_len, &is_pmem);
  dmesh = (REAL *)aligned_alloc(ALIGN_BYTE, nsize*sizeof(REAL));
  
  printf("# Mapped length : %ld\n", mapped_len);
  printf("# Is_pmem : %d\n", is_pmem);

  REAL *dram_REAL;
  dram_REAL = (REAL *)aligned_alloc(ALIGN_BYTE, nsize*sizeof(REAL));
  for(int64_t i=0;i<nsize;i++) dram_REAL[i] = (float)i;

  size_t nelem_fp = mapped_len/sizeof(REAL);
  assert(nelem_fp >= nsize);

  size_t nelem_fp_half = nelem_fp/2;

  static double bandwidth_i[NMEASURE];
  
#if WRITE_IMM_TO_PMEM
  // writing immediate value into PMEM
  printf("# Writing immediate values to PMEM with a for-loop.\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {
    double pmem_start_real_time = get_realtime();
#pragma omp parallel for
    for(int64_t i=0;i<nsize;i++) {
      pmesh[i] = (float)i;
    }
    double pmem_end_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (pmem_end_real_time - pmem_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, pmem_end_real_time - pmem_start_real_time, bandwidth);fflush(stdout);
#endif    
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));
#endif

#if WRITE_IMM_TO_DRAM
  // writing immediate value into DRAM
  printf("# Writing immediate values to DRAM with a for-loop.\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {
    double dram_start_real_time = get_realtime();
#pragma omp parallel for
    for(int64_t i=0;i<nsize;i++) {
      dmesh[i] = (float)i*iloop;
    }
    double dram_stop_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (dram_stop_real_time-dram_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE    
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, dram_stop_real_time-dram_start_real_time, bandwidth);fflush(stdout);
#endif
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));  
#endif

#if COPY_DRAM_TO_DRAM_MEMCPY
  // copying data from DRAM to DRAM with memcpy()
  printf("# Copying data from DRAM to DRAM with memcpy()\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {
    double D2D_copy_start_real_time = get_realtime();

    omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int32_t i=0;i<nthread;i++) {
      memcpy(dmesh+i*stride, dram_REAL+i*stride, sizeof(REAL)*stride);
    }
    double D2D_copy_stop_real_time  = get_realtime();
    
    bandwidth = data_size_in_GiB / (D2D_copy_stop_real_time-D2D_copy_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE    
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, D2D_copy_stop_real_time-D2D_copy_start_real_time, bandwidth);fflush(stdout);
#endif
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));  
#endif

#if COPY_DRAM_TO_DRAM_LOOP
  // copying data from DRAM to DRAM with a for-loop
  printf("# Copying data from DRAM to DRAM with a for-loop\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {
    double D2D_copy_start_real_time = get_realtime();
#pragma omp parallel for
    for(int64_t i=0;i<nsize;i++) {
      dmesh[i] = dram_REAL[i];
    }
    double D2D_copy_stop_real_time  = get_realtime();
    
    bandwidth = data_size_in_GiB / (D2D_copy_stop_real_time-D2D_copy_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, D2D_copy_stop_real_time-D2D_copy_start_real_time, bandwidth);fflush(stdout);
#endif
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));
#endif  

#if COPY_DRAM_TO_PMEM_MEMCPY
  // copying data from DRAM to PMEM with pmem_memcpy() with pmem_memcpy() in the non-tempral model
  printf("# Copying data from DRAM to PMEM with pmem_memcpy()\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {  
    double D2P_start_real_time = get_realtime();
    omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int32_t i=0;i<nthread;i++) {
      pmem_memcpy(pmesh+i*stride, dram_REAL+i*stride, sizeof(REAL)*stride, PMEM_F_MEM_NONTEMPORAL);
    }
    double D2P_stop_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (D2P_stop_real_time - D2P_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, D2P_stop_real_time-D2P_start_real_time, bandwidth);fflush(stdout);
#endif    
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));  
#endif


#if COPY_DRAM_TO_PMEM_LOOP
  // copying data from DRAM to PMEM with a for-loop
  printf("# Copying data from DRAM to PMEM with a for-loop\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {  
    double D2P_start_real_time = get_realtime();
#pragma omp parallel for
    for(int32_t i=0;i<nsize;i++) {
      pmesh[i] = dram_REAL[i];
    }
    double D2P_stop_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (D2P_stop_real_time - D2P_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE    
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, D2P_stop_real_time-D2P_start_real_time, bandwidth);fflush(stdout);
#endif    
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));
#endif

#if COPY_PMEM_TO_PMEM_MEMCPY
  // copying data from PMEM to PMEM with pmem_memcpy() in the non-tempral model
  printf("# Copying data from PMEM to PMEM with pmem_memcpy()\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {  
    double P2P_start_real_time = get_realtime();
    omp_set_num_threads(nthread);
#pragma omp parallel for
    for(int32_t i=0;i<nthread;i++) {
      pmem_memcpy(pmesh+i*stride, pmesh+i*stride+nelem_fp_half, sizeof(REAL)*stride, PMEM_F_MEM_NONTEMPORAL);
    }
    double P2P_stop_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (P2P_stop_real_time - P2P_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, P2P_stop_real_time-P2P_start_real_time, bandwidth);fflush(stdout);
#endif    
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));    
#endif

#if COPY_PMEM_TO_PMEM_LOOP
  // copying data from PMEM to PMEM with a for-loop
  printf("# Copying data from PMEM to PMEM with a for-loop\n");
  for(int32_t iloop=0;iloop<NMEASURE;iloop++) {  
    double P2P_start_real_time = get_realtime();
#pragma omp parallel for
    for(int32_t i=0;i<nsize;i++) {
      pmesh[i] = pmesh[i+nelem_fp_half];
    }
    double P2P_stop_real_time = get_realtime();

    bandwidth = data_size_in_GiB / (P2P_stop_real_time - P2P_start_real_time);
    bandwidth_i[iloop] = bandwidth;

#if VERBOSE    
    printf("%f GiB FP data written in %10.2e sec with a bandwidth of %10.2e GiB/s.\n",
	   data_size_in_GiB, P2P_stop_real_time-P2P_start_real_time, bandwidth);fflush(stdout);
#endif    
  }
  printf("%12.4e %12.4e\n", data_size_in_GiB, get_average(bandwidth_i, NMEASURE));  
#endif

  free(dram_REAL);
  free(dmesh);
  pmem_unmap((void *)pmesh, mapped_len);

  exit(EXIT_SUCCESS);
}
