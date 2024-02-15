#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <libpmem.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>

#define ALIGN_BYTE (8)
#define NMEASURE (16)

#define COPY_DRAM_TO_DRAM_bcast (1)
#define COPY_PMEM_TO_PMEM_bcast (1)
#define BANDWIDTH_NKIND_MAX (2)

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
  double ave = 0.0;
  for (int32_t i = 0; i < n; i++)
    ave += input[i];

  return (ave / (double)n);
}

// typedef float REAL;
typedef double REAL;

int main(int argc, char **argv)
{
  size_t mapped_len;
  int is_pmem;
  REAL *pmesh, *dmesh;

  static double bandwidth[BANDWIDTH_NKIND_MAX];
  
  int32_t myrank, nproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int32_t nshift = atoi(argv[1]);
  int64_t nsize = 1L << nshift;

  double data_size_in_GiB = (float)(sizeof(REAL) * nsize) / (float)(1024 * 1024 * 1024);
  if(myrank == 0) {
    printf("# data size : %12.4e MiB \n",
	   (float)(sizeof(REAL) * nsize) / (float)(1024 * 1024));
  }
  struct timespec ts;
  clock_getres(CLOCK_REALTIME, &ts);

  pmesh = pmem_map_file("/dev/dax0.0", 0, 0, 0, &mapped_len, &is_pmem);
  dmesh = (REAL *)aligned_alloc(ALIGN_BYTE, nsize*sizeof(REAL));
  if(myrank == 0) {
    printf("# Mapped length : %ld\n", mapped_len);
    printf("# Is_pmem : %d\n", is_pmem);
  }

  for(int64_t i=0; i<nsize; i++) {
    dmesh[i] = (float)i;
    pmesh[i] = (float)i;
  }

  static double bandwidth_i[NMEASURE];

#if COPY_DRAM_TO_DRAM_bcast
  double D2D_bcast_start_real_time;
  double D2D_bcast_end_real_time;
  if(myrank == 0) printf("# DRAM to DRAM bcast\n");

  for (int32_t iloop=0; iloop<NMEASURE; iloop++) {
    // start timer
    if (myrank == 0) D2D_bcast_start_real_time = get_realtime();

    MPI_Bcast(dmesh, nsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // stop timer + calc. bandwidth
    if(myrank == 0) {
      D2D_bcast_end_real_time = get_realtime();
      bandwidth_i[iloop] = data_size_in_GiB / (D2D_bcast_end_real_time - D2D_bcast_start_real_time);
    }
  }

  if(myrank == 0) bandwidth[0] = get_average(bandwidth_i, NMEASURE);
#endif

#if COPY_PMEM_TO_PMEM_bcast
  double P2P_bcast_start_real_time;
  double P2P_bcast_end_real_time;
  if(myrank == 0) printf("# PMEM to PMEM bcast\n");

  for(int32_t iloop=0; iloop<NMEASURE; iloop++) {
    // start timer
    if(myrank == 0) P2P_bcast_start_real_time = get_realtime();

    MPI_Bcast(pmesh, nsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // stop timer + calc. bandwidth
    if(myrank == 0) {
      P2P_bcast_end_real_time = get_realtime();
      bandwidth_i[iloop] = data_size_in_GiB / (P2P_bcast_end_real_time - P2P_bcast_start_real_time);
    }
  }

  if(myrank == 0) bandwidth[1] = get_average(bandwidth_i, NMEASURE);
#endif

  if(myrank == 0) {
    FILE *fp = fopen("bcast_pmem.dat","a");
    //    fprintf(fp, "#data_size [GiB] D2D_BW P2P_BW\n");
    fprintf(fp, "%12.4e %12.4e %12.4e\n", data_size_in_GiB, bandwidth[0], bandwidth[1]);
    fclose(fp);
  }  

  free(dmesh);
  pmem_unmap((void *)pmesh, mapped_len);

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
