#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// definition of __PMEM__, __PMEM_MEMCPY__, and __PMEM_LOOP__
// __PMEM__ : the distribution function is placed on the persistent memory
// __PMEM_MEMCPY__ : the D2P- and P2D-copy operations are performed with
// pmem_memcpy() and memcpy(), respectively.
// __PMEM_LOOP__ : the D2P- and P2D-copy operations are performed with a loop.

#if defined(__PMEM_MEMCPY__) || defined(__PMEM_LOOP__)
#define __PMEM__
#endif

#ifdef __PMEM__
#include <libpmem.h>
#endif

#include "vlasov_3d.h"

#define DF(ix, iy, iz) (df[iz + NMESH_Z * (iy + NMESH_Y * ix)])

#ifdef __PMEM__
#define DRAM_DF_vel(ivx, ivy, ivz) \
  (df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))])
#endif

#define ALIGN_SIZE (64)
#define NMEASURE (5)

float timing(struct timespec _start, struct timespec _stop) {
  uint64_t start_time = _start.tv_sec * 1000 + _start.tv_nsec / 1000000;
  uint64_t stop_time = _stop.tv_sec * 1000 + _stop.tv_nsec / 1000000;
  return (stop_time - start_time) * 1.0e-3;
}

void calc_upwind(float *df_tmp_, float *df_cpy_, double cfl_) {
  int NPADD = 4;
  for (int ivv = 0; ivv < NMESH_VX; ivv++) {
    int iv = ivv + NPADD;
    df_tmp_[iv] = df_cpy_[iv] - cfl_ * (df_cpy_[iv] - df_cpy_[iv - 1]);
  }
}

void integrate_vx_loop3(struct pos_grid *df, double cfl) {
  // loop 3 : as-is code of the integrate_vx without SIMD instructions
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;

#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                // DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
                // printf("%f %f\n", DF(ix, iy, iz).vel_grid[ivx][ivy][ivz],
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }

            }  // ivz
          }    // ivy
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);

  }  // end omp parallel
}

void integrate_vx_loop4(struct pos_grid *df, double cfl) {
  // loop 4 : Data transfer from PMEM to DMEM is performed inside
  // the velocity loop (innermost region) both in the memcpy and loop methods.
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;

#pragma omp parallel
  {
    float *df_tmp, *df_cpy;
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
#ifdef __PMEM_MEMCPY__
              memcpy(df_cpy + NPADD, DF(ix, iy, iz).vel_grid,
                     sizeof(float) * NMESH_VX);
#elif __PMEM_LOOP__
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
#endif
              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {
#ifdef __PMEM_MEMCPY__
              pmem_memcpy(DF(ix, iy, iz).vel_grid, df_tmp + NPADD,
                          sizeof(float) * NMESH_VX, PMEM_F_MEM_NONTEMPORAL);
#elif __PMEM_LOOP__
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
              }  // ivx
#endif
            }  // ivz
          }    // ivy
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);

  }  // end omp parallel
#endif
}

void integrate_vx_loop5(struct pos_grid *df, double cfl) {
  // loop 5 : Data transfer from PMEM to DMEM is performed
  // in the middle of spatial and velocity loops. Buffer region in the DMEM
  // is allocated inside the spatial loop.
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;
#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
          float *df_dram = aligned_alloc(
              ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
          memcpy(df_dram, DF(ix, iy, iz).vel_grid,
                 sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
#elif __PMEM_LOOP__
          float *df_dram = aligned_alloc(
              ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))] =
                    DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
            }
          }
#endif
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DRAM_DF_vel(ivx, ivy, ivz);
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DRAM_DF_vel(ivx, ivy, ivz) = df_tmp[ivx_wp];
                // printf("%f %f\n", DRAM_DF_vel(ivx, ivy, ivz),
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }  // ivx
            }    // ivz
          }      // ivy
#ifdef __PMEM_MEMCPY__
          pmem_memcpy(DF(ix, iy, iz).vel_grid, df_dram,
                      sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ,
                      PMEM_F_MEM_NONTEMPORAL);
          free(df_dram);
#elif __PMEM_LOOP__
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] =
                    df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))];
              }
            }
          }
          free(df_dram);
#endif
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);

  }  // end omp parallel
#endif
}

void integrate_vx_loop6(struct pos_grid *df, double cfl) {
// loop 6 : Same as loop 5 but the buffer region in DMEM is allocated
// outside the spatial loop.
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;

#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    float *df_dram = aligned_alloc(
        ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
          memcpy(df_dram, DF(ix, iy, iz).vel_grid,
                 sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
#elif __PMEM_LOOP__
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))] =
                    DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
            }
          }
#endif
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DRAM_DF_vel(ivx, ivy, ivz);
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DRAM_DF_vel(ivx, ivy, ivz) = df_tmp[ivx_wp];
                // printf("%f %f\n", DRAM_DF_vel(ivx, ivy, ivz),
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }  // ivx
            }    // ivz
          }      // ivy
#ifdef __PMEM_MEMCPY__
          pmem_memcpy(DF(ix, iy, iz).vel_grid, df_dram,
                      sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ,
                      PMEM_F_MEM_NONTEMPORAL);
#elif __PMEM_LOOP__
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] =
                    df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))];
              }
            }
          }
#endif
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);
    free(df_dram);
  }  // end omp parallel
#endif
}

void integrate_vx_loop7(struct pos_grid *df, double cfl) {
  // loop 7 : Same as loop 6 but direct write back df_tmp to PMEM
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;

#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    float *df_dram = aligned_alloc(
        ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

// #pragma omp for schedule(auto) collapse(3) nowait
#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
          memcpy(df_dram, DF(ix, iy, iz).vel_grid,
                 sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
#elif __PMEM_LOOP__
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))] =
                    DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
            }
          }
#endif
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DRAM_DF_vel(ivx, ivy, ivz);
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {
#ifdef __PMEM_MEMCPY__
              pmem_memcpy(DF(ix, iy, iz).vel_grid, df_tmp + NPADD,
                          sizeof(float) * NMESH_VX, PMEM_F_MEM_NONTEMPORAL);
#elif __PMEM_LOOP__
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
                // printf("%f %f\n", DRAM_DF_vel(ivx, ivy, ivz),
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }  // ivx
#endif
            }  // ivz
          }    // ivy
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);
    free(df_dram);
  }  // end omp parallel
#endif
}

void integrate_vx_loop8(struct pos_grid *df, double cfl) {
  // loop 8 : Same as loop 7 but when write back df_tmp to PMEM,without
  // pmem_memcpy
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;

#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    float *df_dram = aligned_alloc(
        ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

// #pragma omp for schedule(auto) collapse(3) nowait
#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
          memcpy(df_dram, DF(ix, iy, iz).vel_grid,
                 sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
#elif __PMEM_LOOP__
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))] =
                    DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
            }
          }
#endif
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DRAM_DF_vel(ivx, ivy, ivz);
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux

              // for(int32_t ivx=0;ivx<NMESH_VX;ivx++) {

              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
                // printf("%f %f\n", DRAM_DF_vel(ivx, ivy, ivz),
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }  // ivx
            }    // ivz
          }      // ivy
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);
    free(df_dram);
  }  // end omp parallel
#endif
}

void integrate_vx_loop9(struct pos_grid *df, double cfl) {
  // loop 9 : Same as loop 5 but direct write back df_tmp to PMEM
#ifdef __PMEM__
  int32_t NPADD = 4;

  int32_t vx_start, vx_end;
  int32_t vy_start, vy_end;
  int32_t vz_start, vz_end;

  vx_start = 0;
  vx_end = NMESH_VX;
  vy_start = 0;
  vy_end = NMESH_VY;
  vz_start = 0;
  vz_end = NMESH_VZ;

  const int32_t NMESH_VX_WP = NMESH_VX + NPADD + NPADD;
#pragma omp parallel
  {
    // struct vflux *flux;
    float *df_tmp, *df_cpy;

    // flux = (struct vflux *)malloc(sizeof(struct vflux) * NMESH_VX_WP);
    df_cpy = (float *)malloc(sizeof(float) * NMESH_VX_WP);
    df_tmp = (float *)malloc(sizeof(float) * NMESH_VX_WP);

    for (int32_t ivx = 0; ivx < NMESH_VX_WP; ivx++) {
      df_tmp[ivx] = 0.0;  // for measures of nan=0*nan.
    }

#pragma omp for schedule(auto) collapse(3)
    for (int32_t ix = 0; ix < NMESH_X; ix++) {
      for (int32_t iy = 0; iy < NMESH_Y; iy++) {
        for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
          float *df_dram = aligned_alloc(
              ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
          memcpy(df_dram, DF(ix, iy, iz).vel_grid,
                 sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
#elif __PMEM_LOOP__
          float *df_dram = aligned_alloc(
              ALIGN_SIZE, sizeof(float) * NMESH_VX * NMESH_VY * NMESH_VZ);
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                df_dram[(ivz) + NMESH_VZ * ((ivy) + NMESH_VY * (ivx))] =
                    DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
              }
            }
          }
#endif
          for (int32_t ivy = vy_start; ivy < vy_end; ivy++) {
            for (int32_t ivz = vz_start; ivz < vz_end; ivz++) {
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                df_cpy[ivx_wp] = DRAM_DF_vel(ivx, ivy, ivz);
              }

              for (int32_t ipadd = 0; ipadd < NPADD; ipadd++) {
                df_cpy[ipadd] = df_cpy[NPADD];
                df_cpy[ipadd + NMESH_VX + NPADD] = df_cpy[NMESH_VX + NPADD - 1];
              }
              calc_upwind(df_tmp, df_cpy, cfl);
              // /* Semi-Lagrange scheme */
              // calc_flux_vel(df_cpy, flux, &ta);
              // check_positivity_vel(df_check, df_tmp, df_cpy, flux,
              // &ta);//flux
#ifdef __PMEM_MEMCPY__
              pmem_memcpy(DF(ix, iy, iz).vel_grid, df_tmp + NPADD,
                          sizeof(float) * NMESH_VX, PMEM_F_MEM_NONTEMPORAL);
#elif __PMEM_LOOP__
              for (int32_t ivx = vx_start; ivx < vx_end; ivx++) {
                int32_t ivx_wp = ivx + NPADD;
                DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = df_tmp[ivx_wp];
                // printf("%f %f\n", DRAM_DF_vel(ivx, ivy, ivz),
                //        df_cpy[ivx_wp] -
                //            cfl * (df_cpy[ivx_wp + 1] - df_cpy[ivx_wp -
                //            1]));
              }  // ivx
#endif
            }  // ivz
          }    // ivy
          free(df_dram);
        }
      }
    }  // ix*iy*iz, end omp for

    free(df_cpy);
    free(df_tmp);

  }  // end omp parallel
#endif
}

int main(int argc, char **argv) {
  struct pos_grid *df;

  int32_t is_pmem;
  size_t mapped_len;

#ifdef __PMEM__
  int64_t data_len = NMESH_POS * NMESH_VEL * sizeof(float);
  df = pmem_map_file("/dev/dax0.0", 0, 0, 0, &mapped_len, &is_pmem);
  printf("mapped_len = %ld \n", mapped_len);
  printf("is_pmem = %d\n", is_pmem);
  fflush(stdout);
  assert(data_len < mapped_len);
  assert(is_pmem == 1);
#else
  df = (struct pos_grid *)malloc(sizeof(struct pos_grid) * NMESH_POS);
#endif

  printf("# data size : %12.4e GiB \n",
         (float)(NMESH_POS) * (float)(NMESH_VEL) * sizeof(float) / (1 << 30));

  struct timespec ts_loop1_start, ts_loop1_stop;
  struct timespec ts_loop2_start, ts_loop2_stop;
  struct timespec ts_loop3_start, ts_loop3_stop;
  struct timespec ts_loop4_start, ts_loop4_stop;
  struct timespec ts_loop5_start, ts_loop5_stop;
  struct timespec ts_loop6_start, ts_loop6_stop;
  struct timespec ts_loop7_start, ts_loop7_stop;
  struct timespec ts_loop8_start, ts_loop8_stop;
  struct timespec ts_loop9_start, ts_loop9_stop;

  float *dmem_buf = (float *)malloc(sizeof(float) * NMESH_VEL);

  // loop 1 : substituting immediate values to the DF

  clock_gettime(CLOCK_MONOTONIC, &ts_loop1_start);
#pragma omp parallel for collapse(3) schedule(auto)
  for (int32_t ix = 0; ix < NMESH_X; ix++) {
    for (int32_t iy = 0; iy < NMESH_Y; iy++) {
      for (int32_t iz = 0; iz < NMESH_Z; iz++) {
        int32_t im = iz + NMESH_Z * (iy + NMESH_Y * ix);

        for (int32_t ivx = 0; ivx < NMESH_VX; ivx++) {
          for (int32_t ivy = 0; ivy < NMESH_VY; ivy++) {
            for (int32_t ivz = 0; ivz < NMESH_VZ; ivz++) {
              int32_t ivm = ivz + NMESH_VZ * (ivy + NMESH_VY * ivx);
#if defined(__PMEM_MEMCPY__) || defined(__PMEM_LOOP__)
              dmem_buf[ivm] = (float)(im) * (float)(ivm);
#else
              DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] =
                  (float)(im) * (float)(ivm);
#endif
            }
          }
        }

#ifdef __PMEM_MEMCPY__
        pmem_memcpy(DF(ix, iy, iz).vel_grid, dmem_buf,
                    sizeof(float) * NMESH_VEL, PMEM_F_MEM_NONTEMPORAL);
#elif __PMEM_LOOP__
        for (int32_t ivx = 0; ivx < NMESH_VX; ivx++) {
          for (int32_t ivy = 0; ivy < NMESH_VY; ivy++) {
            for (int32_t ivz = 0; ivz < NMESH_VZ; ivz++) {
              int32_t ivm = ivz + NMESH_VZ * (ivy + NMESH_VY * ivx);
              DF(ix, iy, iz).vel_grid[ivx][ivy][ivz] = dmem_buf[ivm];
            }
          }
        }
#endif
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_loop1_stop);
  printf("loop 1 : %14.6e\n", timing(ts_loop1_start, ts_loop1_stop));

  // loop 2 : fetching data from the DF
  clock_gettime(CLOCK_MONOTONIC, &ts_loop2_start);
#pragma omp parallel for collapse(3) schedule(auto)
  for (int32_t ix = 0; ix < NMESH_X; ix++) {
    for (int32_t iy = 0; iy < NMESH_Y; iy++) {
      for (int32_t iz = 0; iz < NMESH_Z; iz++) {
#ifdef __PMEM_MEMCPY__
        memcpy(dmem_buf, DF(ix, iy, iz).vel_grid, sizeof(float) * NMESH_VEL);
#elif __PMEM_LOOP__
        for (int32_t ivx = 0; ivx < NMESH_VX; ivx++) {
          for (int32_t ivy = 0; ivy < NMESH_VY; ivy++) {
            for (int32_t ivz = 0; ivz < NMESH_VZ; ivz++) {
              int32_t ivm = ivz + NMESH_VZ * (ivy + NMESH_VY * ivx);
              dmem_buf[ivm] = DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
            }
          }
        }
#endif

        float dens = 0.0;

        for (int32_t ivx = 0; ivx < NMESH_VX; ivx++) {
          for (int32_t ivy = 0; ivy < NMESH_VY; ivy++) {
            for (int32_t ivz = 0; ivz < NMESH_VZ; ivz++) {
#ifdef __PMEM__
              dens += dmem_buf[ivz + NMESH_VZ * (ivy + NMESH_VY * ivx)];
#else
              dens += DF(ix, iy, iz).vel_grid[ivx][ivy][ivz];
#endif
            }
          }
        }

        DF(ix, iy, iz).dens = dens;
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_loop2_stop);
  printf("loop 2 : %14.6e\n", timing(ts_loop2_start, ts_loop2_stop));

  const double cfl = 0.25;
  // loop 3 : as-is code of the integrate_vx without SIMD instructions
  float time_loop3 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop3_start);
    integrate_vx_loop3(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop3_stop);
    time_loop3 += timing(ts_loop3_start, ts_loop3_stop);
  }
  printf("loop 3 : %14.6e\n", time_loop3 / (float)NMEASURE);

  // // loop 4 : Data transfer from PMEM to DMEM is performed inside
  // // the velocity loop (innermost region) both in the memcpy and loop
  // methods.
  float time_loop4 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop4_start);
    integrate_vx_loop4(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop4_stop);
    time_loop4 += timing(ts_loop4_start, ts_loop4_stop);
  }
  printf("loop 4 : %14.6e\n", time_loop4 / (float)NMEASURE);

  // loop 5 : Data transfer from PMEM to DMEM is performed
  // in the middle of spatial and velocity loops. Buffer region in the DMEM
  // is allocated inside the spatial loop.
  float time_loop5 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop5_start);
    integrate_vx_loop5(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop5_stop);
    time_loop5 += timing(ts_loop5_start, ts_loop5_stop);
  }
  printf("loop 5 : %14.6e\n", time_loop5 / (float)NMEASURE);

  // loop 6 : Same as loop 5 but the buffer region in DMEM is allocated
  // outside the spatial loop.
  float time_loop6 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop6_start);
    integrate_vx_loop6(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop6_stop);
    time_loop6 += timing(ts_loop6_start, ts_loop6_stop);
  }
  printf("loop 6 : %14.6e\n", time_loop6 / (float)NMEASURE);

  // loop 7 : Same as loop 6 but direct write back df_tmp to PMEM
  float time_loop7 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop7_start);
    integrate_vx_loop7(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop7_stop);
    time_loop7 += timing(ts_loop7_start, ts_loop7_stop);
  }
  printf("loop 7 : %14.6e\n", time_loop7 / (float)NMEASURE);

  // loop 8 : Same as loop 6 but when write back df_tmp to PMEM,without
  // pmem_memcpy
  float time_loop8 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop8_start);
    integrate_vx_loop8(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop8_stop);
    time_loop8 += timing(ts_loop8_start, ts_loop8_stop);
  }
  printf("loop 8 : %14.6e\n", time_loop8 / (float)NMEASURE);

  // loop 9 : Same as loop 5 but direct write back df_tmp to PMEM
  float time_loop9 = 0;
  for (int n = 0; n < NMEASURE; n++) {
    clock_gettime(CLOCK_MONOTONIC, &ts_loop9_start);
    integrate_vx_loop9(df, cfl);
    clock_gettime(CLOCK_MONOTONIC, &ts_loop9_stop);
    time_loop9 += timing(ts_loop9_start, ts_loop9_stop);
  }
  printf("loop 9 : %14.6e\n", time_loop9 / (float)NMEASURE);

#ifdef __PMEM__
  pmem_unmap(df, mapped_len);
#else
  free(df);
#endif
}
