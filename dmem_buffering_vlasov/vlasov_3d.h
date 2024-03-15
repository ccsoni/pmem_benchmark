#pragma once

#define NMESH_VX (64L)
#define NMESH_VY (64L)
#define NMESH_VZ (64L)
#define NMESH_VEL (NMESH_VX*NMESH_VY*NMESH_VZ)

#define NMESH_X (32L)
#define NMESH_Y (32L)
#define NMESH_Z (32L)
#define NMESH_POS (NMESH_X*NMESH_Y*NMESH_Z)

struct pos_grid {
  float vel_grid[NMESH_VX][NMESH_VY][NMESH_VZ];
  float dens, pot;
  float xvel_mean, yvel_mean, zvel_mean;
  float xacc, yacc, zacc;
};
