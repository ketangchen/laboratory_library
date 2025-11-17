/*
   AFL++ Lattice-based Mutation Strategy with Multi-Armed Bandit
   -------------------------------------------------------------
   
   This module implements a lattice-theory based approach to mutation strategy
   selection, combining mutation vectors with Multi-Armed Bandit (MAB) optimization.
   
   Copyright 2024 AFLplusplus Project. All rights reserved.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:
   
     https://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef _AFL_LATTICE_MAB_H
#define _AFL_LATTICE_MAB_H

#include "config.h"
#include "types.h"
/* Don't include afl-mutations.h here to avoid duplicate symbols */
/* Include it in the .c file instead */
#include <math.h>
#include <stdint.h>

/* Forward declaration */
struct afl_state;

/* Dimension of mutation vectors */
#define MUTATION_VECTOR_DIM 8

/* Maximum number of arms in MAB */
#define MAB_MAX_ARMS 256

/* MAB algorithm parameters */
#define MAB_ALPHA 0.1          /* Learning rate for UCB */
#define MAB_EXPLORATION_C 2.0  /* Exploration constant for UCB */
#define MAB_MIN_PULLS 5        /* Minimum pulls before using UCB */

/* Lattice parameters */
#define LATTICE_MAX_VECTORS 1024
#define LATTICE_ORTHOGONALITY_THRESHOLD 0.1

/* Mutation Vector: represents a mutation operation as a structured vector */
typedef struct mutation_vector {
  
  u32 mut_type;                    /* Mutation type (MUT_*) */
  u32 position;                    /* Position in input (normalized 0-100) */
  u32 magnitude;                   /* Magnitude of mutation (0-100) */
  u32 dimension;                   /* Dimension affected (bit/byte/word/dword) */
  u32 direction;                   /* Direction (add/sub/flip/overwrite) */
  u32 context;                     /* Context (text/binary/ascii) */
  u32 frequency;                   /* Frequency of this mutation */
  u32 effectiveness;               /* Historical effectiveness score */
  
} mutation_vector_t;

/* Lattice point: a point in the mutation strategy space */
typedef struct lattice_point {
  
  mutation_vector_t vector;        /* The mutation vector */
  u32               id;            /* Unique identifier */
  double            coordinates[MUTATION_VECTOR_DIM];  /* Coordinates in lattice */
  u32               neighbors[8];  /* Neighbor point IDs */
  u32               neighbor_count;
  double            density;       /* Local density */
  
} lattice_point_t;

/* Multi-Armed Bandit arm */
typedef struct mab_arm {
  
  u32    arm_id;                   /* Arm identifier (maps to mutation type) */
  u64    pulls;                    /* Number of times pulled */
  double total_reward;             /* Cumulative reward */
  double avg_reward;               /* Average reward */
  double ucb_value;                /* Upper Confidence Bound value */
  double variance;                 /* Reward variance */
  u64    last_pull_time;           /* Last time this arm was pulled */
  u32    consecutive_failures;     /* Consecutive failures */
  
} mab_arm_t;

/* Lattice structure */
typedef struct mutation_lattice {
  
  lattice_point_t *points;         /* Array of lattice points */
  u32              point_count;    /* Number of points */
  u32              max_points;     /* Maximum capacity */
  double           bounds[MUTATION_VECTOR_DIM][2];  /* Bounds for each dimension */
  u8               initialized;    /* Initialization flag */
  
} mutation_lattice_t;

/* MAB structure */
typedef struct mutation_mab {
  
  mab_arm_t *arms;                 /* Array of arms */
  u32        arm_count;            /* Number of arms */
  u64        total_pulls;          /* Total number of pulls */
  double     total_reward;         /* Total cumulative reward */
  u8         initialized;          /* Initialization flag */
  
} mutation_mab_t;

/* Lattice-MAB integration structure */
typedef struct lattice_mab_state {
  
  mutation_lattice_t lattice;      /* The mutation lattice */
  mutation_mab_t     mab;          /* The MAB algorithm */
  
  /* Statistics */
  u64 lattice_selections;          /* Times lattice-based selection used */
  u64 mab_selections;              /* Times MAB selection used */
  u64 hybrid_selections;           /* Times hybrid selection used */
  
  /* Configuration */
  u8  use_lattice;                 /* Enable lattice-based selection */
  u8  use_mab;                     /* Enable MAB selection */
  u8  use_hybrid;                  /* Enable hybrid approach */
  u8  adaptive_mode;               /* Adaptive mode selection */
  
  /* Performance tracking */
  double avg_reward;               /* Average reward */
  double best_reward;              /* Best reward achieved */
  u32    best_arm_id;              /* Best performing arm ID */
  
} lattice_mab_state_t;

/* Function declarations */

/* Initialize lattice-MAB system */
void lattice_mab_init(struct afl_state *afl);

/* Cleanup lattice-MAB system */
void lattice_mab_deinit(struct afl_state *afl);

/* Convert mutation type to vector representation */
mutation_vector_t mutation_type_to_vector(u32 mut_type, struct afl_state *afl);

/* Add vector to lattice */
u32 lattice_add_vector(mutation_lattice_t *lattice, mutation_vector_t *vec);

/* Find nearest neighbor in lattice */
u32 lattice_find_nearest(mutation_lattice_t *lattice, mutation_vector_t *vec);

/* Calculate orthogonality between two vectors */
double lattice_orthogonality(mutation_vector_t *v1, mutation_vector_t *v2);

/* Calculate distance between two vectors */
double lattice_distance(mutation_vector_t *v1, mutation_vector_t *v2);

/* Initialize MAB */
void mab_init(mutation_mab_t *mab, u32 arm_count);

/* Select arm using UCB algorithm */
u32 mab_select_arm_ucb(mutation_mab_t *mab);

/* Select arm using epsilon-greedy */
u32 mab_select_arm_epsilon(mutation_mab_t *mab, double epsilon);

/* Update arm reward */
void mab_update_reward(mutation_mab_t *mab, u32 arm_id, double reward);

/* Get best arm */
u32 mab_get_best_arm(mutation_mab_t *mab);

/* Main selection function: choose mutation strategy using lattice-MAB */
u32 lattice_mab_select_mutation(struct afl_state *afl, u32 *mutation_array, u32 array_size);

/* Update reward based on fuzzing result */
void lattice_mab_update_reward(struct afl_state *afl, u32 mut_type, u8 found_new_path, 
                                u8 found_crash, u32 exec_time_us);

/* Calculate reward from fuzzing outcome */
double calculate_mutation_reward(u8 found_new_path, u8 found_crash, 
                                  u32 exec_time_us, u32 base_time_us);

/* Build initial lattice from mutation arrays */
void lattice_build_from_arrays(mutation_lattice_t *lattice, u32 *array, u32 size);

/* Get lattice statistics */
void lattice_get_stats(mutation_lattice_t *lattice, u32 *point_count, 
                       double *avg_density, double *max_density);

/* Get MAB statistics */
void mab_get_stats(mutation_mab_t *mab, u32 *total_pulls, double *avg_reward, 
                   u32 *best_arm);

#endif /* !_AFL_LATTICE_MAB_H */

