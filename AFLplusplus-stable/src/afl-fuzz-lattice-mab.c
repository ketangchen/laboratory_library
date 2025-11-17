/*
   AFL++ Lattice-based Mutation Strategy with Multi-Armed Bandit
   -------------------------------------------------------------
   
   Implementation of lattice-theory based mutation strategy selection
   with Multi-Armed Bandit optimization.
   
   Copyright 2024 AFLplusplus Project. All rights reserved.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:
   
     https://www.apache.org/licenses/LICENSE-2.0
*/

#include "afl-fuzz.h"
#include "afl-lattice-mab.h"
/* We need MUT_* enum values from afl-mutations.h */
/* The arrays in afl-mutations.h cause duplicate symbols when included in multiple .c files */
/* Solution: declare arrays as extern BEFORE including to make them available to inline functions */
/* Then define AFL_MUTATIONS_ARRAYS_DEFINED to skip array definitions in the header */
extern s8  interesting_8[];
extern s16 interesting_16[];
extern s32 interesting_32[];
extern u32 text_array[];
extern u32 binary_array[];
extern u32 normal_splice_array[];
extern u32 full_splice_array[];
extern u32 mutation_strategy_exploration_text[];
extern u32 mutation_strategy_exploration_binary[];
extern u32 mutation_strategy_exploitation_text[];
extern u32 mutation_strategy_exploitation_binary[];

#define AFL_MUTATIONS_ARRAYS_DEFINED
#include "afl-mutations.h"
#undef AFL_MUTATIONS_ARRAYS_DEFINED

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* Initialize lattice-MAB system */
void lattice_mab_init(struct afl_state *afl) {

  if (!afl) { return; }

  lattice_mab_state_t *lm = ck_alloc(sizeof(lattice_mab_state_t));
  if (!lm) { PFATAL("Failed to allocate lattice_mab_state"); }

  memset(lm, 0, sizeof(lattice_mab_state_t));

  /* Initialize lattice */
  lm->lattice.points = ck_alloc(sizeof(lattice_point_t) * LATTICE_MAX_VECTORS);
  if (!lm->lattice.points) { PFATAL("Failed to allocate lattice points"); }
  
  lm->lattice.max_points = LATTICE_MAX_VECTORS;
  lm->lattice.point_count = 0;
  lm->lattice.initialized = 1;

  /* Set bounds for each dimension */
  for (u32 i = 0; i < MUTATION_VECTOR_DIM; ++i) {
    lm->lattice.bounds[i][0] = 0.0;
    lm->lattice.bounds[i][1] = 100.0;
  }

  /* Initialize MAB */
  u32 arm_count = MUT_MAX;  /* One arm per mutation type */
  if (arm_count > MAB_MAX_ARMS) { arm_count = MAB_MAX_ARMS; }
  
  mab_init(&lm->mab, arm_count);

  /* Configuration: enable hybrid approach by default, unless disabled via env */
  char *disable_lm = getenv("AFL_DISABLE_LATTICE_MAB");
  if (disable_lm && atoi(disable_lm) > 0) {
    lm->use_lattice = 0;
    lm->use_mab = 0;
    lm->use_hybrid = 0;
  } else {
    lm->use_lattice = 1;
    lm->use_mab = 1;
    lm->use_hybrid = 1;
  }
  lm->adaptive_mode = 1;

  afl->lattice_mab = lm;

}

/* Cleanup lattice-MAB system */
void lattice_mab_deinit(struct afl_state *afl) {

  if (!afl || !afl->lattice_mab) { return; }

  lattice_mab_state_t *lm = afl->lattice_mab;

  if (lm->lattice.points) {
    ck_free(lm->lattice.points);
  }

  if (lm->mab.arms) {
    ck_free(lm->mab.arms);
  }

  ck_free(lm);
  afl->lattice_mab = NULL;

}

/* Convert mutation type to vector representation */
mutation_vector_t mutation_type_to_vector(u32 mut_type, struct afl_state *afl) {

  mutation_vector_t vec;
  memset(&vec, 0, sizeof(mutation_vector_t));

  vec.mut_type = mut_type;

  /* Map mutation type to vector dimensions */
  switch (mut_type) {

    case MUT_FLIPBIT:
      vec.dimension = 0;  /* bit */
      vec.direction = 2;  /* flip */
      vec.magnitude = 1;
      break;

    case MUT_INTERESTING8:
    case MUT_INTERESTING16:
    case MUT_INTERESTING32:
    case MUT_INTERESTING16BE:
    case MUT_INTERESTING32BE:
      vec.dimension = (mut_type == MUT_INTERESTING8) ? 1 : 
                      ((mut_type == MUT_INTERESTING16 || mut_type == MUT_INTERESTING16BE) ? 2 : 4);
      vec.direction = 3;  /* overwrite */
      vec.magnitude = 50;
      break;

    case MUT_ARITH8:
    case MUT_ARITH8_:
    case MUT_ARITH16:
    case MUT_ARITH16_:
    case MUT_ARITH32:
    case MUT_ARITH32_:
    case MUT_ARITH16BE:
    case MUT_ARITH16BE_:
    case MUT_ARITH32BE:
    case MUT_ARITH32BE_:
      vec.dimension = (mut_type == MUT_ARITH8 || mut_type == MUT_ARITH8_) ? 1 :
                      ((mut_type == MUT_ARITH16 || mut_type == MUT_ARITH16_ ||
                        mut_type == MUT_ARITH16BE || mut_type == MUT_ARITH16BE_) ? 2 : 4);
      vec.direction = (mut_type == MUT_ARITH8 || mut_type == MUT_ARITH16 || 
                       mut_type == MUT_ARITH32 || mut_type == MUT_ARITH16BE || 
                       mut_type == MUT_ARITH32BE) ? 0 : 1;  /* add : sub */
      vec.magnitude = 10;
      break;

    case MUT_CLONE_COPY:
    case MUT_CLONE_FIXED:
    case MUT_OVERWRITE_COPY:
    case MUT_OVERWRITE_FIXED:
      vec.dimension = 1;  /* byte */
      vec.direction = 3;  /* overwrite */
      vec.magnitude = 20;
      break;

    case MUT_DEL:
    case MUT_DELONE:
      vec.dimension = 1;
      vec.direction = 4;  /* delete */
      vec.magnitude = 5;
      break;

    case MUT_INSERTONE:
      vec.dimension = 1;
      vec.direction = 5;  /* insert */
      vec.magnitude = 1;
      break;

    default:
      vec.dimension = 1;
      vec.direction = 0;
      vec.magnitude = 10;
      break;

  }

  /* Set context based on input mode */
  if (afl) {
    vec.context = afl->input_mode;  /* 0=default, 1=text, 2=binary */
  }

  return vec;

}

/* Calculate dot product of two vectors */
static double vector_dot_product(mutation_vector_t *v1, mutation_vector_t *v2) {

  double dot = 0.0;
  dot += (double)v1->position * (double)v2->position;
  dot += (double)v1->magnitude * (double)v2->magnitude;
  dot += (double)v1->dimension * (double)v2->dimension;
  dot += (double)v1->direction * (double)v2->direction;
  dot += (double)v1->context * (double)v2->context;
  dot += (double)v1->frequency * (double)v2->frequency;
  dot += (double)v1->effectiveness * (double)v2->effectiveness;
  
  return dot;

}

/* Calculate vector magnitude */
static double vector_magnitude(mutation_vector_t *v) {

  double mag = 0.0;
  mag += (double)v->position * (double)v->position;
  mag += (double)v->magnitude * (double)v->magnitude;
  mag += (double)v->dimension * (double)v->dimension;
  mag += (double)v->direction * (double)v->direction;
  mag += (double)v->context * (double)v->context;
  mag += (double)v->frequency * (double)v->frequency;
  mag += (double)v->effectiveness * (double)v->effectiveness;
  
  return sqrt(mag);

}

/* Calculate orthogonality between two vectors */
double lattice_orthogonality(mutation_vector_t *v1, mutation_vector_t *v2) {

  if (!v1 || !v2) { return 0.0; }

  double dot = vector_dot_product(v1, v2);
  double mag1 = vector_magnitude(v1);
  double mag2 = vector_magnitude(v2);

  if (mag1 == 0.0 || mag2 == 0.0) { return 0.0; }

  double cos_theta = dot / (mag1 * mag2);
  double orthogonality = fabs(cos_theta);  /* 0 = orthogonal, 1 = parallel */

  return orthogonality;

}

/* Calculate distance between two vectors */
double lattice_distance(mutation_vector_t *v1, mutation_vector_t *v2) {

  if (!v1 || !v2) { return 1e10; }

  double dist = 0.0;
  double d;

  d = (double)v1->position - (double)v2->position;
  dist += d * d;
  
  d = (double)v1->magnitude - (double)v2->magnitude;
  dist += d * d;
  
  d = (double)v1->dimension - (double)v2->dimension;
  dist += d * d;
  
  d = (double)v1->direction - (double)v2->direction;
  dist += d * d;
  
  d = (double)v1->context - (double)v2->context;
  dist += d * d;
  
  d = (double)v1->frequency - (double)v2->frequency;
  dist += d * d;
  
  d = (double)v1->effectiveness - (double)v2->effectiveness;
  dist += d * d;

  return sqrt(dist);

}

/* Add vector to lattice */
u32 lattice_add_vector(mutation_lattice_t *lattice, mutation_vector_t *vec) {

  if (!lattice || !vec || !lattice->initialized) { return 0; }
  if (lattice->point_count >= lattice->max_points) { return 0; }

  lattice_point_t *point = &lattice->points[lattice->point_count];
  
  point->vector = *vec;
  point->id = lattice->point_count;
  
  /* Set coordinates */
  point->coordinates[0] = (double)vec->position;
  point->coordinates[1] = (double)vec->magnitude;
  point->coordinates[2] = (double)vec->dimension;
  point->coordinates[3] = (double)vec->direction;
  point->coordinates[4] = (double)vec->context;
  point->coordinates[5] = (double)vec->frequency;
  point->coordinates[6] = (double)vec->effectiveness;
  point->coordinates[7] = (double)vec->mut_type;

  point->neighbor_count = 0;
  point->density = 0.0;

  lattice->point_count++;
  return point->id;

}

/* Find nearest neighbor in lattice */
u32 lattice_find_nearest(mutation_lattice_t *lattice, mutation_vector_t *vec) {

  if (!lattice || !vec || lattice->point_count == 0) { return 0; }

  u32    nearest_id = 0;
  double min_dist = 1e10;

  for (u32 i = 0; i < lattice->point_count; ++i) {

    double dist = lattice_distance(vec, &lattice->points[i].vector);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_id = i;
    }

  }

  return nearest_id;

}

/* Initialize MAB */
void mab_init(mutation_mab_t *mab, u32 arm_count) {

  if (!mab) { return; }

  if (arm_count > MAB_MAX_ARMS) { arm_count = MAB_MAX_ARMS; }

  mab->arms = ck_alloc(sizeof(mab_arm_t) * arm_count);
  if (!mab->arms) { PFATAL("Failed to allocate MAB arms"); }

  mab->arm_count = arm_count;
  mab->total_pulls = 0;
  mab->total_reward = 0.0;
  mab->initialized = 1;

  /* Initialize all arms */
  for (u32 i = 0; i < arm_count; ++i) {
    mab->arms[i].arm_id = i;
    mab->arms[i].pulls = 0;
    mab->arms[i].total_reward = 0.0;
    mab->arms[i].avg_reward = 0.0;
    mab->arms[i].ucb_value = 1e10;  /* High initial value for exploration */
    mab->arms[i].variance = 0.0;
    mab->arms[i].last_pull_time = 0;
    mab->arms[i].consecutive_failures = 0;
  }

}

/* Select arm using UCB algorithm */
u32 mab_select_arm_ucb(mutation_mab_t *mab) {

  if (!mab || !mab->initialized || mab->arm_count == 0) { return 0; }

  u32 selected_arm = 0;
  double max_ucb = -1e10;

  for (u32 i = 0; i < mab->arm_count; ++i) {

    mab_arm_t *arm = &mab->arms[i];
    double ucb;

    if (arm->pulls < MAB_MIN_PULLS) {
      /* Exploration phase: prioritize under-explored arms */
      ucb = 1e10 - (double)arm->pulls;
    } else {
      /* UCB formula: avg_reward + C * sqrt(ln(total_pulls) / pulls) */
      double exploration = MAB_EXPLORATION_C * 
                          sqrt(log((double)mab->total_pulls + 1.0) / 
                               ((double)arm->pulls + 1.0));
      ucb = arm->avg_reward + exploration;
    }

    arm->ucb_value = ucb;

    if (ucb > max_ucb) {
      max_ucb = ucb;
      selected_arm = i;
    }

  }

  return selected_arm;

}

/* Select arm using epsilon-greedy */
u32 mab_select_arm_epsilon(mutation_mab_t *mab, double epsilon) {

  if (!mab || !mab->initialized || mab->arm_count == 0) { return 0; }

  /* Epsilon-greedy: with probability epsilon, explore randomly */
  double r = (double)rand() / (double)RAND_MAX;
  
  if (r < epsilon) {
    /* Explore: select random arm */
    return rand() % mab->arm_count;
  } else {
    /* Exploit: select best arm */
    return mab_get_best_arm(mab);
  }

}

/* Update arm reward */
void mab_update_reward(mutation_mab_t *mab, u32 arm_id, double reward) {

  if (!mab || !mab->initialized || arm_id >= mab->arm_count) { return; }

  mab_arm_t *arm = &mab->arms[arm_id];

  arm->pulls++;
  mab->total_pulls++;
  
  arm->total_reward += reward;
  mab->total_reward += reward;

  /* Update average reward using exponential moving average */
  if (arm->pulls == 1) {
    arm->avg_reward = reward;
  } else {
    arm->avg_reward = (1.0 - MAB_ALPHA) * arm->avg_reward + MAB_ALPHA * reward;
  }

  /* Update variance (simplified) */
  double diff = reward - arm->avg_reward;
  arm->variance = (1.0 - MAB_ALPHA) * arm->variance + MAB_ALPHA * diff * diff;

  /* Track failures */
  if (reward < 0.1) {
    arm->consecutive_failures++;
  } else {
    arm->consecutive_failures = 0;
  }

}

/* Get best arm */
u32 mab_get_best_arm(mutation_mab_t *mab) {

  if (!mab || !mab->initialized || mab->arm_count == 0) { return 0; }

  u32    best_arm = 0;
  double max_reward = -1e10;

  for (u32 i = 0; i < mab->arm_count; ++i) {
    if (mab->arms[i].pulls > 0 && mab->arms[i].avg_reward > max_reward) {
      max_reward = mab->arms[i].avg_reward;
      best_arm = i;
    }
  }

  return best_arm;

}

/* Calculate reward from fuzzing outcome */
double calculate_mutation_reward(u8 found_new_path, u8 found_crash, 
                                  u32 exec_time_us, u32 base_time_us) {

  double reward = 0.0;

  /* High reward for finding crashes */
  if (found_crash) {
    reward += 100.0;
  }

  /* Medium reward for finding new paths */
  if (found_new_path) {
    reward += 10.0;
  }

  /* Small penalty for slow execution */
  if (base_time_us > 0 && exec_time_us > base_time_us) {
    double time_penalty = (double)(exec_time_us - base_time_us) / (double)base_time_us;
    reward -= time_penalty * 0.1;
  }

  return reward;

}

/* Main selection function: choose mutation strategy using lattice-MAB */
u32 lattice_mab_select_mutation(struct afl_state *afl, u32 *mutation_array, u32 array_size) {

  if (!afl || !afl->lattice_mab || !mutation_array || array_size == 0) {
    /* Fallback to random selection */
    return rand() % array_size;
  }

  lattice_mab_state_t *lm = afl->lattice_mab;

  if (!lm->use_lattice && !lm->use_mab) {
    /* Both disabled: use random */
    return rand() % array_size;
  }

  u32 selected_index = 0;

  if (lm->use_hybrid && lm->use_lattice && lm->use_mab) {
    /* Hybrid approach: combine lattice and MAB */
    
    /* 70% MAB, 30% lattice-based exploration */
    double r = (double)rand() / (double)RAND_MAX;
    
    if (r < 0.7) {
      /* Use MAB */
      u32 mut_type = mab_select_arm_ucb(&lm->mab);
      if (mut_type < MUT_MAX) {
        /* Find this mutation type in the array */
        for (u32 i = 0; i < array_size; ++i) {
          if (mutation_array[i] == mut_type) {
            selected_index = i;
            lm->mab_selections++;
            break;
          }
        }
      }
    } else {
      /* Use lattice-based selection */
      /* Select a random mutation from array and find similar ones in lattice */
      u32 random_idx = rand() % array_size;
      u32 mut_type = mutation_array[random_idx];
      
      mutation_vector_t vec = mutation_type_to_vector(mut_type, afl);
      u32 nearest_id = lattice_find_nearest(&lm->lattice, &vec);
      
      if (nearest_id < lm->lattice.point_count) {
        u32 nearest_mut_type = lm->lattice.points[nearest_id].vector.mut_type;
        /* Find in array */
        for (u32 i = 0; i < array_size; ++i) {
          if (mutation_array[i] == nearest_mut_type) {
            selected_index = i;
            lm->lattice_selections++;
            break;
          }
        }
      } else {
        selected_index = random_idx;
      }
    }
    
    lm->hybrid_selections++;
    
  } else if (lm->use_mab) {
    /* Pure MAB approach */
    u32 mut_type = mab_select_arm_ucb(&lm->mab);
    if (mut_type < MUT_MAX) {
      for (u32 i = 0; i < array_size; ++i) {
        if (mutation_array[i] == mut_type) {
          selected_index = i;
          lm->mab_selections++;
          break;
        }
      }
    }
  } else if (lm->use_lattice) {
    /* Pure lattice approach */
    u32 random_idx = rand() % array_size;
    u32 mut_type = mutation_array[random_idx];
    
    mutation_vector_t vec = mutation_type_to_vector(mut_type, afl);
    u32 nearest_id = lattice_find_nearest(&lm->lattice, &vec);
    
    if (nearest_id < lm->lattice.point_count) {
      u32 nearest_mut_type = lm->lattice.points[nearest_id].vector.mut_type;
      for (u32 i = 0; i < array_size; ++i) {
        if (mutation_array[i] == nearest_mut_type) {
          selected_index = i;
          lm->lattice_selections++;
          break;
        }
      }
    } else {
      selected_index = random_idx;
    }
  }

  /* Ensure valid index */
  if (selected_index >= array_size) {
    selected_index = rand() % array_size;
  }

  return selected_index;

}

/* Update reward based on fuzzing result */
void lattice_mab_update_reward(struct afl_state *afl, u32 mut_type, u8 found_new_path, 
                                u8 found_crash, u32 exec_time_us) {

  if (!afl || !afl->lattice_mab) { return; }

  lattice_mab_state_t *lm = afl->lattice_mab;

  /* Calculate reward */
  u32 base_time_us = 1000;  /* 1ms baseline */
  double reward = calculate_mutation_reward(found_new_path, found_crash, 
                                            exec_time_us, base_time_us);

  /* Update MAB */
  if (lm->use_mab && mut_type < lm->mab.arm_count) {
    mab_update_reward(&lm->mab, mut_type, reward);
  }

  /* Update lattice vector effectiveness */
  if (lm->use_lattice) {
    mutation_vector_t vec = mutation_type_to_vector(mut_type, afl);
    vec.effectiveness = (u32)(reward * 10.0);  /* Scale to integer */
    
    /* Update or add to lattice */
    u32 nearest_id = lattice_find_nearest(&lm->lattice, &vec);
    if (nearest_id < lm->lattice.point_count) {
      /* Update existing point */
      lm->lattice.points[nearest_id].vector.effectiveness = vec.effectiveness;
      lm->lattice.points[nearest_id].coordinates[6] = (double)vec.effectiveness;
    } else {
      /* Add new point */
      lattice_add_vector(&lm->lattice, &vec);
    }
  }

  /* Update statistics */
  lm->avg_reward = (lm->avg_reward * 0.99) + (reward * 0.01);
  if (reward > lm->best_reward) {
    lm->best_reward = reward;
    lm->best_arm_id = mut_type;
  }

}

/* Build initial lattice from mutation arrays */
void lattice_build_from_arrays(mutation_lattice_t *lattice, u32 *array, u32 size) {

  if (!lattice || !array || size == 0) { return; }

  for (u32 i = 0; i < size && lattice->point_count < lattice->max_points; ++i) {
    mutation_vector_t vec;
    memset(&vec, 0, sizeof(mutation_vector_t));
    vec.mut_type = array[i];
    vec.frequency = 1;
    vec.effectiveness = 50;  /* Initial neutral effectiveness */
    
    lattice_add_vector(lattice, &vec);
  }

}

/* Get lattice statistics */
void lattice_get_stats(mutation_lattice_t *lattice, u32 *point_count, 
                       double *avg_density, double *max_density) {

  if (!lattice) { return; }

  if (point_count) { *point_count = lattice->point_count; }
  if (avg_density) { *avg_density = 0.0; }
  if (max_density) { *max_density = 0.0; }

  if (lattice->point_count == 0) { return; }

  double total_density = 0.0;
  double max = 0.0;

  for (u32 i = 0; i < lattice->point_count; ++i) {
    total_density += lattice->points[i].density;
    if (lattice->points[i].density > max) {
      max = lattice->points[i].density;
    }
  }

  if (avg_density) { *avg_density = total_density / (double)lattice->point_count; }
  if (max_density) { *max_density = max; }

}

/* Get MAB statistics */
void mab_get_stats(mutation_mab_t *mab, u32 *total_pulls, double *avg_reward, 
                   u32 *best_arm) {

  if (!mab) { return; }

  if (total_pulls) { *total_pulls = mab->total_pulls; }
  if (avg_reward) { 
    *avg_reward = (mab->total_pulls > 0) ? 
                  (mab->total_reward / (double)mab->total_pulls) : 0.0;
  }
  if (best_arm) { *best_arm = mab_get_best_arm(mab); }

}

