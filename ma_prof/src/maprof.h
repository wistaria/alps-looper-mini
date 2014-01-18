#ifndef MAPROF_H
#define MAPROF_H

/* ---------- measuring ---------- */

typedef enum {
  MAPROF_ROOT,   /**< rank0 */
  MAPROF_AVE,    /**< average */
  MAPROF_MIN,    /**< minimum */
  MAPROF_MAX,    /**< maximum */
  MAPROF_SD,     /**< standard deviation */
} maprof_stat_type;

#ifdef __cplusplus
extern "C" {
#endif

void maprof_time_start(int id);

void maprof_time_stop(int id);

void maprof_add_fp_ops(int id, double ops);

void maprof_add_ld_ops(int id, double ops);

void maprof_add_st_ops(int id, double ops);

void maprof_add_ld_min_ops(int id, double ops);

void maprof_add_st_min_ops(int id, double ops);

double maprof_get_time(int id, maprof_stat_type type);

double maprof_get_flops(int id, maprof_stat_type type);

double maprof_get_throughput(int id, maprof_stat_type type);

double maprof_get_effective_throughput(int id, maprof_stat_type type);

void maprof_print(int id, const char *name);

void maprof_print_time(int id, const char *name);

void maprof_print_time_mpi(int id, const char *name);

void maprof_print_time_mpi_full(int id, const char *name);


/* ---------- reporting ---------- */

void maprof_setup(const char *app_name, const char *app_version);

void maprof_output();

void maprof_app_add_str(const char *key, const char *str);

void maprof_app_add_int(const char *key, int n);

void maprof_app_add_float(const char *key, double r);

void maprof_add_section(const char *name, int id);

void maprof_profile_add_problem_size(const char *key, int n);

void maprof_profile_add_str(const char *key, const char *str);

void maprof_profile_add_int(const char *key, int n);

void maprof_profile_add_float(const char *key, double r);


/* ---------- for fortran interface ---------- */

void maprof_set_num_threads(int n);

#ifdef __cplusplus
}
#endif

#endif /* MAPROF_H */
