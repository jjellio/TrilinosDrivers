#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

// from nersc
// http://www.nersc.gov/assets/Uploads/affinity.c.txt

/* Borrowed from util-linux-2.13-pre7/schedutils/taskset.c */
static char *cpuset_to_cstr(cpu_set_t *mask, char *str)
{
 char *ptr = str;
 int i, j, entry_made = 0;
 for (i = 0; i < CPU_SETSIZE; i++) {
 if (CPU_ISSET(i, mask)) {
 int run = 0;
 entry_made = 1;
 for (j = i + 1; j < CPU_SETSIZE; j++) {
 if (CPU_ISSET(j, mask)) run++;
 else break;
 }
 if (!run)
 sprintf(ptr, "%d,", i);
 else if (run == 1) {
 sprintf(ptr, "%d,%d,", i, i + 1);
 i++;
 } else {
 sprintf(ptr, "%d-%d,", i, i + run);
 i += run;
 }
 while (*ptr != 0) ptr++;
 }
 }
 ptr -= entry_made;
 *ptr = 0;
 return(str);
}
void report_num_threads(int level)
{
 #pragma omp single
 {
 printf("Level %d: number of threads in the team - %d\n", level, omp_get_num_threads());
 }
 }
/* Based on Cray's original xthi.c */
int main(int argc, char *argv[])
{
 int rank, thread1, thread2, requested, provided;
 cpu_set_t coremask1, coremask2;
 char clbuf1[7 * CPU_SETSIZE], hnbuf1[64];
 char clbuf2[7 * CPU_SETSIZE], hnbuf2[64];
 MPI_Init_thread(&argc, &argv, 3, &provided);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 memset(clbuf1, 0, sizeof(clbuf1));
 memset(clbuf2, 0, sizeof(clbuf2));
 memset(hnbuf1, 0, sizeof(hnbuf1));
 memset(hnbuf2, 0, sizeof(hnbuf2));
 (void)gethostname(hnbuf1, sizeof(hnbuf1));
 (void)gethostname(hnbuf2, sizeof(hnbuf2));
 #pragma omp parallel private(thread1, coremask1, clbuf1)
 {
 thread1 = omp_get_thread_num();
 (void)sched_getaffinity(0, sizeof(coremask1), &coremask1);
 cpuset_to_cstr(&coremask1, clbuf1);
 #pragma omp barrier
 printf("Level 1: rank= %d, thread level 1= %3d, on %s. (core affinity = %s)\n",
 rank, thread1, hnbuf1, clbuf1);
 }
 MPI_Finalize();
 return(0);
}

