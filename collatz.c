#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define STUDENT_ID_LAST4 56
#define BASE_N 10000000UL
#define MOD_CHECKSUM 1000000007UL

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

int main(void) {
    uint64_t N = BASE_N + (STUDENT_ID_LAST4 * 1000UL);
    printf("=== OpenMP Amdahl Reality Gap Lab ===\n");
    printf("Student ID Last 4: %04d | Target Workload N: %lu\n\n", STUDENT_ID_LAST4, N);

    // 1. Sequential Baseline
    printf("--- Phase 2: Sequential Baseline ---\n");
    uint32_t max_steps_seq = 0;
    uint64_t sum_steps_seq = 0;

    double t_start = omp_get_wtime();
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t steps = collatz_steps(i);
        if (steps > max_steps_seq) max_steps_seq = steps;
        sum_steps_seq = (sum_steps_seq + steps) % MOD_CHECKSUM;
    }
    double t_seq = omp_get_wtime() - t_start;
    printf("Sequential Time (T_seq): %.4f sec | Max Steps: %u | Checksum: %lu\n\n", 
           t_seq, max_steps_seq, sum_steps_seq);

    // 2. Parallel Scaling Benchmark
    printf("--- Phase 3: Parallel Scaling Benchmark ---\n");
    int threads_to_test[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(threads_to_test) / sizeof(threads_to_test[0]);

    for (int t = 0; t < num_tests; t++) {
        int k = threads_to_test[t];
        if (k > omp_get_max_threads()) break;

        double run_times[3];
        for (int r = 0; r < 3; r++) {
            double start = omp_get_wtime();
            uint32_t max_s = 0;
            #pragma omp parallel for num_threads(k) reduction(max:max_s)
            for (uint64_t i = 1; i <= N; i++) {
                uint32_t steps = collatz_steps(i);
                if (steps > max_s) max_s = steps;
            }
            run_times[r] = omp_get_wtime() - start;
        }
        double avg_t = (run_times[1] + run_times[2]) / 2.0;
        printf("Threads k=%2d | Run1: %.4fs | Run2: %.4fs | Run3: %.4fs | Avg T_k: %.4fs\n",
               k, run_times[0], run_times[1], run_times[2], avg_t);
    }

    // 3. Phase 4 Exp A: False Sharing
    printf("\n--- Phase 4: Experiment A (False Sharing) ---\n");
    int max_threads = omp_get_max_threads();
    
    int *naive_hits = (int*)calloc(max_threads, sizeof(int));
    double t_fs_start = omp_get_wtime();
    #pragma omp parallel for num_threads(max_threads)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > 100) {
            naive_hits[omp_get_thread_num()]++;
        }
    }
    double t_fs_naive = omp_get_wtime() - t_fs_start;
    free(naive_hits);

    int total_hits = 0;
    double t_red_start = omp_get_wtime();
    #pragma omp parallel for num_threads(max_threads) reduction(+:total_hits)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > 100) {
            total_hits++;
        }
    }
    double t_fs_reduction = omp_get_wtime() - t_red_start;

    printf("Naive (False Sharing) Time : %.4f sec\n", t_fs_naive);
    printf("Reduction (Mitigated) Time  : %.4f sec\n", t_fs_reduction);
    printf("Speedup Penalty Ratio       : %.2fx slower\n\n", t_fs_naive / t_fs_reduction);

    // 4. Phase 4 Exp B: Loop Scheduling
    printf("--- Phase 4: Experiment B (Loop Scheduling) ---\n");
    uint32_t dummy = 0;
    double s_time, e_time;

    // Static default
    s_time = omp_get_wtime();
    #pragma omp parallel for schedule(static) num_threads(max_threads) reduction(max:dummy)
    for (uint64_t i = 1; i <= N; i++) { uint32_t s = collatz_steps(i); if(s>dummy) dummy=s; }
    e_time = omp_get_wtime() - s_time;
    printf("Schedule: static (default)          | Time: %.4f sec\n", e_time);

    // Static 1000
    s_time = omp_get_wtime();
    #pragma omp parallel for schedule(static, 1000) num_threads(max_threads) reduction(max:dummy)
    for (uint64_t i = 1; i <= N; i++) { uint32_t s = collatz_steps(i); if(s>dummy) dummy=s; }
    e_time = omp_get_wtime() - s_time;
    printf("Schedule: static, 1000             | Time: %.4f sec\n", e_time);

    // Dynamic 100
    s_time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 100) num_threads(max_threads) reduction(max:dummy)
    for (uint64_t i = 1; i <= N; i++) { uint32_t s = collatz_steps(i); if(s>dummy) dummy=s; }
    e_time = omp_get_wtime() - s_time;
    printf("Schedule: dynamic, 100            | Time: %.4f sec\n", e_time);

    // Dynamic 10000
    s_time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 10000) num_threads(max_threads) reduction(max:dummy)
    for (uint64_t i = 1; i <= N; i++) { uint32_t s = collatz_steps(i); if(s>dummy) dummy=s; }
    e_time = omp_get_wtime() - s_time;
    printf("Schedule: dynamic, 10000          | Time: %.4f sec\n", e_time);

    // Guided
    s_time = omp_get_wtime();
    #pragma omp parallel for schedule(guided) num_threads(max_threads) reduction(max:dummy)
    for (uint64_t i = 1; i <= N; i++) { uint32_t s = collatz_steps(i); if(s>dummy) dummy=s; }
    e_time = omp_get_wtime() - s_time;
    printf("Schedule: guided                   | Time: %.4f sec\n", e_time);

    return 0;
}