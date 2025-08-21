#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "thrust_wrappers.h"

void run_exclusive_scan(unsigned int* input, unsigned int* output, int count) {
    thrust::exclusive_scan(thrust::device, input, input + count, output);
}

unsigned int run_reduce(unsigned int* input, int count) {
    return thrust::reduce(thrust::device, input, input + count, 0u);
}
