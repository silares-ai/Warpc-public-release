// cublas_q7: cublas_q3 (cublas_bench with CQ_GAP, CQ_NOBLOCK, CQ_WARM) and CQ_ROUNDS=R, R rounds of
// [iters] calls back to back with an event between rounds and no wait, each round's time a call printed.
// Built on the node: g++ -O2 -o cublas_q7 cublas_q7.cpp -I$P/cu13/include -L$P/cu13/lib
// -l:libcublasLt.so.13 -l:libcudart.so.13 -ldl, P the pip nvidia directory.
// CQ_GAP=S idles S seconds before each window; CQ_NOBLOCK=1 leaves the blocker out; CQ_WARM=MS runs the
// algorithm back to back for about MS milliseconds before its timed calls, not waited for (the steady state).
// cublas_q: cublas_bench with each timed window queued behind a 16384-cubed
// GEMM on buffers of its own, so the timed calls run back to back on the
// device whatever launching them costs; prints microseconds a call.
// cuBLASLt on the same problem as warpc: C[M,N] f32 row-major = A[M,K] f16
// row-major x Bt[N,K]^T f16 row-major, alpha 1, beta 0. Every algorithm the
// heuristic returns is timed between CUDA events; each result is checked
// exactly on sampled entries (integer inputs in [-3,3]).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define BK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

static int val(uint64_t i, uint64_t salt) { return (int)(((i * 2654435761ull + salt) >> 13) % 7) - 3; }

int main(int argc, char** argv) {
  int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  int iters = argc > 4 ? atoi(argv[4]) : 50;
  int only = argc > 5 ? atoi(argv[5]) : -1;
  std::vector<__half> ha((size_t)M * K), hb((size_t)N * K);
  for (size_t i = 0; i < ha.size(); i++) ha[i] = __int2half_rn(val(i, 1));
  for (size_t i = 0; i < hb.size(); i++) hb[i] = __int2half_rn(val(i, 7));
  __half *a, *bt;
  float* c;
  CK(cudaMalloc(&a, ha.size() * 2));
  CK(cudaMalloc(&bt, hb.size() * 2));
  CK(cudaMalloc(&c, (size_t)M * N * 4));
  CK(cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(bt, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));

  cublasLtHandle_t lt;
  BK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  BK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tT = CUBLAS_OP_T, tN = CUBLAS_OP_N;
  BK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tT, sizeof tT));
  BK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof tN));
  // column-major view: C^T[N,M] = (Bt as K x N)^T (A as K x M)
  cublasLtMatrixLayout_t la, lb, lc;
  BK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K));
  BK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
  BK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
  size_t ws = 256u << 20;
  void* wsp;
  CK(cudaMalloc(&wsp, ws));
  cublasLtMatmulPreference_t pref;
  BK(cublasLtMatmulPreferenceCreate(&pref));
  BK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t res[32];
  int n = 0;
  BK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 32, res, &n));
  float alpha = 1, beta = 0;
  // the blocker: 16384 cubed, the first algorithm the heuristic returns
  const int Q = 16384;
  __half *qa, *qb; float* qc;
  CK(cudaMalloc(&qa, (size_t)Q * Q * 2)); CK(cudaMalloc(&qb, (size_t)Q * Q * 2)); CK(cudaMalloc(&qc, (size_t)Q * Q * 4));
  CK(cudaMemset(qa, 0, (size_t)Q * Q * 2)); CK(cudaMemset(qb, 0, (size_t)Q * Q * 2));
  cublasLtMatrixLayout_t qla, qlb, qlc;
  BK(cublasLtMatrixLayoutCreate(&qla, CUDA_R_16F, Q, Q, Q));
  BK(cublasLtMatrixLayoutCreate(&qlb, CUDA_R_16F, Q, Q, Q));
  BK(cublasLtMatrixLayoutCreate(&qlc, CUDA_R_32F, Q, Q, Q));
  cublasLtMatmulHeuristicResult_t qres[1]; int qn = 0;
  BK(cublasLtMatmulAlgoGetHeuristic(lt, op, qla, qlb, qlc, qlc, pref, 1, qres, &qn));
  auto block = [&] { return cublasLtMatmul(lt, op, &alpha, qb, qla, qa, qlb, &beta, qc, qlc, qc, qlc, &qres[0].algo, wsp, ws, 0); };
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  double flop = 2.0 * M * N * K;
  double best = 0;
  int best_j = -1;
  std::vector<float> hc((size_t)M * N);
  for (int j = 0; j < n; j++) {
    if (only >= 0 && j != only) continue;
    auto go = [&] { return cublasLtMatmul(lt, op, &alpha, bt, la, a, lb, &beta, c, lc, c, lc, &res[j].algo, wsp, ws, 0); };
    CK(cudaMemset(c, 0, (size_t)M * N * 4));
    if (go() != CUBLAS_STATUS_SUCCESS) { printf("algo %d: launch failed\n", j); continue; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hc.data(), c, hc.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int s = 0; s < 64; s++) {
      int i = (int)((s * 7919ull + 13) % M), jj = (int)((s * 104729ull + 5) % N);
      long ref = 0;
      for (int k = 0; k < K; k++) ref += (long)val((size_t)i * K + k, 1) * val((size_t)jj * K + k, 7);
      if ((double)hc[(size_t)i * N + jj] != (double)ref) bad++;
    }
    for (int w = 0; w < 3; w++) go();
    if (getenv("CQ_GAP")) { CK(cudaDeviceSynchronize()); usleep((useconds_t)(atof(getenv("CQ_GAP")) * 1e6)); }
    if (!getenv("CQ_NOBLOCK")) BK(block());
    if (getenv("CQ_WARM")) {
      float est; cudaEventRecord(e0); for (int w = 0; w < 10; w++) go(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
      cudaEventElapsedTime(&est, e0, e1); est /= 10;
      int nw = (int)(atof(getenv("CQ_WARM")) / est); if (nw < 1) nw = 1;
      for (int w = 0; w < nw; w++) go();
    }
    // CQ_ROUNDS=R: R rounds of [iters] calls back to back, an event between
    // rounds and no wait, each round's time a call printed in order
    if (getenv("CQ_ROUNDS")) {
      int R = atoi(getenv("CQ_ROUNDS"));
      std::vector<cudaEvent_t> ev(R + 1);
      for (auto& x : ev) cudaEventCreate(&x);
      cudaEventRecord(ev[0]);
      for (int q = 0; q < R; q++) { for (int it = 0; it < iters; it++) go(); cudaEventRecord(ev[q + 1]); }
      CK(cudaEventSynchronize(ev[R]));
      for (int q = 0; q < R; q++) { float m; cudaEventElapsedTime(&m, ev[q], ev[q + 1]); printf("round %d %.2f us\n", q, 1000.0 * m / iters); }
    }
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++) go();
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    double tf = flop / (ms / iters) / 1e9;
    printf("algo %2d: %8.1f TFLOP/s  %9.2f us  %s\n", j, tf, 1000.0 * ms / iters, bad ? "WRONG" : "exact");
    if (!bad && tf > best) { best = tf; best_j = j; }
  }
  printf("%dx%dx%d cublasLt default %s, best %.1f TFLOP/s (algo %d of %d)\n", M, N, K, n ? "algo 0" : "none", best, best_j, n);
  return 0;
}
