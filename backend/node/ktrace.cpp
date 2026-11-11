// ktrace: a CUPTI injection library (CUDA_INJECTION64_PATH=libktrace.so) recording every kernel's start
// and end on the GPU's clock (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, which does not serialize kernels).
// At exit it prints, over the last KT_LAST kernels (default 200) of the most frequent name: the median
// kernel duration, and the gap from each kernel's end to the next one's start -- negative when the next
// kernel began before the one ahead of it ended.
// Built: g++ -O2 -std=c++17 -shared -fPIC -DKT=<newest CUpti_ActivityKernel struct> -o libktrace.so ktrace.cpp
//        -I$P/cu13/include -L$P/cu13/lib -l:libcupti.so.13
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cupti.h>

struct K { uint64_t start, end; std::string name; };
static std::vector<K> ks;

static void CUPTIAPI requested(uint8_t** buf, size_t* size, size_t* max) {
  *size = 8 << 20; *buf = (uint8_t*)aligned_alloc(8, *size); *max = 0;
}
static void CUPTIAPI completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS)
    if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto* k = (KT*)r;
      ks.push_back({k->start, k->end, k->name ? k->name : "?"});
    }
  free(buf);
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
static void report() {
  cuptiActivityFlushAll(1);
  std::sort(ks.begin(), ks.end(), [](const K& a, const K& b) { return a.start < b.start; });
  std::map<std::string, int> count;
  for (auto& k : ks) count[k.name]++;
  std::string top; int most = 0;
  for (auto& [n, c] : count) if (c > most) { most = c; top = n; }
  printf("ktrace: %zu kernels, %zu names\n", ks.size(), count.size());
  for (auto& [n, c] : count) printf("ktrace:   %6d  %.90s\n", c, n.c_str());
  std::vector<K> v;
  for (auto& k : ks) if (k.name == top) v.push_back(k);
  size_t last = getenv("KT_LAST") ? atoi(getenv("KT_LAST")) : 200;
  if (v.size() > last) v.erase(v.begin(), v.end() - last);
  std::vector<double> dur, gap;
  int overlap = 0;
  for (size_t i = 0; i < v.size(); i++) {
    dur.push_back((v[i].end - v[i].start) / 1e3);
    if (i + 1 < v.size()) {
      double g = ((double)v[i + 1].start - (double)v[i].end) / 1e3;
      gap.push_back(g); if (g < 0) overlap++;
    }
  }
  double span = v.size() > 1 ? (v.back().start - v.front().start) / 1e3 / (v.size() - 1) : 0;
  printf("ktrace: last %zu of the most frequent: duration median %.2f us, start to start %.2f us, gap end to next start median %.2f us, min %.2f, max %.2f; %d of %zu next kernels began before the one ahead ended\n",
         v.size(), median(dur), span, median(gap), gap.empty() ? 0 : *std::min_element(gap.begin(), gap.end()),
         gap.empty() ? 0 : *std::max_element(gap.begin(), gap.end()), overlap, gap.size());
  // every kernel in start order, whatever its name: where a call is several kernels, each kernel
  // against the one that began before it
  std::vector<K> w(ks.end() - std::min(ks.size(), last * count.size()), ks.end());
  std::vector<double> g2;
  int o2 = 0;
  for (size_t i = 0; i + 1 < w.size(); i++) {
    double g = ((double)w[i + 1].start - (double)w[i].end) / 1e3;
    g2.push_back(g); if (g < 0) o2++;
  }
  printf("ktrace: last %zu of every name, in start order: gap end to next start median %.2f us; %d of %zu next kernels began before the one ahead ended\n",
         w.size(), median(g2), o2, g2.size());
  // a call that runs each name once, each kernel alone (CUDA_LAUNCH_BLOCKING=1): its GPU time is the
  // sum of the names' median durations over their last KT_LAST kernels
  double call = 0;
  for (auto& [n, c] : count) {
    std::vector<double> d;
    for (auto& k : ks) if (k.name == n) d.push_back((k.end - k.start) / 1e3);
    if (d.size() > last) d.erase(d.begin(), d.end() - last);
    call += median(d);
  }
  printf("ktrace: a call, each name once: %.2f us of GPU time\n", call);
  fflush(stdout);
}
extern "C" int InitializeInjection(void) {
  cuptiActivityRegisterCallbacks(requested, completed);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  atexit(report);
  return 1;
}
