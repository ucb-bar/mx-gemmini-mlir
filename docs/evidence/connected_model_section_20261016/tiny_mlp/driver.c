#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void run_chain(const float *,const float *,const float *,float *,float *);
static float middle[360448] __attribute__((aligned(64)));
static float output[360448] __attribute__((aligned(64)));
static uint64_t cycle(void) { uint64_t n;
  asm volatile("rdcycle %0" : "=r"(n) :: "memory"); return n; }
int main(void) {
  const float *a=(const float *)payload, *b=a+360448, *c=b+360448;
  const uint32_t *golden=(const uint32_t *)(c+360448);
  for(unsigned run=0;run<3;++run) {
    uint64_t start=cycle(); run_chain(a,b,c,middle,output);
    uint64_t elapsed=cycle()-start;
    unsigned mismatches=0; uint32_t hash=2166136261u;
    for(unsigned i=0;i<360448;++i) {
      union {float f;uint32_t u;} word; word.f=output[i];
      mismatches+=word.u!=golden[i]; hash=(hash^word.u)*16777619u;
    }
    printf("SECTION run=%u cycles=%lu mismatches=%u hash=%u\n",run,elapsed,mismatches,hash);
    if(mismatches) return 3;
  }
  return 0;
}
