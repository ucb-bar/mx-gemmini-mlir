#include <stdint.h>
#include <stdio.h>
#define ELEMENTS 133120
#define LHS_ELEMENTS 133120
#define RHS_ELEMENTS 4160
#define RANK 4
#define LHS_RANK 4
#define RHS_RANK 4
void mx_model_elementwise(const float *, const float *, float *);
static float a[LHS_ELEMENTS] __attribute__((aligned(64)));
static float b[RHS_ELEMENTS] __attribute__((aligned(64)));
static float out[ELEMENTS] __attribute__((aligned(64)));
static const unsigned dims[RANK] = {1, 32, 64, 65};
static const unsigned lhs_dims[LHS_RANK ? LHS_RANK : 1] = {1, 32, 64, 65};
static const unsigned lhs_axes[LHS_RANK ? LHS_RANK : 1] = {0, 1, 2, 3};
static const unsigned rhs_dims[RHS_RANK ? RHS_RANK : 1] = {1, 1, 64, 65};
static const unsigned rhs_axes[RHS_RANK ? RHS_RANK : 1] = {0, 1, 2, 3};
static uint32_t bits(float value) { union {float f; uint32_t u;} w; w.f=value; return w.u; }
static float widen(uint16_t value) {union {float f; uint32_t u;} w; w.u=(uint32_t)value<<16; return w.f;}
static uint16_t narrow(float value) {uint32_t u=bits(value); u+=0x7fffu+((u>>16)&1u); return u>>16;}
static unsigned reference_lhs_index(unsigned index) {
  unsigned coords[RANK], remainder = index;
  for (int axis=RANK-1; axis>=0; --axis) {
    coords[axis] = remainder % dims[axis];
    remainder /= dims[axis];
  }
  unsigned flat = 0;
  for (unsigned axis=0; axis<LHS_RANK; ++axis) {
    unsigned coordinate = lhs_dims[axis] == 1 ? 0 : coords[lhs_axes[axis]];
    flat = flat * lhs_dims[axis] + coordinate;
  }
  return flat;
}
static unsigned reference_rhs_index(unsigned index) {
  unsigned coords[RANK], remainder = index;
  for (int axis=RANK-1; axis>=0; --axis) {
    coords[axis] = remainder % dims[axis];
    remainder /= dims[axis];
  }
  unsigned flat = 0;
  for (unsigned axis=0; axis<RHS_RANK; ++axis) {
    unsigned coordinate = rhs_dims[axis] == 1 ? 0 : coords[rhs_axes[axis]];
    flat = flat * rhs_dims[axis] + coordinate;
  }
  return flat;
}
int main(void) {
  for (unsigned index=0; index<LHS_ELEMENTS; ++index) {
    float av=((int)((index*17u)%257u)-128)/64.0f;
    a[index] = av;
  }
  for (unsigned index=0; index<RHS_ELEMENTS; ++index) {
    float bv=(index % 17u == 0u ? -__builtin_inff() : ((int)((index*11u)%197u)-98)/64.0f);
    b[index] = bv;
  }
  mx_model_elementwise(a,b,out);
  unsigned mismatches=0, max_relative_ppm=0;
  for (unsigned index=0; index<ELEMENTS; ++index) {
    unsigned a_index=reference_lhs_index(index);
    unsigned b_index=reference_rhs_index(index);
    uint32_t want=(uint32_t)narrow(widen(narrow(a[a_index])) + widen(narrow(b[b_index])))<<16;
    mismatches += bits(out[index]) != want;
  }
  printf("VPU ADD %s mismatches=%u\n", mismatches ? "FAIL" : "PASS", mismatches);
  return mismatches ? 3 : 0;
}
