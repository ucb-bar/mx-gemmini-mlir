#include <stdint.h>
#include <stdio.h>
#define DIM 16
#define BANK_ROWS 4096
#define SPIKE_SIM 1
#define printf(...) ((void)0)
#include "baremetal/mxgemmini/include/mx_native.h"
void mx_issue_0(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,32,(const uint8_t *)b,368,(uint16_t *)c,736,(const uint8_t *)as,64,(const uint8_t *)bs,736,64,32,736)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_1(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,1440,(const uint8_t *)b,368,(uint16_t *)c,736,(const uint8_t *)as,64,(const uint8_t *)bs,736,64,1440,736)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_2(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,736,(const uint8_t *)b,368,(uint16_t *)c,736,(const uint8_t *)as,64,(const uint8_t *)bs,736,64,736,736)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_3(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,736,(const uint8_t *)b,480,(uint16_t *)c,960,(const uint8_t *)as,64,(const uint8_t *)bs,960,64,736,960)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_4(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,736,(const uint8_t *)b,160,(uint16_t *)c,320,(const uint8_t *)as,64,(const uint8_t *)bs,320,64,736,320)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_5(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,960,(const uint8_t *)b,368,(uint16_t *)c,736,(const uint8_t *)as,64,(const uint8_t *)bs,736,64,960,736)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_6(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,736,(const uint8_t *)b,1024,(uint16_t *)c,2048,(const uint8_t *)as,64,(const uint8_t *)bs,2048,64,736,2048)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_7(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,2048,(const uint8_t *)b,368,(uint16_t *)c,736,(const uint8_t *)as,64,(const uint8_t *)bs,736,64,2048,736)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_8(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,320,(const uint8_t *)b,160,(uint16_t *)c,320,(const uint8_t *)as,128,(const uint8_t *)bs,320,128,320,320)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_9(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_fp4((const uint8_t *)a,736,(const uint8_t *)b,16,(uint16_t *)c,32,(const uint8_t *)as,64,(const uint8_t *)bs,32,64,736,32)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_10(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_ex((const uint8_t *)a,64,(const uint8_t *)b,176,(uint16_t *)c,176,(const uint8_t *)as,64,(const uint8_t *)bs,176,64,64,176,0,0)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_11(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_ex((const uint8_t *)a,192,(const uint8_t *)b,64,(uint16_t *)c,64,(const uint8_t *)as,64,(const uint8_t *)bs,64,64,192,64,0,0)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_12(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_ex((const uint8_t *)a,64,(const uint8_t *)b,128,(uint16_t *)c,128,(const uint8_t *)as,64,(const uint8_t *)bs,128,64,64,128,0,0)) __builtin_trap();
  gemmini_fence();
}
void mx_issue_13(const void *a,const void *as,const void *b,const void *bs,const void *c,const void *ss) {
  (void)ss;
  if (mxn_matmul_ex((const uint8_t *)a,128,(const uint8_t *)b,64,(uint16_t *)c,64,(const uint8_t *)as,64,(const uint8_t *)bs,64,64,128,64,0,0)) __builtin_trap();
  gemmini_fence();
}
