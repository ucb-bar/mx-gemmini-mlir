/* Generated multi-program stage adapter. */
#include <stddef.h>
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"
extern void merlin_invoke(void **);
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];
static size_t arg_bytes(const merlin_arg_t *arg) {
  size_t n = (size_t)arg->elem_size;
  for (int i = 0; i < arg->rank; i++) n *= (size_t)arg->dims[i];
  return n;
}
void merlin_stage_0_reset(void) { merlin_reset_session(); }
int merlin_stage_0_run(const void *weights, long step, int validate) {
  merlin_prepare_step(step);
  merlin_run_multi_with(MERLIN_ARGS, MERLIN_N_ARGS, weights, MERLIN_INPUT_PTR,
                        MERLIN_OUTPUT_PTR, DESCS, merlin_invoke);
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR,
                          MERLIN_N_STATE_PAIRS, MERLIN_STATE_INPUT_ARGS,
                          MERLIN_STATE_OUTPUT_INDICES) != 0) return -1;
#endif
  if (validate) merlin_validate_step(step);
  return 0;
}
void *merlin_stage_0_input(int arg) {
  return arg >= 0 && arg < MERLIN_N_ARGS && MERLIN_ARGS[arg].kind == MERLIN_INPUT
      ? MERLIN_INPUT_PTR[arg] : 0;
}
size_t merlin_stage_0_input_bytes(int arg) {
  return arg >= 0 && arg < MERLIN_N_ARGS && MERLIN_ARGS[arg].kind == MERLIN_INPUT
      ? arg_bytes(&MERLIN_ARGS[arg]) : 0;
}
static int output_arg(int wanted) {
  int seen = 0;
  for (int arg = 0; arg < MERLIN_N_ARGS; arg++)
    if (MERLIN_ARGS[arg].kind == MERLIN_OUTPUT && seen++ == wanted) return arg;
  return -1;
}
void *merlin_stage_0_output(int output) {
  return output >= 0 && output < MERLIN_N_OUTPUTS ? MERLIN_OUTPUT_PTR[output] : 0;
}
size_t merlin_stage_0_output_bytes(int output) {
  int arg = output_arg(output); return arg >= 0 ? arg_bytes(&MERLIN_ARGS[arg]) : 0;
}
long merlin_stage_0_quality_steps(void) { return merlin_quality_steps(); }
long merlin_stage_0_quality_min_cos_ppm(void) { return merlin_quality_min_cos_ppm(); }
long merlin_stage_0_quality_max_rel_ppm(void) { return merlin_quality_max_rel_ppm(); }
long merlin_stage_0_quality_top1(void) { return merlin_quality_top1(); }
long merlin_stage_0_correctness_steps(void) { return merlin_correctness_steps(); }
long merlin_stage_0_correctness_min_cos_ppm(void) { return merlin_correctness_min_cos_ppm(); }
long merlin_stage_0_correctness_max_rel_ppm(void) { return merlin_correctness_max_rel_ppm(); }
long merlin_stage_0_correctness_top1(void) { return merlin_correctness_top1(); }
size_t merlin_stage_0_output_elems(int output) {
  int arg = output_arg(output); if (arg < 0) return 0;
  size_t n = 1; for (int i = 0; i < MERLIN_ARGS[arg].rank; i++) n *= MERLIN_ARGS[arg].dims[i];
  return n;
}
long merlin_stage_0_output_lastdim(int output) {
  int arg = output_arg(output); return arg >= 0 && MERLIN_ARGS[arg].rank > 0
      ? MERLIN_ARGS[arg].dims[MERLIN_ARGS[arg].rank - 1] : 1;
}
