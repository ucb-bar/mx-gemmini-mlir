/* Full output oracle for the canonical streamed 64x2048x64 flash wrapper. */
#define main source_main
#include "attn_flash_llama.c"
#undef main

int main(void) {
  if (source_main()) return 1;
  uint64_t hash = fnv(O_hw, sizeof(O_hw));
  printf("O hash %016llx\n", (unsigned long long)hash);
  printf("O BF16 words ");
  for (int i = 0; i < SQ * D; i++)
    printf("%04x", ((uint16_t *)O_hw)[i]);
  printf("\n");
  return hash != EXP_FLASH_HASH_O;
}
