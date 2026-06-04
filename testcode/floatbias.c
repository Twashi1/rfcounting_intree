#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;

  int x = strtol(argv[1], NULL, 10);
  int y = x + 1;
  int z = (x + 3) << 3;
  int w = z * 2;

  float x0 = (float)x;
  float x1 = x0 * 1.5f;
  float x2 = powf(x0, 2.3f);
  float x3 = x1 - x2;

  printf("[FloatBias program] Results: %i %i %f\n", w, y, x3);

  return 0;
}
