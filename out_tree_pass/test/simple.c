#include <stdio.h>

int main() {
    int sum = 0;
    int n = 0;

    printf("Enter loop count: ");
    if (scanf("%d", &n) != 1) return 1;

    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            sum += i;
            printf("Even: %d, sum=%d\n", i, sum);
        } else {
            sum -= i;
            printf("Odd: %d, sum=%d\n", i, sum);
        }
    }

    if (sum > 0) {
        printf("Final sum is positive: %d\n", sum);
    } else if (sum < 0) {
        printf("Final sum is negative: %d\n", sum);
    } else {
        printf("Final sum is zero.\n");
    }

    return 0;
}
