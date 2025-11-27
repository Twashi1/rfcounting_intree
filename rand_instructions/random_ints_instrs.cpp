#include <cstdint>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>

#if defined(_MSC_VER)
    #define NOINLINE __declspec(noinline)
#else
    #define NOINLINE __attribute__((noinline))
#endif

// volatile to prevent full optimisations
volatile std::uint64_t g_acc0 = 0;
volatile std::uint64_t g_acc1 = 0;
volatile std::int64_t g_signed = 0;

NOINLINE void save_values_codegen(volatile std::uint64_t x, volatile std::uint64_t y, volatile std::int64_t z) {
    g_acc0 ^= x;
    g_acc1 += y;
    g_signed += z;
}

NOINLINE void block_wrap(volatile std::uint64_t& x, volatile std::uint64_t& y, volatile std::int64_t& z) {
    	volatile std::uint64_t s = static_cast<std::uint64_t>(x + 0x9E67'3A22ULL);
	volatile std::uint64_t t = static_cast<std::uint64_t>(y ^ 0x1234'5678ULL);
	volatile std::int64_t u = static_cast<std::int64_t>(z - 0x1938FLL);
    // block 0
	x %= (146|1);
	x /= (127|1);
	x--;
	x = ~x;
	x += (179);
	x ^= y;
	x += (6);
	x--;
	x += (136);
	x = ~x;
	save_values_codegen(x, y, z);
    // block 1
	x ^= y;
	x |= y;
	switch (x & 7) { case 0: x += 237; break; case 1: x -= 237; break; case 2: x *= 237; break; default: x ^= 237; break; };
	x <<= (6);
	x /= (191|1);
	x = x * x + y;
	x |= y;
	x = x * x + y;
	x = ~x;
	y = y + (x & 2863311530);
	save_values_codegen(x, y, z);
    // block 2
	switch (x & 7) { case 0: x += 226; break; case 1: x -= 226; break; case 2: x *= 226; break; default: x ^= 226; break; };
	x *= (113);
	x <<= (4);
	x = ~x;
	x -= (79);
	x <<= (5);
	x += (198);
	x--;
	z = z + static_cast<std::int64_t>(x - y);
	x--;
	save_values_codegen(x, y, z);
    // block 3
	x = x * x + y;
	y = y + (x & 13107);
	x++;
	x = x * x + y;
	y = y + (x & 2863311530);
	z = z + static_cast<std::int64_t>(x - y);
	x <<= (5);
	x *= (205);
	x -= (216);
	x *= (223);
	save_values_codegen(x, y, z);
    // block 4
	x += (194);
	x &= y;
	x++;
	x <<= (5);
	x &= y;
	z = z + static_cast<std::int64_t>(x - y);
	x = ~x;
	x--;
	x &= y;
	y = y + (x & 15);
	save_values_codegen(x, y, z);
    // block 5
	x += (102);
	x <<= (6);
	switch (x & 7) { case 0: x += 214; break; case 1: x -= 214; break; case 2: x *= 214; break; default: x ^= 214; break; };
	x ^= y;
	x--;
	y = y + (x & 2863311530);
	x %= (248|1);
	x |= y;
	x |= y;
	x &= y;
	save_values_codegen(x, y, z);
    // block 6
	x >>= (5);
	x <<= (6);
	x--;
	x >>= (6);
	x >>= (1);
	x |= y;
	if (x & 2863311530) x += y ^ 157; x -= y & 157;
	x += (41);
	x %= (87|1);
	x &= y;
	save_values_codegen(x, y, z);
    // block 7
	switch (x & 7) { case 0: x += 89; break; case 1: x -= 89; break; case 2: x *= 89; break; default: x ^= 89; break; };
	switch (x & 7) { case 0: x += 61; break; case 1: x -= 61; break; case 2: x *= 61; break; default: x ^= 61; break; };
	x *= (35);
	switch (x & 7) { case 0: x += 55; break; case 1: x -= 55; break; case 2: x *= 55; break; default: x ^= 55; break; };
	x = x * x + y;
	if (x & 2863311530) x += y ^ 88; x -= y & 88;
	x ^= y;
	x %= (149|1);
	if (x & 255) x += y ^ 11; x -= y & 11;
	x--;
	save_values_codegen(x, y, z);
    // block 8
	if (x & 13107) x += y ^ 30; x -= y & 30;
	x *= (147);
	x *= (244);
	x |= y;
	z = z + static_cast<std::int64_t>(x - y);
	x -= (212);
	x += (24);
	x -= (49);
	x <<= (4);
	x ^= y;
	save_values_codegen(x, y, z);
    // block 9
	y = y + (x & 2863311530);
	switch (x & 7) { case 0: x += 65; break; case 1: x -= 65; break; case 2: x *= 65; break; default: x ^= 65; break; };
	if (x & 2863311530) x += y ^ 26; x -= y & 26;
	x -= (7);
	if (x & 2863311530) x += y ^ 116; x -= y & 116;
	x--;
	z = z + static_cast<std::int64_t>(x - y);
	switch (x & 7) { case 0: x += 223; break; case 1: x -= 223; break; case 2: x *= 223; break; default: x ^= 223; break; };
	x++;
	x |= y;
	save_values_codegen(x, y, z);
}

int main(int argc, char** argv) {
    unsigned seed = static_cast<unsigned>(time(nullptr));

    std::srand(seed);

    // random values
    volatile std::uint64_t a = 0x980F'1BA3ULL;
    volatile std::uint64_t b = 0x2FA1'BB14ULL;
    volatile std::int64_t c = -0x014A'BA32LL;

    block_wrap(a, b, c);

    save_values_codegen(a, b, c);

    std::cout << "g_acc0: " << g_acc0 << ", g_acc1: " << g_acc1 << ", g_signed: " << g_signed << std::endl;
    return 0;
}
