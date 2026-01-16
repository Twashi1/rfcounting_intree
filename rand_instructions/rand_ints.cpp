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
    // TODO: this is bad, just use a different operation
    g_acc0 ^= *reinterpret_cast<volatile std::uint64_t*>(&x);
    g_acc1 += y;
    g_signed += z;
}

NOINLINE void block_wrap(volatile std::uint64_t& x, volatile std::uint64_t& y, volatile std::int64_t& z) {
    	volatile std::uint64_t s = static_cast<std::uint64_t>(x + 0x4386BBC4CD613E30LL);
	volatile std::uint64_t t = static_cast<std::uint64_t>(y + -0x61D01476BEB3CBC4LL);
	volatile std::int64_t u = static_cast<std::int64_t>(z - 0x7311D8A3C2CE6F44ULL);
    // block 0
	x ^= y;
	x += (6);
	x--;
	x += (136);
	x = ~x;
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
	switch (x & 7) { case 0: x += 226; break; case 1: x -= 226; break; case 2: x *= 226; break; default: x ^= 226; break; };
	x *= (113);
	x <<= (4);
	x = ~x;
	x -= (79);
	save_values_codegen(x, y, z);
    // block 1
	x <<= (5);
	x += (198);
	x--;
	z = z + static_cast<std::int64_t>(x - y);
	x--;
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
	x += (194);
	x &= y;
	x++;
	x <<= (5);
	x &= y;
	save_values_codegen(x, y, z);
    // block 2
	z = z + static_cast<std::int64_t>(x - y);
	x = ~x;
	x--;
	x &= y;
	y = y + (x & 15);
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
	x >>= (5);
	x <<= (6);
	x--;
	x >>= (6);
	x >>= (1);
	save_values_codegen(x, y, z);
    // block 3
	x |= y;
	if (x & 2863311530) x += y ^ 157; x -= y & 157;
	x += (41);
	x %= (87|1);
	x &= y;
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
	if (x & 13107) x += y ^ 30; x -= y & 30;
	x *= (147);
	x *= (244);
	x |= y;
	z = z + static_cast<std::int64_t>(x - y);
	save_values_codegen(x, y, z);
    // block 4
	x -= (212);
	x += (24);
	x -= (49);
	x <<= (4);
	x ^= y;
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
	x *= (210);
	z = z + static_cast<std::int64_t>(x - y);
	x ^= y;
	if (x & 2863311530) x += y ^ 48; x -= y & 48;
	x ^= y;
	save_values_codegen(x, y, z);
    // block 5
	x ^= y;
	x--;
	x *= (6);
	x |= y;
	x = ~x;
	x = x * x + y;
	x = x * x + y;
	x <<= (2);
	if (x & 2863311530) x += y ^ 79; x -= y & 79;
	x %= (229|1);
	switch (x & 7) { case 0: x += 234; break; case 1: x -= 234; break; case 2: x *= 234; break; default: x ^= 234; break; };
	if (x & 15) x += y ^ 211; x -= y & 211;
	x <<= (4);
	x -= (183);
	x &= y;
	y = y + (x & 13107);
	switch (x & 7) { case 0: x += 117; break; case 1: x -= 117; break; case 2: x *= 117; break; default: x ^= 117; break; };
	if (x & 13107) x += y ^ 44; x -= y & 44;
	x += (204);
	x += (16);
	save_values_codegen(x, y, z);
    // block 6
	x %= (152|1);
	x &= y;
	x--;
	x ^= y;
	x = x * x + y;
	x ^= y;
	x = ~x;
	if (x & 2863311530) x += y ^ 144; x -= y & 144;
	x ^= y;
	x <<= (2);
	x |= y;
	x <<= (6);
	x *= (220);
	x <<= (3);
	x >>= (5);
	x = ~x;
	x--;
	x = x * x + y;
	x /= (69|1);
	x %= (249|1);
	save_values_codegen(x, y, z);
    // block 7
	z = z + static_cast<std::int64_t>(x - y);
	y = y + (x & 2863311530);
	z = z + static_cast<std::int64_t>(x - y);
	x >>= (4);
	x /= (170|1);
	x ^= y;
	x >>= (4);
	x += (161);
	x &= y;
	x %= (139|1);
	x |= y;
	x <<= (3);
	y = y + (x & 15);
	x--;
	x += (31);
	switch (x & 7) { case 0: x += 76; break; case 1: x -= 76; break; case 2: x *= 76; break; default: x ^= 76; break; };
	x *= (129);
	y = y + (x & 2863311530);
	x = x * x + y;
	z = z + static_cast<std::int64_t>(x - y);
	save_values_codegen(x, y, z);
    // block 8
	x |= y;
	x = ~x;
	x--;
	x &= y;
	z = z + static_cast<std::int64_t>(x - y);
	x |= y;
	x = x * x + y;
	x--;
	if (x & 255) x += y ^ 221; x -= y & 221;
	x = ~x;
	x |= y;
	x %= (163|1);
	x <<= (1);
	x++;
	switch (x & 7) { case 0: x += 78; break; case 1: x -= 78; break; case 2: x *= 78; break; default: x ^= 78; break; };
	x &= y;
	x = x * x + y;
	y = y + (x & 255);
	z = z + static_cast<std::int64_t>(x - y);
	x *= (103);
	save_values_codegen(x, y, z);
    // block 9
	x |= y;
	x ^= y;
	x *= (20);
	z = z + static_cast<std::int64_t>(x - y);
	x <<= (6);
	x++;
	x--;
	x = ~x;
	x ^= y;
	x += (219);
	if (x & 15) x += y ^ 238; x -= y & 238;
	x &= y;
	z = z + static_cast<std::int64_t>(x - y);
	x &= y;
	x %= (200|1);
	x++;
	x /= (184|1);
	x *= (28);
	if (x & 15) x += y ^ 127; x -= y & 127;
	x -= (15);
	save_values_codegen(x, y, z);
    // block 10
	x >>= (1);
	x = x * x + y;
	if (x & 255) x += y ^ 170; x -= y & 170;
	x <<= (2);
	x ^= y;
	x <<= (2);
	x |= y;
	x >>= (6);
	if (x & 255) x += y ^ 128; x -= y & 128;
	x >>= (1);
	x += (205);
	if (x & 2863311530) x += y ^ 228; x -= y & 228;
	x >>= (2);
	x += (4);
	switch (x & 7) { case 0: x += 15; break; case 1: x -= 15; break; case 2: x *= 15; break; default: x ^= 15; break; };
	x &= y;
	x |= y;
	switch (x & 7) { case 0: x += 16; break; case 1: x -= 16; break; case 2: x *= 16; break; default: x ^= 16; break; };
	x -= (239);
	y = y + (x & 255);
	save_values_codegen(x, y, z);
    // block 11
	x = ~x;
	x >>= (4);
	if (x & 2863311530) x += y ^ 162; x -= y & 162;
	x ^= y;
	x++;
	x++;
	switch (x & 7) { case 0: x += 109; break; case 1: x -= 109; break; case 2: x *= 109; break; default: x ^= 109; break; };
	x &= y;
	x &= y;
	x -= (236);
	x -= (14);
	x = x * x + y;
	x /= (253|1);
	x %= (137|1);
	x %= (230|1);
	x += (189);
	x *= (65);
	x |= y;
	x &= y;
	x &= y;
	save_values_codegen(x, y, z);
    // block 12
	x |= y;
	x = x * x + y;
	if (x & 13107) x += y ^ 131; x -= y & 131;
	x /= (34|1);
	x = x * x + y;
	x |= y;
	x++;
	x /= (105|1);
	x *= (12);
	y = y + (x & 2863311530);
	x &= y;
	x = x * x + y;
	if (x & 255) x += y ^ 201; x -= y & 201;
	z = z + static_cast<std::int64_t>(x - y);
	z = z + static_cast<std::int64_t>(x - y);
	x *= (237);
	x %= (13|1);
	x &= y;
	x++;
	x--;
	save_values_codegen(x, y, z);
    // block 13
	switch (x & 7) { case 0: x += 130; break; case 1: x -= 130; break; case 2: x *= 130; break; default: x ^= 130; break; };
	x %= (65|1);
	x %= (233|1);
	y = y + (x & 255);
	x /= (251|1);
	x /= (53|1);
	x = x * x + y;
	x >>= (2);
	x += (152);
	x |= y;
	x ^= y;
	switch (x & 7) { case 0: x += 234; break; case 1: x -= 234; break; case 2: x *= 234; break; default: x ^= 234; break; };
	x *= (209);
	x >>= (6);
	x = x * x + y;
	x--;
	x -= (91);
	x >>= (3);
	switch (x & 7) { case 0: x += 31; break; case 1: x -= 31; break; case 2: x *= 31; break; default: x ^= 31; break; };
	switch (x & 7) { case 0: x += 166; break; case 1: x -= 166; break; case 2: x *= 166; break; default: x ^= 166; break; };
	save_values_codegen(x, y, z);
    // block 14
	x = x * x + y;
	x |= y;
	x = x * x + y;
	switch (x & 7) { case 0: x += 198; break; case 1: x -= 198; break; case 2: x *= 198; break; default: x ^= 198; break; };
	x += (255);
	x++;
	x += (231);
	x += (99);
	x &= y;
	if (x & 255) x += y ^ 100; x -= y & 100;
	x = ~x;
	x %= (5|1);
	x++;
	y = y + (x & 2863311530);
	y = y + (x & 13107);
	if (x & 15) x += y ^ 199; x -= y & 199;
	x = ~x;
	x *= (17);
	x %= (59|1);
	x /= (65|1);
	save_values_codegen(x, y, z);
    // block 15
	x /= (103|1);
	x += (23);
	switch (x & 7) { case 0: x += 56; break; case 1: x -= 56; break; case 2: x *= 56; break; default: x ^= 56; break; };
	x++;
	switch (x & 7) { case 0: x += 174; break; case 1: x -= 174; break; case 2: x *= 174; break; default: x ^= 174; break; };
	x &= y;
	x = ~x;
	x >>= (6);
	x--;
	x |= y;
	x--;
	x /= (39|1);
	x <<= (3);
	switch (x & 7) { case 0: x += 74; break; case 1: x -= 74; break; case 2: x *= 74; break; default: x ^= 74; break; };
	if (x & 255) x += y ^ 221; x -= y & 221;
	x += (194);
	x &= y;
	x |= y;
	x = x * x + y;
	y = y + (x & 2863311530);
	save_values_codegen(x, y, z);
    // block 16
	x >>= (5);
	switch (x & 7) { case 0: x += 214; break; case 1: x -= 214; break; case 2: x *= 214; break; default: x ^= 214; break; };
	y = y + (x & 13107);
	x |= y;
	x = ~x;
	x ^= y;
	z = z + static_cast<std::int64_t>(x - y);
	x = x * x + y;
	x >>= (5);
	x = ~x;
	x %= (179|1);
	x *= (163);
	x <<= (1);
	x >>= (5);
	y = y + (x & 255);
	x |= y;
	x *= (191);
	if (x & 13107) x += y ^ 69; x -= y & 69;
	x += (32);
	x %= (30|1);
	save_values_codegen(x, y, z);
    // block 17
	x -= (89);
	x /= (77|1);
	x &= y;
	x += (21);
	x++;
	x /= (174|1);
	x += (132);
	x++;
	x &= y;
	y = y + (x & 2863311530);
	x ^= y;
	x -= (154);
	x <<= (2);
	x &= y;
	switch (x & 7) { case 0: x += 70; break; case 1: x -= 70; break; case 2: x *= 70; break; default: x ^= 70; break; };
	x = ~x;
	x++;
	switch (x & 7) { case 0: x += 143; break; case 1: x -= 143; break; case 2: x *= 143; break; default: x ^= 143; break; };
	x |= y;
	x %= (20|1);
	save_values_codegen(x, y, z);
    // block 18
	x >>= (3);
	x |= y;
	z = z + static_cast<std::int64_t>(x - y);
	x &= y;
	switch (x & 7) { case 0: x += 244; break; case 1: x -= 244; break; case 2: x *= 244; break; default: x ^= 244; break; };
	x--;
	z = z + static_cast<std::int64_t>(x - y);
	x &= y;
	x--;
	x /= (121|1);
	x -= (213);
	x = x * x + y;
	x++;
	x ^= y;
	x <<= (1);
	y = y + (x & 2863311530);
	x &= y;
	y = y + (x & 2863311530);
	x |= y;
	x ^= y;
	save_values_codegen(x, y, z);
    // block 19
	x -= (88);
	x = x * x + y;
	if (x & 255) x += y ^ 194; x -= y & 194;
	x += (123);
	x <<= (6);
	x /= (64|1);
	x ^= y;
	x = ~x;
	y = y + (x & 255);
	x = ~x;
	x %= (4|1);
	x /= (205|1);
	x <<= (4);
	x %= (40|1);
	x &= y;
	x ^= y;
	x += (140);
	x <<= (2);
	x ^= y;
	x <<= (5);
	save_values_codegen(x, y, z);
}

int main(int argc, char** argv) {
    unsigned seed = static_cast<unsigned>(time(nullptr));

    std::srand(seed);

    // random values
    volatile std::uint64_t a = -0x67F8D173CA4066D3LL;
    volatile std::uint64_t b = 0x44647159C324C985LL;
    volatile std::int64_t c = 0x7204E52DB2221A58ULL;

    block_wrap(a, b, c);

    save_values_codegen(a, b, c);

    std::cout << "g_acc0: " << g_acc0 << ", g_acc1: " << g_acc1 << ", g_signed: " << g_signed << std::endl;
    return 0;
}
