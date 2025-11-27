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
volatile double g_acc1 = 0.0;
volatile float g_signed = 0.0f;

NOINLINE void save_values_codegen(volatile double x, volatile double y, volatile float z) {
    // TODO: this is bad, just use a different operation
    g_acc0 ^= *reinterpret_cast<volatile std::uint64_t*>(&x);
    g_acc1 += y;
    g_signed += z;
}

NOINLINE void block_wrap(volatile double& x, volatile double& y, volatile float& z) {
    	volatile double s = static_cast<double>(x + 0.1343642441124012);
	volatile double t = static_cast<double>(y + 0.8474337369372327);
	volatile float u = static_cast<float>(z - 0.76377462f);
    // block 0
	x = x * x;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x == y);
	y = y + x * x;
	z = (x + y) * (x - y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x > y);
	x = static_cast<double>(x < y);
	save_values_codegen(x, y, z);
    // block 1
	x = x * x;
	z = (x + y) * (x - y);
	x = static_cast<double>(x < y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x -= 0.4389616300445631;
	x *= 0.5209384176131452;
	x = static_cast<double>(x > y);
	save_values_codegen(x, y, z);
    // block 2
	y = y + x * x;
	x = x * x;
	x += 0.7705231398308006;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x > y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x += 0.4811018174142402;
	x = static_cast<double>(x < y);
	z = (x + y) * (x - y);
	save_values_codegen(x, y, z);
    // block 3
	x = static_cast<double>(x > y);
	x = static_cast<double>(x < y);
	x += 0.841744832274096;
	x -= 0.868078090374847;
	y = y + x * x;
	z = (x + y) * (x - y);
	y = y + x * x;
	x = x * x;
	x = static_cast<double>(x > y);
	if (z > 0.0) z = x + y; else z = x -y;
	save_values_codegen(x, y, z);
    // block 4
	x -= 0.25345814898474883;
	x = static_cast<double>(x == y);
	x += 0.3973153691475223;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x < y);
	x = 0.5238954398669781 != 0.0 ? x / 0.5238954398669781 : 0.0;
	x = x * x;
	x += 0.7374512500957098;
	x += 0.30638662033324593;
	y = y + x * x;
	save_values_codegen(x, y, z);
    // block 5
	x *= 0.008480262463668842;
	x = static_cast<double>(x == y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	x *= 0.2008530114407594;
	z = (x + y) * (x - y);
	if (z > 0.0) z = x + y; else z = x -y;
	save_values_codegen(x, y, z);
    // block 6
	x = static_cast<double>(x < y);
	x -= 0.13301701225112483;
	y = y + x * x;
	z = (x + y) * (x - y);
	x = 0.8674198235869027 != 0.0 ? x / 0.8674198235869027 : 0.0;
	x *= 0.57995705717184;
	x += 0.4065989261358486;
	x *= 0.3408974641165834;
	x -= 0.5707815255990233;
	y = y + x * x;
	save_values_codegen(x, y, z);
    // block 7
	x = static_cast<double>(x > y);
	y = y + x * x;
	x -= 0.41353443350719277;
	x = 0.7855121343445666 != 0.0 ? x / 0.7855121343445666 : 0.0;
	x *= 0.1155581805922733;
	x *= 0.7440064165370084;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x -= 0.2076146204177305;
	x += 0.010506151518672291;
	save_values_codegen(x, y, z);
    // block 8
	x = static_cast<double>(x > y);
	x -= 0.9133920171659404;
	x -= 0.25006282489890796;
	x = x * x;
	y = y + x * x;
	y = y + x * x;
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x *= 0.3167351468856021;
	x = 0.33433340565076186 != 0.0 ? x / 0.33433340565076186 : 0.0;
	save_values_codegen(x, y, z);
    // block 9
	x = 0.22015542552153777 != 0.0 ? x / 0.22015542552153777 : 0.0;
	x -= 0.2680638175547476;
	x += 0.6353820935630572;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x *= 0.9848958711440725;
	x -= 0.7093239911970365;
	x *= 0.8937587976957898;
	z = (x + y) * (x - y);
	save_values_codegen(x, y, z);
}

int main(int argc, char** argv) {
    unsigned seed = static_cast<unsigned>(time(nullptr));

    std::srand(seed);

    // random values
    volatile double a = 0.2550690257394217;
    volatile double b = 0.4954350870919410;
    volatile float c = 0.44949106f;

    block_wrap(a, b, c);

    save_values_codegen(a, b, c);

    std::cout << "g_acc0: " << g_acc0 << ", g_acc1: " << g_acc1 << ", g_signed: " << g_signed << std::endl;
    return 0;
}
