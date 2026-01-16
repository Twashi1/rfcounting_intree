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
volatile float g_acc1 = 0.0f;
volatile float g_signed = 0.0f;

NOINLINE void save_values_codegen(volatile float x, volatile float y, volatile float z) {
    // TODO: this is bad, just use a different operation
    g_acc0 ^= *reinterpret_cast<volatile std::uint64_t*>(&x);
    g_acc1 += y;
    g_signed += z;
}

NOINLINE void block_wrap(volatile float& x, volatile float& y, volatile float& z) {
    	volatile float s = static_cast<float>(x + 0.13436424f);
	volatile float t = static_cast<float>(y + 0.84743374f);
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
    // block 1
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
    // block 2
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
    // block 3
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
    // block 4
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
    // block 5
	x *= 0.29892637902973584;
	x += 0.7148244519688113;
	y = y + x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x *= 0.2579690924717717;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	x = 0.48599670087901625 != 0.0 ? x / 0.48599670087901625 : 0.0;
	x = static_cast<double>(x < y);
	x = 0.3129848800232429 != 0.0 ? x / 0.3129848800232429 : 0.0;
	x = 0.7128461108524063 != 0.0 ? x / 0.7128461108524063 : 0.0;
	x = x * x;
	x *= 0.5116318148823668;
	y = y + x * x;
	x *= 0.7012826629078087;
	x = static_cast<double>(x == y);
	x *= 0.15577922203090344;
	save_values_codegen(x, y, z);
    // block 6
	x = static_cast<double>(x == y);
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	if (z > 0.0) z = x + y; else z = x -y;
	x *= 0.6224025481111165;
	x -= 0.43126050085335277;
	x -= 0.6605097077449822;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x = x * x;
	x = 0.17285225007520322 != 0.0 ? x / 0.17285225007520322 : 0.0;
	y = y + x * x;
	x *= 0.5453770038258688;
	x -= 0.7690673858593793;
	x = 0.2840474457335592 != 0.0 ? x / 0.2840474457335592 : 0.0;
	x -= 0.5693085770850274;
	save_values_codegen(x, y, z);
    // block 7
	x = x * x;
	x -= 0.5003714738318865;
	if (z > 0.0) z = x + y; else z = x -y;
	x += 0.12390078188713138;
	z = (x + y) * (x - y);
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = x * x;
	x = static_cast<double>(x > y);
	y = y + x * x;
	z = (x + y) * (x - y);
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	x = x * x;
	x = x * x;
	x *= 0.7674997901425722;
	z = (x + y) * (x - y);
	x = x * x;
	y = y + x * x;
	save_values_codegen(x, y, z);
    // block 8
	y = y + x * x;
	x -= 0.3551773135885514;
	x += 0.16413876047107268;
	x -= 0.40192372825721256;
	x = static_cast<double>(x == y);
	x = 0.8860252896990286 != 0.0 ? x / 0.8860252896990286 : 0.0;
	x -= 0.6991950335404097;
	x = static_cast<double>(x > y);
	x *= 0.29687625656326855;
	z = (x + y) * (x - y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	x *= 0.778463238016501;
	z = (x + y) * (x - y);
	x -= 0.7181581423147705;
	save_values_codegen(x, y, z);
    // block 9
	x -= 0.10638543387964139;
	x = static_cast<double>(x > y);
	x += 0.8094777956419704;
	x = x * x;
	z = (x + y) * (x - y);
	x = static_cast<double>(x == y);
	if (z > 0.0) z = x + y; else z = x -y;
	x *= 0.9734382274129786;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x > y);
	x = static_cast<double>(x == y);
	x += 0.015411829929371956;
	z = (x + y) * (x - y);
	x = 0.39992859333804187 != 0.0 ? x / 0.39992859333804187 : 0.0;
	x += 0.015181052589951283;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	x += 0.5369806070689243;
	x += 0.9333806207298506;
	x -= 0.1901109295161283;
	save_values_codegen(x, y, z);
    // block 10
	y = y + x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = x * x;
	x = static_cast<double>(x == y);
	z = (x + y) * (x - y);
	z = (x + y) * (x - y);
	x = static_cast<double>(x < y);
	y = y + x * x;
	y = y + x * x;
	x += 0.9186473950993009;
	x += 0.63712281056656;
	x = static_cast<double>(x < y);
	x *= 0.5314650538056048;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x -= 0.3018396126791336;
	y = y + x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x -= 0.42483036101897353;
	z = (x + y) * (x - y);
	save_values_codegen(x, y, z);
    // block 11
	x = static_cast<double>(x == y);
	x = static_cast<double>(x < y);
	y = y + x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	z = (x + y) * (x - y);
	y = y + x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	z = (x + y) * (x - y);
	x = static_cast<double>(x < y);
	z = (x + y) * (x - y);
	x = static_cast<double>(x > y);
	x -= 0.9222893236500579;
	x += 0.5237117222858407;
	x = 0.7029162166549554 != 0.0 ? x / 0.7029162166549554 : 0.0;
	z = (x + y) * (x - y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	y = y + x * x;
	x -= 0.18460034404937076;
	x -= 0.978596065533965;
	save_values_codegen(x, y, z);
    // block 12
	x -= 0.20434474394990176;
	x -= 0.7942122507887109;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x > y);
	x = 0.5980780012429903 != 0.0 ? x / 0.5980780012429903 : 0.0;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x > y);
	x = 0.008279858133068752 != 0.0 ? x / 0.008279858133068752 : 0.0;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x == y);
	x += 0.3518262717957913;
	x = 0.9605219758998921 != 0.0 ? x / 0.9605219758998921 : 0.0;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x == y);
	y = y + x * x;
	x = static_cast<double>(x < y);
	x = x * x;
	z = (x + y) * (x - y);
	save_values_codegen(x, y, z);
    // block 13
	x -= 0.9251636496933648;
	y = y + x * x;
	z = (x + y) * (x - y);
	z = (x + y) * (x - y);
	x *= 0.8140526358800608;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x < y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x > y);
	x -= 0.06444119013451133;
	x = 0.7301654668609503 != 0.0 ? x / 0.7301654668609503 : 0.0;
	x *= 0.47978705744969674;
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	y = y + x * x;
	x += 0.7864236400590823;
	if (z > 0.0) z = x + y; else z = x -y;
	x = x * x;
	save_values_codegen(x, y, z);
    // block 14
	x = static_cast<double>(x == y);
	if (z > 0.0) z = x + y; else z = x -y;
	x *= 0.5207426269174332;
	x = static_cast<double>(x > y);
	x = static_cast<double>(x == y);
	x += 0.7577761485350656;
	y = y + x * x;
	y = y + x * x;
	x = static_cast<double>(x < y);
	if (z > 0.0) z = x + y; else z = x -y;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	if (z > 0.0) z = x + y; else z = x -y;
	y = y + x * x;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	z = (x + y) * (x - y);
	x -= 0.12792226588170752;
	save_values_codegen(x, y, z);
    // block 15
	x = 0.08802876349734245 != 0.0 ? x / 0.08802876349734245 : 0.0;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x == y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x == y);
	y = y + x * x;
	x = static_cast<double>(x == y);
	x -= 0.3298344116436186;
	y = y + x * x;
	z = (x + y) * (x - y);
	y = y + x * x;
	z = (x + y) * (x - y);
	z = (x + y) * (x - y);
	x -= 0.6796441847743212;
	x = static_cast<double>(x < y);
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x == y);
	x = 0.6327294059296804 != 0.0 ? x / 0.6327294059296804 : 0.0;
	x += 0.6000914164522895;
	save_values_codegen(x, y, z);
    // block 16
	x = 0.2150232188197363 != 0.0 ? x / 0.2150232188197363 : 0.0;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x -= 0.7449157638361875;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x == y);
	x *= 0.074346899341591;
	x = x * x;
	z = (x + y) * (x - y);
	x *= 0.8491095140212457;
	if (z > 0.0) z = x + y; else z = x -y;
	y = y + x * x;
	x = static_cast<double>(x < y);
	if (z > 0.0) z = x + y; else z = x -y;
	x += 0.7792182447906874;
	y = y + x * x;
	x = static_cast<double>(x == y);
	x += 0.6460830174694423;
	x += 0.8317651384407699;
	x = static_cast<double>(x < y);
	save_values_codegen(x, y, z);
    // block 17
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	x -= 0.5623565610644854;
	x -= 0.7768544334216305;
	if (z > 0.0) z = x + y; else z = x -y;
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x > y);
	x += 0.3414971804099921;
	x = static_cast<double>(x < y);
	x -= 0.9751989661364331;
	x = 0.38323864066304636 != 0.0 ? x / 0.38323864066304636 : 0.0;
	x = 0.09969648903871986 != 0.0 ? x / 0.09969648903871986 : 0.0;
	x = x * x;
	x = static_cast<double>(x > y);
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x == y);
	x *= 0.011942322374289605;
	x += 0.6524016192152049;
	if (z > 0.0) z = x + y; else z = x -y;
	save_values_codegen(x, y, z);
    // block 18
	x *= 0.9721002692327158;
	x += 0.46443128136208256;
	x = static_cast<double>(x < y);
	if (z > 0.0) z = x + y; else z = x -y;
	z = (x + y) * (x - y);
	x *= 0.37568325240427813;
	if (z > 0.0) z = x + y; else z = x -y;
	x = static_cast<double>(x < y);
	x = 0.3981631083747522 != 0.0 ? x / 0.3981631083747522 : 0.0;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	y = y + x * x;
	x *= 0.5949667329579694;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x = 0.4953940497109195 != 0.0 ? x / 0.4953940497109195 : 0.0;
	x *= 0.25945266013498836;
	x = static_cast<double>(x == y);
	x *= 0.2584684659168843;
	save_values_codegen(x, y, z);
    // block 19
	if (z > 0.0) z = x + y; else z = x -y;
	x *= 0.9538556123544408;
	x = 0.07045972944573031 != 0.0 ? x / 0.07045972944573031 : 0.0;
	x = static_cast<double>(x < y);
	x = static_cast<double>(x < y);
	x -= 0.6254050875808675;
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	x = x * x;
	if (z > 0.0) z = x + y; else z = x -y;
	z = (x + y) * (x - y);
	y = y + x * x;
	x = static_cast<double>(x < y);
	x = 0.7053799812752958 != 0.0 ? x / 0.7053799812752958 : 0.0;
	x = static_cast<double>(x == y);
	x = static_cast<double>(x > y);
	z = (x + y) * (x - y);
	x -= 0.8539871307856776;
	x = 0.5042343550005652 != 0.0 ? x / 0.5042343550005652 : 0.0;
	x = 0.8192821811677478 != 0.0 ? x / 0.8192821811677478 : 0.0;
	save_values_codegen(x, y, z);
}

int main(int argc, char** argv) {
    unsigned seed = static_cast<unsigned>(time(nullptr));

    std::srand(seed);

    // random values
    volatile float a = 0.25506903f;
    volatile float b = 0.49543509f;
    volatile float c = 0.44949106f;

    block_wrap(a, b, c);

    save_values_codegen(a, b, c);

    std::cout << "g_acc0: " << g_acc0 << ", g_acc1: " << g_acc1 << ", g_signed: " << g_signed << std::endl;
    return 0;
}
