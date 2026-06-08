
./objects/gemm.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <main>:
       0: 55                           	pushq	%rbp
       1: 48 89 e5                     	movq	%rsp, %rbp
       4: 48 83 ec 60                  	subq	$0x60, %rsp
       8: b8 01 00 00 00               	movl	$0x1, %eax
       d: b8 5a 00 00 00               	movl	$0x5a, %eax
      12: 0f a2                        	cpuid
      14: c7 45 b4 00 00 00 00         	movl	$0x0, -0x4c(%rbp)
      1b: 89 7d d4                     	movl	%edi, -0x2c(%rbp)
      1e: 48 89 75 b8                  	movq	%rsi, -0x48(%rbp)
      22: c7 45 f8 e8 03 00 00         	movl	$0x3e8, -0x8(%rbp)      # imm = 0x3E8
      29: c7 45 fc 4c 04 00 00         	movl	$0x44c, -0x4(%rbp)      # imm = 0x44C
      30: c7 45 f4 b0 04 00 00         	movl	$0x4b0, -0xc(%rbp)      # imm = 0x4B0
      37: bf e0 c8 10 00               	movl	$0x10c8e0, %edi         # imm = 0x10C8E0
      3c: be 08 00 00 00               	movl	$0x8, %esi
      41: e8 00 00 00 00               	callq	0x46 <main+0x46>
      46: 48 89 45 e8                  	movq	%rax, -0x18(%rbp)
      4a: bf 80 4f 12 00               	movl	$0x124f80, %edi         # imm = 0x124F80
      4f: be 08 00 00 00               	movl	$0x8, %esi
      54: e8 00 00 00 00               	callq	0x59 <main+0x59>
      59: 48 89 45 d8                  	movq	%rax, -0x28(%rbp)
      5d: bf 40 24 14 00               	movl	$0x142440, %edi         # imm = 0x142440
      62: be 08 00 00 00               	movl	$0x8, %esi
      67: e8 00 00 00 00               	callq	0x6c <main+0x6c>
      6c: 48 89 45 e0                  	movq	%rax, -0x20(%rbp)
      70: 8b 7d f8                     	movl	-0x8(%rbp), %edi
      73: 8b 75 fc                     	movl	-0x4(%rbp), %esi
      76: 8b 55 f4                     	movl	-0xc(%rbp), %edx
      79: 4c 8b 4d e8                  	movq	-0x18(%rbp), %r9
      7d: 48 8b 45 d8                  	movq	-0x28(%rbp), %rax
      81: 4c 8b 55 e0                  	movq	-0x20(%rbp), %r10
      85: 48 8d 4d c0                  	leaq	-0x40(%rbp), %rcx
      89: 4c 8d 45 c8                  	leaq	-0x38(%rbp), %r8
      8d: 48 89 04 24                  	movq	%rax, (%rsp)
      91: 4c 89 54 24 08               	movq	%r10, 0x8(%rsp)
      96: e8 a5 00 00 00               	callq	0x140 <init_array>
      9b: 8b 7d f8                     	movl	-0x8(%rbp), %edi
      9e: 8b 75 fc                     	movl	-0x4(%rbp), %esi
      a1: 8b 55 f4                     	movl	-0xc(%rbp), %edx
      a4: f2 0f 10 45 c0               	movsd	-0x40(%rbp), %xmm0
      a9: f2 0f 10 4d c8               	movsd	-0x38(%rbp), %xmm1
      ae: 48 8b 4d e8                  	movq	-0x18(%rbp), %rcx
      b2: 4c 8b 45 d8                  	movq	-0x28(%rbp), %r8
      b6: 4c 8b 4d e0                  	movq	-0x20(%rbp), %r9
      ba: e8 51 03 00 00               	callq	0x410 <kernel_gemm>
      bf: 83 7d d4 2a                  	cmpl	$0x2a, -0x2c(%rbp)
      c3: 7e 3d                        	jle	0x102 <main+0x102>
      c5: b8 01 00 00 00               	movl	$0x1, %eax
      ca: b8 5a 00 00 00               	movl	$0x5a, %eax
      cf: 0f a2                        	cpuid
      d1: 48 8b 45 b8                  	movq	-0x48(%rbp), %rax
      d5: 48 8b 38                     	movq	(%rax), %rdi
      d8: be 00 00 00 00               	movl	$0x0, %esi
      dd: e8 00 00 00 00               	callq	0xe2 <main+0xe2>
      e2: 83 f8 00                     	cmpl	$0x0, %eax
      e5: 75 1b                        	jne	0x102 <main+0x102>
      e7: b8 01 00 00 00               	movl	$0x1, %eax
      ec: b8 5a 00 00 00               	movl	$0x5a, %eax
      f1: 0f a2                        	cpuid
      f3: 8b 7d f8                     	movl	-0x8(%rbp), %edi
      f6: 8b 75 fc                     	movl	-0x4(%rbp), %esi
      f9: 48 8b 55 e8                  	movq	-0x18(%rbp), %rdx
      fd: e8 0e 05 00 00               	callq	0x610 <print_array>
     102: b8 01 00 00 00               	movl	$0x1, %eax
     107: b8 5a 00 00 00               	movl	$0x5a, %eax
     10c: 0f a2                        	cpuid
     10e: 48 8b 7d e8                  	movq	-0x18(%rbp), %rdi
     112: e8 00 00 00 00               	callq	0x117 <main+0x117>
     117: 48 8b 7d d8                  	movq	-0x28(%rbp), %rdi
     11b: e8 00 00 00 00               	callq	0x120 <main+0x120>
     120: 48 8b 7d e0                  	movq	-0x20(%rbp), %rdi
     124: e8 00 00 00 00               	callq	0x129 <main+0x129>
     129: 31 c0                        	xorl	%eax, %eax
     12b: 48 83 c4 60                  	addq	$0x60, %rsp
     12f: 5d                           	popq	%rbp
     130: c3                           	retq
     131: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00 	nopw	%cs:(%rax,%rax)

0000000000000140 <init_array>:
     140: 55                           	pushq	%rbp
     141: 48 89 e5                     	movq	%rsp, %rbp
     144: b8 01 00 00 00               	movl	$0x1, %eax
     149: b8 5a 00 00 00               	movl	$0x5a, %eax
     14e: 0f a2                        	cpuid
     150: 48 8b 45 18                  	movq	0x18(%rbp), %rax
     154: 48 8b 45 10                  	movq	0x10(%rbp), %rax
     158: 89 7d ec                     	movl	%edi, -0x14(%rbp)
     15b: 89 75 f0                     	movl	%esi, -0x10(%rbp)
     15e: 89 55 f4                     	movl	%edx, -0xc(%rbp)
     161: 48 89 4d d0                  	movq	%rcx, -0x30(%rbp)
     165: 4c 89 45 d8                  	movq	%r8, -0x28(%rbp)
     169: 4c 89 4d e0                  	movq	%r9, -0x20(%rbp)
     16d: 48 8b 45 d0                  	movq	-0x30(%rbp), %rax
     171: f2 0f 10 05 00 00 00 00      	movsd	(%rip), %xmm0           # 0x179 <init_array+0x39>
     179: f2 0f 11 00                  	movsd	%xmm0, (%rax)
     17d: 48 8b 45 d8                  	movq	-0x28(%rbp), %rax
     181: f2 0f 10 05 00 00 00 00      	movsd	(%rip), %xmm0           # 0x189 <init_array+0x49>
     189: f2 0f 11 00                  	movsd	%xmm0, (%rax)
     18d: c7 45 f8 00 00 00 00         	movl	$0x0, -0x8(%rbp)
     194: b8 01 00 00 00               	movl	$0x1, %eax
     199: b8 5a 00 00 00               	movl	$0x5a, %eax
     19e: 0f a2                        	cpuid
     1a0: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     1a3: 3b 45 ec                     	cmpl	-0x14(%rbp), %eax
     1a6: 0f 8d a8 00 00 00            	jge	0x254 <init_array+0x114>
     1ac: b8 01 00 00 00               	movl	$0x1, %eax
     1b1: b8 5a 00 00 00               	movl	$0x5a, %eax
     1b6: 0f a2                        	cpuid
     1b8: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     1bf: b8 01 00 00 00               	movl	$0x1, %eax
     1c4: b8 5a 00 00 00               	movl	$0x5a, %eax
     1c9: 0f a2                        	cpuid
     1cb: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     1ce: 3b 45 f0                     	cmpl	-0x10(%rbp), %eax
     1d1: 7d 59                        	jge	0x22c <init_array+0xec>
     1d3: b8 01 00 00 00               	movl	$0x1, %eax
     1d8: b8 5a 00 00 00               	movl	$0x5a, %eax
     1dd: 0f a2                        	cpuid
     1df: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     1e2: 0f af 45 fc                  	imull	-0x4(%rbp), %eax
     1e6: 83 c0 01                     	addl	$0x1, %eax
     1e9: 99                           	cltd
     1ea: f7 7d ec                     	idivl	-0x14(%rbp)
     1ed: f2 0f 2a c2                  	cvtsi2sd	%edx, %xmm0
     1f1: f2 0f 2a 4d ec               	cvtsi2sdl	-0x14(%rbp), %xmm1
     1f6: f2 0f 5e c1                  	divsd	%xmm1, %xmm0
     1fa: 48 8b 45 e0                  	movq	-0x20(%rbp), %rax
     1fe: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     202: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     209: 48 01 c8                     	addq	%rcx, %rax
     20c: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     210: f2 0f 11 04 c8               	movsd	%xmm0, (%rax,%rcx,8)
     215: b8 01 00 00 00               	movl	$0x1, %eax
     21a: b8 5a 00 00 00               	movl	$0x5a, %eax
     21f: 0f a2                        	cpuid
     221: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     224: 83 c0 01                     	addl	$0x1, %eax
     227: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     22a: eb 93                        	jmp	0x1bf <init_array+0x7f>
     22c: b8 01 00 00 00               	movl	$0x1, %eax
     231: b8 5a 00 00 00               	movl	$0x5a, %eax
     236: 0f a2                        	cpuid
     238: eb 00                        	jmp	0x23a <init_array+0xfa>
     23a: b8 01 00 00 00               	movl	$0x1, %eax
     23f: b8 5a 00 00 00               	movl	$0x5a, %eax
     244: 0f a2                        	cpuid
     246: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     249: 83 c0 01                     	addl	$0x1, %eax
     24c: 89 45 f8                     	movl	%eax, -0x8(%rbp)
     24f: e9 40 ff ff ff               	jmp	0x194 <init_array+0x54>
     254: b8 01 00 00 00               	movl	$0x1, %eax
     259: b8 5a 00 00 00               	movl	$0x5a, %eax
     25e: 0f a2                        	cpuid
     260: c7 45 f8 00 00 00 00         	movl	$0x0, -0x8(%rbp)
     267: b8 01 00 00 00               	movl	$0x1, %eax
     26c: b8 5a 00 00 00               	movl	$0x5a, %eax
     271: 0f a2                        	cpuid
     273: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     276: 3b 45 ec                     	cmpl	-0x14(%rbp), %eax
     279: 0f 8d aa 00 00 00            	jge	0x329 <init_array+0x1e9>
     27f: b8 01 00 00 00               	movl	$0x1, %eax
     284: b8 5a 00 00 00               	movl	$0x5a, %eax
     289: 0f a2                        	cpuid
     28b: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     292: b8 01 00 00 00               	movl	$0x1, %eax
     297: b8 5a 00 00 00               	movl	$0x5a, %eax
     29c: 0f a2                        	cpuid
     29e: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     2a1: 3b 45 f4                     	cmpl	-0xc(%rbp), %eax
     2a4: 7d 5b                        	jge	0x301 <init_array+0x1c1>
     2a6: b8 01 00 00 00               	movl	$0x1, %eax
     2ab: b8 5a 00 00 00               	movl	$0x5a, %eax
     2b0: 0f a2                        	cpuid
     2b2: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     2b5: 8b 4d fc                     	movl	-0x4(%rbp), %ecx
     2b8: 83 c1 01                     	addl	$0x1, %ecx
     2bb: 0f af c1                     	imull	%ecx, %eax
     2be: 99                           	cltd
     2bf: f7 7d f4                     	idivl	-0xc(%rbp)
     2c2: f2 0f 2a c2                  	cvtsi2sd	%edx, %xmm0
     2c6: f2 0f 2a 4d f4               	cvtsi2sdl	-0xc(%rbp), %xmm1
     2cb: f2 0f 5e c1                  	divsd	%xmm1, %xmm0
     2cf: 48 8b 45 10                  	movq	0x10(%rbp), %rax
     2d3: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     2d7: 48 69 c9 80 25 00 00         	imulq	$0x2580, %rcx, %rcx     # imm = 0x2580
     2de: 48 01 c8                     	addq	%rcx, %rax
     2e1: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     2e5: f2 0f 11 04 c8               	movsd	%xmm0, (%rax,%rcx,8)
     2ea: b8 01 00 00 00               	movl	$0x1, %eax
     2ef: b8 5a 00 00 00               	movl	$0x5a, %eax
     2f4: 0f a2                        	cpuid
     2f6: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     2f9: 83 c0 01                     	addl	$0x1, %eax
     2fc: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     2ff: eb 91                        	jmp	0x292 <init_array+0x152>
     301: b8 01 00 00 00               	movl	$0x1, %eax
     306: b8 5a 00 00 00               	movl	$0x5a, %eax
     30b: 0f a2                        	cpuid
     30d: eb 00                        	jmp	0x30f <init_array+0x1cf>
     30f: b8 01 00 00 00               	movl	$0x1, %eax
     314: b8 5a 00 00 00               	movl	$0x5a, %eax
     319: 0f a2                        	cpuid
     31b: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     31e: 83 c0 01                     	addl	$0x1, %eax
     321: 89 45 f8                     	movl	%eax, -0x8(%rbp)
     324: e9 3e ff ff ff               	jmp	0x267 <init_array+0x127>
     329: b8 01 00 00 00               	movl	$0x1, %eax
     32e: b8 5a 00 00 00               	movl	$0x5a, %eax
     333: 0f a2                        	cpuid
     335: c7 45 f8 00 00 00 00         	movl	$0x0, -0x8(%rbp)
     33c: b8 01 00 00 00               	movl	$0x1, %eax
     341: b8 5a 00 00 00               	movl	$0x5a, %eax
     346: 0f a2                        	cpuid
     348: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     34b: 3b 45 f4                     	cmpl	-0xc(%rbp), %eax
     34e: 0f 8d aa 00 00 00            	jge	0x3fe <init_array+0x2be>
     354: b8 01 00 00 00               	movl	$0x1, %eax
     359: b8 5a 00 00 00               	movl	$0x5a, %eax
     35e: 0f a2                        	cpuid
     360: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     367: b8 01 00 00 00               	movl	$0x1, %eax
     36c: b8 5a 00 00 00               	movl	$0x5a, %eax
     371: 0f a2                        	cpuid
     373: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     376: 3b 45 f0                     	cmpl	-0x10(%rbp), %eax
     379: 7d 5b                        	jge	0x3d6 <init_array+0x296>
     37b: b8 01 00 00 00               	movl	$0x1, %eax
     380: b8 5a 00 00 00               	movl	$0x5a, %eax
     385: 0f a2                        	cpuid
     387: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     38a: 8b 4d fc                     	movl	-0x4(%rbp), %ecx
     38d: 83 c1 02                     	addl	$0x2, %ecx
     390: 0f af c1                     	imull	%ecx, %eax
     393: 99                           	cltd
     394: f7 7d f0                     	idivl	-0x10(%rbp)
     397: f2 0f 2a c2                  	cvtsi2sd	%edx, %xmm0
     39b: f2 0f 2a 4d f0               	cvtsi2sdl	-0x10(%rbp), %xmm1
     3a0: f2 0f 5e c1                  	divsd	%xmm1, %xmm0
     3a4: 48 8b 45 18                  	movq	0x18(%rbp), %rax
     3a8: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     3ac: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     3b3: 48 01 c8                     	addq	%rcx, %rax
     3b6: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     3ba: f2 0f 11 04 c8               	movsd	%xmm0, (%rax,%rcx,8)
     3bf: b8 01 00 00 00               	movl	$0x1, %eax
     3c4: b8 5a 00 00 00               	movl	$0x5a, %eax
     3c9: 0f a2                        	cpuid
     3cb: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     3ce: 83 c0 01                     	addl	$0x1, %eax
     3d1: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     3d4: eb 91                        	jmp	0x367 <init_array+0x227>
     3d6: b8 01 00 00 00               	movl	$0x1, %eax
     3db: b8 5a 00 00 00               	movl	$0x5a, %eax
     3e0: 0f a2                        	cpuid
     3e2: eb 00                        	jmp	0x3e4 <init_array+0x2a4>
     3e4: b8 01 00 00 00               	movl	$0x1, %eax
     3e9: b8 5a 00 00 00               	movl	$0x5a, %eax
     3ee: 0f a2                        	cpuid
     3f0: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     3f3: 83 c0 01                     	addl	$0x1, %eax
     3f6: 89 45 f8                     	movl	%eax, -0x8(%rbp)
     3f9: e9 3e ff ff ff               	jmp	0x33c <init_array+0x1fc>
     3fe: b8 01 00 00 00               	movl	$0x1, %eax
     403: b8 5a 00 00 00               	movl	$0x5a, %eax
     408: 0f a2                        	cpuid
     40a: 5d                           	popq	%rbp
     40b: c3                           	retq
     40c: 0f 1f 40 00                  	nopl	(%rax)

0000000000000410 <kernel_gemm>:
     410: 55                           	pushq	%rbp
     411: 48 89 e5                     	movq	%rsp, %rbp
     414: b8 01 00 00 00               	movl	$0x1, %eax
     419: b8 5a 00 00 00               	movl	$0x5a, %eax
     41e: 0f a2                        	cpuid
     420: 89 7d e8                     	movl	%edi, -0x18(%rbp)
     423: 89 75 f0                     	movl	%esi, -0x10(%rbp)
     426: 89 55 ec                     	movl	%edx, -0x14(%rbp)
     429: f2 0f 11 45 c0               	movsd	%xmm0, -0x40(%rbp)
     42e: f2 0f 11 4d c8               	movsd	%xmm1, -0x38(%rbp)
     433: 48 89 4d e0                  	movq	%rcx, -0x20(%rbp)
     437: 4c 89 45 d0                  	movq	%r8, -0x30(%rbp)
     43b: 4c 89 4d d8                  	movq	%r9, -0x28(%rbp)
     43f: c7 45 f8 00 00 00 00         	movl	$0x0, -0x8(%rbp)
     446: b8 01 00 00 00               	movl	$0x1, %eax
     44b: b8 5a 00 00 00               	movl	$0x5a, %eax
     450: 0f a2                        	cpuid
     452: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     455: 3b 45 e8                     	cmpl	-0x18(%rbp), %eax
     458: 0f 8d 9e 01 00 00            	jge	0x5fc <kernel_gemm+0x1ec>
     45e: b8 01 00 00 00               	movl	$0x1, %eax
     463: b8 5a 00 00 00               	movl	$0x5a, %eax
     468: 0f a2                        	cpuid
     46a: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     471: b8 01 00 00 00               	movl	$0x1, %eax
     476: b8 5a 00 00 00               	movl	$0x5a, %eax
     47b: 0f a2                        	cpuid
     47d: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     480: 3b 45 f0                     	cmpl	-0x10(%rbp), %eax
     483: 7d 48                        	jge	0x4cd <kernel_gemm+0xbd>
     485: b8 01 00 00 00               	movl	$0x1, %eax
     48a: b8 5a 00 00 00               	movl	$0x5a, %eax
     48f: 0f a2                        	cpuid
     491: f2 0f 10 45 c8               	movsd	-0x38(%rbp), %xmm0
     496: 48 8b 45 e0                  	movq	-0x20(%rbp), %rax
     49a: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     49e: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     4a5: 48 01 c8                     	addq	%rcx, %rax
     4a8: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     4ac: f2 0f 59 04 c8               	mulsd	(%rax,%rcx,8), %xmm0
     4b1: f2 0f 11 04 c8               	movsd	%xmm0, (%rax,%rcx,8)
     4b6: b8 01 00 00 00               	movl	$0x1, %eax
     4bb: b8 5a 00 00 00               	movl	$0x5a, %eax
     4c0: 0f a2                        	cpuid
     4c2: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     4c5: 83 c0 01                     	addl	$0x1, %eax
     4c8: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     4cb: eb a4                        	jmp	0x471 <kernel_gemm+0x61>
     4cd: b8 01 00 00 00               	movl	$0x1, %eax
     4d2: b8 5a 00 00 00               	movl	$0x5a, %eax
     4d7: 0f a2                        	cpuid
     4d9: c7 45 f4 00 00 00 00         	movl	$0x0, -0xc(%rbp)
     4e0: b8 01 00 00 00               	movl	$0x1, %eax
     4e5: b8 5a 00 00 00               	movl	$0x5a, %eax
     4ea: 0f a2                        	cpuid
     4ec: 8b 45 f4                     	movl	-0xc(%rbp), %eax
     4ef: 3b 45 ec                     	cmpl	-0x14(%rbp), %eax
     4f2: 0f 8d dc 00 00 00            	jge	0x5d4 <kernel_gemm+0x1c4>
     4f8: b8 01 00 00 00               	movl	$0x1, %eax
     4fd: b8 5a 00 00 00               	movl	$0x5a, %eax
     502: 0f a2                        	cpuid
     504: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     50b: b8 01 00 00 00               	movl	$0x1, %eax
     510: b8 5a 00 00 00               	movl	$0x5a, %eax
     515: 0f a2                        	cpuid
     517: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     51a: 3b 45 f0                     	cmpl	-0x10(%rbp), %eax
     51d: 0f 8d 89 00 00 00            	jge	0x5ac <kernel_gemm+0x19c>
     523: b8 01 00 00 00               	movl	$0x1, %eax
     528: b8 5a 00 00 00               	movl	$0x5a, %eax
     52d: 0f a2                        	cpuid
     52f: f2 0f 10 45 c0               	movsd	-0x40(%rbp), %xmm0
     534: 48 8b 45 d0                  	movq	-0x30(%rbp), %rax
     538: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     53c: 48 69 c9 80 25 00 00         	imulq	$0x2580, %rcx, %rcx     # imm = 0x2580
     543: 48 01 c8                     	addq	%rcx, %rax
     546: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
     54a: f2 0f 59 04 c8               	mulsd	(%rax,%rcx,8), %xmm0
     54f: 48 8b 45 d8                  	movq	-0x28(%rbp), %rax
     553: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
     557: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     55e: 48 01 c8                     	addq	%rcx, %rax
     561: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     565: f2 0f 10 0c c8               	movsd	(%rax,%rcx,8), %xmm1
     56a: 48 8b 45 e0                  	movq	-0x20(%rbp), %rax
     56e: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     572: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     579: 48 01 c8                     	addq	%rcx, %rax
     57c: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     580: f2 0f 10 14 c8               	movsd	(%rax,%rcx,8), %xmm2
     585: f2 0f 59 c1                  	mulsd	%xmm1, %xmm0
     589: f2 0f 58 c2                  	addsd	%xmm2, %xmm0
     58d: f2 0f 11 04 c8               	movsd	%xmm0, (%rax,%rcx,8)
     592: b8 01 00 00 00               	movl	$0x1, %eax
     597: b8 5a 00 00 00               	movl	$0x5a, %eax
     59c: 0f a2                        	cpuid
     59e: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     5a1: 83 c0 01                     	addl	$0x1, %eax
     5a4: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     5a7: e9 5f ff ff ff               	jmp	0x50b <kernel_gemm+0xfb>
     5ac: b8 01 00 00 00               	movl	$0x1, %eax
     5b1: b8 5a 00 00 00               	movl	$0x5a, %eax
     5b6: 0f a2                        	cpuid
     5b8: eb 00                        	jmp	0x5ba <kernel_gemm+0x1aa>
     5ba: b8 01 00 00 00               	movl	$0x1, %eax
     5bf: b8 5a 00 00 00               	movl	$0x5a, %eax
     5c4: 0f a2                        	cpuid
     5c6: 8b 45 f4                     	movl	-0xc(%rbp), %eax
     5c9: 83 c0 01                     	addl	$0x1, %eax
     5cc: 89 45 f4                     	movl	%eax, -0xc(%rbp)
     5cf: e9 0c ff ff ff               	jmp	0x4e0 <kernel_gemm+0xd0>
     5d4: b8 01 00 00 00               	movl	$0x1, %eax
     5d9: b8 5a 00 00 00               	movl	$0x5a, %eax
     5de: 0f a2                        	cpuid
     5e0: eb 00                        	jmp	0x5e2 <kernel_gemm+0x1d2>
     5e2: b8 01 00 00 00               	movl	$0x1, %eax
     5e7: b8 5a 00 00 00               	movl	$0x5a, %eax
     5ec: 0f a2                        	cpuid
     5ee: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     5f1: 83 c0 01                     	addl	$0x1, %eax
     5f4: 89 45 f8                     	movl	%eax, -0x8(%rbp)
     5f7: e9 4a fe ff ff               	jmp	0x446 <kernel_gemm+0x36>
     5fc: b8 01 00 00 00               	movl	$0x1, %eax
     601: b8 5a 00 00 00               	movl	$0x5a, %eax
     606: 0f a2                        	cpuid
     608: 5d                           	popq	%rbp
     609: c3                           	retq
     60a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)

0000000000000610 <print_array>:
     610: 55                           	pushq	%rbp
     611: 48 89 e5                     	movq	%rsp, %rbp
     614: 48 83 ec 20                  	subq	$0x20, %rsp
     618: b8 01 00 00 00               	movl	$0x1, %eax
     61d: b8 5a 00 00 00               	movl	$0x5a, %eax
     622: 0f a2                        	cpuid
     624: 89 7d f4                     	movl	%edi, -0xc(%rbp)
     627: 89 75 f0                     	movl	%esi, -0x10(%rbp)
     62a: 48 89 55 e8                  	movq	%rdx, -0x18(%rbp)
     62e: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x635 <print_array+0x25>
     635: 48 8b 38                     	movq	(%rax), %rdi
     638: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     642: b0 00                        	movb	$0x0, %al
     644: e8 00 00 00 00               	callq	0x649 <print_array+0x39>
     649: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x650 <print_array+0x40>
     650: 48 8b 38                     	movq	(%rax), %rdi
     653: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     65d: 48 ba 00 00 00 00 00 00 00 00	movabsq	$0x0, %rdx
     667: b0 00                        	movb	$0x0, %al
     669: e8 00 00 00 00               	callq	0x66e <print_array+0x5e>
     66e: c7 45 f8 00 00 00 00         	movl	$0x0, -0x8(%rbp)
     675: b8 01 00 00 00               	movl	$0x1, %eax
     67a: b8 5a 00 00 00               	movl	$0x5a, %eax
     67f: 0f a2                        	cpuid
     681: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     684: 3b 45 f4                     	cmpl	-0xc(%rbp), %eax
     687: 0f 8d f9 00 00 00            	jge	0x786 <print_array+0x176>
     68d: b8 01 00 00 00               	movl	$0x1, %eax
     692: b8 5a 00 00 00               	movl	$0x5a, %eax
     697: 0f a2                        	cpuid
     699: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     6a0: b8 01 00 00 00               	movl	$0x1, %eax
     6a5: b8 5a 00 00 00               	movl	$0x5a, %eax
     6aa: 0f a2                        	cpuid
     6ac: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     6af: 3b 45 f0                     	cmpl	-0x10(%rbp), %eax
     6b2: 0f 8d a6 00 00 00            	jge	0x75e <print_array+0x14e>
     6b8: b8 01 00 00 00               	movl	$0x1, %eax
     6bd: b8 5a 00 00 00               	movl	$0x5a, %eax
     6c2: 0f a2                        	cpuid
     6c4: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     6c7: 0f af 45 f4                  	imull	-0xc(%rbp), %eax
     6cb: 03 45 fc                     	addl	-0x4(%rbp), %eax
     6ce: b9 14 00 00 00               	movl	$0x14, %ecx
     6d3: 99                           	cltd
     6d4: f7 f9                        	idivl	%ecx
     6d6: 83 fa 00                     	cmpl	$0x0, %edx
     6d9: 75 27                        	jne	0x702 <print_array+0xf2>
     6db: b8 01 00 00 00               	movl	$0x1, %eax
     6e0: b8 5a 00 00 00               	movl	$0x5a, %eax
     6e5: 0f a2                        	cpuid
     6e7: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x6ee <print_array+0xde>
     6ee: 48 8b 38                     	movq	(%rax), %rdi
     6f1: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     6fb: b0 00                        	movb	$0x0, %al
     6fd: e8 00 00 00 00               	callq	0x702 <print_array+0xf2>
     702: b8 01 00 00 00               	movl	$0x1, %eax
     707: b8 5a 00 00 00               	movl	$0x5a, %eax
     70c: 0f a2                        	cpuid
     70e: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x715 <print_array+0x105>
     715: 48 8b 38                     	movq	(%rax), %rdi
     718: 48 8b 45 e8                  	movq	-0x18(%rbp), %rax
     71c: 48 63 4d f8                  	movslq	-0x8(%rbp), %rcx
     720: 48 69 c9 60 22 00 00         	imulq	$0x2260, %rcx, %rcx     # imm = 0x2260
     727: 48 01 c8                     	addq	%rcx, %rax
     72a: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     72e: f2 0f 10 04 c8               	movsd	(%rax,%rcx,8), %xmm0
     733: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     73d: b0 01                        	movb	$0x1, %al
     73f: e8 00 00 00 00               	callq	0x744 <print_array+0x134>
     744: b8 01 00 00 00               	movl	$0x1, %eax
     749: b8 5a 00 00 00               	movl	$0x5a, %eax
     74e: 0f a2                        	cpuid
     750: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     753: 83 c0 01                     	addl	$0x1, %eax
     756: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     759: e9 42 ff ff ff               	jmp	0x6a0 <print_array+0x90>
     75e: b8 01 00 00 00               	movl	$0x1, %eax
     763: b8 5a 00 00 00               	movl	$0x5a, %eax
     768: 0f a2                        	cpuid
     76a: eb 00                        	jmp	0x76c <print_array+0x15c>
     76c: b8 01 00 00 00               	movl	$0x1, %eax
     771: b8 5a 00 00 00               	movl	$0x5a, %eax
     776: 0f a2                        	cpuid
     778: 8b 45 f8                     	movl	-0x8(%rbp), %eax
     77b: 83 c0 01                     	addl	$0x1, %eax
     77e: 89 45 f8                     	movl	%eax, -0x8(%rbp)
     781: e9 ef fe ff ff               	jmp	0x675 <print_array+0x65>
     786: b8 01 00 00 00               	movl	$0x1, %eax
     78b: b8 5a 00 00 00               	movl	$0x5a, %eax
     790: 0f a2                        	cpuid
     792: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x799 <print_array+0x189>
     799: 48 8b 38                     	movq	(%rax), %rdi
     79c: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     7a6: 48 ba 00 00 00 00 00 00 00 00	movabsq	$0x0, %rdx
     7b0: b0 00                        	movb	$0x0, %al
     7b2: e8 00 00 00 00               	callq	0x7b7 <print_array+0x1a7>
     7b7: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0x7be <print_array+0x1ae>
     7be: 48 8b 38                     	movq	(%rax), %rdi
     7c1: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     7cb: b0 00                        	movb	$0x0, %al
     7cd: e8 00 00 00 00               	callq	0x7d2 <print_array+0x1c2>
     7d2: 48 83 c4 20                  	addq	$0x20, %rsp
     7d6: 5d                           	popq	%rbp
     7d7: c3                           	retq
     7d8: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)

00000000000007e0 <polybench_flush_cache>:
     7e0: 55                           	pushq	%rbp
     7e1: 48 89 e5                     	movq	%rsp, %rbp
     7e4: 48 83 ec 20                  	subq	$0x20, %rsp
     7e8: b8 01 00 00 00               	movl	$0x1, %eax
     7ed: b8 5a 00 00 00               	movl	$0x5a, %eax
     7f2: 0f a2                        	cpuid
     7f4: c7 45 f8 00 01 40 00         	movl	$0x400100, -0x8(%rbp)   # imm = 0x400100
     7fb: 48 63 7d f8                  	movslq	-0x8(%rbp), %rdi
     7ff: be 08 00 00 00               	movl	$0x8, %esi
     804: e8 00 00 00 00               	callq	0x809 <polybench_flush_cache+0x29>
     809: 48 89 45 e8                  	movq	%rax, -0x18(%rbp)
     80d: 0f 57 c0                     	xorps	%xmm0, %xmm0
     810: f2 0f 11 45 f0               	movsd	%xmm0, -0x10(%rbp)
     815: c7 45 fc 00 00 00 00         	movl	$0x0, -0x4(%rbp)
     81c: b8 01 00 00 00               	movl	$0x1, %eax
     821: b8 5a 00 00 00               	movl	$0x5a, %eax
     826: 0f a2                        	cpuid
     828: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     82b: 3b 45 f8                     	cmpl	-0x8(%rbp), %eax
     82e: 7d 3a                        	jge	0x86a <polybench_flush_cache+0x8a>
     830: b8 01 00 00 00               	movl	$0x1, %eax
     835: b8 5a 00 00 00               	movl	$0x5a, %eax
     83a: 0f a2                        	cpuid
     83c: 48 8b 45 e8                  	movq	-0x18(%rbp), %rax
     840: 48 63 4d fc                  	movslq	-0x4(%rbp), %rcx
     844: f2 0f 10 04 c8               	movsd	(%rax,%rcx,8), %xmm0
     849: f2 0f 58 45 f0               	addsd	-0x10(%rbp), %xmm0
     84e: f2 0f 11 45 f0               	movsd	%xmm0, -0x10(%rbp)
     853: b8 01 00 00 00               	movl	$0x1, %eax
     858: b8 5a 00 00 00               	movl	$0x5a, %eax
     85d: 0f a2                        	cpuid
     85f: 8b 45 fc                     	movl	-0x4(%rbp), %eax
     862: 83 c0 01                     	addl	$0x1, %eax
     865: 89 45 fc                     	movl	%eax, -0x4(%rbp)
     868: eb b2                        	jmp	0x81c <polybench_flush_cache+0x3c>
     86a: b8 01 00 00 00               	movl	$0x1, %eax
     86f: b8 5a 00 00 00               	movl	$0x5a, %eax
     874: 0f a2                        	cpuid
     876: f2 0f 10 05 00 00 00 00      	movsd	(%rip), %xmm0           # 0x87e <polybench_flush_cache+0x9e>
     87e: 66 0f 2e 45 f0               	ucomisd	-0x10(%rbp), %xmm0
     883: 72 0e                        	jb	0x893 <polybench_flush_cache+0xb3>
     885: b8 01 00 00 00               	movl	$0x1, %eax
     88a: b8 5a 00 00 00               	movl	$0x5a, %eax
     88f: 0f a2                        	cpuid
     891: eb 34                        	jmp	0x8c7 <polybench_flush_cache+0xe7>
     893: b8 01 00 00 00               	movl	$0x1, %eax
     898: b8 5a 00 00 00               	movl	$0x5a, %eax
     89d: 0f a2                        	cpuid
     89f: 48 bf 00 00 00 00 00 00 00 00	movabsq	$0x0, %rdi
     8a9: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     8b3: 48 b9 00 00 00 00 00 00 00 00	movabsq	$0x0, %rcx
     8bd: ba 7b 00 00 00               	movl	$0x7b, %edx
     8c2: e8 00 00 00 00               	callq	0x8c7 <polybench_flush_cache+0xe7>
     8c7: b8 01 00 00 00               	movl	$0x1, %eax
     8cc: b8 5a 00 00 00               	movl	$0x5a, %eax
     8d1: 0f a2                        	cpuid
     8d3: 48 8b 7d e8                  	movq	-0x18(%rbp), %rdi
     8d7: e8 00 00 00 00               	callq	0x8dc <polybench_flush_cache+0xfc>
     8dc: 48 83 c4 20                  	addq	$0x20, %rsp
     8e0: 5d                           	popq	%rbp
     8e1: c3                           	retq
     8e2: 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00    	nopw	%cs:(%rax,%rax)

00000000000008f0 <polybench_prepare_instruments>:
     8f0: 55                           	pushq	%rbp
     8f1: 48 89 e5                     	movq	%rsp, %rbp
     8f4: b8 01 00 00 00               	movl	$0x1, %eax
     8f9: b8 5a 00 00 00               	movl	$0x5a, %eax
     8fe: 0f a2                        	cpuid
     900: e8 00 00 00 00               	callq	0x905 <polybench_prepare_instruments+0x15>
     905: 5d                           	popq	%rbp
     906: c3                           	retq
     907: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)

0000000000000910 <polybench_timer_start>:
     910: 55                           	pushq	%rbp
     911: 48 89 e5                     	movq	%rsp, %rbp
     914: b8 01 00 00 00               	movl	$0x1, %eax
     919: b8 5a 00 00 00               	movl	$0x5a, %eax
     91e: 0f a2                        	cpuid
     920: e8 00 00 00 00               	callq	0x925 <polybench_timer_start+0x15>
     925: e8 16 00 00 00               	callq	0x940 <rtclock>
     92a: f2 0f 11 04 25 00 00 00 00   	movsd	%xmm0, 0x0
     933: 5d                           	popq	%rbp
     934: c3                           	retq
     935: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)

0000000000000940 <rtclock>:
     940: 55                           	pushq	%rbp
     941: 48 89 e5                     	movq	%rsp, %rbp
     944: b8 01 00 00 00               	movl	$0x1, %eax
     949: b8 5a 00 00 00               	movl	$0x5a, %eax
     94e: 0f a2                        	cpuid
     950: 0f 57 c0                     	xorps	%xmm0, %xmm0
     953: 5d                           	popq	%rbp
     954: c3                           	retq
     955: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)

0000000000000960 <polybench_timer_stop>:
     960: 55                           	pushq	%rbp
     961: 48 89 e5                     	movq	%rsp, %rbp
     964: b8 01 00 00 00               	movl	$0x1, %eax
     969: b8 5a 00 00 00               	movl	$0x5a, %eax
     96e: 0f a2                        	cpuid
     970: e8 cb ff ff ff               	callq	0x940 <rtclock>
     975: f2 0f 11 04 25 00 00 00 00   	movsd	%xmm0, 0x0
     97e: 5d                           	popq	%rbp
     97f: c3                           	retq

0000000000000980 <polybench_timer_print>:
     980: 55                           	pushq	%rbp
     981: 48 89 e5                     	movq	%rsp, %rbp
     984: b8 01 00 00 00               	movl	$0x1, %eax
     989: b8 5a 00 00 00               	movl	$0x5a, %eax
     98e: 0f a2                        	cpuid
     990: f2 0f 10 04 25 00 00 00 00   	movsd	0x0, %xmm0
     999: f2 0f 5c 04 25 00 00 00 00   	subsd	0x0, %xmm0
     9a2: 48 bf 00 00 00 00 00 00 00 00	movabsq	$0x0, %rdi
     9ac: b0 01                        	movb	$0x1, %al
     9ae: e8 00 00 00 00               	callq	0x9b3 <polybench_timer_print+0x33>
     9b3: 5d                           	popq	%rbp
     9b4: c3                           	retq
     9b5: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)

00000000000009c0 <polybench_free_data>:
     9c0: 55                           	pushq	%rbp
     9c1: 48 89 e5                     	movq	%rsp, %rbp
     9c4: 48 83 ec 10                  	subq	$0x10, %rsp
     9c8: b8 01 00 00 00               	movl	$0x1, %eax
     9cd: b8 5a 00 00 00               	movl	$0x5a, %eax
     9d2: 0f a2                        	cpuid
     9d4: 48 89 7d f8                  	movq	%rdi, -0x8(%rbp)
     9d8: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
     9dc: e8 00 00 00 00               	callq	0x9e1 <polybench_free_data+0x21>
     9e1: 48 83 c4 10                  	addq	$0x10, %rsp
     9e5: 5d                           	popq	%rbp
     9e6: c3                           	retq
     9e7: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)

00000000000009f0 <polybench_alloc_data>:
     9f0: 55                           	pushq	%rbp
     9f1: 48 89 e5                     	movq	%rsp, %rbp
     9f4: 48 83 ec 20                  	subq	$0x20, %rsp
     9f8: b8 01 00 00 00               	movl	$0x1, %eax
     9fd: b8 5a 00 00 00               	movl	$0x5a, %eax
     a02: 0f a2                        	cpuid
     a04: 48 89 7d e0                  	movq	%rdi, -0x20(%rbp)
     a08: 89 75 f4                     	movl	%esi, -0xc(%rbp)
     a0b: 48 8b 45 e0                  	movq	-0x20(%rbp), %rax
     a0f: 48 89 45 f8                  	movq	%rax, -0x8(%rbp)
     a13: 48 63 45 f4                  	movslq	-0xc(%rbp), %rax
     a17: 48 0f af 45 f8               	imulq	-0x8(%rbp), %rax
     a1c: 48 89 45 f8                  	movq	%rax, -0x8(%rbp)
     a20: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
     a24: e8 17 00 00 00               	callq	0xa40 <xmalloc>
     a29: 48 89 45 e8                  	movq	%rax, -0x18(%rbp)
     a2d: 48 8b 45 e8                  	movq	-0x18(%rbp), %rax
     a31: 48 83 c4 20                  	addq	$0x20, %rsp
     a35: 5d                           	popq	%rbp
     a36: c3                           	retq
     a37: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)

0000000000000a40 <xmalloc>:
     a40: 55                           	pushq	%rbp
     a41: 48 89 e5                     	movq	%rsp, %rbp
     a44: 48 83 ec 20                  	subq	$0x20, %rsp
     a48: b8 01 00 00 00               	movl	$0x1, %eax
     a4d: b8 5a 00 00 00               	movl	$0x5a, %eax
     a52: 0f a2                        	cpuid
     a54: 48 89 7d e0                  	movq	%rdi, -0x20(%rbp)
     a58: 48 c7 45 f8 00 00 00 00      	movq	$0x0, -0x8(%rbp)
     a60: 48 8b 04 25 00 00 00 00      	movq	0x0, %rax
     a68: 48 83 c0 00                  	addq	$0x0, %rax
     a6c: 48 89 04 25 00 00 00 00      	movq	%rax, 0x0
     a74: 48 8b 45 e0                  	movq	-0x20(%rbp), %rax
     a78: 48 03 04 25 00 00 00 00      	addq	0x0, %rax
     a80: 48 89 45 e8                  	movq	%rax, -0x18(%rbp)
     a84: 48 8b 55 e8                  	movq	-0x18(%rbp), %rdx
     a88: 48 8d 7d f8                  	leaq	-0x8(%rbp), %rdi
     a8c: be 00 10 00 00               	movl	$0x1000, %esi           # imm = 0x1000
     a91: e8 00 00 00 00               	callq	0xa96 <xmalloc+0x56>
     a96: 89 45 f4                     	movl	%eax, -0xc(%rbp)
     a99: 48 83 7d f8 00               	cmpq	$0x0, -0x8(%rbp)
     a9e: 74 12                        	je	0xab2 <xmalloc+0x72>
     aa0: b8 01 00 00 00               	movl	$0x1, %eax
     aa5: b8 5a 00 00 00               	movl	$0x5a, %eax
     aaa: 0f a2                        	cpuid
     aac: 83 7d f4 00                  	cmpl	$0x0, -0xc(%rbp)
     ab0: 74 31                        	je	0xae3 <xmalloc+0xa3>
     ab2: b8 01 00 00 00               	movl	$0x1, %eax
     ab7: b8 5a 00 00 00               	movl	$0x5a, %eax
     abc: 0f a2                        	cpuid
     abe: 48 8b 05 00 00 00 00         	movq	(%rip), %rax            # 0xac5 <xmalloc+0x85>
     ac5: 48 8b 38                     	movq	(%rax), %rdi
     ac8: 48 be 00 00 00 00 00 00 00 00	movabsq	$0x0, %rsi
     ad2: b0 00                        	movb	$0x0, %al
     ad4: e8 00 00 00 00               	callq	0xad9 <xmalloc+0x99>
     ad9: bf 01 00 00 00               	movl	$0x1, %edi
     ade: e8 00 00 00 00               	callq	0xae3 <xmalloc+0xa3>
     ae3: b8 01 00 00 00               	movl	$0x1, %eax
     ae8: b8 5a 00 00 00               	movl	$0x5a, %eax
     aed: 0f a2                        	cpuid
     aef: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
     af3: 48 83 c4 20                  	addq	$0x20, %rsp
     af7: 5d                           	popq	%rbp
     af8: c3                           	retq
