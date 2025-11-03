	.file	"simple.c"
	.text
	.globl	addmul                          # -- Begin function addmul
	.p2align	4
	.type	addmul,@function
addmul:                                 # @addmul
	.cfi_startproc
# %bb.0:                                # %entry
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	leal	(%rsi,%rdi), %ecx
	movl	%esi, %eax
	imull	%edi, %eax
	cmpl	%esi, %edi
	cmovgl	%ecx, %eax
	retq
.Lfunc_end0:
	.size	addmul, .Lfunc_end0-addmul
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	movl	$64, %eax
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 21.1.4 (https://www.github.com/llvm/llvm-project.git 222fc11f2b8f25f6a0f4976272ef1bb7bf49521d)"
	.section	".note.GNU-stack","",@progbits
