/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 2009-2011 by Synge Todo <wistaria@comp-phys.org>,
*                            Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

#ifndef LOOPER_ATOMIC_IMPL_H
#define LOOPER_ATOMIC_IMPL_H

//
// compare_and_swap
//

#if defined(__linux__) && defined(__x86_64__) && defined(__FCC_VERSION) && !defined(_GNU_SOURCE)

// compare_and_swap assembler code written by Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>

// argument 1: %rdi (int& variable)
// argument 2: %esi (int  oldval)
// argument 3: %edx (int  newval)

asm(".align  16");
asm(".section        .gnu.linkonce.t.compare_and_swap,\"ax\",@progbits");
asm(".weak   compare_and_swap");
asm("compare_and_swap:");
asm("  movl    %esi, %eax");      // copy oldval to %eax
asm("  lock");
asm("  cmpxchgl %edx, (%rdi)");   // compare oldval(%eax) and *variable
asm("  sete %dl");                // set %dl value of ZF
asm("  xorl %eax, %eax");         // set %eax 0 (x86 common practice)
asm("  testb %dl, %dl");          // %dl && %dl. If 0, ZF=1 elseif 1, ZF=0
asm("  setne %al");               // retrun value (%al is a part of %eax)
asm("  ret");
asm(".size   compare_and_swap,.-compare_and_swap");
asm(".type   compare_and_swap,@function");


#elif defined(__linux__) && defined(__sparc) && defined(__FCC_VERSION) && !defined(_GNU_SOURCE)

asm(".section \".text\"");
asm(".align 4");
asm(".global atomic_cas_uint");
asm(".type atomic_cas_uint, #function");
asm("atomic_cas_uint:");
asm("      cas   [%o0],%o1,%o2");
asm("      retl");
asm("      mov   %o2,%o0");
asm(".size  atomic_cas_uint, (.-atomic_cas_uint)");

#endif

#endif // LOOPER_ATOMIC_IMPL_H
