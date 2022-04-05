# CUDA inline assembly和PTX ISA

## inline assembly

官方参考：https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

CUDA inline assembly使用语法和C/C++相同（除不允许设置clobbered registers）

```c++
       asm ( "assembler template string" 
           : "constraint"(output operands)                  /* optional */
           : "constraint"(input operands)                   /* optional */
           );
```

### 引用

`"assembler template string"`中可以用`%n`（n为从0开始的整数）引用操作数。编号顺序和在出现在输出操作数、输入操作数中的先后顺序一致。

### constriant

规定使用相应的PTX寄存器类型

```c++
"h" = .u16 reg	// u表示无符号 16表示位数
"r" = .u32 reg
"l" = .u64 reg
"f" = .f32 reg  // f表示浮点寄存器
"d" = .f64 reg
```

### 补充（modifier）

由于官方参考关于操作数（operand）的**修饰符**描述较少，这里稍作简略描述（参考[OpenXL C/C++ inline asemmbly](https://www.ibm.com/docs/en/openxl-c-and-cpp-aix/17.1.0?topic=features-inline-assembly-statements)）：

- `=`：表示该操作数**只写**（write-only），之前的值会被弃用，被这一新的输出数据取而代之
- `+`：表示该操作数**既可读又可写**

输出操作数必须带上修饰符`=`或`+`，输入操作数由于官方文档并未说明可直接认为修饰符可选（或没有）



## PTX ISA

### 概述

PTX：Parallel Thread Execution

PTX ISA比CUDA C++更底层，但编程模型完全相同，仅在一些**称谓**上有所区别，例如

CTA（Cooperative Thread Array）：原来的Block（PS：Grid这一概念依然保持不变）



### PTX ASM Example

```c++
        asm("{\n\t"
                ".reg .s32 b;\n\t"
                ".reg .pred p;\n\t"
                "add.cc.u32 %1, %1, %2;\n\t"
                "addc.s32 b, 0, 0;\n\t"
                "sub.cc.u32 %0, %0, %2;\n\t"
                "subc.cc.u32 %1, %1, 0;\n\t"
                "subc.s32 b, b, 0;\n\t"
                "setp.eq.s32 p, b, 1;\n\t"
                "@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
                "@p addc.u32 %1, %1, 0;\n\t"
                "}"
                : "+r"(x[0]), "+r"(x[1])
                : "r"(x[2]));
```



### 变量声明

Example第2、3行均为变量声明语句，单独来看

```assembly
.reg .s32 b;
.reg .pred p;
```

由三部分组成：`State Space + Type + Identifier`

#### State Space

规定存储位置，带有`.`前缀，区别标识符

| 名称    | 描述                                    |
| ------- | --------------------------------------- |
| .reg    | 寄存器，访存快                          |
| .sreg   | 特殊寄存器。只读；预定义；平台有关      |
| .const  | 共享内存（Shared memory），只读         |
| .global | 全局内存（Global memory），全部线程共享 |
| .local  | 本地内存（Local memory），每个线程私有  |
| .param  | Kernel parameter                        |
| .shared | 可按地址访问的共享内存，一个CTA内共享   |

具体特性参见[官方文档State Spaces](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces)

#### Type

| 基本类型       | 对应符号                 |
| -------------- | ------------------------ |
| 有符号整数     | .s8, .s16, .s32, .s64    |
| 无符号整数     | .u8, .u16, .u32, .u64    |
| 浮点数         | .f16, .f16x2, .f32, .f64 |
| 位串（无类型） | .b8, .b16, .b32, .b64    |
| 谓词           | .pred                    |

#### Identifier

标识符基本和所有语言规定相同，细微区别参见[官方文档Identifier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#identifiers)

---

回到例子

```assembly
.reg .s32 b;	// 有符号32位整数变量b，使用寄存器存放
.reg .pred p;	// 谓词变量p（用于存放谓词逻辑结果，布尔值），使用寄存器存放
```

关于谓词的使用方法下文会有详细描述



### 指令集

#### 指令格式

```assembly
@p opcode;
@p opcode a;
@p opcode d, a;
@p opcode d, a, b;
@p opcode d, a, b, c;
```

- 注意指令的操作数个数，以及操作数的含义（源、目的等等）
- 最左边的`@p`是可选的`guard predicate`，即根据对应谓词结果选择是否执行该条指令

#### 指令类型信息

多类型指令必须带上类型及大小描述符，例如Example中第4行`add.cc.u32 %1, %1, %2;`中`.u32`表示进行无符号32位整数的加法。

#### 扩展精度的整数运算

主要用于处理进位，例如Example中的4、5行：

```assembly
add.cc.u32 %1, %1, %2;
addc.s32 b, 0, 0;
```

`add.cc`表示该条指令会改写条件码（Condition Code，简称CC）寄存器中的进位标志位（Carry Flag，简称CF）；`addc`表示执行带进位的加法，也就是说除了源操作数外还会加上`CC.CF`。

其它扩展运算指令参考官方文档[Extended-Precision Integer Arithmetic Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions)

#### 谓词逻辑执行分支

谓词寄存器本质上是虚拟的寄存器，用于处理PTX中的分支（类比其他ISA的条件跳转指令`beq`等）

1. 声明谓词寄存器

   如Example中第3行`.reg .pred p`，声明谓词变量p

2. `step`指令给谓词变量绑定具体谓词逻辑

   如Example中第9行`setp.eq.s32 p, b, 1`，`setp`指set predicate register

   基本语法格式：`setp.CmpOp.type         p, a, b`

   - `type`：规定源操作数`a,b`的类型
   - `CmpOp`：比较运算符
   - `p`必须是`.pred`类型变量

   关于`step`指令的详细用法参见官方文档[Comparison and Selection Instructions: setp](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)

3. 设置条件分支

   如Example中10、11行

   ```assembly
   @p add.cc.u32 %0, %0, 0xffffffff;
   @p addc.u32 %1, %1, 0;
   ```

   `@p`表示当`p=True`时，执行该条指令；而`@!p`则表示`p=False`时执行。



### From PTX ASM Eample to C-like language

```c++
// 函数功能：X mod p
// p = 0xFFFFFFFF00000001
// x = {x[0], x[1], x[2]}
// x[0]: LSW(least significant word) word=32-bit
void _uint96_modP(uint32 *x) {
	x[1] += x[2];
    int b = /*上条指令(i.e., x[1] + x[2])发生溢出*/ ? 1 : 0;
    x[0] -= x[2];
    x[1] -= /*上条指令(i.e., x[0] - x[2])发生溢出*/ ? 1 : 0;
    if (b == 1) {
        x[0] += 0xFFFFFFFF; // x[0] += UINT_MAX
        x[1] += /*上条指令发生溢出*/ ? 1 : 0;
    }
}
```

$$
\begin{split}
X&=x_0+x_1\cdot 2^{32}+x_2\cdot 2^{64} \mod {(P=2^{64}-2^{32}+1)}\\
 &=x_0+x_1\cdot 2^{32}+x_2\cdot 2^{64}-x_2P \mod P\\
 &=(x_0-x_2)+(x_1+x_2)\cdot 2^{32} \mod P
\end{split}
$$

代码中`x[0] = x[0] - x[2]`，`x[0]`位长为32，可能发生溢出即实际所求为模$2^{32}$的结果$[x_0-x_2]_{2^{32}}=x_0-x_2+2^{32}$，额外加的$2^{32}$相当于从高32位$x_1$借位，体现在上面代码第9行；

同样，`x[1] = x[1] + x[2]`，也可能溢出$[x_1+x_2]_{2^{32}}=x_1+x_2-2^{32}$

于是$X$中应该继续处理多减掉的$2^{32}$，
$$
\begin{split}
 &([x_1+x_2]_{2^{32}}+2^{32})\cdot2^{32}\mod P\\
=&[x_1+x_2]_{2^{32}}\cdot2^{32}+\cdot2^{64}-P\mod P\\
=&[x_1+x_2]_{2^{32}}+(2^{32} - 1) \mod P
\end{split}
$$
$2^{32} - 1=0\text {xffff ffff}$应该加到低32位，对应代码第11、12行（包括处理来自低32位的进位）

----

## Example2

```c++
 uint64 _mul_modP(uint64 x, uint64 y) {
	volatile register uint32 mul[4]; // NEVER REMOVE VOLATILE HERE!!!
    // 128-bit = 64-bit * 64-bit
    asm("mul.lo.u32 %0, %4, %6;\n\t"
        "mul.hi.u32 %1, %4, %6;\n\t"
        "mul.lo.u32 %2, %5, %7;\n\t"
        "mul.hi.u32 %3, %5, %7;\n\t"
        "mad.lo.cc.u32 %1, %4, %7, %1;\n\t"
        "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
        "addc.u32 %3, %3, 0;\n\t"
        "mad.lo.cc.u32 %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
        "addc.u32 %3, %3, 0;\n\t"
        : "+r"(mul[0]), "+r"(mul[1]), "+r"(mul[2]), "+r"(mul[3])
        : "r"(((uint32 *)&x)[0]), "r"(((uint32 *)&x)[1]),
        "r"(((uint32 *)&y)[0]), "r"(((uint32 *)&y)[1]));
 	/* ... */
 }

// output: %0 = mul[0], %1 = mul[1], %2 = mul[2], %3 = mul[3]
// input: %4 = x[0..31], %5 = x[32...63], %6 = y[0...31], %7 = y[32...63]
```

### mul指令

语法如下：

```assembly
mul.mode.type  d, a, b;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

- type规定操作数类型和大小
- mode有三种：当使用`.wide`时，要求目的操作数d的长度为源操作数a,b的两倍，PTX规定此时type只能使用位长16和32的类型；当使用`.hi`和`.lo`时对type不作要求，此时d的位长和a,b相同，分别取高位和低位。

### mad指令

"multiplication and addition"的意思，自然有四个操作数，具体语法如下：

```assembly
mad.mode.type  d, a, b, c;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

- mode和type规定决定乘法的中间值，完全同上；c的位宽和d相同
- 由于涉及加法，同样有和CC寄存器相关的扩展指令：mad.cc、madc等

### 翻译为C-like language

```c++
// output: %0 = mul[0], %1 = mul[1], %2 = mul[2], %3 = mul[3]
// input: %4 = x[0..31], %5 = x[32...63], %6 = y[0...31], %7 = y[32...63]
__inline__ __device__
uint64 _mul_modP(uint64 x, uint64 y) {
	volatile register uint32 mul[4]; // NEVER REMOVE VOLATILE HERE!!!
	mul[0] = (x[0:31] * y[0:31])[0:31];
	mul[1] = (x[0:31] * y[0:31])[32:63];
	mul[2] = (x[32:63] * y[32:63])[0:31];
	mul[3] = (x[32:63] * y[32:63])[32:63];
 	mul[1] += (x[0:31] * y[32:63])[0:31];
 	mul[2] += (x[0:31] * y[32:63])[32:63] + /*上条指令发生溢出*/ ? 1 : 0;
	mul[3] += /*上条指令发生溢出*/ ? 1 : 0;
 	mul[1] += (x[32:63] * y[0:31])[0:31];
	mul[2] += (x[32:63] * y[0:31])[32:63] + /*上条指令发生溢出*/ ? 1 : 0;
	mul[3] += /*上条指令发生溢出*/ ? 1 : 0;
    _uint128_modP(mul);
    if (*(uint64 *)mul > valP)
            *(uint64 *)mul -= valP;
        return *(uint64 *)mul;
}
```

$X=x_0 + x_1\cdot2^{32},Y=y_0+y_1\cdot2^{32}$，注意到$x_i,y_i, i =0,1$都是一字长（32位）
$$
\begin{split}
XY
&=(x_0 + x_1\cdot2^{32})(y_0+y_1\cdot2^{32})\\
&=x_0y_0+(x_0y_1+x_1y_0)\cdot2^{32}+x_1y_1\cdot2^{64}
\end{split}
$$
乘法结果为4字长，可以复用`_uint128_modP`，而$x_iy_j$为2字长，继续分解
$$
XY=mul_0+(mul_1+x_0y_1[0:31]+x_1y_0[0:31])\cdot2^{32}+(mul_2+x_0y_1[32:63]+x_1y_0[32:63])\cdot2^{64}+mul_3\cdot2^{96}
$$
其中$mul_{0\sim3}$的计算对应代码6~9行；$mul_1+x_0y_1[0:31]$和$mul_2+x_0y_1[32:63]$（包括加法进位处理）对应代码10~12行，$mul_1+x_1y_0[0:31]$和$mul_2+x_1y_0[32:63]$（包括加法进位处理）对应代码13~15行
