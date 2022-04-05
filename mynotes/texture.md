## Texture Memory

作为图形世界里的一个特征，纹理是在多边形上拉伸、旋转和粘贴的图像，以形成我们熟悉的3D图形。而在GPU计算中使用纹理内存是对CUDA程序员的进阶建议，纹理内存允许像数组一样随机访问并且通过cache提高带宽。

总体来说，CUDA提供了两套不同的API访问纹理内存：

- texture reference API（支持所有设备）
- texture object API（仅支持capacity $\ge3.\text x$ ）

设备执行kernel时，通过Texture Functions（显然属于device functions）访问纹理内存，引出一个称呼`texture fetch`——调用texture function之一访问纹理内存的过程。

每一个`texture fetch`都规定了一个参数`texture object`（如果使用texture object API）或`texture reference `（如果使用texture reference API）

### texture object / texture reference

1. `texture`：希望访问的一段纹理内存，可以是设备中一段线性内存或是一个CUDA数组

   - `texture object`：运行时创建，`texture`在其创建时也随之指定
   - `texture reference`：编译时创建，通过运行时绑定`texture reference`到某段纹理内存完成`texture`的指定；不同的`texture reference`可以绑定到相同`texture`或是内存有重叠的`texture`

2. `dimensionality`：定义`texture`的维度，将`texture`看作一维、二维或三维数组。

   - `texels`：texture elements的简称，即数组中的元素
   - `texture width & texture height & texture depth`：数组各维度的大小

3. `type`：`texel`的类型，只能为以下几种类型

   - 基本整数类型（8位、16位、32位和64位整型）

   - 单精度浮点数类型

   - 由基本整型和单精度度浮点数生成的内置向量类型，且组件个数只能为1、2和4

     PS：内置向量类型实际为结构体，CUDA允许其有2、3和4个成员，并且通过`x`，`y`，`z`和`w`分别访问第一，第二，第三和第四个成员；构造函数形如`make_<type name>`，例如`int2 make_int2(int x, int y)`生成一个`int2`类型的值`(x,y)`。

     详细用法参见官方文档[Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types)

4. `read mode`：读数据模式

   - `cudaReadModeElementType`：不发生转换
   - `cudaReadModeNormalizedFloat`：当`type`为16-bit或8-bit整数（`short/ushort`或`char/uchar`，i.e.不适用于32位整数）时，`texture fetch`返回的值均匀地映射为$[0.0, 1.0]$（对于无符号类型，i.e. `ushort/uchar`）或$[-1.0, 1.0]$（对于有符号数，i.e. `short/char`）中的浮点数

5. 是否标准化`texture`坐标

   - 非标准化坐标：$[0, N-1]$
   - 标准化坐标：$[0.0,1.0-\frac 1 N]$，其中N是某一维的长度

6. `addressing mode`：译址模式，规定坐标越界的行为

   - `cudaAddressModeClamp`：**默认模式**，要求坐标必须有效。对于非标准化坐标$[0, N)$，对于标准化坐标$[0.0, 1.0)$
   - `cudaAddressModeBorder`：允许坐标越界，且越界时返回0
   - `cudaAddressModeWrap`：仅使用标准化坐标时可用，对于坐标x，只取其小数部分$\langle x\rangle = x-\lfloor x\rfloor$
   - `cudaAddressModeMirror`：仅使用标准化坐标时可用，对于坐标x，取离x最近偶数与x的绝对值（i.e.，当$\lfloor x\rfloor$为偶数时，取$\langle x\rangle$；当$\lfloor x\rfloor$为奇数时，取$1-\langle x\rangle$）

7. `filtering mode`

   - `cudaFilterModePoint`：返回其数组下标离给定坐标最近的`texel`
   - `cudaFilterModeLinear`：（要求`texture fetch`必须返回浮点数）考虑给定坐标附近的`texels`，以线性差值的形式返回。当纹理数组为一维时，返回两个数的线性差值；二维时，返回四个数的；三维时，返回八个数的

## Texture Object

`texture object`由`cudaCreateTextureObject()`函数根据资源描述符（`struct cudaResourceDesc`）和纹理描述符（`struct cudaTextureDesc`）创建而来

### 资源描述符（struct cudaResourceDesc）

```c++
struct cudaResourceDesc {
    enum cudaResourceType resType;

    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
};
```

1. `cudaResourceType`：使用何种资源，选定后只有联合结构体`res`中的一个结构有效

   - `cudaResourceTypeArray`：使用CUDA数组
   - `cudaResourceTypeMipmappedArray`：使用CUDA映射数组（mipmapped array）
   - `cudaResourceTypeLinear`：使用设备上一段线性内存。
   - `cudaResourceTypePitch2D`

2. CUDA数组

   - `cudaResourceDesc::resType`设置为`cudaResourceTypeArray`
   - `cudaResourceDesc::res::array::array`必须设置为有效的CUDA数组

3. CUDA映射数组（mipmapped array）

   - `cudaResourceDesc::resType`设置为`cudaResourceTypeMipmappedArray`
   - `cudaResourceDesc::res::mipmap::mipmap`必须设置为有效
   - `texture`必须使用标准化坐标（也即`cudaTextureDesc::normalizedCoords`必须置为1）

4. 设备上的线性内存

   - `cudaResourceDesc::resType`设置为`cudaResourceTypeLinear`
   - `cudaResourceDesc::res::linear::devPtr`必须指向设备上一段有效内存
   - ` cudaResourceDesc::res::linear::desc`描述`texel`的类型和位长（当`texel`为内置向量类型时，每个维度的位长均要记录）
   - `cudaResourceDesc::res::linear::sizeInBytes`表示这段线性内存所占字节空间

   PS1：`# of texels = sizeInBytes / sizeof(type)`，注意元素个数不能超过`cudaDeviceProp::maxTexture1DLinear`（也可以由`cudaDeviceGetTexture1DLinearMaxWidth()`函数查询设备信息获得）

   PS2：`struct cudaChannelFormatDesc`定义如下

   ```c++
   struct cudaChannelFormatDesc {
   	int x, y, z, w;
       enum cudaChannelFormatKind f;
   };
   ```

   这里`f`只能为`cudaChannelFormatKindSigned`，`cudaChannelFormatKindUnsigned`， `cudaChannelFormatKindFloat`之一（顾名思义，分别表示无符号数，有符号数和浮点数）；`x`，`y`，`z`和`w`分别表示该`texel`各组件的位长

### 纹理描述符（struct  cudaTextureDesc）

```c++
struct cudaTextureDesc {
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    float                       borderColor[4];
    int                         normalizedCoords;	// 0 or 1
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
    int                         disableTrilinearOptimization;
    int                         seamlessCubemap;
};
```

1. `cudaTextureDesc::addressMode`：（**当`cudaResourceDesc::resType`为`cudaResourceTypeLinear`时，忽略这一项设置**）译址模式，在Texture Memory一小节中有详细讲述

   ```c++
   enum cudaTextureAddressMode {
       cudaAddressModeWrap   = 0,
       cudaAddressModeClamp  = 1,
       cudaAddressModeMirror = 2,
       cudaAddressModeBorder = 3
   };
   ```

2. `cudaTextureDesc::filterMode`：（**当`cudaResourceDesc::resType`为`cudaResourceTypeLinear`时，忽略这一项设置**）过滤模式，在Texture Memory一小节中有详细讲述

   ```c++
   enum cudaTextureFilterMode {
   	cudaFilterModePoint  = 0,
   	cudaFilterModeLinear = 1
   };
   ```

3. `cudaTextureDesc::readMode`：读数据模式，在Texture Memory一小节中有详细讲述

   ```c++
   enum cudaTextureReadMode {
       cudaReadModeElementType     = 0,
       cudaReadModeNormalizedFloat = 1
   };
   ```

4. `cudaTextureDesc::borderColor`：只有`cudaTextureDesc::addressMode`设置为`cudaAddressModeBorder`时有效。数组borderColor的4个元素依次表示RGBA的浮点值

5. `cudaTextureDesc::normalizedCoords`：是否使用标准化坐标

6. 其他不太常用的省略，具体参考[cudaCreateTextureObject函数的API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html)

## Texture Reference

与`texture object`不同，`texture reference`的某些属性在运行时是不变的，必须在编译期间明确。这些属性在**声明`texture reference`**时指定

### `texture reference`的声明

1. 必须在文件作用域声明（file scope, i.e. global scope，全局作用域）
2. 声明后本质作为静态全局变量，不能当做函数的参数
3. 声明为`texture`类型变量：`texture<DataType, Type, ReadMode> texRef;`
   - `DataType`：指定`texel`的类型
   - `Type`：（可选参数，默认为`cudaTextureType1D`）指定`texture reference`的类型。设置为`cudaTextureType1D`，`cudaTextureType2D`和`cudaTextureType3D`分别表示一维，二维和三维纹理
   - `ReadMode`：（可选参数，默认为`cudaReadModeElementType`）指定读数据模式。在Texture Memory一小节中有详细讲述

### struct textureReference

`texture reference`声明时指定一些运行时不变的属性，而其它属性则是在host运行时可变的，这些属性在`struct textureReference`中描述

```c++
struct textureReference {
    int                          normalized;	// 0 or 1
    enum cudaTextureFilterMode   filterMode;
    enum cudaTextureAddressMode  addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
    int                          sRGB;
    unsigned int                 maxAnisotropy;
    enum cudaTextureFilterMode   mipmapFilterMode;
    float                        mipmapLevelBias;
    float                        minMipmapLevelClamp;
    float                        maxMipmapLevelClamp;
}
```

1. `normalized`：作用和描述`textureObject`的描述符`cudaTextureDesc`中的`normalizedCoords`相同

2. `filterMode`、`addressMode`等属性也和描述符`cudaTextureDesc`中属性完全对应，直接参考上文，这里不再赘述

3. `channelDesc`：描述`texel`的格式，`texture object`中资源类型使用设备上线性内存时PS2中有补充讲述，其类型实际为结构体`struct cudaChannelFormatDesc`

   ```c++
   struct cudaChannelFormatDesc {
     int x, y, z, w;
     enum cudaChannelFormatKind f;
   };
   ```

   - 其指定内容必须和声明时的`DataType`匹配
   - 由于`DataType`可能是最多有4个组件的CUDA内置向量类型，所以成员`x`，`y`，`z`和`w`分别表示各个组件的位长
   - 枚举类型`enum cudaChannelFormatKind`可使用三个值：`cudaChannelFormatKindSigned`，`cudaChannelFormatKindUnsigned`和`cudaChannelFormatKindFloat`分别表示这些组件为有符号整数，无符号整数和浮点数类型

PS：CUDA C++的高层接口实现的`texture`类型实际继承C风格的结构体`struct textureReference`

### Bind/UnBind

在kernel可以使用`texture reference`从纹理内存中读数据时，必须使用`cudaBindTexture()`或`cudaBindTexture2D()`将**设备上的**线性内存绑定到纹理上；或者使用`cudaBindTextureToArray()`绑定为CUDA数据

1. 低层级（C风格）接口的`cudaBindTexture()`

   ```c
   __host__ cudaError_t cudaBindTexture ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size = UINT_MAX )
   ```

   将`devPtr`指向的`size`大小内存区域绑定到纹理引用`texref`上

   - `desc`描述读数据时，如何解释一个元素（`texel`的格式）
   - `texref`之前绑定的内存区域会被解除绑定
   - `offset`：由于硬件会强制纹理内存的基址对齐，所以`cudaBindTexture()`返回的`*offset`表示对绑定内存区域的一个位移修正，使用`texture functions`进行texture fetch时必须考虑该值。`cudaMalloc()`返回的设备内存指针确保基址对齐，可以不用考虑`offset`参数（直接设置为0或`NULL`）

2. 高层级（C++风格）接口的`cudaBindTexture()`

   ```c++
   template < class T, int dim, enum cudaTextureReadMode readMode >
   __host__ cudaError_t cudaBindTexture ( size_t* offset, const texture < T, dim, readMode > & tex, const void* devPtr, size_t size = UINT_MAX ) [inline]
       
   template < class T, int dim, enum cudaTextureReadMode readMode >
   __host__ cudaError_t cudaBindTexture ( size_t* offset, const texture < T, dim, readMode > & tex, const void* devPtr, const cudaChannelFormatDesc& desc, size_t size = UINT_MAX ) [inline]
   ```

   `const cudaChannelFormatDesc& desc`变为C++引用类型

---

## Example：specify texture

### Texture Object

```c++
// 使用设备上的线性内存
#define N 1024

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch<float>(tex, i);
  // do some work using x ...
}

void call_kernel(cudaTextureObject_t tex) {
  dim3 block(128,1,1);
  dim3 grid(N/block.x,1,1);
  kernel <<<grid, block>>>(tex);
}

int main() {
  // declare and allocate memory
  float *buffer;
  cudaMalloc(&buffer, N*sizeof(float));

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = buffer;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = N*sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tex=0;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  call_kernel(tex); // pass texture as argument

  // destroy texture object
  cudaDestroyTextureObject(tex);

  cudaFree(buffer);
}
```

### Texture Reference

```c++
// Example1: low-level API，使用设备上的线性内存

// 在全局作用域声明
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// 从texture类型变量中对应的textureReference结构体指针
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);	
// 修改textureReference中的可改属性...

// 根据CUDA基本类型生成cudaChannelFormatDesc类型的texel描述结构体
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

// bind
size_t offset;
cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc, width, height, pitch);

//////////////////////////////////////////////////////////////////////////////////
// Example2: 完整模板
#define N 1024
texture<float, 1, cudaReadModeElementType> tex;

// texture reference name must be known at compile time
__global__ void kernel() {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch(tex, i);
  // do some work using x...
}

void call_kernel(float *buffer) {
  // bind texture to buffer
  cudaBindTexture(0, tex, buffer, N*sizeof(float));

  dim3 block(128,1,1);
  dim3 grid(N/block.x,1,1);
  kernel <<<grid, block>>>();

  // unbind texture from buffer
  cudaUnbindTexture(tex);
}

int main() {
  // declare and allocate memory
  float *buffer;
  cudaMalloc(&buffer, N*sizeof(float));
  call_kernel(buffer);
  cudaFree(buffer);
}
```

---

## Texture Function

`texture fetch`通过调用`texture function`访问纹理内存，其返回值的具体计算规则见官方文档[Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching)

### tex1Dfetch (texture object  API)

```c++
template<class T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);
```

- 注意`x`为整型，函数中1D说明texture object被指定为一维数组
- texture**必须是**设备上的线性内存

该函数只对非标准化坐标的纹理起作用，所以纹理描述符（`struct  cudaTextureDesc`）中的译址模式（address mode）只能为`cudaAddressModeClamp`和`cudaAddressModeBorder`，并且不会执行任何纹理过滤（texture filtering）

### 其它

其它`texture function`参见官方文档[Texture Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-functions)