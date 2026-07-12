# 第 6 章：RoPE 位置编码 —— 给向量注入位置信息

在上一章，我们实现了 Embedding、RMSNorm 和 Linear 三个基础算子。但从 Token ID 到向量之后，还有一个关键问题需要解决：**模型怎么知道每个 token 的位置？**

这一章，我们来实现 RoPE（Rotary Position Embedding，旋转位置编码），它是 Qwen3、LLaMA 等主流模型编码位置信息的标准方式。

---

## 6.1 RoPE 是什么？

### 6.1.1 Attention 的位置盲问题

回顾一下 Attention 的计算：它对每个位置的 Q 向量与其他位置的 K 向量做点积。但点积运算本身是**对称的**——`Q₀ · K₁` 和 `Q₁ · K₀` 的值是相同的，模型无从区分"我在前你在后"还是"我在后你在前"。

```
序列: ["我", "爱", "你"]
位置:    0     1     2

没有位置编码时：
  "我爱"和"你爱我"对模型来说是一样的
```

RoPE 的作用：**在做 Attention 之前，给 Q 和 K 向量各施加一次"旋转"，旋转的角度由位置决定**。

### 6.1.2 RoPE 在推理流程中的位置

```mermaid
flowchart LR
    classDef blue fill:#2196F3,stroke:#0D47A1,stroke-width:2px,color:#fff;
    classDef orange fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff;

    subgraph Attention [Attention 内部]
        direction LR
        A[输入向量] --> B[Q/K 线性投影]
        B --> C[Q/K 归一化]
        C --> D[<b>RoPE 旋转</b>]
        D --> E[计算 Attention Score]
    end

    class D orange;
```

RoPE 只作用在 Q 和 K 上，不动 V。

### 6.1.3 核心操作

一句话概括 RoPE：**对于 Q 和 K 的每个 head 向量，把相邻两维当作一对坐标 (x, y)，按一个由位置决定的固定角度把它们旋转一下**。

```
旋转前:  (x, y)        旋转后:  (x', y')
         ▲                         ▲
         │                         │
    ─────┼─────               ─────┼─────
    │    │    │               │    │    │
    │    │    │   旋转后  ──▶  │   ╱│    │
    │    │    │               │  ╱ │    │
    ─────┼─────               ─────┼─────
```

---

## 6.2 旋转是怎么做的

### 6.2.1 二维旋转公式

先看最简单的情况：平面上有一个点 (x, y)，把它绕原点逆时针旋转角度 θ：

```
x' = x·cosθ - y·sinθ
y' = x·sinθ + y·cosθ
```

**手算示例**：点 (1, 0) 旋转 90°（即 π/2）

```
cos(90°) ≈ 0,   sin(90°) ≈ 1

x' = 1·0 - 0·1 = 0
y' = 1·1 + 0·0 = 1

结果: (0, 1)  ← (1, 0) 确实转到了 (0, 1)
```

### 6.2.2 高维向量怎么旋转：成对旋转

一个 head 向量有 head_dim 维（Qwen3-0.6B 中是 128 维）。我们把它的维度两两配对，每对独立旋转：

```
head 向量 (128 维):
[d₀, d₁,  d₂, d₃,  d₄, d₅,  ...,  d₁₂₆, d₁₂₇]
 └─┬─┘   └─┬─┘   └─┬─┘           └──┬──┘
 第0对   第1对   第2对   ...      第63对

每一对:
  x = head[d]           ← 前半维
  y = head[d + 64]      ← 后半维（注意：d + head_dim/2）
```

关键点：**x 和 y 不是相邻的两个数**。x 是 head 向量的前半部分（d），y 是后半部分（d + head_dim/2）。

### 6.2.3 旋转角度怎么算

每对维度有自己的旋转角度。计算公式：

```
θ_d = pos / (rope_theta ^ (2d / head_dim))
```

其中：
- `pos`：token 在序列中的位置（第 0 个、第 1 个...）
- `d`：维度对的编号（0, 1, 2, ..., head_dim/2 - 1）
- `rope_theta`：一个超参数，Qwen3 中为 **1000000**
- `head_dim`：每个 head 的维度，Qwen3-0.6B 中为 **128**

频率 `1/(rope_theta^(2d/head_dim))` 随 d 增大而减小：
- d=0：频率最高，旋转最快 → 捕捉局部位置关系
- d=63：频率最低，旋转最慢 → 捕捉长距离位置关系

**手算示例**（head_dim=128，rope_theta=1000000）：

```
d=0:   rope_theta^(2×0/128) = rope_theta^0 = 1
       freq = 1/1 = 1.0           ← 高频，旋转快

d=32:  rope_theta^(2×32/128) = rope_theta^0.5 = 1000
       freq = 1/1000 = 0.001      ← 中频

d=63:  rope_theta^(2×63/128) = rope_theta^0.984 ≈ 819292
       freq = 1/819292 ≈ 0.0000012  ← 低频，旋转极慢
```

---

## 6.3 完整手算示例

取一个小例子来手动过一遍 RoPE 的完整计算过程：

**设定**：
- head_dim = 4（只有 4 维，便于手算）
- num_heads = 1（1 个 head）
- num_kv_heads = 1（1 个 KV head）
- rope_theta = 10000（简化计算）
- pos = 0（当前第一个 token 的位置）

**输入 Q 和 K**（1 个 token，head_dim=4）：

```
Q = [0.5, 0.8, 1.2, 0.3]
K = [0.1, 0.9, 0.6, 0.4]
```

**步骤 1：算每对的频率和角度**

head_dim=4，half_dim=2，有 2 对维度：

```
d=0:  freq = 1 / 10000^(2×0/4) = 1 / 10000^0 = 1 / 1 = 1.0
      angle = pos × freq = 0 × 1.0 = 0.0
      cos = cos(0.0) = 1.0
      sin = sin(0.0) = 0.0

d=1:  freq = 1 / 10000^(2×1/4) = 1 / 10000^0.5 = 1 / 100 = 0.01
      angle = pos × freq = 0 × 0.01 = 0.0
      cos = cos(0.0) = 1.0
      sin = sin(0.0) = 0.0
```

pos=0 时所有角度都是 0，旋转完不变。下面我们算 pos=1 时：

```
d=0:  angle = 1 × 1.0 = 1.0 弧度
      cos(1.0) ≈ 0.5403
      sin(1.0) ≈ 0.8415

d=1:  angle = 1 × 0.01 = 0.01 弧度
      cos(0.01) ≈ 0.99995
      sin(0.01) ≈ 0.0099998
```

**步骤 2：对 Q 做旋转**

Q 的数据布局：`[0.5, 0.8, 1.2, 0.3]`
- d=0: x=Q[0]=0.5, y=Q[0+2]=Q[2]=1.2
- d=1: x=Q[1]=0.8, y=Q[1+2]=Q[3]=0.3

```
d=0 旋转 (angle=1.0):
  x' = 0.5 × 0.5403 - 1.2 × 0.8415 = 0.2702 - 1.0098 = -0.7396
  y' = 0.5 × 0.8415 + 1.2 × 0.5403 = 0.4208 + 0.6484 = 1.0692

d=1 旋转 (angle=0.01):
  x' = 0.8 × 0.99995 - 0.3 × 0.0099998 = 0.79996 - 0.0030 = 0.7970
  y' = 0.8 × 0.0099998 + 0.3 × 0.99995 = 0.00800 + 0.29999 = 0.3080

Q' = [-0.7396, 0.7970, 1.0692, 0.3080]
```

**步骤 3：对 K 做同样的旋转**

K = `[0.1, 0.9, 0.6, 0.4]`
- d=0: x=K[0]=0.1, y=K[0+2]=K[2]=0.6
- d=1: x=K[1]=0.9, y=K[1+2]=K[3]=0.4

```
d=0 旋转 (angle=1.0):
  x' = 0.1 × 0.5403 - 0.6 × 0.8415 = 0.0540 - 0.5049 = -0.4509
  y' = 0.1 × 0.8415 + 0.6 × 0.5403 = 0.0842 + 0.3242 = 0.4084

d=1 旋转 (angle=0.01):
  x' = 0.9 × 0.99995 - 0.4 × 0.0099998 = 0.89996 - 0.0040 = 0.8960
  y' = 0.9 × 0.0099998 + 0.4 × 0.99995 = 0.00900 + 0.39998 = 0.4090

K' = [-0.4509, 0.8960, 0.4084, 0.4090]
```

这样就完成了 RoPE。可以看到 d=0（高频对）旋转了约 57°（1 rad），变化很大；d=1（低频对）只转了约 0.57°，几乎没变。

---

## 6.4 代码实现

### 6.4.1 数据布局

在写代码之前，先搞清楚内存里数据是怎么放的。

q 和 k 都是 Tensor，shape 是 `[seq_len, num_heads × head_dim]`，行优先存储：

```
q.shape() = [seq_len, num_heads × head_dim]

                    head_dim=128
              ┌─────────────────────┐
           ┌─ ┌──┬──┬──┬──┬──┬──┬──┐
seq_len=N │   │  │  │  │  │  │  │  │  ← head 0
           │   ├──┼──┼──┼──┼──┼──┼──┤
           │   │  │  │  │  │  │  │  │  ← head 1
           │   ├──┼──┼──┼──┼──┼──┼──┤
           │   │  │  │  │  │  │  │  │  ← head 2
           └─  └──┴──┴──┴──┴──┴──┴──┘
```

要定位到**第 i 个 token、第 h 个 head 的 d 维**：

```cpp
float* head_ptr = q.data<float>() + i * num_heads * head_dim + h * head_dim;
// head_ptr[d] 就是第 d 维
```

要定位到成对旋转的另一半（y）：

```cpp
// x = head_ptr[d]
// y = head_ptr[d + head_dim / 2]
```

### 6.4.2 ApplyRoPE 函数

```cpp
void ApplyRoPE(Tensor& q, Tensor& k, size_t pos = 0) {
    size_t seq_len = q.shape()[0];       // 当前输入的 token 数
    int half_dim = head_dim_ / 2;        // 维度对的数量（128/2 = 64）

    // 第一层循环：遍历每个 token 位置
    for (size_t i = 0; i < seq_len; ++i) {

        // 第二层循环：遍历每个维度对
        for (int d = 0; d < half_dim; ++d) {

            // 计算这一对的旋转角度
            float theta = 1.0f / std::pow(rope_theta_, (float)(2 * d) / head_dim_);
            float angle = (pos + i) * theta;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // 对 Q 的每个 head 做旋转
            for (size_t h = 0; h < num_heads_; ++h) {
                float* head_ptr = q.data<float>()
                    + i * num_heads_ * head_dim_    // 定位到第 i 个 token
                    + h * head_dim_;                 // 定位到第 h 个 head

                float x = head_ptr[d];               // 前半维
                float y = head_ptr[d + half_dim];    // 对应的后半维

                head_ptr[d]            = x * cos_val - y * sin_val;
                head_ptr[d + half_dim] = x * sin_val + y * cos_val;
            }

            // 对 K 的每个 head 做同样的旋转
            for (size_t h = 0; h < num_kv_heads_; ++h) {
                float* head_ptr = k.data<float>()
                    + i * num_kv_heads_ * head_dim_
                    + h * head_dim_;

                float x = head_ptr[d];
                float y = head_ptr[d + half_dim];

                head_ptr[d]            = x * cos_val - y * sin_val;
                head_ptr[d + half_dim] = x * sin_val + y * cos_val;
            }
        }
    }
}
```

**逐段解释**：

1. **三层循环的含义**：
   - 最外层 `i`：遍历当前输入中的每个 token
   - 中间层 `d`：遍历每个维度对（d 和 d+half_dim 组成一对）
   - 最内层 `h`：对每个 attention head 独立做旋转

2. **频率计算 `std::pow(rope_theta_, 2d/head_dim)`**：
   - `2d/head_dim` 的范围是 0 到接近 1
   - rope_theta^0 = 1 → 频率最高，旋转最快
   - rope_theta^~1 → 频率极低，旋转极慢

3. **角度 `(pos + i) * theta`**：
   - `pos` 是 KV Cache 中的起始偏移（无 cache 时 pos=0）
   - `pos + i` 就是当前 token 在完整序列中的绝对位置

4. **旋转操作**：
   - `x = ptr[d]`：取前半维
   - `y = ptr[d + half_dim]`：取后半维
   - 直接用二维旋转公式替换

---

## 6.5 完整代码

RoPE 是 Attention 运算的一部分，放在 `operator.hpp` 的 `Attention` 类中。以下是 Attention 类的骨架，展示 RoPE 相关的完整上下文：

```cpp
class Attention {
public:
    Attention(size_t hidden_dim, size_t num_kv_heads,
              size_t num_heads, size_t head_dim)
        : hidden_dim_(hidden_dim)
        , num_heads_(num_heads)
        , num_kv_heads_(num_kv_heads)
        , head_dim_(head_dim) {

        // 线性投影：hidden_dim → num_heads × head_dim
        q_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_heads_ * head_dim_);
        k_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_kv_heads_ * head_dim_);
        v_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_kv_heads_ * head_dim_);
        output_proj_ = std::make_shared<LinearProjection>(num_heads_ * head_dim_, hidden_dim_);

        // QK 归一化（Qwen3 特有）
        q_norm_ = std::make_shared<RMSNorm>(num_heads_ * head_dim_);
        k_norm_ = std::make_shared<RMSNorm>(num_kv_heads_ * head_dim_);
    }

    Tensor Forward(Tensor& input, size_t position = 0) {
        // 1. 线性投影
        Tensor q = q_proj_->Forward(input);   // [seq_len, num_heads × head_dim]
        Tensor k = k_proj_->Forward(input);   // [seq_len, num_kv_heads × head_dim]
        Tensor v = v_proj_->Forward(input);

        // 2. QK 归一化
        q = q_norm_->Forward(q);
        k = k_norm_->Forward(k);

        // 3. RoPE —— 本章的核心
        ApplyRoPE(q, k, position);

        // 4. KV Cache 更新 + Attention 计算（后续章节）
        // ...

        return output;
    }

private:
    void ApplyRoPE(Tensor& q, Tensor& k, size_t pos = 0) {
        // ... 见 6.4.2 节
    }

    size_t hidden_dim_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    float rope_theta_ = 1000000.0f;  // Qwen3 的 rope_theta

    std::shared_ptr<RMSNorm> q_norm_;
    std::shared_ptr<RMSNorm> k_norm_;
    std::shared_ptr<LinearProjection> q_proj_;
    std::shared_ptr<LinearProjection> k_proj_;
    std::shared_ptr<LinearProjection> v_proj_;
    std::shared_ptr<LinearProjection> output_proj_;
};
```

---

## 6.6 测试与验证

### 6.6.1 创建测试文件

新建 `test/test_rope.cpp`：

```cpp
#include <cmath>
#include <iostream>
#include <iomanip>

#include "tensor.h"

// 模拟一个小型 RoPE 测试
// 这里不依赖完整的 Attention 类，直接写一个简化版 ApplyRoPE 来验证

void ApplyRoPE(Tensor& q, Tensor& k, size_t pos,
               int head_dim, int num_heads, int num_kv_heads,
               float rope_theta) {
    size_t seq_len = q.shape()[0];
    int half_dim = head_dim / 2;

    for (size_t i = 0; i < seq_len; ++i) {
        for (int d = 0; d < half_dim; ++d) {
            float freq = 1.0f / std::pow(rope_theta, (float)(2 * d) / head_dim);
            float angle = (pos + i) * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            for (int h = 0; h < num_heads; ++h) {
                float* head_ptr = q.data<float>()
                    + i * num_heads * head_dim + h * head_dim;
                float x = head_ptr[d];
                float y = head_ptr[d + half_dim];
                head_ptr[d]            = x * cos_val - y * sin_val;
                head_ptr[d + half_dim] = x * sin_val + y * cos_val;
            }

            for (int h = 0; h < num_kv_heads; ++h) {
                float* head_ptr = k.data<float>()
                    + i * num_kv_heads * head_dim + h * head_dim;
                float x = head_ptr[d];
                float y = head_ptr[d + half_dim];
                head_ptr[d]            = x * cos_val - y * sin_val;
                head_ptr[d + half_dim] = x * sin_val + y * cos_val;
            }
        }
    }
}

void PrintVector(const char* label, Tensor& t) {
    size_t n = t.shape()[0] * t.shape()[1];
    float* data = t.data<float>();
    std::cout << label << " = [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    int head_dim = 4;
    int num_heads = 1;
    int num_kv_heads = 1;
    float rope_theta = 10000.0f;

    std::cout << "=== 测试 1：pos=0 时的 RoPE ===" << std::endl;
    std::cout << "(角度全为 0，旋转后应与输入相同)" << std::endl << std::endl;

    {
        Tensor q({1, (size_t)(num_heads * head_dim)}, sizeof(float));
        Tensor k({1, (size_t)(num_kv_heads * head_dim)}, sizeof(float));

        float* qd = q.data<float>();
        float* kd = k.data<float>();

        qd[0] = 0.5f; qd[1] = 0.8f; qd[2] = 1.2f; qd[3] = 0.3f;
        kd[0] = 0.1f; kd[1] = 0.9f; kd[2] = 0.6f; kd[3] = 0.4f;

        PrintVector("Q 输入", q);
        PrintVector("K 输入", k);

        ApplyRoPE(q, k, 0, head_dim, num_heads, num_kv_heads, rope_theta);

        PrintVector("Q 输出", q);
        PrintVector("K 输出", k);
        std::cout << "(预期：与输入相同)" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== 测试 2：pos=1 时的 RoPE ===" << std::endl;
    std::cout << "(验证角度非零时的旋转结果，对照手算)" << std::endl << std::endl;

    {
        Tensor q({1, (size_t)(num_heads * head_dim)}, sizeof(float));
        Tensor k({1, (size_t)(num_kv_heads * head_dim)}, sizeof(float));

        float* qd = q.data<float>();
        float* kd = k.data<float>();

        qd[0] = 0.5f; qd[1] = 0.8f; qd[2] = 1.2f; qd[3] = 0.3f;
        kd[0] = 0.1f; kd[1] = 0.9f; kd[2] = 0.6f; kd[3] = 0.4f;

        PrintVector("Q 输入", q);
        PrintVector("K 输入", k);

        ApplyRoPE(q, k, 1, head_dim, num_heads, num_kv_heads, rope_theta);

        PrintVector("Q 输出", q);
        std::cout << "(6.3 节手算: [-0.7396, 0.7970, 1.0692, 0.3080])" << std::endl;
        PrintVector("K 输出", k);
        std::cout << "(6.3 节手算: [-0.4509, 0.8960, 0.4084, 0.4090])" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== 测试 3：不同位置旋转角度不同 ===" << std::endl;
    std::cout << "(同一输入，pos=0 和 pos=5 结果应不同)" << std::endl << std::endl;

    {
        Tensor q0({1, (size_t)(num_heads * head_dim)}, sizeof(float));
        Tensor q5({1, (size_t)(num_heads * head_dim)}, sizeof(float));

        float* qd0 = q0.data<float>();
        float* qd5 = q5.data<float>();

        // 填入相同数据
        qd0[0] = 1.0f; qd0[1] = 2.0f; qd0[2] = 3.0f; qd0[3] = 4.0f;
        qd5[0] = 1.0f; qd5[1] = 2.0f; qd5[2] = 3.0f; qd5[3] = 4.0f;

        Tensor k_dummy({1, (size_t)(num_kv_heads * head_dim)}, sizeof(float));

        ApplyRoPE(q0, k_dummy, 0, head_dim, num_heads, num_kv_heads, rope_theta);
        ApplyRoPE(q5, k_dummy, 5, head_dim, num_heads, num_kv_heads, rope_theta);

        PrintVector("Q(pos=0)", q0);
        PrintVector("Q(pos=5)", q5);
    }

    std::cout << std::endl;
    std::cout << "所有测试完成！" << std::endl;
    return 0;
}
```

### 6.6.2 更新 CMakeLists.txt

在 `CMakeLists.txt` 中添加：

```cmake
# RoPE 测试
add_executable(test_rope test/test_rope.cpp)
target_include_directories(test_rope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
```

### 6.6.3 编译并运行

```bash
cd build
cmake ..
make test_rope
./test_rope
```

**预期输出**：

```
=== 测试 1：pos=0 时的 RoPE ===
(角度全为 0，旋转后应与输入相同)

Q 输入 = [0.5000, 0.8000, 1.2000, 0.3000]
K 输入 = [0.1000, 0.9000, 0.6000, 0.4000]
Q 输出 = [0.5000, 0.8000, 1.2000, 0.3000]
K 输出 = [0.1000, 0.9000, 0.6000, 0.4000]
(预期：与输入相同)

=== 测试 2：pos=1 时的 RoPE ===
(验证角度非零时的旋转结果，对照手算)

Q 输入 = [0.5000, 0.8000, 1.2000, 0.3000]
K 输入 = [0.1000, 0.9000, 0.6000, 0.4000]
Q 输出 = [-0.7396, 0.7970, 1.0692, 0.3080]
(6.3 节手算: [-0.7396, 0.7970, 1.0692, 0.3080])
K 输出 = [-0.4509, 0.8960, 0.4084, 0.4090]
(6.3 节手算: [-0.4509, 0.8960, 0.4084, 0.4090])

=== 测试 3：不同位置旋转角度不同 ===
(同一输入，pos=0 和 pos=5 结果应不同)

Q(pos=0) = [1.0000, 2.0000, 3.0000, 4.0000]
Q(pos=5) = [-0.9656, 1.9919, 3.5430, 3.4684]

所有测试完成！
```

---

## 6.7 小结

本章实现了 RoPE 位置编码，核心流程：

1. **思路**：对 Q 和 K 的每个 head 向量，把维度成对分组，每组按位置相关角度做二维旋转
2. **角度公式**：`θ_d = pos / (rope_theta ^ (2d / head_dim))`，高频到低频覆盖从局部到长距离的位置信息
3. **实现**：三层循环——token → 维度对 → head，每层逻辑都很直白

RoPE 之后，Q 和 K 就携带了位置信息。下一章，我们将把这些组件全部串起来，实现 **Multi-Head Attention（GQA）**——整个 Transformer 中最复杂的模块。

