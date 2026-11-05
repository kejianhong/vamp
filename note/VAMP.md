# VAMP 论文 III. METHOD 与源码对照分析

本文对照论文 *Motions in Microseconds via Vectorized Sampling-Based Planning*（ICRA 2024，Thomason / Kingston / Kavraki）第 III 章 METHOD 的每一个小节，逐条说明其**方法原理**以及在本仓库 `src/impl` 中的**实现方式与代码位置**。

论文 PDF：`Motions-in-Microseconds-via-Vectorized-Sampling-Based-Planning.pdf`

---

## 0. 总览映射表

| 论文小节                            | 核心机制                              | 主要源码                                                              |
| ----------------------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| III.A Vectorized Motion Planning    | SoA 内存布局 + SIMD 抽象层            | `src/impl/vamp/vector.hh`、`vector/interface.hh`、`vector/isa/avx.hh` |
| III.B Vectorized Forward Kinematics | tracing compiler 生成展开的无分支 FK  | `src/impl/vamp/robots/panda.hh` 等生成文件（`sphere_fk` / `fkcc`）    |
| III.C Vectorized Collision Checking | 球近似 + rake 批量 + 层级 mid-phase   | `src/impl/vamp/collision/shapes.hh`、`validity.hh`、`sphere_*.hh`     |
| III.D Vectorized Motion Validation  | 空间分布式（等间距）批量检查          | `src/impl/vamp/planning/validate.hh`                                  |
| III.E Bringing it Together          | 不改算法的 RRT-Connect / PRM + 简化器 | `src/impl/vamp/planning/rrtc.hh`、`prm.hh`、`simplify.hh`、`nn.hh`    |

---

## III.A Vectorized Motion Planning（SoA 布局）

### 原理

论文强调 "vectorized" 有两个含义：

1. **SIMD 意义**上，一个 vector 是**固定长度、同标量类型**的值集合，一个 vectorized operation 并行作用在全部值上；
2. 把 SBMP 的基元（FK、CC、state/edge validity）从"一次一个状态"提升为"一次一组状态"，从而既可用 SIMD，又能**与顺序代码自由交错**（fine-grained parallelism），而不需要线程/锁/通信。

为了让数据并行，内存布局必须从 AoS 改成 **SoA**（论文 Fig. 1）：**行=维度，列=配置**。AoS（每个配置一块连续内存）是最"自然"的布局，但不利于 SIMD，因为跨配置访问会变成 gather。

### 代码实现

**1. ISA 抽象层** — `src/impl/vamp/vector.hh:16-38`，编译器预定义的宏（preprocessor macro）。该文件中的所有代码没有一行是运行时执行的代码，全部是预处理和编译期“起名字”。这正是C++元编程（meta-programming)的基本功：使用**类型系统+模板在编译期做抽象**，让运行时代码既快又跨平台。

```cpp
#if defined(__EMSCRIPTEN__)   ... WasmFloatVec
#elif defined(__x86_64__)     using SimdFloatT = __m256;      // AVX2, 8 lanes
#elif defined(__ARM_NEON)     using SimdFloatT = float32x4_t; // 4 lanes
using FloatVector<scalars_per_row, num_rows> = Vector<SIMDVector<SimdFloatT>, num_rows, scalars_per_row>;
```

`rake` 就是向量宽度（值为一个 SIMD 寄存器能装几个标量)：`src/impl/vamp/bindings/python/robot_helper.hh:58` 中

```cpp
static constexpr std::size_t rake = vamp::FloatVectorWidth;
```

→ x86 AVX2 下为 8，ARM NEON 下为 4。这正是论文所说的 "8-wide SIMD" 及其在低功耗单板机上的对应形态。

**2. 一个向量长的物理表示** — `vector/interface.hh:30-49`

```cpp
struct VectorSignature {
    num_scalars                 = num_rows * num_scalars_per_row;
    num_vectors_per_row         = fit_scalars_to_vectors(num_scalars_per_row, S::VectorWidth);
    num_scalars_per_row_rounded = num_vectors_per_row * S::VectorWidth;
    using DataT = std::array<typename S::VectorT, num_vectors>;   // <-- SoA 实体
};
```

即一个 `Vector` 由 `num_rows × num_vectors_per_row` 个 ISA 寄存器拼成；**每一行是独立的寄存器组**，这就是 SoA。`VectorInterface`（CRTP，同文件 51–805 行）把所有逐元素算子一次性写出来（`add` / `sub` / `mul` / `div` / `cmp_*` / `blend` / `max` / `clamp` / `gather` / `hsum` / `l2_norm` / `sqrt` / `sin` …），派生类只需提供 ISA 后端。

**3. ISA 后端** — `vector/isa/avx.hh`

`SIMDVector<__m256>`（231 行起）与 `SIMDVector<__m256i>`（15 行起）把每个算子映射到一条 intrinsic（`_mm256_add_ps`、`_mm256_cmp_ps`、`_mm256_movemask_ps`、`_mm256_i32gather_ps` 等），`VectorWidth = 8`、`Alignment = 32`。两处值得注意的性能取舍：

- `sqrt` 用 `_mm256_mul_ps(v, _mm256_rsqrt_ps(v))` 近似（`avx.hh:412`）；
- AVX2 没有向量 `cos`，在 `interface.hh:452-458` 用 `sin(x + π/2)` 加区间折叠实现；`sin` 是移植的 cephes 多项式（`avx.hh:456-548`）。

**4. SoA 在机器人上的具体形态** — `src/impl/vamp/robots/panda.hh:30-48`

```cpp
using Configuration            = FloatVector<7>;             // 单个构型：7 个标量塞进 1 个 8-lane 向量
template <std::size_t rake>
using ConfigurationBlock<rake> = FloatVector<rake, 7>;       // Vector<S, num_rows=7, scalars_per_row=rake>

template <std::size_t rake>
struct Spheres {                                             // 每球一行
    FloatVector<rake, n_spheres> x, y, z, r;                 // n_spheres = 59 (Panda)
};
```

`ConfigurationBlock<rake>` 有 **7 行（每个关节一行）× 每行 rake 条 lane**，一个 block 里躺着 rake 个构型 —— **行=维度、lane=构型**，与论文 Fig. 1 的 SoA 一一对应。`Spheres` 同样每球一行，所以 FK 的输出天然是 SoA，喂给碰撞检查时零拷贝。

这也解释了论文所说的 "planning-specific independence patterns"：`VectorSignature` 的注释（`interface.hh:42-47`）明确记录了 `num_scalars_per_row < VectorWidth` 时会 over-packing，作者**有意保留**该行为，因为紧凑打包会让 row 访问/抽取变得更贵。

---

## III.B Vectorized Forward Kinematics（tracing compiler）

### 原理

朴素向量化 FK 是"一次算一个构型的位姿"。VAMP 反其道而行：**把每个运算按顺序执行，但在多个构型上同时执行**。

难点在于常规 FK 实现会用动态分支与关节类型多态（模板化 robot link，如 KDL），编译器无法证明后序位姿依赖前序位姿，于是会**为整组构型产生虚假依赖，拖慢整个向量**。

论文的解法是一个 **tracing compiler**：

- 读入 URDF，trace 机器人运动学函数的实际运算；
- 据此自动生成 (1) 以"构型批次"为元素的向量化结构表示、(2) 计算该函数所需的最少算子集合；
- 因为 trace 的是**实际运算**，可以**完全展开（unrolled）**、避开分支与冗余依赖，再做常量折叠、代数化简、冗余常量消除等优化；
- 最终输出**直筒子（straight-line）代码**，与 KDL 的 CRTP 方案相比没有精度损失。

### 代码实现

生成物就在仓库里。注意：**编译器本体在独立仓库 [`cricket`](https://github.com/CoMMALab/cricket)**，球分解工具是 [`foam`](https://github.com/CoMMALab/foam)，见 `README.md:224-225`；本仓库只含生成结果。

`src/impl/vamp/robots/panda.hh` 中的四个函数：

| 函数          | 行号  | 作用                          |
| ------------- | ----- | ----------------------------- |
| `sphere_fk`   | 117   | 只做 FK，输出`Spheres&`       |
| `fkcc`        | 5227  | FK + 碰撞检查融合，**热路径** |
| `fkcc_attach` | 10265 | 带末端附着物版本              |
| `fkcc_debug`  | 468   | 调试（可定位碰撞球）          |

看 `panda.hh:119-130`：

```cpp
std::array<FloatVector<rake, 1>, 22>  v;     // SSA 临时寄存器
std::array<FloatVector<rake, 1>, 236> y;     // 跨 link 存活的值
v[0] = sin(x[0]);   v[1] = -v[0];
y[4] = -0.08 * v[1];
v[2] = cos(x[0]);
y[5] = -0.08 * v[2];
...
```

特征完全吻合论文描述：**只有标量-向量乘加、没有任何循环、没有任何分支**，`v[]` / `y[]` 就是展开后的直线代码寄存器分配。

常量折叠 / 代数化简的直接证据：

- `v[6] = 4.89663865010925e-12 * v[5];` —— 这是 URDF 里 `sin(π/2)` 的浮点残差，被常量传播下来但没被折成 0（说明做了常量传播与代数化简，但未做到符号级的零消除）；
- 固定基座球被**完全折叠成常数**（`fetch.hh:17760-17775`：`y[500]=0.01312...; y[501]=0.; y[503]=0.07;`）；
- 只保留依赖闭包内的子图：`sphere_fk` 只用 22 个 `v` / 236 个 `y`，而 `fkcc` 用 57 个 `v` / 280 个 `y` —— 这就是论文说的 "skip a large number of irrelevant operations"。

---

## III.C Vectorized Collision Checking（球近似 + 层级 mid-phase）

### 原理

三角网格的窄相检测是为**单个**构型优化的，无法向量化。论文改用经典思路（Bradshaw & O'Sullivan 的中轴球逼近）：**机器人和环境都用基元（球、胶囊、长方体）表示**，这样"球 vs 基元"的相交测试天然可以逐 lane 并行。

在此之上叠加两层优化：

1. **rake**（论文 Fig. 2）：一次检查 rake 个空间上分散的构型，任一 lane 碰撞就**整批拒绝**；
2. **层级 mid-phase**：先为每个 link 生成一个保守的单一大球（over-approximation），用它粗筛；通过后才查该 link 的细球。这与传统 broadphase 的体积层次不同 —— **不需要递归树结构，也不需要在 FK 之后做任何更新**。

### 代码实现

**几何表示** — `src/impl/vamp/collision/shapes.hh`

定义 `Sphere` / `Cylinder`（别名 `Capsule`）/ `Cuboid` / `HeightField` / `ConvexPolytope`，每个都继承 `Shape` 并**预计算 `min_distance`**（到原点的最小距离，`shapes.hh:21`），例如球是 `sqrt(x²+y²+z²) - r`（`shapes.hh:242`）。由 `collision/factory.hh` 从 URDF / 几何描述构建。

`collision/environment.hh` 把环境按类型分成多个 `std::vector`，并提供 `sort()` **按 `min_distance` 升序排序**（`environment.hh:45-75`）—— 这是下面提前 break 的前提。

**主检测循环** — `src/impl/vamp/collision/validity.hh:48-165`

全篇 SIMD 味道最浓的一段：

```cpp
const auto max_extent = collision::sqrt(collision::dot_3(sx,sy,sz, sx,sy,sz)) + sr;  // rake 个 lane 各自算上界
for (const auto &es : e.spheres) {
    const auto diff = es.min_distance - max_extent;
    if (diff.test_zero()) break;   // 所有 lane 的 diff 均 >= 0 → 该物体及后续物体都不可能撞
    if (not collision::sphere_sphere_sql2(es, sx, sy, sz, sr).test_zero()) return true;
}
```

两处关键点：

- `max_extent` 用**一条向量运算**算出全部 lane 的上界；
- `diff = min_distance - max_extent` 的物理含义是「**环境物体到原点的最近距离 − 机器人球到原点的最远距离**」。由三角不等式，对物体内任意点 `p` 与球内任意点 `q` 都有 `|p − q| ≥ |p| − |q| ≥ diff`，因此 **`diff` 是球与物体之间最小间距的一个保守下界**：
  - `diff > 0` → 两者至少相隔 `diff`，**必然不相交**，可整个跳过该物体；
  - `diff ≤ 0` → 该界失效，无法判定，必须进入窄相精确检测。
- `test_zero()` 的真实语义是 **VTESTPS 指令：所有 lane 的符号位为 0**，即所有 lane **非负**（实测：`+0.0` 返回 true，`-0.0` 返回 false）。所以 `diff.test_zero()` 表示"**rake 个 lane 全部满足 `diff ≥ 0`**"。
- 由于 `Environment::sort()` 已按 `min_distance` **升序**排序（构造环境时在 `bindings/python/environment.cc:213-252` 调用），一旦最近的物体都满足该判定，后续所有物体必然同样满足 → 可直接 `break` 跳出整个循环。这就是配合排序的整批 broadphase 剪枝。

各几何的窄相都返回 `|d|² - (r+R)²` 的符号，用 `test_zero()` 判定：

| 几何对      | 位置                      | 说明                                                                                                                               |
| ----------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 球-球       | `sphere_sphere.hh:10`     | `sphere_sphere_sql2`，包装为 `sphere_sphere_self_collision`（`validity.hh:24`）                                                    |
| 球-胶囊     | `sphere_capsule.hh:9`     | 先`dot` 再 `clamp(0,1)` 求线段参数 `cdf`；另有 z 轴对齐快速路径 `sphere_z_aligned_capsule`                                         |
| 球-长方体   | `sphere_cuboid.hh:9`      | 投影到三轴取`abs` 减半长后 `max(0)` 再平方和，**全程无 sqrt**；同样有 z-aligned 版本                                               |
| 球-高度场   | `sphere_heightfield.hh:9` | `clamp`/`floor` 求网格索引后 `DataT::gather(a.data.data(), indices)` —— SIMD gather 的直接用例                                     |
| 球-凸多面体 | `sphere_polytope.hh`      | 论文之后新增（见 commit`27cb9b6`）；先用 OBB 做 SIMD broadphase（复用 `sphere_cuboid`），再对存活 lane 逐个做**标量 GJK** 距离查询 |

`sphere_polytope.hh:54-56` 的注释很诚实：*"I tried many times to figure out a good block-parallel way to do this, but it was always faster to do simple sequential GJK."*

**层级 mid-phase** — 直接体现在生成代码的嵌套 `if` 里

`panda.hh:5634` 起：

```cpp
// panda_rightfinger
if (sphere_environment_in_collision(environment, y[276], y[277], y[278], y[279]))  // 粗球 r≈0.0235 (行 5613)
{
    if (sphere_environment_in_collision(environment, y[228], y[229], y[230], y[231]))  // 细球 r=0.012 (行 5608)
        return false;
    ...
}
```

自碰撞同理（`panda.hh:10111` 起）：注释 `// panda_link5 vs. panda_rightfinger`，外层先测 `y[256..259] × y[276..279]` 这一对粗球，命中后才展开 `y[68..115]` 与手指细球的两两组合。

**早退语义**：`fkcc` 返回 `false` 时不保证能定位是哪个 lane / 哪个球碰撞 —— `sphere_polytope.hh:14-16` 的注释明确写了这一点；需要定位时用单独的慢版本 `fkcc_debug`。

### ⚠️ 与论文陈述的一处偏差（实测）

论文 III.C 末尾称编译器会 *"re-order instructions topologically, to interleave each sphere's collision check within the generated FK code, placing checks immediately after the position of the sphere has been computed"*。

但逐行核对 4 个机器人的 `fkcc` 后，**检查语句全部出现在最后一个 `y[i]` 赋值之后**：

| 文件        | 首个检查行 | 最后一个`y[]` 赋值行 | 之前出现的检查数 |
| ----------- | ---------- | -------------------- | ---------------- |
| `panda.hh`  | 5634       | 5627                 | 0                |
| `ur5.hh`    | 3598       | 3591                 | 0                |
| `fetch.hh`  | 17782      | 17775                | 0                |
| `baxter.hh` | 13379      | 13372                | 0                |

即当前生成物是 **"整套 FK → 环境检查 → 自碰撞检查"分块**的，而非把检查插进坐标计算流中。论文描述的是 tracing compiler 具备的**指令重排能力/设计意图**，仓库里现存的生成结果并未体现。（可能是生成器版本演进，或重排后又被重新分块化。）

复现该结论的脚本：

```python
import re, glob
for path in sorted(glob.glob('src/impl/vamp/robots/*.hh')):
    f = open(path).read().split('\n')
    for start in [i for i, l in enumerate(f) if 'static bool fkcc(' in l]:
        end = next((i for i in range(start+1, len(f)) if 'static bool fkcc_attach(' in f[i]), len(f))
        checks  = [i+1 for i in range(start, end) if 'sphere_environment_in_collision' in f[i]
                                                or 'sphere_sphere_self_collision' in f[i]]
        assigns = [i+1 for i in range(start, end) if re.match(r'\s+y\[\d+\] =', f[i])]
        if checks and assigns:
            print(path, start+1, checks[0], assigns[-1],
                  sum(1 for c in checks if c < assigns[-1]))
```

---

## III.D Vectorized Motion Validation（空间分布式 rake）

### 原理

运动验证需要把连续运动离散化并逐点检查，是 SBMP 的时间大头。关键洞见：

- 假设碰撞在运动上近似均匀分布，则**一个非法运动期望需要检查的离散点数约为 n/2**（n 为离散点数）；
- 因此与其顺序检查前 rake 个状态，不如**空间上分布式地**选 rake 个等间距状态：一次检查 `{0, 1/rake, 2/rake, …, (rake-1)/rake}`；
- 如果全通过，再以 `n/rake` 次迭代"往回梳"（comb through）剩余状态 —— 代价仅是**一次检查**；
- 一旦某个 block 不合法就**整批拒绝**，不需要定位是哪个 lane，从而实现 **aggressive early termination**。

论文称这一策略为 "rake"（耙子），论文 Fig. 2 用两连杆机构做了图示。

### 代码实现 — `src/impl/vamp/planning/validate.hh`

```cpp
template <std::size_t n> struct Percents {                      // 行 12-21
    static constexpr auto percents = generate_percents<n>(...);  // {1/n, 2/n, ..., n/n}
};

template <typename Robot, std::size_t rake, std::size_t resolution>
inline constexpr auto validate_vector(start, vector, distance, environment) -> bool {
    const auto percents = FloatVector<rake>(Percents<rake>::percents);      // 行 31
    typename Robot::template ConfigurationBlock<rake> block;
    for (auto i = 0U; i < Robot::dimension; ++i)
        block[i] = start.broadcast(i) + (vector.broadcast(i) * percents);   // 行 38 一行生成 rake 个构型

    const std::size_t n = std::max(std::ceil(distance / (float)rake * resolution), 1.F);  // 行 41

    bool valid = Robot::template fkcc<rake>(environment, block);            // 第一次 rake 检查
    if (not valid or n == 1) return valid;                                  // 行 45

    const auto backstep = vector / (rake * n);                              // 行 50
    for (auto i = 1U; i < n; ++i) {
        for (auto j = 0U; j < Robot::dimension; ++j)
            block[j] = block[j] - backstep.broadcast(j);                    // 整体回退 1/(rake*n)
        if (not Robot::template fkcc<rake>(environment, block)) return false;  // 整批拒绝
    }
    return true;
}

template <...> validate_motion(start, goal, environment) {                  // 行 69-77
    auto vector = goal - start;
    return validate_vector(start, vector, vector.l2_norm(), environment);
}
```

逐点对应论文：

| 论文表述                                                       | 代码                                                                                                                    |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| "simultaneously check states {0, 1/rake, …, (rake-1)/rake}"    | `percents` + 第 38 行的 fuse-multiply-add，一条向量指令生成 rake 个构型（`start.broadcast(i)` 把第 i 个关节广播成一行） |
| "for at most n/rake iterations"                                | `n = ⌈distance/rake × resolution⌉`                                                                                      |
| "comb through the remaining states by incrementing each index" | `backstep = vector/(rake*n)`，每次循环整体回退 `1/(rake·n)`（见下节详解）                                               |
| aggressive early termination                                   | 第 45、60 行的`return false` 立刻冒泡                                                                                   |

### "comb through" 到底是什么

**先澄清一个常见误解**：这里处理的不是"采样得到的点"，也不是"相邻两个点之间成对检查"，而是**一条边** —— 构型空间中从 `start` 到 `goal` 的直线段。函数把这条线段离散成许多**单个构型**，每个构型**独立**做一次碰撞检测（点测试），不存在任何"成对"运算。调用方传入的也都是图中的边或路径段：

| 调用点                     | 传入的 `start` / `goal`                    |
| -------------------------- | ------------------------------------------ |
| `rrtc.hh:136`              | 树节点 → steer（截断到 `range`）后的扩展点 |
| `rrtc.hh:176`              | 已确认节点 → 另一棵树的延伸步长            |
| `prm.hh:138` / `:266`      | 两个 roadmap 节点                          |
| `simplify.hh:104` / `:131` | 路径上两个顶点（shortcut 尝试）            |

**离散化本身**（`validate_motion` → `validate_vector`）：

```text
q(t) = start + t · (goal − start),   t ∈ (0, 1]
```

即沿直线做线性插值。`distance = |goal − start|`（`l2_norm()`），`resolution`（Panda = 32）决定每单位构型距离检查多少个状态，总检查点数约为 `rake × n ≈ distance × resolution`，格点间距 `1/(rake·n)`。

**第一遍：铺开"耙子"**

`percents = {1/8, 2/8, …, 8/8}` 让一个 block 的 8 条 lane 落在整条线段上**均匀分布的 8 个点**，一次 `fkcc` 就同时检查它们。如果这 8 个点都安全，只能说明"大概率"没撞 —— **耙齿之间还留着 7 段空隙**。

**comb through：整把梳子平移一格，逐遍补空隙**

`backstep = vector / (rake*n)`，每次循环把 block 的**所有 lane 同时**减去这个位移，相当于把整把梳子沿线段平移一个格点：

```cpp
const auto backstep = vector / (rake * n);
for (auto i = 1U; i < n; ++i) {
    for (auto j = 0U; j < Robot::dimension; ++j)
        block[j] = block[j] - backstep.broadcast(j);   // 整把梳子左移一格
    if (not fkcc(environment, block)) return false;    // 再查 rake 个点
}
```

以 `rake = 8`、`distance × resolution = 24`（故 `n = 3`、格距 `1/24`）为例：

| 遍数 `i` | 被检查的 `t` 值（每遍 8 个）                          |
| -------- | ----------------------------------------------------- |
| 0        | `3/24, 6/24, 9/24, 12/24, 15/24, 18/24, 21/24, 24/24` |
| 1        | `2/24, 5/24, 8/24, 11/24, 14/24, 17/24, 20/24, 23/24` |
| 2        | `1/24, 4/24, 7/24, 10/24, 13/24, 16/24, 19/24, 22/24` |

三遍合起来，`1/24 … 24/24` **每个格点恰好被检查一次**。一般地，第 `i` 遍第 `j` 条 lane 覆盖：

```text
t(i, j) = ((j+1)·n − i) / (rake·n),   i ∈ [0, n−1],  j ∈ [0, rake−1]
```

令 `m = (j+1)·n − i`，则 `m` 恰好遍历 `[1, rake·n]` 且互不重叠 —— 覆盖是**精确且无冗余**的。这正是论文说的 *"comb through the remaining states by incrementing each index"*：像梳子一样反复梳理，每遍把上一遍齿间漏掉的点补上。

**为什么值得这样绕一圈**：若按顺序检查，一个非法运动期望要查 `N/2` 个点才撞上；而 rake 让**每一批 8 个点都散布在整条线段上**，任一 lane 命中就整批返回 `false`（`validate.hh:60`），期望只需极少批就能发现非法运动 —— 这就是 aggressive early termination。合法运动仍需跑满 `n` 遍，但每遍都是全 SIMD 的、整批通过的判定极便宜。

**两个容易忽略的细节**：

1. 这是**纯离散检测**，不是连续碰撞检测（CCD）：格点之间是否真的安全，靠 `resolution` 足够细来保证；`fkcc` 只对具体构型判定，不对线段扫掠体判定。
2. 与论文表述的细微差异：论文写的是同时检查 `{0, 1/rake, …, (rake−1)/rake}`，代码实际是 `{1/rake, 2/rake, …, 1}` —— 把起点换成了终点。因此 `t = 0`（起点）不在本函数检查范围内，其合法性由调用方（树/路图中已存在的节点）保证。

`vector.l2_norm()`（`interface.hh:402`，内部 `squared_l2_norm() → mul + hsum`）正是论文 III.E 所说的"L2 范数用水平求和廉价地算出来"。

**其余调用点**（见上表之外的路径段验证）：`simplify.hh:38-39`（B-spline 平滑中的两段检查）、`simplify.hh:178-179`（扰动后的两段检查）、`plan.hh:162`（`Path::validate`）。

---

## III.E Bringing it Together: Design of the Planner

### 原理

算法层面**不做任何改动** —— 论文明确说明所有性能提升都归因于向量化基元。另外利用 Euclidean 配置空间假设：

- 线性插值退化为**简单地把两个 SoA 向量相加**；
- L2 范数用**水平求和**廉价计算。

由此可以低成本地生成中间配置、计算 NN 距离。最后用两种简化器（随机 shortcut、B-spline 平滑）。

### 代码实现

**1. 最近邻与距离** — `src/impl/vamp/planning/nn.hh`

`NNFloatArray<dim>` 就是裸 `float*`，避免拷贝。`Space` 的 `distance`（`nn.hh:52-56`）：

```cpp
const auto diff = vamp::FloatVector<kDimensions>(a.v) - vamp::FloatVector<kDimensions>(b.v);
return diff.l2_norm();
```

即论文所说"线性插值 = 简单地把向量相加 + L2 范数用水平求和算"。索引结构用 `nigh` 库的 `KDTreeBatch<128>`（`nn.hh:89-95`）。

**2. RRT-Connect** — `src/impl/vamp/planning/rrtc.hh` + `rrtc_settings.hh`

默认参数（`rrtc_settings.hh`）：

```cpp
float range = 2.;
bool  dynamic_domain = true;  float radius = 4.; float alpha = 1e-4; float min_radius = 1.;
bool  balance = true;         float tree_ratio = 1.;
std::size_t max_iterations = 100000; std::size_t max_samples = 100000;
bool  start_tree_first = true;
```

对应 README 中的 *dynamic-domain balanced RRT-Connect*。流程与论文"无算法改动"一致：

| 行号         | 行为                                                  |
| ------------ | ----------------------------------------------------- |
| `rrtc.hh:62` | 先试起点-目标直线                                     |
| `:100-108`   | 按两棵树大小比做平衡交换                              |
| `:123-126`   | dynamic-domain 剪枝                                   |
| `:136`       | 用`validate_vector` 做 raked 扩展                     |
| `:158-193`   | 以`n_extensions = ⌈d/range⌉` 向另一棵树延伸并判断连通 |
| `:193`       | 连通后从两棵树回溯拼接路径                            |

**3. PRM** — `src/impl/vamp/planning/prm.hh` + `roadmap.hh`

`:111` 采样 → `:117-122` 单点检查（这里把同一构型 `temp.broadcast(i)` 复制到 block 的每个 lane，复用同一个 `fkcc` 入口）→ `:138` `validate_motion` 检查边 → `:178` `utils::astar`。

`:112-114` 的注释说明他们**主动放弃了 edge sampling**（"gross hack to get around the instruction cache issue ... edge sampling, while valid, wastes too much effort with our current validation API"）—— 这点与论文"不改算法"的叙述略有张力。

**4. 简化器** — `src/impl/vamp/planning/simplify.hh`

四个可组合 routine，由 `SimplifySettings.operations` 组合（`:215-220`）：

| routine                | 行号 | 说明                                         |
| ---------------------- | ---- | -------------------------------------------- |
| `smooth_bspline`       | 15   | B-spline 平滑，对应论文的 B-spline smoothing |
| `reduce_path_vertices` | 56   | 随机区间删点                                 |
| `shortcut_path`        | 116  | 确定性 shortcut                              |
| `perturb_path`         | 144  | 随机扰动中间点                               |

`src/impl/vamp/planning/plan.hh` 的 `Path` 提供 `subdivide()` / `interpolate_to_n_states()` / `interpolate_to_resolution()` / `validate()`。

**5. 采样基元也是向量化的** — `src/impl/vamp/random/xorshift.hh`

`next()` 一次产生 8 个 64-bit 随机数（`avx_xorshift128plus`），再 `FloatVector::map_to_range(buffer, 0.F, 1.F)` 映射到 `[0,1)` 并 `Robot::scale_configuration`（用 `s_m` / `s_a` 把单位超立方体映射到关节限位，见 `panda.hh:77-97`）。

这是论文"sharing perspective 可推广到并行模型 / GPU"以及"所有 planner 共享同一 Halton 序列"（`random/halton.hh`）的实现基础。

---

## 小结：值得注意的对应与偏差

1. **`rake` 即向量宽度**（AVX2 下 8 / NEON 下 4），贯穿 `validate.hh` 与生成代码的 `ConfigurationBlock<rake>`，是全文 "rake" 一词的唯一来源。
2. **论文 III.C 的"检查与 FK 交错"在现存生成代码中未体现**：4 个机器人的 `fkcc` 都是"先算完全部 `y[]`，再做环境检查，再做自碰撞检查"（见上表）。
3. **`fkcc` 的早退是"整批拒绝"语义**：返回 `false` 时不保证能定位到哪个 lane / 哪个球碰撞；`sphere_polytope.hh:14-16` 注释明确说明了这一点，定位需用 `fkcc_debug`。
4. **`VectorSignature` 保留了已知的 over-packing 行为**（`interface.hh:42-47`），作者在注释中说明是刻意为之（紧凑打包会让 row 访问/抽取变贵）。
5. **多面体 / GJK 路径**（`gjk.hh`、`polytope_convert.hh`、`sphere_polytope.hh`）是论文之后的新增功能（commit `27cb9b6`），论文正文只覆盖 spheres / capsules / cuboids。
6. **`sphere_fk` 与 `fkcc` 是两个函数**：前者只做 FK 输出 `Spheres`（供 `vamp.<robot>.fk`、pointcloud 自过滤等使用），后者是 FK + CC 融合的热路径；两者由同一个 tracing compiler 从同一依赖图生成。
7. **`cos` 用 `sin(x + π/2)` 实现、`sqrt` 用 `rsqrt` 近似**（`avx.hh:412`、`interface.hh:452-458`）：以精度换速度。对碰撞判定的符号位足够，但边界情形可能因近似而抖动。

---

## 附：源码目录速查（`src/impl/vamp`）

```
vector.hh / vector/
    interface.hh     SIMD 抽象接口（CRTP），所有逐元素算子
    isa/avx.hh       x86 AVX2 后端（VectorWidth = 8）
    isa/neon.hh      ARM NEON 后端（VectorWidth = 4）
    isa/wasm.hh      WASM 后端
    utils.hh, math.hh
collision/
    shapes.hh        基元定义 + min_distance 预计算
    factory.hh       从 URDF / 几何描述构建环境
    environment.hh   Environment<DataT>，按类型分桶 + sort()
    validity.hh      球 vs 环境的 SIMD 主循环
    sphere_sphere.hh / sphere_capsule.hh / sphere_cuboid.hh
    sphere_heightfield.hh / sphere_polytope.hh / gjk.hh
    filter.hh        Morton 曲线点云滤波
    capt.hh          CAPT（接触空间近似）
planning/
    validate.hh      raked 运动验证（III.D 核心）
    rrtc.hh          RRT-Connect
    prm.hh, roadmap.hh
    fcit.hh, aorrtc.hh, grrtstar.hh
    simplify.hh      四个简化 routine
    nn.hh            KDTree 最近邻 + SIMD L2 距离
    plan.hh          Path / PlanningResult / Roadmap
random/
    xorshift.hh      SIMD xorshift128+
    halton.hh        多维 Halton 序列
    rng.hh, distribution.hh
robots/
    panda.hh / fetch.hh / baxter.hh / ur5.hh / sphere.hh
                     cricket tracing compiler 生成的 FK + CC 代码
bindings/
    python/          nanobind 绑定（robot_helper.hh 定义 rake / resolution 绑定）
    wasm/            WASM 绑定
```
