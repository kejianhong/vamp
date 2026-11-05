# VAMP 路径简化器（Simplify）四个 routine 详解

本文对应论文 *Motions in Microseconds via Vectorized Sampling-Based Planning* §III.E 中提到的路径简化（path simplification）部分，逐条说明 `src/impl/vamp/planning/simplify.hh` 里四个可组合 routine 的作用、原理与代码实现。

涉及文件：

|                     文件                      |                      内容                       |
| :-------------------------------------------: | :---------------------------------------------: |
|     `src/impl/vamp/planning/simplify.hh`      |       四个 routine + `simplify()` 调度器        |
| `src/impl/vamp/planning/simplify_settings.hh` |                参数结构与默认值                 |
|       `src/impl/vamp/planning/plan.hh`        | `Path`（`subdivide` / `interpolate` / `erase`） |
|     `src/impl/vamp/planning/validate.hh`      |   rake 化运动验证（所有 routine 的安全判据）    |

---

## 0. 调度框架：`simplify()`

四个 routine 都是独立函数，由 `simplify()`（`simplify.hh:192-260`）统一调度：

```cpp
// 1) 先试直线：若首尾直连可行，直接返回两点路径
if (path.size() == 2 or (path.size() > 2 and
        validate_motion<Robot, rake, resolution>(path.front(), path.back(), environment)))
{ result.path = {front, back}; return result; }

result.path = path;

// 2) 可选：按目标点数重采样
if (settings.interpolate) result.path.interpolate_to_n_states(settings.interpolate);

// 3) 最多 max_iterations 轮；每轮按 operations 顺序执行全部 routine
for (auto i = 0U; i < settings.max_iterations; ++i) {
    bool any = false;
    for (const auto &op : settings.operations)
        any |= operations.find(op)->second();     // 每个 routine 返回"是否改了路径"
    if (not any) break;                           // 一轮下来毫无变化 → 收敛，提前结束
}
```

关键约定：

| 项                    | 说明                                                                                |
| --------------------- | ----------------------------------------------------------------------------------- |
| 每个 routine 的返回值 | `bool`：本次调用**是否修改了路径**                                                  |
| 收敛条件              | 一整轮里**所有** routine 都返回 `false` 时停止                                      |
| 默认配置              | `max_iterations = 5`，`operations = {SHORTCUT, BSPLINE}`                            |
| 碰撞检查              | 全部走 `validate_motion<Robot, rake, resolution>`，即享受 §III.D 的 rake 离散化加速 |
| 就地修改              | 所有 routine 都**原地修改** `Path&`，不是函数式返回新路径                           |
| 顺序敏感              | `operations` 严格按用户给定顺序执行，顺序不同结果不同                               |

默认配置说明：VAMP 默认只开启 **shortcut + B-spline**，与论文 §III.E 说的 "randomized shortcut" 和 "B-spline smoothing" 对应；`REDUCE`、`PERTURB` 需要用户在 Python 侧显式写进 `settings.operations`。

参数默认值（`simplify_settings.hh:15-51`）：

```cpp
struct BSplineSettings { size_t max_steps{1}; float min_change{0.1}; float midpoint_interpolation{0.5}; };
struct ReduceSettings  { size_t max_steps{10}; size_t max_empty_steps{5}; float range_ratio{1./2.}; };
struct ShortcutSettings { };                                   // 空
struct PerturbSettings { size_t max_steps{10}; size_t max_empty_steps{5};
                         size_t perturbation_attempts{5}; float range{0.1}; };

struct SimplifySettings {
    size_t max_iterations{5};
    size_t interpolate{0};
    std::vector<SimplifyRoutine> operations{{SHORTCUT, BSPLINE}};
    ReduceSettings reduce; ShortcutSettings shortcut;
    BSplineSettings bspline; PerturbSettings perturb;
};
```

---

## 1. `SHORTCUT` — 确定性贪心捷径

**位置**：`simplify.hh:115-141`

### 作用

删除绕路的中间顶点：只要两点之间的**直线段**无碰撞，就把中间的点全部丢掉。

### 原理与代码

```cpp
for (auto i = 0U; i < path.size() - 2; ++i)          // 从前往后
    for (auto j = path.size() - 1; j > i + 1; --j)   // 从后往前找"最远可达"
        if (validate_motion(path[i], path[j], env)) {
            path.erase(begin() + i + 1, begin() + j); // 删掉 (i, j) 之间所有点
            result = true;
            break;                                    // 跳到下一个 i
        }
```

- 对固定 `i`，`j` **从最远往近扫**，第一次成功就连到**最远的可行点** —— 这是贪心（局部最优捷径），不是全局最优。
- 删除后 `path.size()` 变小，外层的 `i < path.size() - 2` 每轮重新求值，所以不会越界。
- **完全确定**：不用 RNG，`ShortcutSettings` 是个空结构体。
- 复杂度：最坏 `O(n²)` 次 `validate_motion`。
- 效果：路径长度单调不增，点数单调不增。
- 只跳出内循环，不跳出外循环。当删除[i+1, j-1]这些点后，由于`path.size()`会重新计算，`i+1`天然地更新到`path[j]`的位置，可以继续修剪。

> 它是默认 operations 的**第一个**，所以每轮先把上一轮 BSPLINE 产生的冗余点压掉。

---

## 2. `BSPLINE` — 细分 + 拉普拉斯平滑

**位置**：`simplify.hh:14-53`

### 作用

平滑路径、削减拐角（降低曲率，让轨迹更适合执行）。注意它**只加点（修改顶点)、不删点**，属于"细化/精修"类操作。

- 严格缩短路径；
- 当corner-cutting执行下去，则对应的极限曲线就是光滑的B-spline（执行1次时只是一次拉普拉斯变换)。路径拐角被磨圆，曲率降低（速度/加速度突变小)；

### 原理：Chaikin 式 corner cutting

每个 step 做两件事。

**① `path.subdivide()`**（`plan.hh:34-49`）：在每两个相邻点之间插入中点，点数从 `n` 变成 `2n-1`：

```text
原顶点：  p0      p1      p2      ...   p_{n-1}
细分后：  p0  m0  p1  m1  p2  ...  m_{n-2}  p_{n-1}
          ↑偶数下标=原顶点    ↑奇数下标=新插入的中点
```

**② 只移动偶数下标的原顶点**（`index = 2, 4, …`）：

```cpp
temp_1   = path[index].interpolate(path[index-1], 0.5);   // 0.5·p + 0.5·前中点
temp_2   = path[index].interpolate(path[index+1], 0.5);   // 0.5·p + 0.5·后中点
midpoint = temp_1.interpolate(temp_2, 0.5);               // 0.5·temp_1 + 0.5·temp_2
```

`interpolate` 的定义是 `a.interpolate(b, α) = a + (b − a)·α`（`interface.hh:422`），代入化简得到：

```text
new p_k = 0.125·p_{k-1} + 0.75·p_k + 0.125·p_{k+1}
```

即**把顶点沿邻居方向做一次加权拉普拉斯平滑**（自己占 3/4，两个邻居各占 1/8）。反复执行就是经典的 corner-cutting 细分，极限曲线是一条 B-spline —— 这就是函数名的由来。

- 相邻的中点（奇数下标）**从不被修改**，所以更新顺序无关、无需 Jacobi/Gauss-Seidel 处理。
- 首尾两端点（下标 0 和 `2n-2`）固定不动。

**提交条件**（`simplify.hh:37-43`）：

```cpp
if (path[index].distance(midpoint) > settings.min_change and      // 位移足够大 (默认 0.1)
    validate_motion(path[index-1], midpoint, env) and             // 新前半段无碰撞
    validate_motion(midpoint, path[index+1], env))                // 新后半段无碰撞
{ path[index] = midpoint; changed |= (updated = true); }
```

即：**位移太小就没必要动**（避免无意义抖动），且新生成的两段都必须通过碰撞检查才接受 —— 平滑永远不会把路径推进障碍物。

- 循环：若某个 step 里**没有任何顶点被更新**，提前 `break`；否则最多 `max_steps` 次（默认 **1**）。

### ⚠️ 一个容易忽略的点

`path.subdivide()` 是**无条件**执行的，在检查 `updated` 之前。所以即使一个顶点都没移动、函数返回 `false`，路径点数也已经翻倍了。也就是说 BSPLINE 单独使用时路径会"越来越密"，必须靠 SHORTCUT / REDUCE 把它压回去 —— 这正是默认组合 `{SHORTCUT, BSPLINE}` 的意义。

---

## 3. `REDUCE` — 随机区间删点

**位置**：`simplify.hh:55-113`

### 作用

随机地删除一段连续顶点中的冗余点，是 SHORTCUT 的"随机化 + 局部窗口"版本。

### 原理与代码

```cpp
int range = 1 + floor(0.5 + n * settings.range_ratio);         // range_ratio = 0.5 → 约半个路径长

auto point_0 = rng->dist.uniform_integer(0, max_n);            // 随机锚点
auto point_1 = rng->dist.uniform_integer(max(point_0-range,0),
                                         min(max_n, point_0+range));  // 附近随机点

// 保证两点之间至少隔一个点（否则删了也没意义）
if (abs(point_0 - point_1) < 2) { point_1 = point_0 ± 2; ... }
if (point_0 > point_1) swap(point_0, point_1);

if (validate_motion(path[point_0], path[point_1], env))
{
    path.erase(begin() + point_0 + 1, begin() + point_1);      // 删掉中间所有点
    no_change = 0;  result = true;
}
```

- **随机**选锚点和窗口，因此每次运行结果可能不同（受 RNG 种子影响）。
- `range_ratio = 0.5` 意味着搜索窗口约为整条路径的一半，不是全局。
- 终止条件（默认 `max_steps = 10`、`max_empty_steps = 5`）：

```cpp
for (auto i = 0U, no_change = 0U; i < max_steps or no_change < max_empty_steps; ++i, ++no_change)
```

注意这里是 **OR**：至少跑 `max_steps` 轮；之后只要还在不断成功（`no_change` 被重置为 0），就继续跑；直到**连续 `max_empty_steps` 次失败**才停。`max_steps` 或 `max_empty_steps` 设为 0 时，用 `path.size()` 代替。

- 效果：路径点数减少；路径长度**不保证**减少（随机删点可能删掉一个"绕但不长"的点）。
- `REDUCE`能做的每一次删除，`SHORTCUT`中的候选集中都有。`SHORTCUT`的复杂度是$O(n^2)$，在对时间要求及其严苛的情况下，可以关闭`SHORTCUT`，开启`REDUCE`。

---

## 4. `PERTURB` — 随机局部扰动

**位置**：`simplify.hh:143-190`

### 作用

本质是对路径做**随机坐标下降**，用于在"必须保留某些中间点"的场合继续缩短路径。

### 原理与代码

```cpp
auto k = rng->dist.uniform_integer(1UL, path.size() - 2);      // 随机选一个内部点
auto old_cost = |p_k − p_{k-1}| + |p_{k+1} − p_k|;             // 该点的两段总长

for (attempt = 0; attempt < settings.perturbation_attempts; ++attempt)   // 默认 5 次
{
    auto perturbation = rng->next();
    Robot::scale_configuration(perturbation);                  // ← 见下方 ⚠️
    const auto new_state = perturb_state.interpolate(perturbation, settings.range);  // 朝随机点走 10%

    float new_cost = |new_state − before| + |after − new_state|;

    if (new_cost < old_cost and                                // 必须真正变短
        validate_motion(before, new_state, env) and            // 前半段无碰撞
        validate_motion(after,  new_state, env))               // 后半段无碰撞
    { path[k] = new_state; break; }
}
```

- 只接受**代价严格下降**且两段都无碰撞的移动 —— 所以路径长度单调不增。
- 与 shortcut 的区别：shortcut 只能沿"已有顶点的连线"走，而 perturb 可以在构型空间里**任意方向**微调，因此能利用 shortcut 触及不到的自由度（例如绕开障碍物时把某个弯点往一侧推）。
- 参数：`max_steps = 10`、`max_empty_steps = 5`、`perturbation_attempts = 5`、`range = 0.1`。
- 终止条件与 REDUCE 相同（`or` 结构 + 连续失败计数）。

---

## 5. 四者对比与组合关系

| routine    | 是否随机 | 点数变化         | 是否移动顶点 | 主要作用                      | 默认启用 |
| ---------- | -------- | ---------------- | ------------ | ----------------------------- | -------- |
| `SHORTCUT` | 否       | 减少             | 否           | 删掉绕路段，贪心取最远可达点  | ✅        |
| `BSPLINE`  | 否       | **增加**（细分） | 是           | 拉普拉斯平滑 / corner cutting | ✅        |
| `REDUCE`   | 是       | 减少             | 否           | 随机窗口内删点                | ❌        |
| `PERTURB`  | 是       | 不变             | 是           | 随机坐标下降，微调中间点位置  | ❌        |

**默认组合的协作逻辑**（`{SHORTCUT, BSPLINE}`，最多 5 轮）：

```text
第 1 轮： SHORTCUT  ← 先压掉规划器留下的冗余顶点
         BSPLINE   ← 再细分并平滑（点数翻倍，产生大量新的中点）

第 2 轮： SHORTCUT  ← 把上一步新插入、但"可被直线取代"的中点删掉
         BSPLINE   ← 再平滑一次

...  直到某一轮两个 routine 都没有改动 → 收敛
```

一个典型的等价叙述是 **"先粗化、再细化、再粗化"**：SHORTCUT 负责"拓扑上的化简"（少绕路），BSPLINE 负责"几何上的光顺"（少拐角），两者交替直到不动点。

---

## 6. 补充说明

- **调用入口**：Python 侧的 `simplify` 绑定在 `robot_helper.hh:303`；AORRTC 会在内部自动调用它（`aorrtc.hh:408`、`:480`），不是所有规划器都会做后处理。
- **只做离散验证**：所有 routine 的安全性都依赖 `validate_motion` 的离散采样，`BSPLINE` 平滑后新线段若恰好"跨过"一个薄障碍物，也可能被判为安全 —— 这是 `resolution` 参数的固有局限，与简化器本身无关。
- **`operations` 的顺序有意义**：`for (const auto &op : settings.operations)` 严格按用户给定顺序执行。把 `REDUCE` 放在 `BSPLINE` 前面，效果与反过来并不相同。
- **`ShortcutSettings` 是空的**：`shortcut_path` 甚至把参数注释掉了（`const ShortcutSettings & /*settings*/`），说明目前不暴露任何可调项。
- **Python 侧枚举名**：`BSPLINE` / `REDUCE` / `SHORTCUT` / `PERTURB`（`bindings/python/settings.cc:104-107`），通过 `SimplifySettings.operations` 设置（同文件 `:135`）。
