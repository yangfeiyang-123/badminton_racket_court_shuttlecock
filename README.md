# Badminton Racket / Court / Shuttlecock Physics

MuJoCo 羽毛球物理建模：球场、球拍（含拍弦床）、羽毛球（气动）、拍–球接触与碰撞、
双人对打物理、手–拍握持接触/摩擦，以及 JAX/MJX GPU 并行版本。

本仓库从 MuscleMimic 主仓库
（[asi_strengthen_musclemimic](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic)）
中抽出物理建模相关部分。目录结构与主仓库保持一致（`environment.*`、`src.grip.*` 导入路径不变），
可以直接回合并，也能单独运行大部分测试。

## 目录

| 路径 | 内容 |
| --- | --- |
| `environment/court/` | BWF 标准球场几何（边界线“含线”语义、发球区、落点判定）、球网碰撞体、MJCF 生成与参数校验 |
| `environment/racket/` | 球拍几何/质量/惯量参数，刚性与柔性代理 MJCF，拍弦床接触力模型 `racket_stringbed.py`，设计验证图 |
| `environment/shuttlecock/` | 羽毛球质量/惯量/规则尺寸，气动模型 `shuttlecock_aero.py`（v1 + v2），事件式拍–球碰撞 `shuttlecock_racket_impact.py` |
| `environment/overall_environment/src/badminton_physics.py` | 每个子步的完整物理管线（气动 → 拍弦力 → 事件反弹 → `mj_step`） |
| `environment/overall_environment/src/badminton_physics_mjx.py` | 上述管线的 JAX/MJX 逐公式移植，可 `jit`/`vmap`/`scan` |
| `environment/overall_environment/src/shuttle_feeder.py` | 来球轨迹生成（刚体 + 气动积分的 `calibrated_rigid_body_v3`） |
| `environment/overall_environment/src/contact_graph.py` / `impact_target.py` / `ghost_racket.py` | 手柄/拍面接触统计、击球点与身体尺度目标、参考球拍位姿与速度 |
| `environment/overall_environment/src/racket_attachment.py` / `soft_weld_schedule.py` | 手–拍刚性附着合同（质量、惯量、握持变换，带 SHA-256 指纹），软焊约束 solref/solimp 课程 |
| `environment/double_play/src/rally_physics.py` | 一球多拍的对打子步物理（v2 物理默认开启） |
| `src/grip/` + `configs/racket_handle_params.json` + `configs/right_hand_racket_grip_*` | 右手–拍柄握持：拍柄八角截面几何、接触摩擦/condim 参数、握姿求解与评估 |
| `configs/racket_grip/forehand_clear_grip_v{1,2}_custom.json` | **握拍姿势 preset**：右手 20 个手指关节目标角（rad），绑定一份球拍附着合同的 SHA-256 指纹；v2 为当前默认 |
| `configs/racket_attachment/forehand_clear_rigid_v{2,3,4}_custom.json` | 球拍相对右手 `thirdmc_r` 的固定位姿（位置、朝向）与球拍质量/惯量合同；v4 与 v2 握拍配套 |
| `configs/right_hand_racket_grip_reference.json` | 静态握拍参考：IK 优化后的右手 qpos 与球拍位姿（平均 site 误差 12.5 mm） |
| `configs/right_hand_racket_grip_targets.json`, `environment/holdracket/configs/` | 手部 site ↔ 拍柄目标点对应与权重 |
| `src/grip/racket_grip_preset.py` | 握拍 preset 读取、校验（指纹必须与附着合同一致）、写出 |
| `src/grip/racket_pose_editor.py`, `docs/racket_pose_editor.rst` | 交互式握拍编辑器：调球拍朝向/坐标与逐关节手指角度，保存为新版本 preset |
| `docs/contracts/body_action_modes_and_rigid_racket.md` | 刚性球拍与身体动作模式合同 |

坐标约定：球场 x 沿场长（网在 x=0），y 沿场宽，z 向上；球拍体系 +Y 柄→拍头、+Z 拍面法向；
羽毛球体系 +Z 指向球头（软木）。全部 SI 单位。

## 物理模型摘要

**羽毛球气动**（`shuttlecock_aero.py`）
- 二次阻力 `F_D = -k |v_rel| v_rel`，`k = m g / v_t²`，由终端速度标定（名义 `v_t ≈ 6.86 m/s`，质量 5.19 g）。
- 攻角阻力增益、压力中心位于质心后方产生的回正力矩、角阻尼；总力矩有上限。
- v2（新增字段开启，旧默认值逐位不变）：裙部横流力 `F_N = -k·g_N·|v_rel|·v_⊥`、
  各向异性角阻尼（翻滚阻尼远大于自然轴向自旋）、按回合注入风速的域随机化。
- 注意：MuJoCo freejoint 的角速度是 body 局部系。

**拍弦床接触**（`racket_stringbed.py`）
- 椭圆拍面上的罚函数膜代理：法向刚度 9600 N/m（中心 5 mm 约 48 N），越靠边越硬（上限 2.5 倍），
  法向阻尼 3.0，切向阻尼 0.15，切向力受库仑上限 `μ = 0.08` 约束；作用力在拍、球上等大反向。
- 扫掠穿越检测（v2）：一个子步内越过有限捕获带的高速球按事件反弹处理，防止隧穿。

**事件式碰撞**（`shuttlecock_racket_impact.py`）
- 法向恢复系数 0.50、切向速度保留 0.85（v1）；v2 采用随冲击速度下降的恢复系数
  `e = clip(e0 - s·max(0, |v_n| - v_ref), e_min, e0)`。
- 反弹冲量与球拍上的等大反向点冲量成对施加，并传递到球拍所在运动链；
  v2 闭合软木偏心冲击的角冲量通道。
- 冷却期内抑制连续拍弦力，避免同一击球被重复计算。

**球场**：BWF 名义尺寸、40 mm 线宽归属规则、球网碰撞几何，
地面/网/球头接触为 MuJoCo 原生接触（`solref`、`solimp`、`friction` 见 XML 与 `court_bwf_nominal.json`）。

**手–拍握持**：拍柄 G5 八角截面几何、手部接触 site 与拍柄的切向/扭转/滚动摩擦和 `condim`
由配置生成场景；软焊（weld）强度、`solref`/`solimp` 按课程阶段调度。

## 安装与测试

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[mjx,viz,test]"
pytest
```

## 测试覆盖与独立运行限制

- 仅安装本仓库依赖（无 MuscleMimic）：**97 passed, 1 skipped**。覆盖球场几何/XML、球拍设计与拍弦力、
  羽毛球气动 v1/v2、拍–球事件碰撞、单人子步物理（含 v2）、MJX 与 NumPy 数值一致性、来球生成、
  接触图、击球目标、参考球拍、握拍 preset 读写与编辑器几何。
- `test_double_play_scene.py`、`test_rally_physics.py`、`tests/unit/test_right_hand_racket_grip.py`
  需要重新生成全身/握拍场景，依赖主仓库的 `musclemimic` 包和 `musclemimic_models`；
  未安装时由根目录 `conftest.py` 跳过。在主仓库环境中这 74 项全部通过。
- 预生成的场景 XML（`overall_incoming_hit_scene.xml`、`double_play_scene.xml`、
  `right_hand_racket_grip_scene.xml`）已包含在内，直接加载无需 MuscleMimic；
  重新生成这些场景（`build_*_scene.py`）、`src/grip/render/` 和握拍编辑器的交互预览需要 MuscleMimic。
- 主仓库中握拍 preset 在 reset 时由 `loco_mujoco` 的 `RacketGripInitialStateHandler` 施加到模型，
  该训练框架胶水代码未包含在本仓库。

## License

Apache-2.0（与主仓库一致）。`environment/overall_environment/assets/mimic_msk_model/` 中的骨骼网格来自
MyoSuite / MuscleMimic 肌骨模型，仅用于加载含人体的接触测试场景；
`environment/double_play/assets/mimic_msk_model` 是指向它的符号链接。
