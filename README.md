# DTU Point Cloud Evaluation

**[English](#english) · [中文](#中文)**

## English

Evaluate DTU reconstructed point clouds without MATLAB. Results are in
millimetres; **lower is better**.

| File | Purpose |
| --- | --- |
| `dtu_eval_python.py` | Evaluate one scan on CPU. |
| `dtu_eval_torch.py` | Evaluate one scan with PyTorch on GPU or CPU. |
| `evalply.py` | Evaluate the 22 DTU test scans and report their mean. |

### 1. Prepare

```bash
python -m pip install numpy scipy plyfile
```

Install PyTorch as well for the GPU option. Download **SampleSet** and
**Points** from the [DTU MVS website](https://roboimagedata.compute.dtu.dk/?page_id=36).
The dataset root passed below must contain `ObsMask/` and `Points/stl/`.
PLY coordinates must use the DTU coordinate system and millimetres.

For batch evaluation, place the 22 files `scan1.ply`, `scan4.ply`, …,
`scan118.ply` in one folder.

### 2. Run

**One scan · CPU**

```bash
python dtu_eval_python.py --scan 77 --ply /path/to/scan77.ply --dataset-root "/path/to/MVS Data" --output /path/to/scan77.json
```

**One scan · GPU**

```bash
python dtu_eval_torch.py --scan 77 --ply /path/to/scan77.ply --dataset-root "/path/to/MVS Data" --output /path/to/scan77.json --device cuda
```

**All 22 scans · CPU**

```bash
python evalply.py --data /path/to/plys --dataset-dir "/path/to/MVS Data" --output-dir /path/to/results
```

**All 22 scans · GPU:** add `--backend torch --device cuda` to the batch
command. Add `--scans 77` to try a subset or `--seed 0` to choose the
thinning seed.

The batch command saves `scanN.json` for each scene and
`summary_all22.json` with the **unweighted mean across scenes**.

---

## 中文

无需 MATLAB，评估 DTU 重建点云。结果单位为毫米，**越低越好**。

| 文件 | 用途 |
| --- | --- |
| `dtu_eval_python.py` | 使用 CPU 评估单个场景。 |
| `dtu_eval_torch.py` | 使用 PyTorch 在 GPU 或 CPU 上评估单个场景。 |
| `evalply.py` | 评估 DTU 的 22 个测试场景并计算均值。 |

### 1. 准备

```bash
python -m pip install numpy scipy plyfile
```

使用 GPU 时还需安装 PyTorch。从
[DTU MVS 官网](https://roboimagedata.compute.dtu.dk/?page_id=36)
下载 **SampleSet** 和 **Points**。以下命令中的数据根目录须包含
`ObsMask/` 与 `Points/stl/`。待评估 PLY 须使用 DTU 坐标系，
坐标单位为毫米。

批量评估时，将 `scan1.ply`、`scan4.ply`、……、`scan118.ply`
这 22 个文件放在同一目录。

### 2. 运行

**单场景 · CPU**

```bash
python dtu_eval_python.py --scan 77 --ply /path/to/scan77.ply --dataset-root "/path/to/MVS Data" --output /path/to/scan77.json
```

**单场景 · GPU**

```bash
python dtu_eval_torch.py --scan 77 --ply /path/to/scan77.ply --dataset-root "/path/to/MVS Data" --output /path/to/scan77.json --device cuda
```

**22 场景 · CPU**

```bash
python evalply.py --data /path/to/plys --dataset-dir "/path/to/MVS Data" --output-dir /path/to/results
```

**22 场景 · GPU：**在批量命令后添加
`--backend torch --device cuda`。可用 `--scans 77` 先试运行部分
场景，用 `--seed 0` 指定稀疏化随机种子。

批量运行逐场景保存 `scanN.json`，并在 `summary_all22.json`
中给出**22 个场景的等权均值**。
