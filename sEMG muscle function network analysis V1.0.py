import numpy as np
from scipy import signal
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import csv

# ============================== 配置区 ==============================
N_CH      = 16            # 默认通道数(实际以文件为准)
FILE_PATH = "emg.txt"     # 对话框取消/headless时的回退路径
FS        = 2000          # 默认采样率(若文件含时间列则自动识别覆盖)

# 默认16肌肉命名(文件有表头时自动替换)
MUSCLE_NAMES = ["DEL_L","DEL_R","BB_L","BB_R","TB_L","TB_R","ECRL_L","ECRL_R",
                "RF_L","RF_R","BF_L","BF_R","GC_L","GC_R","TA_L","TA_R"]

# Welch 相干参数
WIN, OVERLAP, NFFT = 1000, 500, 4096   # 窗1000点, 重叠500点, nfft=4096
FMIN, FMAX         = 10, 150           # 连边所用频段

# 阈值方法: 'surrogate'(相位随机化代理数据, 推荐) 或 'fixed'(固定阈值)
THRESH_METHOD = "surrogate"
SURROGATE_N   = 20        # 代理数据次数(通道数少/数据短建议30~50)
THRESH_FIXED  = 0.05      # fixed 模式下的相干阈值

# 分频段网络(可选): None = 只做全频段
BANDS = None
# BANDS = {"alpha": (8, 13), "beta": (13, 30), "gamma": (31, 60)}

# 陈玲玲等(2019)互信息网络参数：等宽直方图估计概率，归一化后TH=0.50二值化
MI_ANALYSIS = True
MI_BINS = 32
MI_THRESHOLD = 0.50

OUT_DIR = "network_output"
rng = np.random.default_rng(42)
# =====================================================================


# ----------------------------- 文件选择 -----------------------------
def _pick_file():
    """弹出文件选择对话框(headless/无tkinter时回退到配置路径)"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        p = filedialog.askopenfilename(
            title="选择sEMG数据文件 (TXT/CSV)",
            filetypes=[("数据文件", "*.txt *.csv *.tsv"), ("所有文件", "*.*")])
        root.destroy()
        if p:
            return p
    except Exception:
        pass
    print(f"[提示] 文件对话框不可用, 回退到配置路径: {FILE_PATH}")
    return FILE_PATH


# ----------------------------- 数据读取 -----------------------------
def load_emg_txt(path=None):
    """读取TXT -> 自动识别编码/表头/肌肉名/时间列/采样率
       返回 (data, names, fs):
         data  - 信号矩阵 (采样点 x 通道), 已去均值
         names - 表头肌肉名列表(无表头时为None)
         fs    - 采样率(文件含时间列时自动识别, 否则用配置FS)"""
    if not path:
        path = _pick_file()
        print(f"[选择] {path}")
    # 1) 读取全文: UTF-8 -> GBK -> Latin-1 依次尝试
    lines = used_enc = None
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            lines = open(path, "r", encoding=enc).read().splitlines()
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise ValueError(f"无法识别文件编码: {path}")
    # 2) 定位表头: 首个"tab分隔后>=4个token且非纯数值"的行
    #    (可跳过 '1.c3d'、空行等; 无表头纯数据文件不会命中)
    h_idx = h_toks = None
    for i, ln in enumerate(lines):
        toks = [t.strip() for t in ln.split("\t") if t.strip()]
        if len(toks) >= 4:
            try:
                [float(t) for t in toks]
                continue                     # 纯数值 -> 数据行
            except ValueError:
                h_idx, h_toks = i, toks    # 非数值 -> 表头
                break
    # 3) 清洗肌肉名: 去'(μV)'等单位、去末尾冒号; 第1列(Time)跳过
    names = None
    if h_toks is not None:
        names = [re.sub(r"\(.*?\)", "", t).strip().rstrip(":").strip()
                 for t in h_toks[1:]]
    # 4) 解析数据行: 只保留全部为数值、且列数匹配的行。
    #    有表头时允许“含时间列”或“不含时间列”，但同一文件不能混用两种列数，
    #    否则 np.array(rows) 会因各行长度不一致而报 inhomogeneous shape 错误。
    n_col = len(names) + 1 if names else None
    rows = []
    start = h_idx + 1 if h_idx is not None else 0
    for ln in lines[start:]:
        parts = ln.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if n_col is not None and len(vals) not in (n_col, n_col - 1):
            continue
        rows.append(vals)
    if not rows:
        raise ValueError(f"未解析到数据行: {path}")

    # 以出现次数最多的列数为准，并明确跳过异常行，避免生成不规则二维数组。
    col_counts = {k: sum(len(r) == k for r in rows)
                  for k in sorted({len(r) for r in rows})}
    if n_col is not None and col_counts.get(n_col, 0) > 0:
        expected_col = n_col
    else:
        expected_col = max(col_counts, key=col_counts.get)
    valid_rows = [r for r in rows if len(r) == expected_col]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"[警告] 跳过 {skipped} 行列数异常的数据行，统一按 {expected_col} 列读取")
    data = np.asarray(valid_rows, dtype=float)

    # 删除含 NaN/Inf 的行，避免后续滤波和相干性计算产生无效结果。
    finite_rows = np.all(np.isfinite(data), axis=1)
    if not np.all(finite_rows):
        print(f"[警告] 删除 {np.sum(~finite_rows)} 行含 NaN/Inf 的数据")
        data = data[finite_rows]
    if data.shape[0] < 2:
        raise ValueError(f"有效数据不足2行: {path}")
    # 5) 时间列处理与采样率识别
    fs = FS
    if names is not None and data.shape[1] == len(names) + 1:
        dt = np.median(np.diff(data[:, 0]))
        if dt > 0:
            fs = 1.0 / dt
        data = data[:, 1:]               # 剔除时间列
    elif data.shape[0] >= 100:           # 无表头但首列像时间列(单调递增)也剔除
        if np.all(np.diff(data[:, 0]) > 0) and abs(np.std(np.diff(data[:, 0]))/np.mean(np.diff(data[:, 0]))) < 0.01:
            dt = np.median(np.diff(data[:, 0]))
            if dt > 0:
                fs = 1.0 / dt
            data = data[:, 1:]
    data = data - np.mean(data, axis=0)
    print(f"[读取] 编码={used_enc}, {data.shape[0]/fs:.2f} s (fs={fs:.0f} Hz), "
          f"{data.shape[1]} 通道" + (f"\n  肌肉名: {names}" if names else " (无表头)"))
    return data, names, fs


# ----------------------------- 预处理 -------------------------------
def preprocess(x, fs=None):
    """50Hz陷波 + 20-450Hz 带通"""
    fs = FS if fs is None else fs
    y = np.asarray(x, dtype=float)
    nyq = fs / 2
    for f0 in (50,):
        b, a = signal.iirnotch(f0, Q=35, fs=fs)
        y = signal.filtfilt(b, a, y)
    b, a = signal.butter(4, [20 / nyq, 450 / nyq], btype="band")
    return signal.filtfilt(b, a, y)


# --------------------------- 相干性计算 -----------------------------
def _welch_params(x, fs):
    """生成安全的Welch参数，短信号也不会出现 noverlap >= nperseg。"""
    nperseg = min(WIN, len(x))
    if nperseg < 2:
        raise ValueError("每个通道至少需要2个有效采样点")
    return dict(fs=fs, window="hamming", nperseg=nperseg,
                noverlap=min(OVERLAP, nperseg - 1), nfft=max(NFFT, nperseg))


def pair_coherence(x, y, fmin=FMIN, fmax=FMAX, fs=None):
    """Welch法相干谱 -> 返回(f, coh)指定频段"""
    fs = FS if fs is None else fs
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("相干性计算要求两个信号长度相同")
    kw = _welch_params(x, fs)
    f, Pxx = signal.welch(x, **kw)
    _,  Pyy = signal.welch(y, **kw)
    _,  Pxy = signal.csd(x, y, **kw)
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-20)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError(f"频段 {fmin}-{fmax} Hz 内没有可用频率点")
    return f[m], np.clip(np.nan_to_num(coh[m]), 0, 1)


def coherence_matrix(emg, fmin=FMIN, fmax=FMAX, fs=None):
    """n通道两两相干 -> 平均相干矩阵 (nxn)。

    所有通道共用一次STFT，再向量化计算互谱；结果与逐对调用Welch/CSD一致，
    但避免在代理数据循环中重复计算同一通道的自谱，显著缩短运行时间。
    """
    fs = FS if fs is None else fs
    emg = np.asarray(emg, dtype=float)
    if emg.ndim != 2 or emg.shape[1] < 2:
        raise ValueError("相干性分析至少需要2个通道")
    kw = _welch_params(emg, fs)
    f, _, Z = signal.stft(emg, detrend="constant", return_onesided=True,
                          boundary=None, padded=False, axis=0, **kw)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError(f"频段 {fmin}-{fmax} Hz 内没有可用频率点")

    # stft(axis=0) 返回形状: (频率, 通道, 窗段)。
    Zb = Z[m]
    Pxy = np.einsum("fit,fjt->fij", Zb, Zb.conj()) / Zb.shape[2]
    P = np.mean(np.abs(Zb) ** 2, axis=2)       # (频率, 通道)
    coh = np.abs(Pxy) ** 2 / (P[:, :, None] * P[:, None, :] + 1e-20)
    C = np.mean(np.clip(np.nan_to_num(coh), 0, 1), axis=0)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    return C


# --------------------- 互信息(陈玲玲等, 2019) -----------------------
def _shannon_entropy(counts):
    """由计数估计Shannon熵 H=-sum(p*log2(p))，单位为bit。"""
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def mutual_information_matrix(emg, bins=MI_BINS):
    """计算通道间互信息矩阵及0~1归一化互信息矩阵。

    文献方法：H(S)、H(Q)、H(S,Q) -> MI=H(S)+H(Q)-H(S,Q)。
    文献未指定概率离散化与归一化公式，这里明确采用每通道等宽直方图，
    再对全部非对角MI值做最小-最大归一化：
        MI_norm=(MI-min(MI_off))/(max(MI_off)-min(MI_off))。
    这样TH=0.50表示保留相对关联较强的一半以上区间，而不是固定的物理量阈值。

    返回 (MI_raw, MI_norm, entropies)；MI_raw单位为bit，MI_norm对角线固定为1。
    常数通道不含信息，其与其他通道的原始MI为0。若所有非对角MI完全相同，
    为避免除以零，非对角归一化值全部设为0。
    """
    x = np.asarray(emg, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("互信息分析需要形状为(采样点, 通道)且至少含2个采样点的数据")
    if not np.all(np.isfinite(x)):
        raise ValueError("互信息分析输入含NaN或Inf，请先清理数据")
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("MI_BINS 必须是大于等于2的整数")

    n_samples, n_ch = x.shape
    codes = np.zeros((n_samples, n_ch), dtype=np.int64)
    H = np.zeros(n_ch, dtype=float)
    for ch in range(n_ch):
        xmin, xmax = np.min(x[:, ch]), np.max(x[:, ch])
        if xmax > xmin:
            edges = np.linspace(xmin, xmax, bins + 1)
            # searchsorted把右端最大值也归入最后一个箱，箱编号固定为0~bins-1。
            codes[:, ch] = np.clip(
                np.searchsorted(edges, x[:, ch], side="right") - 1, 0, bins - 1)
        counts = np.bincount(codes[:, ch], minlength=bins)
        H[ch] = _shannon_entropy(counts)

    MI = np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                mi = H[i]
            else:
                joint_codes = codes[:, i] * bins + codes[:, j]
                joint_counts = np.bincount(joint_codes, minlength=bins * bins)
                H_joint = _shannon_entropy(joint_counts)
                # 直方图估计的舍入误差可能产生极小负值，互信息本身不应为负。
                mi = max(0.0, H[i] + H[j] - H_joint)
            MI[i, j] = MI[j, i] = mi

    MI_norm = np.zeros_like(MI)
    if n_ch > 1:
        iu = np.triu_indices(n_ch, k=1)
        mi_off = MI[iu]
        mi_min, mi_max = float(np.min(mi_off)), float(np.max(mi_off))
        if mi_max > mi_min:
            MI_norm = (MI - mi_min) / (mi_max - mi_min)
            MI_norm = np.clip(MI_norm, 0.0, 1.0)
    np.fill_diagonal(MI_norm, 1.0)  # 对角线不参与建网和归一化范围计算
    return MI, MI_norm, H


# ------------------------- 代理数据(阈值) ---------------------------
def phase_randomize(x):
    """傅里叶相位随机化: 保留功率谱, 破坏时序耦合(构造代理数据)"""
    X = np.fft.rfft(x)
    # 偶数长度保留Nyquist分量；奇数长度的最后一个rfft分量也应随机化。
    stop = len(X) - 1 if len(x) % 2 == 0 else len(X)
    if stop > 1:
        ph = rng.uniform(0, 2 * np.pi, stop - 1)
        X[1:stop] *= np.exp(1j * ph)
    return np.fft.irfft(X, n=len(x))


def surrogate_threshold(emg, n_surr=SURROGATE_N, fmin=FMIN, fmax=FMAX, fs=None):
    """相位随机化代理数据估计显著相干阈值 (均值+2std, 约95%置信度)"""
    n = emg.shape[1]
    if n < 2:
        raise ValueError("代理数据阈值分析至少需要2个通道")
    if n_surr < 1:
        raise ValueError("SURROGATE_N 必须大于等于1")
    vals = []
    iu = np.triu_indices(n, k=1)
    for k in range(n_surr):
        surr = np.column_stack([phase_randomize(emg[:, ch]) for ch in range(n)])
        C_surr = coherence_matrix(surr, fmin, fmax, fs)
        vals.extend(C_surr[iu].tolist())
        print(f"\r[代理阈值] 已完成 {k + 1}/{n_surr} 次", end="", flush=True)
    print()
    th = np.mean(vals) + 2 * np.std(vals)
    print(f"[代理阈值] mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
          f"threshold = {th:.4f}")
    return th


# --------------------------- 网络构建 -------------------------------
def build_graph(C, thresh, names=MUSCLE_NAMES):
    """相干矩阵 -> 二值无向图(保留相干值作为边权重)"""
    n = C.shape[0]
    node_names = list(names[:n]) if names is not None else []
    node_names.extend(f"CH_{i + 1}" for i in range(len(node_names), n))
    # NetworkX会用节点名识别节点；重名必须修正，否则节点数会与相干矩阵不一致。
    seen, used = {}, set()
    for i, name in enumerate(node_names):
        base = str(name) if str(name).strip() else f"CH_{i + 1}"
        seen[base] = seen.get(base, 0) + 1
        candidate = base if seen[base] == 1 else f"{base}_{seen[base]}"
        while candidate in used:
            seen[base] += 1
            candidate = f"{base}_{seen[base]}"
        node_names[i] = candidate
        used.add(candidate)

    G = nx.Graph()
    G.add_nodes_from(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= thresh:
                G.add_edge(node_names[i], node_names[j], weight=float(C[i, j]))
    return G


# --------------------------- 图论指标 -------------------------------
def _avg_path_len(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    lens = [d for s, dd in nx.all_pairs_shortest_path_length(G)
            for t, d in dd.items() if t != s]
    return np.mean(lens) if lens else np.nan


def graph_metrics(G):
    """聚类系数/全局效率/介数中心性/平均路径/平均度 + 小世界sigma"""
    names = list(G.nodes())
    deg = dict(G.degree())
    C   = nx.average_clustering(G)
    E   = nx.global_efficiency(G)
    B   = nx.betweenness_centrality(G, normalized=True)
    L   = _avg_path_len(G)
    k   = np.mean(list(deg.values())) if deg else 0
    sigma = gamma = lam = np.nan
    if G.number_of_edges() > 0 and k >= 2:     # 平均度>=2才做随机网络对比
        Gr  = nx.random_reference(G, niter=20, seed=42)
        Cr, Lr = nx.average_clustering(Gr), _avg_path_len(Gr)
        if Cr > 0 and Lr > 0:
            gamma, lam = C / Cr, L / Lr
            sigma = gamma / lam
    hubs = sorted(names, key=lambda m: (deg.get(m, 0), B.get(m, 0)),
                  reverse=True)[:5]
    return dict(clustering=C, efficiency=E, path_len=L, avg_degree=k,
                gamma=gamma, lambda_=lam, sigma=sigma, n_edges=G.number_of_edges(),
                degree=deg, betweenness=B, hubs=hubs)


# --------------------- 节点收缩法(陈玲玲等, 2019) --------------------
def network_cohesion(G):
    """网络凝聚度 cohesion=1/(n*L)；仅对节点数>=2的连通网络有定义。"""
    n = G.number_of_nodes()
    if n < 2 or not nx.is_connected(G):
        return np.nan
    L = nx.average_shortest_path_length(G)
    return 1.0 / (n * L) if L > 0 else np.nan


def contract_node_and_neighbors(G, node):
    """将node及其全部直接邻居合并为一个节点，并把外部边重连到该节点。"""
    if node not in G:
        raise KeyError(f"节点不存在: {node}")
    merged = {node, *G.neighbors(node)}
    new_node = ("contracted", node)  # 元组标签避免与原肌肉重名
    mapping = {v: (new_node if v in merged else v) for v in G.nodes()}
    H = nx.Graph()
    H.add_nodes_from(mapping.values())
    for u, v in G.edges():
        a, b = mapping[u], mapping[v]
        if a != b:  # 合并集合内部边不再保留，也不产生自环
            H.add_edge(a, b)
    return H


def node_contraction_importance(G):
    """按 IMC(i)=1-cohesion(M)/cohesion(M·i) 评价节点重要度。

    M·i表示收缩节点i及其直接邻居后的网络。IMC越大，收缩后凝聚度
    相对原网络提升越明显，节点越重要。论文公式以连通网络为前提；
    原网络不连通时返回空IMC，不用自定义距离填补来伪造排名。
    """
    base = network_cohesion(G)
    rows = []
    for node in G.nodes():
        H = contract_node_and_neighbors(G, node)
        after = network_cohesion(H)
        valid = np.isfinite(base) and np.isfinite(after) and after > 0
        imc = 1.0 - base / after if valid else np.nan
        rows.append(dict(muscle=str(node), degree=G.degree(node),
                         original_nodes=G.number_of_nodes(), contracted_nodes=H.number_of_nodes(),
                         original_connected=nx.is_connected(G),
                         contracted_connected=nx.is_connected(H),
                         cohesion_before=base, cohesion_after=after, IMC=imc))
    rows.sort(key=lambda r: (np.isfinite(r["IMC"]),
                             r["IMC"] if np.isfinite(r["IMC"]) else -np.inf,
                             r["degree"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, base


def save_contraction_csv(rows, outdir, tag="MI_full"):
    """保存节点收缩关键肌群重要度排名。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"key_muscles_{tag}.csv")
    fields = ["rank", "muscle", "degree", "IMC", "cohesion_before", "cohesion_after",
              "original_nodes", "contracted_nodes", "original_connected",
              "contracted_connected"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(f"[关键肌群] 已保存 {os.path.abspath(path)}")
    return path


def print_threshold_diagnostics(G, threshold):
    """按文献原则检查连通性、孤立节点及平均度是否大于2*ln(n)。"""
    n = G.number_of_nodes()
    degree = np.mean([d for _, d in G.degree()]) if n else 0.0
    degree_ref = 2 * np.log(n) if n > 1 else 0.0
    isolated = list(nx.isolates(G))
    connected = n > 0 and nx.is_connected(G)
    print(f"[TH检查] TH={threshold:.2f}, 连通={connected}, 孤立节点={len(isolated)}, "
          f"平均度={degree:.3f}, 2ln(n)={degree_ref:.3f}, "
          f"平均度条件={'满足' if degree > degree_ref else '不满足'}")
    if isolated:
        print("[TH提示] 孤立节点: " + ", ".join(map(str, isolated)))
    if not connected:
        print("[TH提示] 当前网络不连通；节点收缩IMC按论文定义暂不可计算，"
              "可调整MI_THRESHOLD后重试。")


def print_metrics(tag, M):
    print(f"\n--- [{tag}] 网络指标 ---")
    print(f"  边数 = {M['n_edges']} (平均度 = {M['avg_degree']:.2f})")
    print(f"  聚类系数 C      = {M['clustering']:.4f}")
    print(f"  全局效率 E      = {M['efficiency']:.4f}")
    print(f"  平均路径长度 L  = {M['path_len']:.4f}")
    if not np.isnan(M['sigma']):
        print(f"  小世界: gamma={M['gamma']:.2f}, lambda={M['lambda_']:.2f}, "
              f"sigma={M['sigma']:.2f}  "
              f"({'有小世界属性' if M['sigma'] > 1 else '不明显'})")
    print("  Top5 关键肌(度+介数): " + ", ".join(M['hubs']))


# ---------------------- 导出 Gephi (可选) ---------------------------
def export_gephi(G, outdir, tag="full"):
    """导出 nodes/edges CSV + GEXF(Gephi 的 File->Open 可直接打开)"""
    os.makedirs(outdir, exist_ok=True)
    npath = os.path.join(outdir, f"nodes_{tag}.csv")
    epath = os.path.join(outdir, f"edges_{tag}.csv")
    with open(npath, "w", encoding="utf-8-sig") as f:
        f.write("Id,Label,Degree\n")
        for nd, d in G.degree():
            f.write(f"{nd},{nd},{d}\n")
    with open(epath, "w", encoding="utf-8-sig") as f:
        f.write("Source,Target,Weight\n")
        for u, v, w in G.edges(data="weight"):
            f.write(f"{u},{v},{w:.4f}\n")
    gpath = os.path.join(outdir, f"network_{tag}.gexf")
    nx.write_gexf(G, gpath)
    print(f"[导出] {epath} / {gpath}")


# ----------------------------- 可视化 -------------------------------
def plot_results(C, G, M, outdir, tag="full", save=True, show=False,
                 matrix_title="Inter-muscular coherence matrix"):
    """三联图: (a)关联矩阵热图 (b)网络拓扑图 (c)指标柱状图。

    show=True 时，图片保存后会弹出Matplotlib图片窗口；关闭窗口即可结束程序。
    """
    if save:
        os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(16, 5.5))
    nch = C.shape[0]
    labels = list(G.nodes())
    # (a) 关联矩阵热图
    ax1 = fig.add_subplot(1, 3, 1)
    C_off = C.copy(); np.fill_diagonal(C_off, np.nan)   # 掩掉对角线
    finite_off = C_off[np.isfinite(C_off)]
    vmax = max(0.2, float(np.max(finite_off)) if finite_off.size else 0.2)
    im = ax1.imshow(C_off, cmap="jet", vmin=0, vmax=vmax)
    ax1.set_xticks(range(nch)); ax1.set_yticks(range(nch))
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_title(matrix_title)
    plt.colorbar(im, ax=ax1, fraction=0.046)
    # (b) 网络拓扑图(环形布局, 线粗=关联强度, 点大=度)
    ax2 = fig.add_subplot(1, 3, 2)
    pos = nx.circular_layout(G)
    wmax = max((d["weight"] for *_, d in G.edges(data=True)), default=1)
    nx.draw_networkx_edges(G, pos, ax=ax2,
                           width=[2.5 * d["weight"] / wmax for *_, d in G.edges(data=True)],
                           edge_color="gray", alpha=0.7)
    dmax = max(1, max(M["degree"].values(), default=0))  # 无连边时避免除以0
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                           node_size=[400 * M["degree"][n] / dmax + 150 for n in G.nodes],
                           node_color="tomato", alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
    ax2.set_title(f"Muscle functional network ({tag})")
    ax2.axis("off")
    # (c) 指标柱状图
    ax3 = fig.add_subplot(1, 3, 3)
    labels3 = ["C", "E", "L"]
    vals   = [M["clustering"], M["efficiency"], M["path_len"]]
    ax3.bar(labels3, vals, color=["steelblue", "seagreen", "darkorange"])
    for i, v in enumerate(vals):
        ax3.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax3.set_title("Graph metrics")
    fig.tight_layout()
    if save:
        p = os.path.join(outdir, f"network_{tag}.png")
        fig.savefig(p, dpi=200)
        print(f"[图] 已保存 {os.path.abspath(p)}")
    if show:
        print("[提示] 结果图片已弹出，关闭当前图片窗口后继续")
        plt.show()
    plt.close(fig)


# ------------------------- 互信息网络完整流程 ------------------------
def mutual_information_analysis(emg_f, outdir=OUT_DIR, export=False, show=False):
    """互信息网络、TH=0.50二值化、图论指标和节点收缩关键肌群分析。"""
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[互信息] Shannon熵/联合熵计算中，等宽分箱={MI_BINS} ...")
    MI, MI_norm, H = mutual_information_matrix(emg_f, bins=MI_BINS)
    np.save(os.path.join(outdir, "MI_raw.npy"), MI)
    np.save(os.path.join(outdir, "MI_normalized.npy"), MI_norm)
    np.save(os.path.join(outdir, "MI_channel_entropy.npy"), H)

    G = build_graph(MI_norm, MI_THRESHOLD)
    M = graph_metrics(G)
    print_threshold_diagnostics(G, MI_THRESHOLD)
    print_metrics(f"MI_full(TH={MI_THRESHOLD:.2f})", M)

    ranking, cohesion = node_contraction_importance(G)
    if np.isfinite(cohesion):
        print(f"[节点收缩] 原网络凝聚度 = {cohesion:.6f}")
        valid = [r for r in ranking if np.isfinite(r["IMC"])]
        print("  Top5关键肌群(节点收缩IMC): " + ", ".join(
            f"{r['muscle']}({r['IMC']:.4f})" for r in valid[:5]))
    save_contraction_csv(ranking, outdir, tag="MI_full")
    if export:
        export_gephi(G, outdir, tag="MI_full")
    plot_results(MI_norm, G, M, outdir, tag="MI_full", show=show,
                 matrix_title="Min-max normalized mutual information matrix")
    return dict(MI=MI, MI_norm=MI_norm, entropy=H, G=G, M=M,
                contraction=ranking, cohesion=cohesion)


# ------------------------------ 主流程 ------------------------------
def run(emg, fs=None, outdir=OUT_DIR, export=False, show=False):
    """运行相干性与互信息两套平行分析，返回值仍保持为相干性的(C, G, M)。

    export=True 时导出两套网络的Gephi文件；show=True 时依次弹出结果图片。
    互信息结果通过独立文件保存，也可单独调用mutual_information_analysis获取。
    """
    os.makedirs(outdir, exist_ok=True)
    fs = FS if fs is None else fs
    n = emg.shape[1]
    print(f"[预处理] 陷波50Hz + 带通20-450Hz ...")
    emg_f = np.column_stack([preprocess(emg[:, k], fs) for k in range(n)])

    th = (surrogate_threshold(emg_f, fs=fs) if THRESH_METHOD == "surrogate"
          else THRESH_FIXED)
    bands = BANDS if BANDS else {"full": (FMIN, FMAX)}
    n_pairs = n * (n - 1) // 2
    for tag, (f1, f2) in bands.items():
        print(f"\n[{tag}] 频段 {f1}-{f2} Hz 相干矩阵计算中({n_pairs}对) ...")
        C = coherence_matrix(emg_f, f1, f2, fs)
        G = build_graph(C, th)
        M = graph_metrics(G)
        print_metrics(f"{tag}({f1}-{f2}Hz)", M)
        np.save(os.path.join(outdir, f"coherence_{tag}.npy"), C)
        if export:
            export_gephi(G, outdir, tag)
        plot_results(C, G, M, outdir, tag, show=show)

    # 新增方法与原相干性分析平行运行；文件名带MI，避免覆盖原结果。
    if MI_ANALYSIS:
        mutual_information_analysis(emg_f, outdir=outdir, export=export, show=show)
    return C, G, M


if __name__ == "__main__":
    # 先选择数据文件，再把结果保存到该文件所在目录下的 network_output。
    input_path = _pick_file()
    print(f"[选择] {input_path}")
    data, names, fs = load_emg_txt(input_path)
    if names:                            # 表头肌肉名 -> 替换默认方案
        MUSCLE_NAMES.clear()
        MUSCLE_NAMES.extend(names)
    globals()["FS"] = fs
    result_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), OUT_DIR)
    print(f"[输出目录] {result_dir}")
    run(data, fs=fs, outdir=result_dir, show=True)
