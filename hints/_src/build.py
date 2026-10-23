"""Build hints/*.html from the Markdown sources in hints/_src/.

    python hints/_src/build.py            # writes hints/index.html + one page per drill

Source format: CommonMark + tables, raw HTML allowed, plus fenced containers
(nesting is fine; fenced code blocks are skipped when looking for them):

    ::: hint 1 标题            L1/L2/L3 collapsible hint (spoiler level 1..3)
    ::: answer 标题            click-to-reveal answer
    ::: callout analogy 标题   boxed note; kinds: analogy | warn | info | ok
    ::: steps 标题             step-through widget; optional raw-HTML diagram before the
    --- step 标题 {hl=a,b}     first step, `hl` highlights [data-node] elements in it
    ::: raw                    emitted verbatim (SVG etc.)
    :::                        closes the innermost container

Inline macro: {{cells: t:1 h:2 w+on:3 bad:x plain}} renders a row of colored cells
(classes: t h w on off ok bad gap; `_` or no class = plain cell).

Every page source starts with `lede: one sentence` on its first line.
"""
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

SRC = Path(__file__).resolve().parent
OUT = SRC.parent

# (num, ID, file stem, title, minutes, difficulty, track, book page)
DRILLS = [
    (1, "M01", "m01_rmsnorm_mrope", "RMSNorm 与多模态 RoPE（interleaved / chunked）", 45, "中", "M", 19),
    (2, "M02", "m02_thinker_attention", "Thinker Attention：QK-norm + GQA + KV cache", 60, "中", "M", 23),
    (3, "M03", "m03_sparse_moe", "Sparse MoE：路由、按 expert 分组计算、shared expert", 60, "中", "M", 27),
    (4, "M04", "m04_mm_merge_deepstack", "多模态拼接：placeholder 计数、embedding 合并、DeepStack", 45, "中", "M", 31),
    (5, "M05", "m05_talker_mtp", "Talker MTP：一帧内预测 15 个残差 codebook", 60, "难", "M", 35),
    (6, "M06", "m06_streaming_causal_conv", "Code2Wav 的流式因果卷积与转置卷积", 60, "难", "M", 38),
    (7, "M07", "m07_stage_weight_loading", "按 stage 加载与打包权重", 45, "中", "M", 41),
    (8, "M08", "m08_tiny_thinker_e2e", "串起来：tiny thinker 的 KV-cache greedy 解码", 60, "中", "M", 44),
    (9, "P02", "p02_orchestrator", "Orchestrator v1：线性多 stage 路由", 60, "中", "A", 48),
    (10, "P03", "p03_async_chunk_orchestrator", "Orchestrator v2：async_chunk 预热 + 终态守门", 45, "中", "A", 51),
    (11, "P01", "p01_output_sink", "OutputSink：线程 → asyncio 的批量唤醒", 45, "中", "A", 54),
    (12, "P04", "p04_stage_pool_abort", "StagePool：粘性绑定、两阶段 abort、故障隔离", 60, "中", "A", 57),
    (13, "P05", "p05_rpc_router", "RpcRouter：共享通道上的 RPC 关联", 40, "易", "A", 61),
    (14, "P10", "p10_engine_core_loop", "EngineCore busy loop", 45, "中", "B", 65),
    (15, "P11", "p11_chunk_aware_scheduler", "WAITING_FOR_CHUNK：停靠与归还", 45, "中", "B", 68),
    (16, "P07", "p07_scheduler", "Scheduler：continuous batching + chunked prefill + 抢占", 75, "难", "B", 72),
    (17, "P06", "p06_block_manager", "BlockManager：paged KV + 前缀缓存", 75, "难", "B", 76),
    (18, "P08", "p08_sampler", "批不变的采样器（per-request seed）", 45, "中", "B", 80),
    (19, "P09", "p09_paged_attention", "Paged KV 写入 + decode attention（参考实现）", 45, "中", "B", 84),
    (20, "P12", "p12_shm_connector", "SharedMemoryConnector：跨进程 put/get 协议", 60, "难", "C", 88),
    (21, "P13", "p13_chunk_adapter", "ChunkAdapter：后台收发线程与顺序/清理语义", 75, "难", "C", 91),
    (22, "P15", "p15_talker_inputs", "Thinker→Talker 输入构造（含“差一行”）", 40, "易", "D", 96),
    (23, "P14", "p14_codec_windows", "Codec 窗口规划：首块小、后块大、左上下文", 40, "易", "D", 99),
    (24, "P18", "p18_mrope_positions", "MRoPE 三维位置", 40, "中", "D", 102),
    (25, "P16", "p16_init_planner", "启动规划：设备分组、并行初始化、回滚、准入", 45, "中", "E", 106),
    (26, "P17", "p17_graph_runner", "CUDA graph bucket runner（CPU 模拟）", 40, "易", "E", 109),
]

TRACKS = {
    "M": ("Track M · Qwen-Omni 模型实现", "先把模型本身吃透：每题对拍 HF transformers 4.57 的 Qwen3-Omni（tiny 随机权重，CPU 秒级）。"),
    "A": ("Track A · 前端与 Orchestrator", "API 进程里的并发：线程 ↔ 事件循环、单写者路由、终态、abort、RPC。"),
    "B": ("Track B · Stage 进程与调度", "EngineCore busy loop、停靠/归还、continuous batching、paged KV、采样、attention。"),
    "C": ("Track C · 数据面", "跨进程 payload：共享内存协议、后台收发线程、顺序与清理。"),
    "D": ("Track D · 模型胶水（纯函数）", "最容易“差一行”的地方：thinker→talker 输入、codec 窗口、MRoPE 位置。"),
    "E": ("Track E · 启动与 GPU", "多 stage 共卡启动、显存准入、CUDA graph bucket。"),
}

HINT_NOTES = {
    "1": "I/O 示例 + 关键问题 + 细节（轻度剧透）",
    "2": "思路拆分与常见坑（中度剧透）",
    "3": "详解：答案、逐步追踪、骨架提示（重度剧透）",
}

MD = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(["table", "strikethrough"])

OPEN_RE = re.compile(r"^:::\s+(\w+)\s*(.*)$")
CLOSE_RE = re.compile(r"^:::\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
STEP_RE = re.compile(r"^---\s+step\s*(.*?)\s*(\{hl=([^}]*)\})?\s*$")


def split_blocks(lines: list[str]) -> list:
    """-> list of str (markdown line) | (name, args, children)."""
    out: list = []
    i, in_fence = 0, False
    while i < len(lines):
        line = lines[i]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        m = None if in_fence else OPEN_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        depth, j, inner, f = 1, i + 1, [], False
        while j < len(lines):
            ln = lines[j]
            if FENCE_RE.match(ln):
                f = not f
            elif not f and OPEN_RE.match(ln):
                depth += 1
            elif not f and CLOSE_RE.match(ln):
                depth -= 1
                if depth == 0:
                    break
            inner.append(ln)
            j += 1
        if depth:
            raise SystemExit(f"unclosed ::: {m.group(1)} near: {line!r}")
        out.append((m.group(1), m.group(2).strip(), inner))
        i = j + 1
    return out


CELLS_RE = re.compile(r"\{\{cells:\s*(.*?)\}\}")


def _cells(m: re.Match) -> str:
    """{{cells: t:1 h:2 w+on:3 x _:·}} -> a row of colored cells; class `_` = plain."""
    out = []
    for tok in m.group(1).split():
        cls, sep, text = tok.partition(":")
        if not sep:
            cls, text = "_", tok
        classes = " ".join(c for c in cls.split("+") if c != "_")
        out.append(f'<span class="cell {classes}">{html.escape(text)}</span>')
    return f'<span class="cells">{"".join(out)}</span>'


def md(text: str) -> str:
    out = MD.render(CELLS_RE.sub(_cells, text))
    out = out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return out


def render(lines: list[str]) -> str:
    parts, buf = [], []

    def flush():
        if buf:
            parts.append(md("\n".join(buf)))
            buf.clear()

    for b in split_blocks(lines):
        if isinstance(b, str):
            buf.append(b)
            continue
        flush()
        name, args, inner = b
        if name == "raw":
            parts.append("\n".join(inner))
        elif name == "hint":
            lvl, _, title = args.partition(" ")
            parts.append(
                f'<details class="hint l{lvl}"><summary><span class="lvl">L{lvl}</span>{html.escape(title)}'
                f'<span class="spoiler-note">{HINT_NOTES.get(lvl, "")}</span></summary>'
                f'<div class="body">{render(inner)}</div></details>'
            )
        elif name == "answer":
            title = args or "显示答案"
            parts.append(f'<details class="answer"><summary>{html.escape(title)}</summary>'
                         f'<div class="body">{render(inner)}</div></details>')
        elif name == "callout":
            kind, _, title = args.partition(" ")
            t = f'<div class="callout-title">{html.escape(title)}</div>' if title else ""
            parts.append(f'<div class="callout {kind}">{t}{render(inner)}</div>')
        elif name == "steps":
            parts.append(render_steps(args, inner))
        else:
            raise SystemExit(f"unknown container ::: {name}")
    flush()
    return "\n".join(parts)


def render_steps(title: str, inner: list[str]) -> str:
    pre, steps, cur = [], [], None
    for ln in inner:
        m = STEP_RE.match(ln)
        if m:
            cur = (m.group(1), m.group(3) or "", [])
            steps.append(cur)
        elif cur is None:
            pre.append(ln)
        else:
            cur[2].append(ln)
    # the diagram is raw HTML; an optional `::: raw` wrapper around it is just for readability
    diagram = "\n".join(ln for ln in pre if not (OPEN_RE.match(ln) or CLOSE_RE.match(ln))).strip()
    dia = f'<div class="st-diagram diagram">{diagram}</div>' if diagram else ""
    body = "".join(
        f'<div class="step" data-hl="{html.escape(hl)}"><div class="step-title">{i + 1}. {html.escape(t)}</div>'
        f"{render(lines)}</div>"
        for i, (t, hl, lines) in enumerate(steps)
    )
    return (
        f'<div class="stepper"><div class="st-head"><span class="st-title">▶ {html.escape(title)}</span>'
        f'<span class="st-count"></span><span class="st-ctrl">'
        f'<button type="button" class="btn st-prev">← 上一步</button>'
        f'<button type="button" class="btn primary st-next">下一步 →</button></span></div>'
        f'<div class="st-dots"></div>{dia}<div class="st-body">{body}</div></div>'
    )


def add_heading_ids(body: str) -> tuple[str, list[tuple[str, str]]]:
    toc = []

    def sub(m):
        idx = len(toc) + 1
        text = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        toc.append((f"s{idx}", text))
        return f'<h2 id="s{idx}">{m.group(1)}</h2>'

    return re.sub(r"<h2>(.*?)</h2>", sub, body), toc


def read_src(name: str) -> tuple[str, list[str]]:
    lines = (SRC / f"{name}.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("lede:"), f"{name}.md must start with 'lede:'"
    return lines[0][5:].strip(), lines[1:]


def page(title: str, body: str, *, nav: str = "", desc: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<link rel="stylesheet" href="assets/hints.css">
<script src="assets/hints.js"></script>
</head>
<body>
<header class="topbar">
  <a class="home" href="index.html">nano-vllm-omni · 提示</a>
  <span class="spacer"></span>
  <span class="nav">{nav}</span>
  <button type="button" class="btn" id="theme-btn" title="切换明暗">◐</button>
</header>
<main class="page">
{body}
</main>
</body>
</html>
"""


def drill_page(k: int) -> str:
    num, did, stem, title, mins, diff, track, bp = DRILLS[k]
    lede, lines = read_src(f"{did.lower()}")
    body, toc = add_heading_ids(render(lines))
    prev_d = DRILLS[k - 1] if k > 0 else None
    next_d = DRILLS[k + 1] if k + 1 < len(DRILLS) else None

    def link(d, cls, label):
        if d is None:
            return ""
        return (f'<a class="{cls}" href="{d[1].lower()}_overview.html"><small>{label}</small>'
                f"#{d[0]} · {d[1]} {html.escape(d[3])}</a>")

    nav = ""
    if prev_d:
        nav += f'<a href="{prev_d[1].lower()}_overview.html">← <span class="lbl">{prev_d[1]}</span></a>'
    if next_d:
        nav += f'<a href="{next_d[1].lower()}_overview.html"><span class="lbl">{next_d[1]}</span> →</a>'
    extra = ""
    if did == "P02":
        extra = ('<span class="badge"><a href="p02_flow.html">时序图 p02_flow</a></span>'
                 '<span class="badge"><a href="p02_vs_vllm_omni.html">源码对照</a></span>')
    hero = f"""<section class="hero">
<div class="kicker">#{num} · {TRACKS[track][0]} · 书 p.{bp}</div>
<h1>{did} · {html.escape(title)}</h1>
<p class="lede">{html.escape(lede)}</p>
<div class="badges">
<span class="badge">⏱ <b>{mins} min</b></span>
<span class="badge diff-{diff}">难度 <b>{diff}</b></span>
<span class="badge">实现 <code>mine/{stem}.py</code></span>
<span class="badge">验收 <code>./check.sh {did.lower()}</code></span>
<span class="badge">提示 <b>A-{num}</b> · 参考解 <b>C-{num}</b> <code>solutions/{stem}.py</code></span>
{extra}
<label class="badge done-toggle"><input type="checkbox" id="done-box" data-id="{did}">标记完成</label>
</div>
</section>"""
    toc_html = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    toc_box = (f'<nav class="toc"><b>本页目录</b> · <button type="button" class="btn" id="open-l1">全部展开</button>'
               f"<ol>{toc_html}</ol></nav>")
    pager = f'<nav class="pager">{link(prev_d, "prev", "← 上一题")}{link(next_d, "next", "下一题 →")}</nav>'
    return page(f"{did} 提示 · {title}", hero + toc_box + body + pager, nav=nav, desc=lede)


def special_page(name: str, title: str, kicker: str) -> str:
    lede, lines = read_src(name)
    body, toc = add_heading_ids(render(lines))
    toc_html = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    hero = (f'<section class="hero"><div class="kicker">{kicker}</div><h1>{html.escape(title)}</h1>'
            f'<p class="lede">{html.escape(lede)}</p></section>'
            f'<nav class="toc"><b>本页目录</b><ol>{toc_html}</ol></nav>')
    nav = '<a href="p02_overview.html">← <span class="lbl">P02 提示</span></a>'
    return page(title, hero + body, nav=nav, desc=lede)


def index_page() -> str:
    lede, lines = read_src("index")
    intro, _ = add_heading_ids(render(lines))
    sections = []
    for t, (tname, tdesc) in TRACKS.items():
        ds = [d for d in DRILLS if d[6] == t]
        cards = "".join(
            f'<a class="card" href="{d[1].lower()}_overview.html" data-id="{d[1]}" data-min="{d[4]}">'
            f'<input type="checkbox" class="chk" title="标记完成">'
            f'<span class="num">#{d[0]}</span><span class="id">{d[1]}</span>'
            f'<div class="ttl">{html.escape(d[3])}</div>'
            f'<div class="meta">⏱ {d[4]} min · 难度 {d[5]} · 书 p.{d[7]}<br><code>{d[2]}</code></div></a>'
            for d in ds
        )
        mins = sum(d[4] for d in ds)
        sections.append(
            f'<div class="track-head"><h2 id="track-{t}">{tname}</h2>'
            f'<span class="muted">{len(ds)} 题 · {mins} 分钟</span></div>'
            f'<p class="muted">{tdesc}</p><div class="grid">{cards}</div>'
        )
    total = sum(d[4] for d in DRILLS)
    hero = f"""<section class="hero">
<div class="kicker">《nano-vllm-omni 自学手册》Part I · 交互提示</div>
<h1>26 道一小时核心实现 · 提示总览</h1>
<p class="lede">{html.escape(lede)}</p>
<div class="badges"><span class="badge">共 <b>26</b> 题</span><span class="badge">约 <b>{total // 60} 小时 {total % 60} 分</b></span>
<span class="badge">P02 额外：<a href="p02_flow.html">时序图</a> · <a href="p02_vs_vllm_omni.html">源码对照</a></span></div>
<div class="progress" title="完成进度"><div id="prog-bar"></div></div>
<p class="small muted"><span id="prog-text"></span> · 进度只存在本浏览器的 localStorage 里 ·
<button type="button" class="btn" id="prog-reset">清空</button></p>
</section>"""
    return page("nano-vllm-omni 刷题提示", hero + intro + "".join(sections),
                desc="26 道 nano-vllm-omni 核心实现题的三级提示、逐步图与易错点总览")


def main() -> None:
    for k, d in enumerate(DRILLS):
        if not (SRC / f"{d[1].lower()}.md").exists():
            print(f"skip {d[1]} (no source yet)")
            continue
        (OUT / f"{d[1].lower()}_overview.html").write_text(drill_page(k), encoding="utf-8")
    if (SRC / "p02_flow.md").exists():
        (OUT / "p02_flow.html").write_text(
            special_page("p02_flow", "P02 Orchestrator 时序图", "#9 · P02 · 交互时序图"), encoding="utf-8")
    if (SRC / "p02_vs_vllm_omni.md").exists():
        (OUT / "p02_vs_vllm_omni.html").write_text(
            special_page("p02_vs_vllm_omni", "P02 ↔ vllm-omni 源码对照", "附录 D · 源码对照"), encoding="utf-8")
    (OUT / "index.html").write_text(index_page(), encoding="utf-8")
    print("built", sorted(p.name for p in OUT.glob("*.html")))


if __name__ == "__main__":
    main()
