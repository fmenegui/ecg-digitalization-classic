"""
Pipeline completo: imagem -> 12 derivações (mV, fs Hz).

    from ecgdig import digitize
    res = digitize("ecg.jpg")            # layout automático
    res["signals"]  # (12, N) float32, NaN onde a derivação não aparece
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from .geometry import rectify
from .extract import (ink_map, binarize, find_rows, row_extent, detect_pulse,
                      column_runs, viterbi_track, selected_envelopes, envelope_edges,
                      centerline_from_envelopes, calibrate_pen, mark_occlusions, horizontal_halfwidth,
                      trace_likeness, refine_steep)

from .layouts import LEADS, LIBRARY as LAYOUTS, is_limb
from .layout_engine import Ctx as _LayoutCtx


@dataclass
class Config:
    R: float = 10.0            # px/mm do quadro retificado
    fs: int = 500
    duration: float | None = None  # s exibidos por linha; None = mede (pontas do traço)
    speed: float | None = None     # mm/s; None = lê "NN mm/s" na página, senão QRS/RR
    gain: float | None = None      # mm/mV para todas as derivações; None = mede nos pulsos
    layout: str | list = "auto"    # "auto", nome da biblioteca ou lista de linhas de nomes
    einthoven: bool = True     # fusão por restrições de Einthoven/Goldberger
    fill_limb: bool = False    # reconstrói derivações dos membros onde possível
    ppmm: float | None = None  # px/mm da imagem original, se conhecido (DPI/25.4)
    pen: str = "measure"       # largura horizontal da pena: measure | calibrate | isotropic
    # registro por batimentos só corrige cisalhamento acima disto (mm a 25 mm/s): o instante
    # do QRS difere entre derivações por razões fisiológicas (p90 ~12 ms), o que imita
    # um cisalhamento de até ~0.5 mm; abaixo disso o registro mais atrapalha que ajuda
    reg_min_shift_mm: float = 0.65
    # papel não plano: "poly" (polinômio cúbico global), "cell" (+ campo local em células de
    # 5 mm, acompanha vincos e dobras) ou "none"
    nonrigid: str = "cell"
    major_snap: bool = True   # ajuste pelas linhas grossas (5 mm)
    fine_snap: bool = True    # ajuste pelas linhas finas (1 mm)
    steep_subpx: bool = False  # instante sub-pixel nos trechos íngremes (teste)
    synth: bool = True         # ajuste por síntese: pena + borrão gaussianos (traço fino)
    bridge: bool = True        # Viterbi: salto vertical ligado por tinta fraca custa pouco
    synth_max_shift: float = 1.5
    synth_order: str = "sort"
    fit_kw: dict = None
    rigid_edges: bool = True     # Viterbi: traço íngreme que translada inteiro não paga a continuidade das bordas
    pulse_retry: bool = True     # pulso procurado também a partir da margem comum das linhas
    row_beats: bool = True       # linhas nas pontas sem QRS simultâneos aos das outras saem
    ink_model: str = ""         # caminho do ONNX do segmentador de traço (modo híbrido)
    ink_mix: float = 1.0        # 1 = só a rede; <1 mistura com o mapa clássico
    waveform_model: str = ""    # diretório EMA/config do 5º lugar; opcional, só leitura final
    waveform_source: str = ""   # checkout upstream kaggle_ecg_digitization
    waveform_device: str = "cpu"
    band_sp: float = 1.6        # faixa de busca do traço: +-band_sp espaçamentos de linha
    fit: str = "none"          # "lsf": renderização inversa com o perfil de pena medido


def _pick_rows(peaks, cov, n_expected):
    if len(peaks) <= n_expected:
        return list(peaks)
    # mantém o subconjunto de n linhas com maior cobertura, depois ordena
    best = sorted(peaks, key=lambda p: -cov[p])[:n_expected]
    return sorted(best)


def _row_quick_trace(B, D, rows, R, x0, span_px, full=False):
    """Traço grosseiro (centróide do segmento escolhido pelo Viterbi) de cada linha inteira.
    full=True devolve dicts com cen/top/bot absolutos e x0 da janela."""
    H, W = B.shape
    out = []
    xa, xb = max(0, int(x0)), min(W, int(x0 + span_px) + 1)
    for r, y in enumerate(rows):
        up = (y - rows[r - 1]) if r > 0 else 40 * R
        dn = (rows[r + 1] - y) if r + 1 < len(rows) else 40 * R
        ya, yb = max(0, int(y - 0.9 * up)), min(H, int(y + 0.9 * dn))
        runs = column_runs(B, D, ya, yb, xa, xb)
        others = [float(v) for i2, v in enumerate(rows) if i2 != r]
        sel = viterbi_track(runs, float(y), R, others=others)
        if full:
            cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)])
            top = np.array([runs[i][k][0] if k is not None else np.nan for i, k in enumerate(sel)], float)
            bot = np.array([runs[i][k][1] if k is not None else np.nan for i, k in enumerate(sel)], float)
            out.append(dict(x0=xa, cen=cen, top=top, bot=bot))
        else:
            out.append(np.array([y - runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)]))
    return out


def _corr(a, b, win=151):
    """Correlação após remover a linha de base (mediana móvel ~0.6 s) e preencher falhas."""
    from scipy.ndimage import median_filter
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 50:
        return 0.0
    x = np.arange(len(a))
    out = []
    for v in (a, b):
        v = np.interp(x, x[~np.isnan(v)], v[~np.isnan(v)])
        out.append((v - median_filter(v, size=win, mode="nearest"))[m])
    a, b = out
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])




def _runs_bool(m):
    """Trechos True de um vetor booleano: (inícios, fins exclusivos)."""
    d = np.diff(np.r_[0, np.asarray(m, np.int8), 0])
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def _tick_span(xs, top, bot, xb, R):
    """A marca separadora na fronteira xb: colunas "altas" (topo-base > 1,2 mm) a ±0,4 mm de
    xb, estendidas enquanto topo E base ficam onde estavam (±0,3 mm). A marca é um retângulo;
    um QRS colado tem topo/base andando e não entra. Devolve (x_ini, x_fim) ou None."""
    if not len(xs):
        return None
    hh = bot - top
    tall = hh > 1.2 * R
    c = int(np.argmin(np.abs(xs - xb)))
    win = int(0.4 * R)
    cand = [q for q in range(max(0, c - win), min(len(xs), c + win + 1)) if tall[q]]
    if not cand:
        return None
    q0 = min(cand, key=lambda q: abs(xs[q] - xb))
    t0, b0 = top[q0], bot[q0]
    tol = 0.3 * R

    def same(q):
        return tall[q] and abs(top[q] - t0) < tol and abs(bot[q] - b0) < tol

    a_ = b_ = q0
    while a_ - 1 >= 0 and same(a_ - 1):
        a_ -= 1
    while b_ + 1 < len(xs) and same(b_ + 1):
        b_ += 1
    if xs[b_] - xs[a_] > 1.2 * R:
        return None
    return float(xs[a_]), float(xs[b_])


def _cut_leading_pulse(v, fs):
    """Pulso de calibração que o detector não achou e entrou no traço: no começo da linha,
    platô plano de ~1 mV (0,5-2,5 mV) por >= 40 ms seguido de queda brusca. Tudo até a
    queda (+ 10 ms) vira desconhecido."""
    n = min(len(v), int(0.4 * fs))
    if n < int(0.06 * fs):
        return v
    w = v[:n]
    if not np.isfinite(w).any():
        return v
    k = int(0.04 * fs)
    best = None
    for i in range(0, n - k):
        seg = w[i:i + k]
        if not np.isfinite(seg).all():
            continue
        m = float(np.median(seg))
        if 0.5 <= abs(m) <= 2.5 and np.std(seg) < 0.04:
            # fim do platô: primeira amostra que sai dele por mais de metade da altura
            j = i + k
            while j < n and np.isfinite(w[j]) and abs(w[j] - m) < 0.5 * abs(m):
                j += 1
            drop = j < n and np.isfinite(w[min(n - 1, j + int(0.02 * fs))]) and \
                abs(w[min(n - 1, j + int(0.02 * fs))]) < 0.35 * abs(m)
            if drop:
                best = j
                break
    if best is None:
        return v
    out = v.copy()
    out[:min(len(v), best + int(0.01 * fs))] = np.nan
    return out

def _box_cols(xa, n, boxes, pad=2):
    """Para cada caixa de rótulo que cruza [xa, xa+n): (i0, i1, y0, y1) em índices locais."""
    out = []
    for (bx0, by0, bx1, by1) in boxes:
        i0, i1 = max(0, bx0 - pad - xa), min(n, bx1 + pad + 1 - xa)
        if i1 > i0:
            out.append((i0, i1, by0 - pad, by1 + pad))
    return out


def _drop_label_runs(runs, xa, boxes):
    """Segmentos de tinta inteiramente dentro de um rótulo são letras, não traçado."""
    for i0, i1, y0, y1 in _box_cols(xa, len(runs), boxes):
        for i in range(i0, i1):
            runs[i] = [c for c in runs[i] if not (c[0] >= y0 and c[1] <= y1)]


def _mask_label_edges(job, boxes):
    """Borda do envelope que cai dentro de um rótulo é desconhecida (letra colada ao
    traço); a borda do lado de fora continua valendo (ex.: topo de um R acima do rótulo)."""
    top, bot = job["top"], job["bot"]
    for i0, i1, y0, y1 in _box_cols(job["xa"], len(top), boxes):
        sl = slice(i0, i1)
        with np.errstate(invalid="ignore"):
            t, b = top[sl], bot[sl]
            t[(t >= y0) & (t <= y1)] = np.nan
            b[(b >= y0) & (b <= y1)] = np.nan

def _beat_xs(B, y, half, R, ext=None):
    """Posições x dos complexos (QRS) de uma linha: colunas em que a tinta da faixa se estende
    na vertical bem além do traço típico (subida/descida rápida)."""
    H, W = B.shape
    a, b = max(0, int(y - half)), min(H, int(y + half))
    band = B[a:b]
    if ext:
        band = band[:, max(0, int(ext[0])):min(W, int(ext[1]))]
        x_off = max(0, int(ext[0]))
    else:
        x_off = 0
    has = band.any(0)
    if has.sum() < 20 * R:
        return np.array([])
    idx = np.arange(band.shape[0])[:, None]
    top = np.where(band, idx, 10 ** 6).min(0)
    bot = np.where(band, idx, -1).max(0)
    span = np.where(has, bot - top, 0).astype(float)
    base = float(np.median(span[has]))
    thr = max(2.0 * R, 3.0 * base)
    sp = span > thr
    runs = np.diff(np.r_[0, sp.astype(np.int8), 0])
    st, en = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
    xs, last = [], -10 ** 9
    for s_, e_ in zip(st, en):
        if e_ - s_ > 6 * R:      # largo demais para um QRS (bloco de texto, borda)
            continue
        c = 0.5 * (s_ + e_)
        if c - last > 5 * R:
            xs.append(c + x_off)
            last = c
    return np.array(xs)


def _beat_coherence_drop(B, peaks, ext, half, R):
    """Linhas nas pontas do bloco cujos complexos não coincidem no tempo com os das outras
    linhas: no ECG multicanal as linhas são simultâneas, então os QRS caem nas mesmas colunas.
    Cabeçalho, barra de menu de tela, moldura e régua de texto não seguem esse ritmo.
    Concordância entre duas linhas = F1 dos complexos casados (+-2,5 mm): uma linha de ritmo
    com ruído (moiré) tem complexos a mais, mas contém os da outra (revocação alta)."""
    n = len(peaks)
    keep = np.ones(n, bool)
    if n < 4:
        return keep
    bx = [_beat_xs(B, y, half, R, e) for y, e in zip(peaks, ext)]

    def f1(i, j):
        if len(bx[i]) < 2 or len(bx[j]) < 2:
            return 0.0
        d = np.abs(bx[i][:, None] - bx[j][None, :])
        p_ = float(np.mean(d.min(1) < 2.5 * R)); r_ = float(np.mean(d.min(0) < 2.5 * R))
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)

    F = np.array([[f1(i, j) if i != j else np.nan for j in range(n)] for i in range(n)])
    cand = [i for i in range(n) if len(bx[i]) >= 3]
    if len(cand) < 3:
        return keep
    # consenso: linhas que concordam (F1 >= 0,5) com a maioria das outras candidatas
    cons = [i for i in cand if np.nanmedian([F[i, j] for j in cand if j != i]) >= 0.5]
    if len(cons) < 3:
        return keep
    sp_c = float(np.median(np.diff(np.sort(peaks[cons]))))
    for order in (range(n), range(n - 1, -1, -1)):
        for i in order:
            if i in cons:
                break
            sc = np.nanmedian([F[i, j] for j in cons])
            if _ROW_BEAT_DEBUG:
                print("beat-row", int(peaks[i]), round(float(sc), 2), len(bx[i]))
            # só sai se, além de fora do ritmo, estiver fora do espaçamento do bloco (uma
            # derivação legítima de QRS pequeno fica no passo normal das linhas)
            kp = [j for j in range(n) if keep[j] and j != i]
            gap = float(np.min(np.abs(peaks[kp] - peaks[i]))) if kp else np.inf
            odd = gap < 0.75 * sp_c or gap > 1.5 * sp_c
            # nenhum complexo numa ponta, com todas as linhas do consenso cheias de QRS:
            # faixa de grade/mancha/moldura (uma linha de ECG real quase sempre tem algum)
            odd = odd or (len(bx[i]) == 0 and min(len(bx[j]) for j in cons) >= 5)
            if sc < _ROW_BEAT_THR[0] and odd and keep.sum() > 3:
                keep[i] = False
            else:
                break
    return keep


def _excursions(B, y, half, R, dev_mm=3.0):
    """Quantos trechos da linha têm tinta a >= dev_mm do eixo (dentro da faixa da linha)."""
    H, W = B.shape
    a, b = max(0, int(y - half)), min(H, int(y + half))
    d = int(dev_mm * R)
    band = B[a:b]
    yy = np.arange(a, b)[:, None]
    far = (band & (np.abs(yy - y) >= d)).any(0)
    near = band[max(0, y - a - int(1.5 * R)):y - a + int(1.5 * R)].any(0)
    far &= cv2.dilate(near.astype(np.uint8).reshape(1, -1), np.ones((1, int(2 * R) | 1), np.uint8)).ravel() > 0
    runs = np.diff(np.r_[0, far.astype(np.int8), 0])
    st, en = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
    # trechos separados por menos de 2 mm contam como um
    n, last = 0, -10 ** 9
    for s_, e_ in zip(st, en):
        if s_ - last > 2 * R:
            n += 1
        last = e_
    return n


def _rows_pulses(B, pap, R):
    """Linhas de traçado candidatas (sem texto/bordas), extensões e pulsos."""
    H, W = B.shape
    peaks, cov = find_rows(B, R)
    if len(peaks) == 0:
        return peaks, cov, [], [], 40 * R
    tl, raw = trace_likeness(B, peaks, R, return_raw=True)
    if len(tl) and tl.max() > 0:
        # traçado: longo depois de juntar quebras (tl) OU com pedaços longos/altos sem juntar
        # (raw; traço fraco de scan pode ter tl baixo mas raw alto)
        ok = tl >= 0.4 * tl.max()
        weak = ~ok & (raw >= 0.3 * max(raw.max(), 1e-9))
        # linhas só "fracas" entram se continuarem o espaçamento regular das fortes
        # (cabeçalho com tabela de texto também tem pedaços altos, mas fora do passo)
        if ok.sum() >= 2 and weak.any():
            sp0 = float(np.median(np.diff(peaks[ok])))
            changed = True
            while changed:
                changed = False
                acc = peaks[ok]
                for i in np.nonzero(weak & ~ok)[0]:
                    dmin = np.min(np.abs(acc - peaks[i]))
                    if 0.8 * sp0 <= dmin <= 1.25 * sp0:
                        ok[i] = True; changed = True
        # rodapé/cabeçalho de texto: só é "longo" depois de dilatado (raw << tl), fica na
        # ponta do bloco e mais perto do vizinho que o espaçamento normal das linhas.
        # (traço quebrado de scan também tem raw baixo, mas fica no meio ou bem espaçado)
        idx = np.nonzero(ok)[0]
        if len(idx) >= 3:
            sp = float(np.median(np.diff(peaks[idx])))
            for e, nb in ((idx[0], idx[1]), (idx[-1], idx[-2])):
                if raw[e] < 0.55 * tl[e] and abs(peaks[e] - peaks[nb]) < 0.85 * sp:
                    ok[e] = False
        peaks = peaks[ok]
    ys = np.nonzero(pap.any(1))[0]
    if len(ys):  # bordas da folha / moldura não são linhas de traçado
        peaks = peaks[(peaks > ys[0] + 5 * R) & (peaks < ys[-1] - 5 * R)]
    spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 40 * R
    half = 0.45 * spacing
    ext = [row_extent(B, y, half, R) for y in peaks]
    pulses = []
    starts = [e[0] for e in ext if e]
    xs_med = float(np.median(starts)) if starts else None
    for i, (y, e) in enumerate(zip(peaks, ext)):
        up_sp = (y - peaks[i - 1]) if i > 0 else spacing
        p_ = detect_pulse(B, y, half, e[0], R, up=max(half, min(0.95 * up_sp, 25 * R))) if e else None
        # traço fraco/partido no começo da linha (foto, tela): a extensão começa depois do
        # pulso. As linhas começam na mesma margem: tenta de novo a partir da margem comum
        if p_ is None and e and xs_med is not None and _PULSE_RETRY[0] and 10 * R < e[0] - xs_med < 60 * R:
            p_ = detect_pulse(B, y, half, xs_med, R, up=max(half, min(0.95 * up_sp, 25 * R)))
        if p_ is None and e and _PULSE_RETRY[0]:
            # bordas verticais do pulso claras e partidas (foto de tela, traço fino): fecha
            # quebras maiores na vertical
            p_ = detect_pulse(B, y, half, min(e[0], xs_med) if xs_med is not None else e[0], R,
                              up=max(half, min(0.95 * up_sp, 25 * R)), close_mm=2.5)
            # na segunda tentativa só o retângulo simples: o fechamento largo pode emendar o
            # pulso ao começo do traço e fabricar um "degrau" (ganho dividido falso)
            if p_ is not None and (p_.get("step") or p_.get("half")):
                p_ = None
            if p_ is not None:
                p_ = dict(p_, retry=True)
        pulses.append(p_)
    # pulso achado só na segunda tentativa vale se concordar com os achados na primeira
    # (altura e coluna); sem nenhum da primeira, vale sozinho
    first = [p for p in pulses if p is not None and not p.get("retry")]
    if first:
        h0 = float(np.median([p["foot"] - p["top"] for p in first]))
        x0_ = float(np.median([p["x_start"] for p in first]))
        pulses = [None if (p is not None and p.get("retry") and
                           (abs((p["foot"] - p["top"]) - h0) > 0.2 * h0 or abs(p["x_start"] - x0_) > 3 * R))
                  else p for p in pulses]
    else:
        rt = [p for p in pulses if p is not None]
        # só da segunda tentativa: precisa de pelo menos dois, de mesma altura e coluna
        ok_ = False
        if len(rt) >= 2:
            hs = np.array([p["foot"] - p["top"] for p in rt]); xs_ = np.array([p["x_start"] for p in rt])
            ok_ = np.ptp(hs) <= 0.2 * np.median(hs) and np.ptp(xs_) <= 3 * R
        if not ok_:
            pulses = [None] * len(pulses)
    # linha nas pontas do bloco sem pulso, quando outras têm: cabeçalho isolado, régua de
    # texto colada ao bloco, borda da área impressa/sombra da folha numa foto (quase plana)
    has = [p is not None for p in pulses]
    if sum(has) >= 2:
        # "pulso" longe da coluna dos outros (QR code, caixa do cabeçalho) não conta
        xs_p = np.array([p["x_start"] for p in pulses if p is not None], float)
        med_x = float(np.median(xs_p))
        has = [p is not None and abs(p["x_start"] - med_x) < 3 * R for p in pulses]
    if sum(has) >= 2 and len(peaks) > sum(has) and len(peaks) >= 3:
        exc = [_excursions(B, int(y), half, R) for y in peaks]
        ref = float(np.median([x for x, h in zip(exc, has) if h]))
        len_p = float(np.median([e[1] - e[0] for e, h in zip(ext, has) if h and e]))
        sp_all = float(np.median(np.diff(peaks)))
        keep = np.ones(len(peaks), bool)
        for order in (range(len(peaks)), range(len(peaks) - 1, -1, -1)):
            for i in order:
                if has[i]:
                    break
                others = peaks[keep & (np.arange(len(peaks)) != i)]
                gap = float(np.min(np.abs(others - peaks[i]))) if len(others) else np.inf
                short = not ext[i] or (ext[i][1] - ext[i][0]) < 0.6 * len_p
                flat = exc[i] < 0.4 * ref
                if gap > 1.6 * sp_all or gap < 0.7 * sp_all or (sum(has) >= 3 and (flat or short)):
                    keep[i] = False
                else:
                    break
        if not keep.all() and keep.sum() >= 2:
            peaks = peaks[keep]
            ext = [e for e, k in zip(ext, keep) if k]
            pulses = [p for p, k in zip(pulses, keep) if k]
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else spacing
    if _ROW_BEAT_RULE[0] and len(peaks) >= 4:
        keep = _beat_coherence_drop(B, peaks, ext, half, R)
        if not keep.all():
            peaks = peaks[keep]
            ext = [e for e, k in zip(ext, keep) if k]
            pulses = [p for p, k in zip(pulses, keep) if k]
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else spacing
    return peaks, cov, ext, pulses, spacing


_ROW_BEAT_RULE = [True]
_PULSE_RETRY = [True]
_ROW_BEAT_THR = [0.3]
_ROW_BEAT_DEBUG = False


def _orient(arrs, R):
    """Escolhe a rotação (0/90/180/270) em que os traçados ficam horizontais e os
    pulsos de calibração à esquerda."""
    rect, pap, D, B = arrs
    # traçados = estruturas longas (> 50 mm) numa direção; QRS altos não passam de ~40 mm.
    # Fecha pequenas quebras (traço fino/fraco) antes de medir.
    k = int(1.5 * R) | 1
    Bc = cv2.dilate(B.astype(np.uint8), np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(Bc, connectivity=8)
    w, h = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    sh = w[(w >= 50 * R) & (w > 1.5 * h)].sum()
    sv = h[(h >= 50 * R) & (h > 1.5 * w)].sum()
    cands = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE] if sv > sh else [None, cv2.ROTATE_180]

    def rot(a, c):
        if c is None:
            return a
        if a.dtype == bool:
            return cv2.rotate(a.astype(np.uint8), c) > 0
        return cv2.rotate(np.ascontiguousarray(a), c)

    from .ocr import find_words, read_label

    def upright_labels(Bc):
        # rótulos assimétricos legíveis (aVR, V1...): I/II/III são iguais de cabeça para baixo
        n_ = 0
        tm = text_like_mask(Bc, R)
        for (x0, y0, x1, y1) in find_words(tm, R):
            d = read_label(tm[y0:y1, x0:x1])
            if d:
                k_ = max(d, key=d.get)
                if d[k_] >= 0.6 and k_ not in ("I", "II", "III"):
                    n_ += 1
        return n_

    best, best_score = None, -1
    for c in cands:
        Bc, pc = rot(B, c), rot(pap, c)
        pk, _, _, pulses, _ = _rows_pulses(Bc, pc, R)
        # pulsos à esquerda + rótulos legíveis (página impressa de cabeça para baixo tem
        # pulsos dos dois lados em alguns aparelhos; o texto desempata)
        score = sum(p is not None for p in pulses) + upright_labels(Bc)
        if score > best_score:
            best, best_score = c, score
    return [rot(a, best) for a in arrs], best


def _edge_cands(D, B, rows, ext, pulses, half, lev, R, rad=0.0):
    """Bordas sub-pixel esquerda (início do traço) e direita (fim): listas sem NaN."""
    xl, xe = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad)
    return [v for v in xl if not np.isnan(v)], [v for v in xe if not np.isnan(v)]


def _origin_cands(D, B, rows, ext, pulses, half, lev, R, rad):
    """Candidatos à origem do tempo: inícios de traço (fortes), finais de linha e, nas linhas
    sem início destacado, o fim da descida do pulso (fraca: escolhe o grupo, não dá o valor).
    Sem as fracas, só os finais de linha ou uma borda espúria decidiriam, e a grade escura,
    o ruído ou a borda da folha numa foto podem estar a dezenas de mm do traço."""
    xl, xe, wk = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad, return_weak=True)
    xl = [float(v) for v in xl if not np.isnan(v)]
    # o traço não começa antes do pulso de calibração: "início" à esquerda dos pulsos das
    # outras linhas é a borda da folha/grade numa foto
    xs_p = [p["x_start"] for p in pulses if p is not None]
    if xs_p:
        xl = [v for v in xl if v > float(np.median(xs_p))]
    xe = [float(v) for v in xe if not np.isnan(v)]
    return xl, xe, [float(v) for v in wk if np.isfinite(v)]


def text_like_mask(B, R):
    """Componentes pequenas e isoladas (rótulos, números, marcas): não pertencem ao traço,
    que é uma componente longa. Usado para que bordas não sejam medidas em texto."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = (st[:, cv2.CC_STAT_WIDTH] < 25 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 9 * R)
    small[0] = False
    return small[lab]


def _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad=0.0, return_weak=False):
    """Bordas sub-pixel esquerda (início do traço) e direita (fim), uma por linha (NaN se falta)."""
    tm = getattr(_edge_rows, "text_mask", None)
    if tm is not None and tm.shape == D.shape:
        D = np.where(tm, 0.0, D).astype(np.float32)
    if getattr(_edge_rows, "speckle", False):
        # grade escura (scan P&B): o pontilhado passa do nível no perfil máximo e "estende" o
        # traço até a borda; vale só a tinta que sobreviveu à binarização (e o entorno)
        Bd = cv2.dilate(B.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        D = np.where(Bd, D, 0.0).astype(np.float32)
    H, W = D.shape
    lev_rad_hint = [rad]
    weak = []
    xl, xe = [], []
    for y, e, p in zip(rows, ext, pulses):
        xl.append(np.nan); xe.append(np.nan); weak.append(np.nan)
        if not e:
            continue
        a_, b_ = max(0, int(y - half)), min(H, int(y + half))
        # direita
        x1 = max(0, e[1] - 12)
        blk = D[a_:b_, x1:min(W, e[1] + 4)]
        if blk.size:
            prof = blk.max(0)
            idx = np.nonzero(prof >= lev)[0]
            if len(idx) and idx[-1] + 1 < len(prof):
                k_ = idx[-1]
                xe[-1] = x1 + k_ + (prof[k_] - lev) / max(prof[k_] - prof[k_ + 1], 1e-6)
        # esquerda: depois da lacuna após o pulso; sem pulso, a primeira tinta da linha
        if p is not None:
            x_s = p["x_end"]
            prof = D[a_:b_, x_s:min(W, x_s + int(6 * R))].max(0)
            below = np.nonzero((prof < lev) & (np.r_[prof[1:], 0] < lev))[0]
            after = np.nonzero(prof[below[0]:] >= lev)[0] if len(below) and below[0] <= 4 * R else []
            if not len(after):
                # traço colado ao pulso (sem lacuna): t = 0 provavelmente na descida do
                # pulso — âncora FRACA (em outros formatos o pulso tem uma cauda plana antes
                # do traço e a lacuna pode sumir com borrão)
                weak[-1] = p.get("x_fall", p["x_end"]) - lev_rad_hint[0]
                continue
            k_ = below[0] + after[0]
        else:
            x_s = max(0, e[0] - 6)
            prof = D[a_:b_, x_s:min(W, e[0] + 12)].max(0)
            idx = np.nonzero(prof >= lev)[0]
            if not len(idx) or idx[0] == 0:
                continue
            k_ = idx[0]
        xl[-1] = x_s + k_ - 1 + (lev - prof[k_ - 1]) / max(prof[k_] - prof[k_ - 1], 1e-6)
    if return_weak:
        return np.array(xl), np.array(xe), np.array(weak)
    return np.array(xl), np.array(xe)


def _beats(trace, x_off, R):
    """Posições (x, px) dos complexos QRS: picos da derivada absoluta suavizada (~80 ms)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    t = np.asarray(trace, float)
    ok = ~np.isnan(t)
    if ok.sum() < 50:
        return np.array([])
    x = np.arange(len(t))
    t = np.interp(x, x[ok], t[ok])
    a = uniform_filter1d(np.abs(np.gradient(t)), max(3, int(2 * R)))
    pk, _ = find_peaks(a, height=0.35 * a.max(), distance=int(12 * R))  # >= 480 ms
    # centróide da energia do QRS (+-60 ms): menos dependente da derivação que o pico
    g = np.abs(np.gradient(t))
    h = int(1.5 * R)
    out = []
    for k in pk:
        lo, hi = max(0, k - h), min(len(g), k + h + 1)
        w = g[lo:hi] ** 2
        out.append(float((w * np.arange(lo, hi)).sum() / max(w.sum(), 1e-9)))
    return np.array(out) + x_off


def _register_rows(beats, R, width_px, tol_mm=2.5, min_shift_px=4.0, rows=None):
    """Registro afim por linha contra a referência comum (mediana entre linhas) dos QRS.
    Todas as derivações são simultâneas: o mesmo batimento tem o mesmo instante em
    todas as linhas. Retorna (a_r, b_r) com x_linha = a_r + b_r * x_ref.
    Só corrige quando o deslocamento previsto passa de min_shift_px (senão identidade:
    o viés de fase do QRS entre derivações seria maior que a correção)."""
    n = len(beats)
    a = np.zeros(n); b = np.ones(n)
    nonempty = [bb for bb in beats if len(bb)]
    if n < 2 or not nonempty:
        return a, b
    order = np.sort(np.concatenate(nonempty))
    if len(order) < 4:
        return a, b
    groups, cur = [], [order[0]]
    for v in order[1:]:
        if v - cur[-1] <= tol_mm * R:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    refs = np.array([np.median(g) for g in groups if len(g) >= max(2, n // 2)])
    if len(refs) < 2:
        return a, b
    lo, hi = refs.min(), refs.max()
    valid = np.zeros(n, bool)
    for r, bb in enumerate(beats):
        if len(bb) < 2:
            continue
        d = np.abs(bb[:, None] - refs[None, :])
        j = np.argmin(d, 1)
        m = d[np.arange(len(bb)), j] <= tol_mm * R
        if m.sum() < 2:
            continue
        xo, xr = bb[m], refs[j[m]]
        ar, br = float(np.median(xo - xr)), 1.0
        if m.sum() >= 4 and np.ptp(xr) > 0.4 * width_px:
            A = np.stack([np.ones_like(xr), xr], 1)
            c = np.linalg.lstsq(A, xo, rcond=None)[0]
            res = np.abs(xo - A @ c)
            keep = res <= max(2.5 * np.median(res), 2.0)
            if keep.sum() >= 3:
                c = np.linalg.lstsq(A[keep], xo[keep], rcond=None)[0]
            if 0.97 <= c[1] <= 1.03:
                ar, br = float(c[0]), float(c[1])
        a[r], b[r] = ar, br
        valid[r] = True
    # distorção de página varia suavemente: deslocamento e escala lineares na altura da
    # linha (média o ruído do instante do QRS entre derivações, ~2-3 px)
    if rows is None or valid.sum() < 4:
        # com <= 3 linhas (ex.: 3x4 sem tira de ritmo) cada linha tem derivações diferentes e
        # o ruído do instante do QRS entre derivações não se distingue de cisalhamento
        return np.zeros(n), np.ones(n)
    if True:
        y = np.asarray(rows, float)
        A = np.stack([np.ones(valid.sum()), y[valid]], 1)
        ca = np.linalg.lstsq(A, a[valid], rcond=None)[0]
        cb = np.linalg.lstsq(A, b[valid], rcond=None)[0]
        a = ca[0] + ca[1] * y; b = cb[0] + cb[1] * y
    shift = np.maximum(np.abs(a + (b - 1) * lo), np.abs(a + (b - 1) * hi))
    if shift.max() < min_shift_px:
        return np.zeros(n), np.ones(n)
    return a, b


def _row_time_maps(rows, xl, rad, x0g, R, reg, tol_mm=4.0, weak=None, both_sided=True):
    """Eixo do tempo por linha: x(t) = x0_r + sc_r * t * v * R.
    reg = (a_r, b_r) do registro por batimentos (geometria relativa entre linhas).
    A origem absoluta vem das âncoras esquerdas (fim do pulso / início do traço),
    levadas ao referencial comum; se discordam da votação global por > 0.8 mm, valem elas."""
    a_r, b_r = reg
    if weak is not None and not both_sided:
        xl = np.where(np.isnan(xl), weak, xl)
    Lref = (xl + rad - a_r) / b_r
    ok = ~np.isnan(Lref) & (np.abs(Lref - x0g) < tol_mm * R)
    x0_ref = x0g
    # votação com bordas dos dois lados é a evidência mais forte: mantém. Só sem ela
    # as âncoras esquerdas (inclusive as fracas) podem corrigir a origem.
    if not both_sided and ok.sum() >= 1:
        med = float(np.median(Lref[ok]))
        if abs(med - x0g) > 0.8 * R:
            x0_ref = med
    return a_r + b_r * x0_ref, b_r.copy()


def _estimate_speed(B, D, rows, R, x_left, width_px, return_scores=False):
    """25 ou 50 mm/s pela largura do QRS em mm (25 mm/s: ~1.5-4 mm; 50 mm/s: ~3-8 mm)
    combinada com o intervalo RR (40-150 bpm)."""
    q = _row_quick_trace(B, D, rows, R, max(0, int(x_left)), width_px)
    widths, rrs = [], []
    from scipy.ndimage import uniform_filter1d
    for t in q:
        ok = ~np.isnan(t)
        if ok.sum() < 100:
            continue
        x = np.arange(len(t)); t = np.interp(x, x[ok], t[ok])
        g = np.abs(np.gradient(t))
        a = uniform_filter1d(g, max(3, int(1.5 * R)))
        b = _beats(t, 0, R)
        if len(b) >= 2:
            rrs += list(np.diff(b) / R)
        thr = 0.5 * np.percentile(g, 99.5)
        act = a > 0.25 * a.max()
        s_, e_ = np.nonzero(np.diff(np.r_[0, act.astype(int), 0]) == 1)[0], np.nonzero(np.diff(np.r_[0, act.astype(int), 0]) == -1)[0]
        for i0, i1 in zip(s_, e_):
            if g[i0:i1].max() > thr:
                widths.append((i1 - i0) / R)
    if not widths:
        return (25.0, {}) if return_scores else 25.0
    wq = float(np.median(widths))
    rr = float(np.median(rrs)) if rrs else None
    sc = {}
    for v in (12.5, 25.0, 50.0):
        qrs_ms = wq / v * 1000
        s_ = -((np.log(qrs_ms / 100.0)) / 0.35) ** 2
        if rr:
            hr = 60.0 / (rr / v)
            s_ += -((np.log(hr / 75.0)) / 0.4) ** 2
        sc[v] = s_
    # 25 mm/s é o padrão esmagador: 50 só com evidência clara
    v = 50.0 if sc[50.0] - sc[25.0] > 1.0 else 25.0
    return (v, sc) if return_scores else v


def _vote_x0(xl, xe, rad, span_px, R, weak=()):
    """Votação: cada borda propõe um x0; vence o grupo com mais apoio (+-0.8 mm).
    Com bordas dos dois lados, a média cancela o viés da ponta do traço.
    weak: âncoras fracas (fim da descida do pulso) — votam na escolha do grupo, mas só dão
    o valor quando o grupo não tem nenhuma borda de verdade (elas têm viés de ~0,1 mm)."""
    c = [(v + rad, "L") for v in xl] + [(v - rad - span_px, "R") for v in xe] + \
        [(v + rad, "W") for v in weak]
    if weak:
        # com pulsos, o traço começa no fim do pulso ou poucos mm depois: um final de linha
        # que põe a origem 8 mm além (ou antes do pulso) emendou em outra coisa
        wm = float(np.median(weak)) + rad
        c = [(v, t) for v, t in c if t != "R" or -1.0 * R <= v - wm <= 8.0 * R]
    _vote_x0.both_sided = False
    if not c:
        return None
    vals = np.array([v for v, _ in c])
    sup = [(np.abs(vals - v) < 0.8 * R).sum() + 0.1 * len({t for (u, t) in c if abs(u - v) < 0.8 * R})
           for v in vals]
    v0 = vals[int(np.argmax(sup))]
    grp = [(v, t) for v, t in c if abs(v - v0) < 0.8 * R]
    Ls = [v for v, t in grp if t == "L"]; Rs = [v for v, t in grp if t == "R"]
    Ws = [v for v, t in grp if t == "W"]
    _vote_x0.both_sided = bool(Ls and Rs)
    if Ls and Rs:
        return 0.5 * (np.median(Ls) + np.median(Rs))
    if Ls or Rs:
        return float(np.median(Ls or Rs))
    return float(np.median(Ws))


def digitize(image, cfg: Config | None = None, return_debug: bool = False):
    cfg = cfg or Config()
    if cfg.ink_model and cfg.waveform_model:
        raise ValueError("Avalie ink_model e waveform_model separadamente neste experimento")
    img = cv2.imread(image, cv2.IMREAD_COLOR) if isinstance(image, str) else image
    if img is None:
        raise FileNotFoundError(image)
    R = cfg.R
    _ROW_BEAT_RULE[0] = cfg.row_beats
    _PULSE_RETRY[0] = cfg.pulse_retry
    rect, pap, ginfo = rectify(img, R, cfg.ppmm, cfg.nonrigid, major_snap=cfg.major_snap,
                               fine_snap=cfg.fine_snap)
    D, glev = ink_map(rect, pap, R, return_grid_level=True)
    B = binarize(D, R, glev)
    (rect, pap, D, B), rot = _orient([rect, pap, D, B], R)

    rows_all, cov, ext_all, pulses_all, spacing = _rows_pulses(B, pap, R)
    H, W = B.shape
    if len(rows_all) == 0:
        raise RuntimeError("nenhuma linha de traçado encontrada")

    forced = None
    if cfg.layout != "auto":
        forced = LAYOUTS[cfg.layout] if isinstance(cfg.layout, str) else cfg.layout
        idx = [list(rows_all).index(r) for r in _pick_rows(rows_all, cov, len(forced))]
    else:
        idx = list(range(len(rows_all)))
    rows = [rows_all[i] for i in idx]
    ext = [ext_all[i] for i in idx]
    pulses = [pulses_all[i] for i in idx]
    spacing = float(np.median(np.diff(rows))) if len(rows) > 1 else spacing
    half = 0.45 * spacing
    # pulso no FIM da linha (vários aparelhos imprimem a calibração à direita, ou dos dois
    # lados): procurado na imagem espelhada; serve só para ganho, não para o eixo do tempo
    rpulses = [None] * len(rows)
    Bf = B[:, ::-1]
    for i_, (y, e) in enumerate(zip(rows, ext)):
        if e:
            up_sp = (y - rows[i_ - 1]) if i_ > 0 else spacing
            try:
                rpulses[i_] = detect_pulse(Bf, y, half, W - 1 - e[1], R,
                                           up=max(half, min(0.95 * up_sp, 25 * R)))
            except Exception:
                rpulses[i_] = None

    # espessura provisória (alturas de segmentos em colunas)
    lw_guess = []
    for y, e in zip(rows, ext):
        if not e:
            continue
        mid = (e[0] + e[1]) // 2
        sub = B[max(0, int(y - half)):int(y + half), max(0, mid - int(20 * R)):mid + int(20 * R)]
        for j in range(0, sub.shape[1], 3):
            d = np.diff(np.concatenate([[0], sub[:, j].astype(np.int8), [0]]))
            lw_guess += list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])
    lw0 = float(np.percentile(lw_guess, 30)) if lw_guess else 3.0
    band_vals = np.concatenate([D[max(0, int(y - half)):int(y + half)].ravel() for y in rows])
    lev0 = 0.5 * float(np.percentile(band_vals[band_vals > 0.1], 90)) if (band_vals > 0.1).any() else 0.35

    _edge_rows.text_mask = text_like_mask(B, R)
    _edge_rows.speckle = bool(getattr(ink_map, "dark_grid", False))

    # ---------------------------------------------------- velocidade e duração
    xlr0, xer0, wkr0 = _edge_rows(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2, return_weak=True)
    # início por linha: borda do traço, ou o fim do pulso quando o traço sai colado nele
    xs_p0 = [p["x_start"] for p in pulses if p is not None]
    if xs_p0:  # início à esquerda do pulso não é traço (borda da folha)
        xlr0 = np.where(xlr0 <= float(np.median(xs_p0)), np.nan, xlr0)
    xlr0 = np.where(np.isnan(xlr0), wkr0 + lw0 / 2, xlr0)
    xl_ok = xlr0[~np.isnan(xlr0)]
    xr_ok = xer0[~np.isnan(xer0)]
    left = float(np.median(xl_ok)) if len(xl_ok) else float(np.median([e[0] for e in ext if e]))
    right = float(np.median(xr_ok)) if len(xr_ok) else float(np.median([e[1] for e in ext if e]))
    span_mm = (right - left - lw0) / R
    speed = cfg.speed
    speed_src = "config"
    if speed is None:
        from .ocr import read_speed
        sp_ocr, sp_sc = read_speed(_edge_rows.text_mask, R)
        v_phys, sc_phys = _estimate_speed(B, D, rows, R, left, right - left, return_scores=True)
        # o texto vence, a menos que a fisiologia o contradiga com força (leitura fraca de
        # "25mm/s" degradado vira "12.5" ou "50": QRS de 50 ms / FC de 30 bpm denunciam)
        ok_txt = sp_ocr is not None and (
            sp_sc >= 0.7 or not sc_phys or sc_phys.get(sp_ocr, -np.inf) >= max(sc_phys.values()) - 2.0)
        # 25 mm/s é o padrão esmagador; "50" lido numa foto (o "25" borrado) só vale se a
        # fisiologia (QRS/RR) não preferir 25
        if ok_txt and sp_ocr != 25.0 and sc_phys and sc_phys.get(25.0, -np.inf) > sc_phys.get(sp_ocr, -np.inf) + 0.5:
            ok_txt = False
        if ok_txt:
            speed, speed_src = sp_ocr, "texto"
        else:
            speed, speed_src = v_phys, "QRS/RR"
    speed_nominal = speed
    duration_assumed = False
    if cfg.duration:
        duration = float(cfg.duration)
    else:
        duration = span_mm / speed
        for d_ in (1.25, 2.5, 5.0, 10.0, 12.0, 15.0, 20.0):
            if abs(duration - d_) / d_ < 0.06:
                duration = d_
                break
        else:
            # a mediana dos finais de linha pode cair num artefato (borda, sombra) em metade
            # das linhas: procura a duração padrão que o maior número de finais confirma
            sup = {}
            for d_ in (2.5, 5.0, 10.0):
                exp_ = left + lw0 + d_ * speed * R
                sup[d_] = int((np.abs(xr_ok - exp_) < 1.5 * R).sum())
            d_best = max(sup, key=sup.get) if sup else None
            n_pulses = sum(p is not None for p in pulses)
            if d_best is not None and sup[d_best] >= max(1, int(np.ceil(0.25 * len(xr_ok)))):
                duration = d_best
            elif n_pulses >= 2 and 10.0 < duration <= 12.5 and len(rows) >= 3:
                # início ancorado nos pulsos e linhas "longas demais": os finais emendaram na
                # borda da folha/grade (foto). A grade dá o mm; vale a página padrão de 10 s
                duration = 10.0
            elif len(rows) >= 3:
                # página de várias linhas fora de qualquer duração padrão: quase sempre a
                # escala horizontal está errada (imagem da web redimensionada sem manter a
                # proporção, grade só com linhas grossas). A página padrão tem 10 s; o eixo
                # do tempo se ajusta à largura medida (a escala vertical não muda)
                duration = 10.0
                duration_assumed = True
                speed = span_mm / duration
            else:
                duration = round(duration, 1)
    N = int(round(duration * cfg.fs))
    span_px = (N - 1) / cfg.fs * speed * R

    # ------------------------------------------------------- origem temporal
    xl, xe, wk_ = _origin_cands(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2)
    x0 = _vote_x0(xl, xe, lw0 / 2, span_px, R, weak=wk_)
    if x0 is None:
        x0 = float(np.median([e[0] for e in ext if e])) if any(ext) else 0.0
    # registro por batimentos: traço rápido de cada linha inteira -> QRS -> afim por linha
    xq0 = max(0, int(x0 - 2 * R))
    qfull = _row_quick_trace(B, D, rows, R, xq0, span_px + 4 * R, full=True)
    qtr = [float(y) - q["cen"] for y, q in zip(rows, qfull)]
    reg = _register_rows([_beats(t, xq0, R) for t in qtr], R, span_px, rows=rows,
                         min_shift_px=cfg.reg_min_shift_mm * R * max(1.0, speed / 25.0))
    xlr, _, wkr = _edge_rows(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2, return_weak=True)
    x0_r, sc_r = _row_time_maps(rows, xlr, lw0 / 2, x0, R, reg, weak=wkr,
                                both_sided=getattr(_vote_x0, "both_sided", True))

    # ------------------------------------- ganhos por linha (1 ou 2 níveis) e linha de base
    def _snap(g):
        # ganhos são sempre padronizados; foto em perspectiva encurta o pulso em até ~20%
        gg = min((2.5, 5.0, 10.0, 20.0, 40.0), key=lambda v: abs(np.log(g / v)))
        return gg if abs(g - gg) / gg < 0.22 else float(g)
    std_g = (2.5, 5.0, 10.0, 20.0, 40.0)

    def _std_ok(g, tol=0.22):
        return g is not None and min(abs(g - gg) / gg for gg in std_g) < tol

    def _lv(p):
        if p is None:
            return None, None
        g1_ = (p["foot"] + 0.5 - (p["top"] - 0.5) - lw0) / R
        g2_ = (p["foot"] + 0.5 - (p["top2"] - 0.5) - lw0) / R if p.get("step") else None
        return g1_, g2_
    lv1, lv2, src = [], [], []
    for p, rp in zip(pulses, rpulses):
        a1, a2 = _lv(p)
        b1, b2 = _lv(rp)
        # pulso da esquerda vale se der um ganho padrão; senão tenta o da direita
        if _std_ok(a1) or not _std_ok(b1):
            lv1.append(a1); lv2.append(a2); src.append("L" if a1 is not None else None)
        else:
            lv1.append(b1); lv2.append(None); src.append("R")  # ganho dividido só pelo pulso da esquerda
    # Consenso da página: ganhos são padronizados (2,5/5/10/20/40 mm/mV). Pulso deformado,
    # cortado na borda ou colado ao traço dá 3,7 ou 7 mm/mV; esse nível não vale e a linha
    # herda o ganho padrão mais votado entre as linhas confiáveis (ou 10, se nenhuma for).
    reliable = [_snap(g) for g in lv1 if _std_ok(g)]
    if reliable:
        vals, cnts = np.unique(reliable, return_counts=True)
        g1def = float(vals[np.argmax(cnts)])
    else:
        g1def = 10.0  # nenhum pulso confiável: 10 mm/mV é o padrão de quase todo aparelho
    gain_flags = []
    for r_, g in enumerate(lv1):
        # pulso da direita (menos confiável: costuma encostar no fim do traço) que discorda
        # do consenso da página também cede
        if g is not None and src[r_] == "R" and len(reliable) >= 2 and _snap(g) != g1def:
            lv1[r_] = None
            gain_flags.append(r_)
            continue
        if g is not None and not _std_ok(g):
            lv1[r_] = None
            lv2[r_] = None
            gain_flags.append(r_)
    g2s = [g for g in lv2 if _std_ok(g)]
    g2def = _snap(float(np.median(g2s))) if g2s else None
    levels = []
    for r_ in range(len(rows)):
        g1 = _snap(lv1[r_]) if lv1[r_] is not None else g1def
        g2 = _snap(lv2[r_]) if (lv2[r_] is not None and _std_ok(lv2[r_])) else (
            g2def if pulses[r_] is None else None)
        if cfg.gain:
            g1, g2 = float(cfg.gain), None
        levels.append((g1, g2))
    # o 0 mV vem do platô do pulso da esquerda só se ele for confiável
    pulses_base = [p if (p is not None and src[r_] == "L" and lv1[r_] is not None) else None
                   for r_, p in enumerate(pulses)]
    base_q = []
    for r_, (y, p) in enumerate(zip(rows, pulses_base)):
        if p is not None:
            base_q.append(p["top"] - 0.5 + lw0 / 2 + levels[r_][0] * R)
        else:
            q = qfull[r_]["cen"]; q = q[~np.isnan(q)]
            if len(q):
                hist, edges = np.histogram(q, bins=np.arange(q.min() - 1, q.max() + 2, 1.0))
                hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(-1, 1), (0, 0), 2).ravel()
                base_q.append(float(edges[int(np.argmax(hist))] + 0.5))
            else:
                base_q.append(float(y))

    # ------------------------------------------------------------- layout
    layout_info = None
    label_boxes = []
    if forced is not None:
        rows_spec = [list(r) for r in forced]
        layout = cfg.layout if isinstance(cfg.layout, str) else "custom"
    else:
        ctx = _LayoutCtx(B, D, rows, pulses, R, x0_r, sc_r, speed, duration, qfull, base_q, levels)
        layout_info = ctx.choose()
        # caixas de rótulos lidos com confiança: o traço que atravessa um rótulo (ex.: R
        # passando por cima de "V4") não pode ser confundido com as letras
        label_boxes = [w["box"] for w in ctx.words if w["best"] >= 0.5 and not w["single"]
                       and (w["box"][2] - w["box"][0]) < 15 * R and (w["box"][3] - w["box"][1]) < 8 * R]
        if layout_info is None:
            rows_spec = [[f"row{i + 1}"] for i in range(len(rows))]
            layout = "rows"
        else:
            rows_spec, layout = layout_info["spec"], layout_info["name"]

    def seg_gain(r_, name):
        g1, g2 = levels[r_]
        return g2 if (g2 is not None and not is_limb(name) and not name.startswith("row")) else g1

    names = LEADS if all((nm in LEADS or nm == "-aVR") for row in rows_spec for nm in row) else \
        [nm for row in rows_spec for nm in row]

    # ------------------------------------------------------- extração por segmento
    # Página/layout/pulsos/calibração foram decididos SOMENTE com D/B clássicos.
    # A probabilidade neural orienta o Viterbi, mas as bordas e o ajuste da pena
    # continuam medidos na tinta original: um heatmap não é a largura da pena.
    D_track, B_track = D, B
    if cfg.ink_model:
        from .inkmodel import predict
        if not 0 <= cfg.ink_mix <= 1:
            raise ValueError("ink_mix deve estar em [0,1]")
        pn = predict(rect, cfg.ink_model)
        pn[pap == 0] = 0
        D_track = ((1-cfg.ink_mix)*D + cfg.ink_mix*pn).astype(np.float32)
        B_track = binarize(D_track, R, glev if cfg.ink_mix < 1 else 0)
    waveform = None
    if cfg.waveform_model:
        from .kaggle5 import get_predictor
        waveform = get_predictor(cfg.waveform_model, cfg.waveform_source, cfg.waveform_device)
    hybrid_segments = []
    meas = {l: [] for l in names}  # lista de arrays (N) com NaN
    wts = {l: [] for l in names}   # pesos de confiança correspondentes
    # nível de referência da tinta do traço (para a ponte entre segmentos no Viterbi)
    _bref = None
    if cfg.bridge:
        vb = D[B]
        _bref = float(np.percentile(vb, 75)) if vb.size > 100 else None

    def br_kw(xa_):
        kw_ = dict(D=D, xa=xa_, bridge_ref=_bref, bridge_floor=min(glev, 0.6 * _bref)) if _bref else {}
        kw_['rigid'] = cfg.rigid_edges
        return kw_

    traces_dbg = []
    jobs_dbg = []
    seg_jobs = []
    for r, (y, spec) in enumerate(zip(rows, rows_spec)):
        base0 = base_q[r]
        up = (y - rows[r - 1]) if r > 0 else spacing
        dn = (rows[r + 1] - y) if r + 1 < len(rows) else spacing
        # faixa de busca generosa: ondas grandes (R de 3-5 mV) invadem a linha vizinha;
        # a primeira/última linha pode ir até a borda. O Viterbi + prior de linha de base
        # mantém o traço certo mesmo com os vizinhos dentro da faixa.
        ya = 0 if r == 0 else max(0, int(y - cfg.band_sp * up))
        yb = H if r == len(rows) - 1 else min(H, int(y + cfg.band_sp * dn))
        k = len(spec)
        for c, lead in enumerate(spec):
            n0, n1 = c * N // k, (c + 1) * N // k
            xa = int(np.floor(x0_r[r] + sc_r[r] * n0 / cfg.fs * speed * R)) - 1
            xb = int(np.ceil(x0_r[r] + sc_r[r] * (n1 - 1) / cfg.fs * speed * R)) + 2
            xa, xb = max(0, xa), min(W, xb)
            if xb - xa < 10:
                continue
            runs = column_runs(B_track, D_track, ya, yb, xa, xb)
            _drop_label_runs(runs, xa, label_boxes)
            others = [float(v) for i2, v in enumerate(rows) if i2 != r]
            # na extração final, continuidade das bordas (letras de rótulo coladas ao traço);
            # os traços rápidos usados no layout ficam como antes
            sel = viterbi_track(runs, base0, R, others=others, lw=lw0, w_edge=0.3, **br_kw(xa))
            # restrição fisiológica: o ECG volta à linha de base a cada batimento. Um caminho
            # que passa > 0,8 s inteiro a mais de 4 mm dela seguiu outra coisa (borda da folha,
            # régua de texto, traço vizinho): esses segmentos saem e o rastreio é refeito
            far_len = int(0.8 * speed * R)
            cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)])
            far = np.abs(cen - base0) > 4 * R
            st_, en_ = _runs_bool(far)
            # ... e é PLANO (sem QRS): uma derivação com desnível real de linha de base
            # continua tendo complexos; borda de folha e régua não
            win_ = max(10, int(1.0 * speed * R))

            def _flat(a_, b_):  # em >= 75% das janelas de 1 s a excursão local fica < 2,5 mm
                fl = []
                for w0 in range(a_, b_, win_):
                    seg = cen[w0:min(b_, w0 + win_)]
                    seg = seg[np.isfinite(seg)]
                    if len(seg) > 5:
                        fl.append(np.ptp(np.percentile(seg, [5, 95])) < 2.5 * R)
                return bool(fl) and np.mean(fl) >= 0.75
            bad = [(a_, b_) for a_, b_ in zip(st_, en_) if b_ - a_ >= far_len and _flat(a_, b_)]
            if bad:
                for a_, b_ in bad:
                    for i in range(a_, b_):
                        runs[i] = [rr_ for rr_ in runs[i] if abs(rr_[2] - base0) <= 4 * R]
                sel = viterbi_track(runs, base0, R, others=others, lw=lw0, w_edge=0.3, **br_kw(xa))
            seg_jobs.append(dict(lead=lead, row=r, n0=n0, n1=n1, xa=xa, runs=runs, sel=sel))

    # nível de "tinta cheia" e raio do traço (meia espessura) em trechos planos
    pks = []
    for j in seg_jobs:
        tb, pk = selected_envelopes(D, j["runs"], j["sel"], j["xa"])
        h = tb[:, 1] - tb[:, 0]
        flat = np.abs(np.diff(tb[:, 0], prepend=np.nan)) + np.abs(np.diff(tb[:, 1], append=np.nan)) < 1
        pks.append(pk[flat & (h >= 1)])
    pks = np.concatenate(pks) if pks else np.array([0.8])
    dfull = float(np.nanpercentile(pks, 75)) if pks.size else 0.8
    th = []
    for j in seg_jobs:
        top, bot = envelope_edges(D, j["runs"], j["sel"], j["xa"], dfull)
        j["top"], j["bot"] = top, bot
        dt = np.abs(np.gradient(top)) + np.abs(np.gradient(bot))
        th.append((bot - top)[dt < 0.3])
    th = np.concatenate(th)
    for j in seg_jobs:
        _mask_label_edges(j, label_boxes)
    mark_occlusions(seg_jobs, float(np.median(th)) if th.size else lw0)
    th = th[~np.isnan(th)]
    rad = float(np.percentile(th, 20)) / 2 if th.size else lw0 / 2
    rad_h = horizontal_halfwidth(D, seg_jobs, 0.5 * dfull, rad) if cfg.pen == "measure" else (
        calibrate_pen([(j["top"], j["bot"]) for j in seg_jobs], rad) if cfg.pen == "calibrate" else rad)

    # confiança por coluna: o segmento escolhido é compartilhado com o traço de OUTRA linha
    # (ondas grandes que se cruzam / se fundem) -> a forma ali é ambígua
    for j in seg_jobs:
        n_ = len(j["sel"])
        tb = np.full((n_, 2), np.nan)
        for i_, k_ in enumerate(j["sel"]):
            if k_ is not None:
                tb[i_] = j["runs"][i_][k_][0], j["runs"][i_][k_][1]
        j["tb"] = tb
        j["shared"] = np.zeros(n_, bool)
    for j in seg_jobs:
        for o in seg_jobs:
            if o is j or o["row"] == j["row"]:
                continue
            lo, hi = max(j["xa"], o["xa"]), min(j["xa"] + len(j["sel"]), o["xa"] + len(o["sel"]))
            if hi <= lo:
                continue
            a_ = j["tb"][lo - j["xa"]:hi - j["xa"]]; b_ = o["tb"][lo - o["xa"]:hi - o["xa"]]
            with np.errstate(invalid="ignore"):
                ov = (a_[:, 0] <= b_[:, 1]) & (b_[:, 0] <= a_[:, 1])
            j["shared"][lo - j["xa"]:hi - j["xa"]] |= ov
    for j in seg_jobs:  # dilata ~1 mm: o entorno de uma fusão também é suspeito
        k_ = int(R) | 1
        j["shared"] = cv2.dilate(j["shared"].astype(np.uint8).reshape(1, -1), np.ones((1, k_), np.uint8)).ravel() > 0

    # origem temporal final com o nível de 50% e o raio medidos
    lev = 0.5 * dfull
    xl, xe, wk_ = _origin_cands(D, B, rows, ext, pulses, half, lev, R, rad)
    x0f = _vote_x0(xl, xe, rad, span_px, R, weak=wk_)
    if x0f is not None:
        x0 = float(x0f)
    xlr, _, wkr = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad, return_weak=True)
    x0_r, sc_r = _row_time_maps(rows, xlr, rad, x0, R, reg, weak=wkr,
                                both_sided=getattr(_vote_x0, "both_sided", True))

    # linha de base final com o raio medido: 0 mV = centro do platô 1 + ganho do nível 1
    for r_, p in enumerate(pulses_base):
        if p is not None:
            base_q[r_] = p["top"] - 0.5 + rad + levels[r_][0] * R

    # extensão típica da marca separadora (à esquerda/direita da fronteira) nesta página
    ext_l, ext_r = [], []
    for j in seg_jobs:
        rr = j["row"]
        xs_ = np.arange(j["xa"], j["xa"] + len(j["top"]), dtype=np.float64)
        if j["n0"] > 0:   # começo de derivação: quanto a marca avança depois da fronteira
            xb = x0_r[rr] + sc_r[rr] * j["n0"] / cfg.fs * speed * R
            sp_ = _tick_span(xs_, j["top"], j["bot"], xb, R)
            if sp_ is not None:
                ext_r.append(sp_[1] - xb)
        if j["n1"] < N:   # fim de derivação: quanto a marca começa antes da fronteira
            xb = x0_r[rr] + sc_r[rr] * (j["n1"] - 1) / cfg.fs * speed * R
            sp_ = _tick_span(xs_, j["top"], j["bot"], xb, R)
            if sp_ is not None:
                ext_l.append(xb - sp_[0])
    # percentil 25: um QRS colado à fronteira só alarga a medida, nunca a estreita
    tick_l = float(np.percentile(ext_l, 25)) + 1.0 if len(ext_l) >= 3 else 0.0
    tick_r = float(np.percentile(ext_r, 25)) + 1.0 if len(ext_r) >= 3 else 0.0
    for j in seg_jobs:
        yc, valid = centerline_from_envelopes(j["top"], j["bot"], rad, rh=rad_h)
        if valid.sum() < 5:
            continue
        p = pulses_base[j["row"]]
        gain = seg_gain(j["row"], j["lead"])
        if p is not None:
            base = base_q[j["row"]]
        else:  # sem pulso: moda (isoelétrica) da linha
            vv = yc[valid]
            vv = vv[np.isfinite(vv)]
            if len(vv) < 5:
                continue
            hist, edges = np.histogram(vv, bins=np.arange(vv.min() - 1, vv.max() + 2, 1.0))
            hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(-1, 1), (0, 0), 2).ravel()
            base = float(edges[int(np.argmax(hist))] + 0.5)
            j["base_from_mode"] = True
        xs = np.arange(j["xa"], j["xa"] + len(yc), dtype=np.float64)
        # bordas do segmento: pontas do traço / marcas separadoras / fim do traço vizinho
        # -> descarta ~raio+1 px e extrapola linearmente a partir do interior
        tr = np.ceil(rad_h + 1.0)
        rr = j["row"]
        xs_s = x0_r[rr] + sc_r[rr] * j["n0"] / cfg.fs * speed * R
        xs_e = x0_r[rr] + sc_r[rr] * (j["n1"] - 1) / cfg.fs * speed * R
        lo_cut = xs_s + tr if j["n0"] > 0 else -np.inf          # fronteira interna à esquerda
        hi_cut = xs_e - tr if j["n1"] < N else np.inf           # fronteira interna à direita
        if j["n0"] == 0 and pulses[rr] is not None:
            # começo da linha: nada antes do fim do pulso de calibração (a descida do pulso e
            # o platô não são sinal)
            pe = pulses[rr].get("x_fall", pulses[rr].get("x_end"))
            if pe is not None:
                lo_cut = max(lo_cut, float(pe) + tr)
        # marca separadora na troca de derivação: barra vertical mais larga que a pena em
        # scans e fotos (borrão). As colunas da marca têm o traço "alto" (topo-base de vários
        # mm) colado na fronteira; corta até o fim dela, para não virar um degrau de 0,3 mV
        # marca separadora: corta até a extensão típica da marca nesta página (mediana entre
        # todas as fronteiras), para um QRS colado na fronteira não ser cortado junto
        if j["n0"] > 0:
            sp_ = _tick_span(xs, j["top"], j["bot"], xs_s, R)
            if sp_ is not None:
                lo_cut = max(lo_cut, xs_s + min(sp_[1] - xs_s, tick_r) + 1.0)
        if j["n1"] < N:
            sp_ = _tick_span(xs, j["top"], j["bot"], xs_e, R)
            if sp_ is not None:
                hi_cut = min(hi_cut, xs_e - min(xs_e - sp_[0], tick_l) - 1.0)
        good = valid & (xs > lo_cut) & (xs < hi_cut)
        if good.sum() < 10:
            good = valid
        xg, yg = xs[good], yc[good]
        if cfg.steep_subpx and len(xg) > 4:
            xg, yg = refine_steep(D, xg, yg, rad_h)
        if cfg.synth and len(xg) > 20:
            from .synthesis import synth_refine
            xg, yg = synth_refine(D, xg, yg, rad, max_shift=cfg.synth_max_shift, order=cfg.synth_order)
        if cfg.fit == "lsf" and len(xg) > 30:
            from .fit import lsf_refine
            xg, yg = lsf_refine(D, xg, yg, **(cfg.fit_kw or {}))
        tn = np.arange(j["n0"], j["n1"]) / cfg.fs
        xn = x0_r[rr] + sc_r[rr] * tn * speed * R
        yn = np.interp(xn, xg, yg)
        # extrapolação linear limitada (inclinação amortecida), senão mantém o valor
        K = min(6, len(xg))
        a0 = np.polyfit(xg[:K], yg[:K], 1); a1 = np.polyfit(xg[-K:], yg[-K:], 1)
        yn = np.where(xn < xg[0], yg[0] + 0.5 * a0[0] * (xn - xg[0]), yn)
        yn = np.where(xn > xg[-1], yg[-1] + 0.5 * a1[0] * (xn - xg[-1]), yn)
        # extrapolar só o que a marca separadora esconde (~1 raio de pena); além disso a
        # amostra é desconhecida: uma rampa inventada estraga o 1º batimento e Einthoven
        ext_max = 2.0 * tr
        yn = np.where((xn < xg[0] - ext_max) | (xn > xg[-1] + ext_max), np.nan, yn)
        # na troca de derivação o trecho escondido pela marca é curto (poucos ms): vale o
        # primeiro/último valor limpo, constante (melhor que vazio, que a métrica lê como zero)
        # extrapolação com metade da inclinação das primeiras amostras limpas (um QRS colado
        # à fronteira continua por baixo da marca, mas a curva costuma fazer a volta ali)
        K4 = min(4, len(xg))
        if j["n0"] > 0 and K4 >= 2:
            sl = np.polyfit(xg[:K4], yg[:K4], 1)[0]
            yn = np.where((xn < xg[0]) & (xn >= xs_s - 1.0), yg[0] + 0.5 * sl * (xn - xg[0]), yn)
        if j["n1"] < N and K4 >= 2:
            sl = np.polyfit(xg[-K4:], yg[-K4:], 1)[0]
            yn = np.where((xn > xg[-1]) & (xn <= xs_e + 1.0), yg[-1] + 0.5 * sl * (xn - xg[-1]), yn)
        if waveform is not None:
            # Usa a janela e a escala encontradas pelo clássico, nunca nomes ou
            # posições de uma página 3x4. O backend retorna coordenadas na imagem.
            yn_neural, info = waveform.predict(rect, xn, base, gain*R,
                                               sc_r[rr]*speed*R)
            # Mesma cobertura nas duas variantes; não inventa bordas ausentes.
            yn = np.where(np.isfinite(yn), yn_neural, np.nan)
            hybrid_segments.append(dict(lead=j["lead"], row=rr, **info))
        v = (base - yn) / (gain * R)
        if j["n0"] == 0:
            v = _cut_leading_pulse(v, cfg.fs)
        lead = j["lead"]
        if lead == "-aVR":  # Cabrera: derivação invertida
            lead, v = "aVR", -v
        arr = np.full(N, np.nan, np.float32)
        arr[j["n0"]:j["n1"]] = v
        meas[lead].append(arr)
        w = np.full(N, np.nan, np.float32)
        sh = np.interp(xn, xs, j["shared"].astype(float)) > 0.5
        w[j["n0"]:j["n1"]] = np.where(sh, 0.05, 1.0)
        wts[lead].append(w)
        traces_dbg.append((xn, yn, np.isfinite(yn)) if waveform is not None else (xs, yc, valid))
        if return_debug:
            jobs_dbg.append(dict(lead=lead, row=rr, n0=j["n0"], n1=j["n1"], xn=xn, yn=(base - v * gain * R) if lead == j["lead"] else None,
                                 xg=xg, yg=yg, base=base, gain=gain, top=j["top"], bot=j["bot"], xa=j["xa"]))
    lw = 2 * rad

    sig = np.full((len(names), N), np.nan, np.float32)
    nmeas = np.zeros((len(names), N), np.float32)
    for li, l in enumerate(names):
        if meas[l]:
            st = np.stack(meas[l])
            wt = np.where(np.isnan(st), 0.0, np.nan_to_num(np.stack(wts[l]), nan=0.0))
            sw = wt.sum(0)
            sig[li] = np.where(sw > 0, np.nansum(st * wt, 0) / np.maximum(sw, 1e-9), np.nan)
            nmeas[li] = sw  # "número efetivo de medidas" (peso total) -> variância na fusão
    if cfg.einthoven and names is LEADS:
        sig = einthoven_fuse(sig, nmeas, fill=cfg.fill_limb)

    gains_seg = {f"{r_}:{nm}": seg_gain(r_, nm) for r_, row in enumerate(rows_spec[:len(levels)]) for nm in row}
    res = dict(signals=sig, fs=cfg.fs, leads=list(names), layout=layout, layout_rows=rows_spec,
               layout_info=layout_info, rotation=str(rot),
               gain_mm_per_mv=levels[0][0] if levels else None, gains=gains_seg, duration_s=duration,
               speed_mm_per_s=speed_nominal, speed_source=speed_src, duration_assumed=duration_assumed,
               horizontal_scale=speed / speed_nominal, ppmm=ginfo["ppmm"], angle=ginfo["angle"],
               grid_residual_mm=ginfo["residual_rms_mm"], grid_residual_cell_mm=ginfo.get("residual_cell_rms_mm"),
               grid_snap_mm=ginfo.get("major_snap_mm"), grid_fine_mm=ginfo.get("fine_snap_mm"), line_width_px=lw, line_width_h_px=2 * rad_h,
               pulses_found=sum(p is not None for p in pulses),
               pulses_right=sum(p is not None for p in rpulses), gain_rows_inferred=gain_flags)
    res["hybrid"] = dict(ink_model=bool(cfg.ink_model), waveform_model=bool(cfg.waveform_model),
                         structural_map="classic", segments=hybrid_segments)
    if return_debug:
        dbg = rect.copy()
        for xs, yi, valid in traces_dbg:
            pts = np.stack([xs, yi], 1).astype(np.int32)
            cv2.polylines(dbg, [pts], False, (255, 120, 0), 1, cv2.LINE_AA)
        for y in rows:
            cv2.line(dbg, (0, int(y)), (W, int(y)), (0, 160, 0), 1)
        cv2.line(dbg, (int(x0), 0), (int(x0), H), (0, 160, 0), 1)
        res["debug_image"] = dbg
        res["per_segment"] = {l: [m.copy() for m in v] for l, v in meas.items() if v}
        res["per_segment_weight"] = {l: [m.copy() for m in v] for l, v in wts.items() if v}
        res["internals"] = dict(x0=x0, x0_rows=list(map(float, x0_r)), scale_rows=list(map(float, sc_r)),
                                reg_a=list(map(float, reg[0])), reg_b=list(map(float, reg[1])),
                                xl=xl, xe=xe, ext=ext,
                                rows=list(map(int, rows)), pulses=pulses, rad=rad, rad_h=rad_h)
        res["rectified"] = rect
        res["jobs"] = jobs_dbg
        res["ink"] = D
        res["tracking_ink"] = D_track
    return res


def einthoven_fuse(sig, nmeas, fill=False):
    """Projeção de mínimos quadrados nas restrições lineares exatas:
       I - II + III = 0   e   aVR + aVL + aVF = 0   (e, com fill, as relações de Goldberger)."""
    s = sig.copy()
    I, II, III, aVR, aVL, aVF = range(6)
    var = 1.0 / np.maximum(nmeas, 0.02)  # amostras ocluídas (peso baixo) absorvem a correção

    def project(idx, coef):
        idx = list(idx)
        X = s[idx]
        ok = ~np.isnan(X).any(0)
        if not ok.any():
            return
        a = np.array(coef, np.float32)[:, None]
        v = var[idx][:, ok]
        e = (a * X[:, ok]).sum(0)
        den = (a * a * v).sum(0)
        for j, k in enumerate(idx):
            s[k, ok] = X[j, ok] - a[j, 0] * v[j] * e / den

    project([I, II, III], [1, -1, 1])
    project([aVR, aVL, aVF], [1, 1, 1])
    if fill:
        # completa usando as definições de Goldberger quando I e II existem, etc.
        has = ~np.isnan(s)
        m = has[I] & has[II]
        for k, f in ((III, lambda: s[II] - s[I]), (aVR, lambda: -(s[I] + s[II]) / 2),
                     (aVL, lambda: s[I] - s[II] / 2), (aVF, lambda: s[II] - s[I] / 2)):
            val = f()
            fillm = m & ~has[k]
            s[k, fillm] = val[fillm]
        has = ~np.isnan(s)
        m = has[aVR] & has[aVL] & ~has[I]
        Iv = (s[aVL] - s[aVR]) / 1.5
        IIv = -2 * s[aVR] - Iv
        s[I, m] = Iv[m]
        mm = m & ~has[II]
        s[II, mm] = IIv[mm]
        mm = m & ~has[III]
        s[III, mm] = (IIv - Iv)[mm]
    return s

