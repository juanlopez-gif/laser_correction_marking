"""
VR-6000 Laser Effect Comparison  --  script unificado
======================================================
Un solo script para todos los datasets del experimento.
Para anadir un nuevo dataset, agregar una entrada en GRUPOS (al final).

Genera en ./resultados/:
  comparison_*.png  -- PRE | POST | DELTA + histograma + perfil central
  delta_*.png       -- mapa delta detallado con isolineas
  profiles_*.png    -- N perfiles horizontales PRE vs POST
  report_*.csv      -- metricas numericas completas

Flujo de alineacion:
  1. Coarse search  (Phase-Only Correlation, robusto ante texturas periodicas)
  2. Fine   search  (ventana +-margin px alrededor del estimado grueso)
  3. Correccion Z offset global
  4. Correccion de rotacion residual (si R2 suficiente)
  5. Mapa delta + estadisticas ablacion/deposito
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import matplotlib
import sys as _sys
# Solo forzar Agg cuando pyplot no esta ya inicializado (ej. por un script
# interactivo que importa este modulo despues de haber cargado TkAgg).
if 'matplotlib.pyplot' not in _sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from pathlib import Path
import logging
import csv
import re
from typing import Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ComparisonConfig:
    pixel_size_mm: float = 0.011814       # 11.814 um/px (Keyence VR-6000, 12x)
    smoothing_sigma: float = 1.0
    coarse_downsample: int = 4
    min_snr: float = 1.5
    confidence_threshold: float = 0.30
    min_overlap_ratio: float = 0.40
    rotation_r2_threshold: float = 0.20
    rotation_mag_threshold_deg: float = 0.001
    outlier_sigma: float = 3.0
    laser_effect_threshold_um: float = 2.0
    border_erosion_px: int = 50           # ~0.6mm de borde (50 * 11.814um)
    min_cluster_area_px: int = 25
    stats_percentile: float = 0.5
    n_profiles: int = 5
    profile_smooth_px: int = 5
    max_shift_px: int = 30                # desplazamiento maximo esperado (~0.35mm)
    background_correction_degree: int = 1 # 0=off, 1=plano, 2=cuadratico
    background_sigma_clip: float = 2.5    # sigma para descartar marcas laser del ajuste
    inplane_rotation_search: bool = True  # buscar rotacion en plano (eje Z)
    inplane_angle_range_deg: float = 2.0  # rango de busqueda +-deg
    inplane_angle_step_deg: float = 0.1   # paso de busqueda
    inplane_search_ds: int = 8            # downsample rapido para la busqueda


@dataclass
class AlignmentResult:
    y_pos: int
    x_pos: int
    y_mm: float
    x_mm: float
    confidence: float
    snr: float
    overlap_ratio: float
    delta_z_mm: float
    theta_x_deg: float
    theta_y_deg: float
    r_squared: float
    rotation_corrected: bool
    inplane_angle_deg: float = 0.0


@dataclass
class LaserEffectStats:
    analysis_area_mm2: float
    n_valid_points: int
    mean_diff_um: float
    std_diff_um: float
    rms_diff_um: float
    ablation_area_mm2: float
    deposition_area_mm2: float
    max_ablation_um: float
    max_deposition_um: float
    mean_ablation_um: float
    mean_deposition_um: float
    ablation_volume_mm3: float
    deposition_volume_mm3: float
    delta_z_pre_post_um: float


# ============================================================================
# CSV LOADER
# ============================================================================

def load_vr6000_csv(filepath: str) -> Tuple[np.ndarray, float]:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    pixel_size = None
    for line in lines[:50]:
        if 'xy' in line.lower() and 'calibration' in line.lower():
            nums = re.findall(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
            if nums:
                pixel_size = float(nums[0]) / 1000.0   # um -> mm
                break

    if pixel_size is None:
        logger.warning("XY Calibration no encontrada, usando 0.011814 mm")
        pixel_size = 0.011814

    data_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('Height,') or s.startswith('"Height"'):
            parts = s.split(',')
            if all(p.strip() in ('', '""', '"') for p in parts[1:]):
                data_start = i + 1
                break
    if data_start == 0:
        data_start = 22

    rows, max_cols = [], 0
    for line in lines[data_start:]:
        if not line.strip():
            continue
        vals = [x.strip().strip('"') for x in line.strip().split(',')]
        row = []
        for v in vals:
            try:
                row.append(float(v) if v not in ('', '""') else np.nan)
            except Exception:
                row.append(np.nan)
        if row:
            rows.append(row)
            max_cols = max(max_cols, len(row))

    if not rows:
        raise ValueError(f"Sin datos en {Path(filepath).name}")

    H = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        H[i, :len(row)] = row[:max_cols]

    H[H < -10] = np.nan
    H[H >  10] = np.nan

    logger.info(f"Cargado {Path(filepath).name}: {H.shape}, "
                f"pixel={pixel_size*1000:.3f}um, "
                f"validos={int((~np.isnan(H)).sum()):,}")
    return H, pixel_size


# ============================================================================
# ALIGNMENT  (coarse POC -> fine)
# ============================================================================

def _downsample(H: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return H.copy()
    nr = H.shape[0] // factor
    nc = H.shape[1] // factor
    out = np.full((nr, nc), np.nan)
    for i in range(nr):
        for j in range(nc):
            blk = H[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            v = blk[~np.isnan(blk)]
            if v.size > 0:
                out[i, j] = v.mean()
    return out


def _normalize(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    if valid.sum() < 10:
        return np.zeros_like(data)
    m = np.nanmean(data)
    s = np.nanstd(data)
    n = (data - m) / s if s > 1e-10 else data - m
    n[~valid] = 0.0
    return n


def _smooth(H: np.ndarray, sigma: float) -> np.ndarray:
    filled = np.where(np.isnan(H), 0.0, H)
    return ndimage.gaussian_filter(filled, sigma=sigma, mode='constant', cval=0.0)


def _overlap_region(shape_pre, shape_post, y_shift, x_shift):
    pre_r0 = max(0, y_shift)
    pre_c0 = max(0, x_shift)
    pre_r1 = min(shape_pre[0], y_shift + shape_post[0])
    pre_c1 = min(shape_pre[1], x_shift + shape_post[1])
    if pre_r1 <= pre_r0 or pre_c1 <= pre_c0:
        return None
    post_r0 = pre_r0 - y_shift
    post_c0 = pre_c0 - x_shift
    post_r1 = post_r0 + (pre_r1 - pre_r0)
    post_c1 = post_c0 + (pre_c1 - pre_c0)
    return (slice(pre_r0, pre_r1), slice(pre_c0, pre_c1),
            slice(post_r0, post_r1), slice(post_c0, post_c1))


def _rotate_image(H: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rota la imagen angle_deg grados (sentido antihorario) preservando NaN.
    Usa interpolacion bilineal (order=1). El NaN se propaga correctamente.
    """
    if abs(angle_deg) < 1e-6:
        return H.copy()
    nan_mask = np.isnan(H)
    filled   = np.where(nan_mask, 0.0, H)
    rotated  = ndimage.rotate(filled, angle_deg, reshape=False,
                               order=1, mode='constant', cval=0.0)
    rot_nan  = ndimage.rotate(nan_mask.astype(np.float32), angle_deg, reshape=False,
                               order=1, mode='constant', cval=1.0)
    rotated[rot_nan > 0.5] = np.nan
    return rotated


def search_inplane_rotation(H_ref: np.ndarray, H_post: np.ndarray,
                             y_pos: int, x_pos: int,
                             config: ComparisonConfig) -> float:
    """
    Busca el angulo de rotacion en el plano XY (eje Z) que minimiza el RMS
    de la diferencia (H_post_rotado - H_ref) en la region de solape.

    Estrategia en dos fases:
      1. Grid coarse (paso config.inplane_angle_step_deg) para ubicar el minimo global.
      2. Refinamiento continuo con scipy.optimize.minimize_scalar alrededor del
         minimo coarse — resolucion efectiva < 0.005 deg, lo que reduce el error
         de alineacion en el borde de 20mm a < 0.002mm ≈ 0.15px.
    """
    from scipy.optimize import minimize_scalar

    logger.info("\n" + "="*60)
    logger.info(f"ROTACION EN PLANO (+-{config.inplane_angle_range_deg} deg)")
    logger.info("="*60)

    ds = config.inplane_search_ds

    def _rms_at_angle(angle):
        H_rot = _rotate_image(H_post, angle)
        slices = _overlap_region(H_ref.shape, H_rot.shape, y_pos, x_pos)
        if slices is None:
            return 1e9
        pre_sr, pre_sc, post_sr, post_sc = slices
        pre_d  = H_ref[pre_sr, pre_sc][::ds, ::ds]
        post_d = H_rot[post_sr, post_sc][::ds, ::ds]
        diff   = post_d - pre_d
        valid  = ~np.isnan(diff)
        if valid.sum() < 50:
            return 1e9
        dz  = np.nanmedian(diff[valid])
        return float(np.sqrt(np.nanmean((diff[valid] - dz) ** 2)))

    # --- Fase 1: grid coarse para encontrar la cuenca del minimo ---
    step   = config.inplane_angle_step_deg
    angles = np.arange(-config.inplane_angle_range_deg,
                        config.inplane_angle_range_deg + step / 2, step)

    best_angle, best_rms = 0.0, np.inf
    for a in angles:
        r = _rms_at_angle(a)
        if r < best_rms:
            best_rms, best_angle = r, float(a)

    logger.info(f"  Grid coarse: mejor angulo = {best_angle:+.2f} deg  "
                f"(RMS coarse = {best_rms*1000:.3f} um)")

    # --- Fase 2: refinamiento continuo en +-step alrededor del minimo ---
    lo = best_angle - step
    hi = best_angle + step
    result = minimize_scalar(_rms_at_angle, bounds=(lo, hi), method='bounded',
                             options={'xatol': 1e-4, 'maxiter': 50})

    if result.success and result.fun < best_rms:
        best_angle = float(result.x)
        best_rms   = float(result.fun)

    logger.info(f"  Refinado   : mejor angulo = {best_angle:+.4f} deg  "
                f"(RMS fino   = {best_rms*1000:.3f} um)")
    return best_angle


def coarse_search(H_ref: np.ndarray, H_query: np.ndarray,
                  config: ComparisonConfig) -> Optional[Tuple[int, int, float, float]]:
    """
    Phase-Only Correlation (POC): suprime frecuencias periodicas dominantes,
    da picos mas nitidos que XC estandar en superficies texturadas.
    Busqueda restringida a +-max_shift_px para evitar picos espurios.
    """
    logger.info("\n" + "="*60)
    logger.info("COARSE SEARCH (POC, ventana +-max_shift_px)")
    logger.info("="*60)

    DS      = config.coarse_downsample
    H_ref_d = _downsample(H_ref,   DS)
    H_q_d   = _downsample(H_query, DS)

    if (~np.isnan(H_q_d)).sum() < 50:
        logger.error("Query insuficiente tras downsampling")
        return None

    logger.info(f"Ref (down): {H_ref_d.shape}  |  Query (down): {H_q_d.shape}")

    ref_n = _normalize(_smooth(H_ref_d, config.smoothing_sigma))
    q_n   = _normalize(_smooth(H_q_d,   config.smoothing_sigma))

    ref_f = np.fft.fft2(ref_n)
    q_f   = np.fft.fft2(q_n)
    cross = ref_f * np.conj(q_f)
    eps   = 1e-10 * (np.abs(cross).max() + 1e-30)
    poc   = np.real(np.fft.ifft2(cross / (np.abs(cross) + eps)))
    poc   = np.fft.fftshift(poc)

    cy0, cx0 = poc.shape[0] // 2, poc.shape[1] // 2

    uy, ux = np.unravel_index(np.argmax(poc), poc.shape)
    logger.info(f"  Pico POC global (sin restriccion): "
                f"Y={(uy - cy0) * DS:+d}px, X={(ux - cx0) * DS:+d}px")

    max_s_c = config.max_shift_px // DS + 1
    r0 = max(0, cy0 - max_s_c); r1 = min(poc.shape[0], cy0 + max_s_c + 1)
    c0 = max(0, cx0 - max_s_c); c1 = min(poc.shape[1], cx0 + max_s_c + 1)
    poc_win = poc[r0:r1, c0:c1]

    cmean = poc.mean()
    cstd  = poc.std()

    top_idx = np.argsort(poc_win.flatten())[-30:][::-1]
    best = None

    for idx in top_idx:
        wy, wx  = np.unravel_index(idx, poc_win.shape)
        cy, cx  = r0 + wy, c0 + wx
        val     = float(poc[cy, cx])
        y_shift = int((cy - cy0) * DS)
        x_shift = int((cx - cx0) * DS)

        slices = _overlap_region(H_ref.shape, H_query.shape, y_shift, x_shift)
        if slices is None:
            continue
        pre_sr, pre_sc, _, _ = slices
        ov_area = (pre_sr.stop - pre_sr.start) * (pre_sc.stop - pre_sc.start)
        q_area  = H_query.shape[0] * H_query.shape[1]
        if ov_area < config.min_overlap_ratio * q_area:
            continue

        snr        = (val - cmean) / cstd if cstd > 0 else 0
        confidence = min(snr / 5.0, 1.0)
        logger.info(f"  Candidato: Y={y_shift:+d}, X={x_shift:+d}, "
                    f"SNR={snr:.2f}, Solape={100*ov_area/q_area:.1f}%")

        if best is None or snr > best[2]:
            best = (y_shift, x_shift, snr, confidence)

    if best is None:
        logger.warning("Sin candidato en ventana — usando (0,0) como fallback")
        return (0, 0, 0.0, 0.0)

    logger.info(f"Mejor grueso: Y={best[0]:+d}, X={best[1]:+d}, "
                f"SNR={best[2]:.2f}, Conf={best[3]:.3f}")
    return best


def fine_search(H_ref: np.ndarray, H_query: np.ndarray,
                coarse_y: int, coarse_x: int,
                config: ComparisonConfig) -> Tuple[int, int, float, float]:
    logger.info("\n" + "="*60)
    logger.info(f"FINE SEARCH alrededor de Y={coarse_y:+d}, X={coarse_x:+d}")
    logger.info("="*60)

    margin = 2 * config.coarse_downsample + 4

    r0 = max(0, coarse_y - margin)
    c0 = max(0, coarse_x - margin)
    r1 = min(H_ref.shape[0], coarse_y + H_query.shape[0] + margin)
    c1 = min(H_ref.shape[1], coarse_x + H_query.shape[1] + margin)

    H_reg    = H_ref[r0:r1, c0:c1]
    reg_norm = _normalize(_smooth(H_reg,   config.smoothing_sigma))
    q_norm   = _normalize(_smooth(H_query, config.smoothing_sigma))
    corr     = fftconvolve(reg_norm, q_norm[::-1, ::-1], mode='full')

    qr, qc = H_query.shape
    cy_lo = max(0, coarse_y - margin - r0 + qr - 1)
    cy_hi = min(corr.shape[0] - 1, coarse_y + margin - r0 + qr - 1)
    cx_lo = max(0, coarse_x - margin - c0 + qc - 1)
    cx_hi = min(corr.shape[1] - 1, coarse_x + margin - c0 + qc - 1)

    corr_w = corr.copy()
    corr_w[:cy_lo, :]      = -np.inf
    corr_w[cy_hi + 1:, :]  = -np.inf
    corr_w[:, :cx_lo]      = -np.inf
    corr_w[:, cx_hi + 1:]  = -np.inf

    max_idx = np.unravel_index(np.argmax(corr_w), corr.shape)
    cval    = float(corr[max_idx])
    cy, cx  = max_idx

    y_final = r0 + cy - (qr - 1)
    x_final = c0 + cx - (qc - 1)

    cmean = np.nanmean(corr)
    cstd  = np.nanstd(corr)
    snr        = (cval - cmean) / cstd if cstd > 0 else 0
    confidence = min(snr / 5.0, 1.0)

    logger.info(f"Refinado: Y={y_final:+d}, X={x_final:+d}, "
                f"SNR={snr:.2f}, Conf={confidence:.3f}")
    return int(y_final), int(x_final), snr, confidence


# ============================================================================
# Z OFFSET & ROTATION
# ============================================================================

def compute_z_offset(H_pre_reg: np.ndarray, H_post_reg: np.ndarray) -> float:
    valid = ~np.isnan(H_pre_reg) & ~np.isnan(H_post_reg)
    if valid.sum() < 100:
        return 0.0
    return float(np.nanmedian(H_post_reg[valid] - H_pre_reg[valid]))


def compute_rotation(H_pre_reg: np.ndarray, H_post_reg: np.ndarray,
                     delta_z: float, pixel_size_mm: float) -> Tuple[float, float, float]:
    valid = ~np.isnan(H_pre_reg) & ~np.isnan(H_post_reg)
    if valid.sum() < 200:
        return 0.0, 0.0, 0.0

    error = H_post_reg[valid] - delta_z - H_pre_reg[valid]
    nr, nc = H_pre_reg.shape
    Y, X = np.meshgrid(np.arange(nr) * pixel_size_mm,
                       np.arange(nc) * pixel_size_mm, indexing='ij')
    y_d = Y[valid].flatten()
    x_d = X[valid].flatten()
    z_d = error.flatten()

    inliers = np.abs(z_d - z_d.mean()) < 3 * z_d.std()
    if inliers.sum() < 200:
        return 0.0, 0.0, 0.0

    A = np.column_stack([y_d[inliers], x_d[inliers], np.ones(inliers.sum())])
    try:
        coeffs, *_ = np.linalg.lstsq(A, z_d[inliers], rcond=None)
        a, b, _ = coeffs
        z_pred = A @ coeffs
        ss_res = np.sum((z_d[inliers] - z_pred) ** 2)
        ss_tot = np.sum((z_d[inliers] - z_d[inliers].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return float(np.degrees(np.arctan(a))), float(np.degrees(np.arctan(b))), float(r2)
    except Exception:
        return 0.0, 0.0, 0.0


def apply_rotation_correction(H: np.ndarray, theta_x: float, theta_y: float,
                               pixel_size_mm: float) -> np.ndarray:
    rows, cols = H.shape
    Y, X = np.meshgrid((np.arange(rows) - rows / 2) * pixel_size_mm,
                       (np.arange(cols) - cols / 2) * pixel_size_mm, indexing='ij')
    Z_corr = Y * np.tan(np.radians(theta_x)) + X * np.tan(np.radians(theta_y))
    H_out = H.copy()
    H_out[~np.isnan(H)] -= Z_corr[~np.isnan(H)]
    return H_out


# ============================================================================
# DIFFERENCE MAP & STATS
# ============================================================================

def _build_interior_mask(H_pre: np.ndarray, H_post: np.ndarray,
                          erosion_px: int) -> np.ndarray:
    if erosion_px <= 0:
        return ~np.isnan(H_pre) & ~np.isnan(H_post)
    struct = ndimage.generate_binary_structure(2, 1)
    e_pre  = ndimage.binary_erosion(~np.isnan(H_pre), structure=struct,
                                    iterations=erosion_px, border_value=0)
    e_post = ndimage.binary_erosion(~np.isnan(H_post), structure=struct,
                                    iterations=erosion_px, border_value=0)
    mask  = e_pre & e_post
    n_rem = int((~np.isnan(H_pre) & ~np.isnan(H_post)).sum()) - int(mask.sum())
    logger.info(f"Erosion borde ({erosion_px}px): eliminados {n_rem:,} px | "
                f"interior: {int(mask.sum()):,} px")
    return mask


def _filter_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    clean = np.zeros_like(mask)
    for lbl, sz in enumerate(sizes, start=1):
        if sz >= min_size:
            clean |= (labeled == lbl)
    rem = int(mask.sum()) - int(clean.sum())
    if rem > 0:
        logger.info(f"  Cluster filter: eliminados {rem:,} px spike (< {min_size}px)")
    return clean


def correct_background_trend(delta_map: np.ndarray,
                             degree: int = 1,
                             sigma_clip: float = 2.5,
                             n_iter: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polynomial background subtraction con sigma-clipping iterativo.

    Problema: entre medida PRE y POST el sample se reposiciona con un micro-tilt
    diferente. La rugosidad superficial (~8um RMS) ahoga la senal de tilt en el
    ajuste de rotacion (R2 bajo), por lo que el tilt queda sin corregir.
    Solucion: ajustar un polinomio al delta_map (donde la textura ya se cancelo)
    excluyendo iterativamente los outliers (marcas laser reales).

    degree=1: plano (tilt X+Y)
    degree=2: cuadratico (tilt + bow/curvatura)

    Devuelve (delta_map_corregido, superficie_ajustada_en_um).
    """
    valid = ~np.isnan(delta_map)
    if valid.sum() < 100 or degree == 0:
        return delta_map.copy(), np.zeros_like(delta_map)

    rows, cols = np.where(valid)
    z = delta_map[rows, cols]

    # coordenadas normalizadas a [0,1] para estabilidad numerica
    r_n = rows / delta_map.shape[0]
    c_n = cols / delta_map.shape[1]

    if degree == 1:
        A = np.column_stack([np.ones(len(r_n)), r_n, c_n])
    else:  # degree 2: plano + terminos cuadraticos
        A = np.column_stack([np.ones(len(r_n)), r_n, c_n,
                             r_n * r_n, c_n * c_n, r_n * c_n])

    inliers = np.ones(len(z), dtype=bool)
    coeffs  = None
    for it in range(n_iter):
        if inliers.sum() < A.shape[1] + 1:
            break
        coeffs, *_ = np.linalg.lstsq(A[inliers], z[inliers], rcond=None)
        resid   = z - A @ coeffs
        std_in  = resid[inliers].std()
        if std_in < 1e-10:
            break
        inliers = np.abs(resid) < sigma_clip * std_in

    if coeffs is None:
        return delta_map.copy(), np.zeros_like(delta_map)

    # construir superficie sobre la imagen completa
    R, C = np.meshgrid(np.arange(delta_map.shape[0]),
                        np.arange(delta_map.shape[1]), indexing='ij')
    R_n = R / delta_map.shape[0]
    C_n = C / delta_map.shape[1]

    if degree == 1:
        surf = coeffs[0] + coeffs[1] * R_n + coeffs[2] * C_n
    else:
        surf = (coeffs[0] + coeffs[1] * R_n + coeffs[2] * C_n
                + coeffs[3] * R_n * R_n + coeffs[4] * C_n * C_n
                + coeffs[5] * R_n * C_n)

    surf_valid   = surf[valid]
    surf_range   = surf_valid.max() - surf_valid.min()
    inlier_pct   = 100.0 * inliers.sum() / len(z)
    logger.info(f"  Background corr (grado {degree}): rango superficie = "
                f"{surf_range:.2f} um | inliers = {inlier_pct:.1f}%")

    corrected = delta_map.copy()
    corrected[valid] -= surf_valid
    return corrected, surf


def compute_difference_map(H_pre_reg: np.ndarray, H_post_reg: np.ndarray,
                            delta_z: float,
                            config: ComparisonConfig) -> Tuple[np.ndarray, LaserEffectStats]:
    nr = min(H_pre_reg.shape[0], H_post_reg.shape[0])
    nc = min(H_pre_reg.shape[1], H_post_reg.shape[1])
    pre  = H_pre_reg[:nr, :nc]
    post = H_post_reg[:nr, :nc] - delta_z

    valid     = _build_interior_mask(pre, post, config.border_erosion_px)
    delta_map = np.full((nr, nc), np.nan)
    delta_map[valid] = (post[valid] - pre[valid]) * 1000.0

    # Correccion de tendencia de fondo: elimina tilt/bow sistematico del
    # reposicionamiento de muestra entre medidas PRE y POST.
    if config.background_correction_degree > 0:
        delta_map, _ = correct_background_trend(
            delta_map,
            degree     = config.background_correction_degree,
            sigma_clip = config.background_sigma_clip,
        )

    thresh   = config.laser_effect_threshold_um
    abl_mask = _filter_clusters(valid & (delta_map < -thresh), config.min_cluster_area_px)
    dep_mask = _filter_clusters(valid & (delta_map >  thresh), config.min_cluster_area_px)

    abl_vals = delta_map[abl_mask]
    dep_vals = delta_map[dep_mask]
    d_vals   = delta_map[valid]
    d_mean   = np.nanmean(d_vals)
    d_std    = np.nanstd(d_vals)
    d_clean  = d_vals[np.abs(d_vals - d_mean) < config.outlier_sigma * d_std]

    pct     = config.stats_percentile
    max_abl = float(np.percentile(abl_vals,     pct)) if len(abl_vals) > 0 else 0.0
    max_dep = float(np.percentile(dep_vals, 100-pct)) if len(dep_vals) > 0 else 0.0

    pixel_area = config.pixel_size_mm ** 2

    logger.info(f"Ablacion  : {int(abl_mask.sum()):,} px | P{pct:.1f}% = {max_abl:.2f} um")
    logger.info(f"Deposicion: {int(dep_mask.sum()):,} px | P{100-pct:.1f}% = {max_dep:.2f} um")

    return delta_map, LaserEffectStats(
        analysis_area_mm2    = float(valid.sum()) * pixel_area,
        n_valid_points       = int(valid.sum()),
        mean_diff_um         = float(d_clean.mean()) if len(d_clean) > 0 else 0.0,
        std_diff_um          = float(d_clean.std())  if len(d_clean) > 0 else 0.0,
        rms_diff_um          = float(np.sqrt((d_clean**2).mean())) if len(d_clean) > 0 else 0.0,
        ablation_area_mm2    = float(abl_mask.sum()  * pixel_area),
        deposition_area_mm2  = float(dep_mask.sum()  * pixel_area),
        max_ablation_um      = max_abl,
        max_deposition_um    = max_dep,
        mean_ablation_um     = float(abl_vals.mean()) if len(abl_vals) > 0 else 0.0,
        mean_deposition_um   = float(dep_vals.mean()) if len(dep_vals) > 0 else 0.0,
        ablation_volume_mm3  = float(-abl_vals.sum() * pixel_area / 1000) if len(abl_vals) > 0 else 0.0,
        deposition_volume_mm3= float( dep_vals.sum() * pixel_area / 1000) if len(dep_vals) > 0 else 0.0,
        delta_z_pre_post_um  = float(delta_z * 1000),
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def _make_vr6000_cmap():
    colors    = ['#00008B', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000']
    positions = [0.0, 0.20, 0.38, 0.56, 0.73, 1.0]
    return LinearSegmentedColormap.from_list('vr6000', list(zip(positions, colors)), N=256)


def _smooth_profile(arr: np.ndarray, px: int) -> np.ndarray:
    if px <= 1:
        return arr.copy()
    out  = np.full_like(arr, np.nan)
    half = px // 2
    for i in range(len(arr)):
        seg   = arr[max(0, i - half): min(len(arr), i + half + 1)]
        valid = seg[~np.isnan(seg)]
        if valid.size > 0:
            out[i] = valid.mean()
    return out


def _safe_name(s: str) -> str:
    return re.sub(r'[^\w\-]', '_', s)


def plot_comparison(H_pre: np.ndarray, H_post: np.ndarray,
                    H_post_final: np.ndarray, delta_map: np.ndarray,
                    alignment: AlignmentResult, stats: LaserEffectStats,
                    pre_name: str, post_name: str,
                    pixel_size_mm: float, output_dir: str) -> str:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmap_h = _make_vr6000_cmap()
    px = pixel_size_mm

    vmin_h = np.nanpercentile(H_pre, 2)
    vmax_h = np.nanpercentile(H_pre, 98)

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1, 1.2, 0.5],
                            hspace=0.38, wspace=0.3)

    for ax, H, title in [
        (fig.add_subplot(gs[0, 0]), H_pre,  f'PRE\n{pre_name}'),
        (fig.add_subplot(gs[0, 1]), H_post, f'POST\n{post_name}'),
    ]:
        im = ax.imshow(H, cmap=cmap_h, aspect='auto', origin='upper',
                       extent=[0, H.shape[1]*px, H.shape[0]*px, 0],
                       vmin=vmin_h, vmax=vmax_h)
        if ax.get_subplotspec().get_geometry()[2] == 0:   # PRE panel only
            rect = Rectangle((max(alignment.x_pos, 0) * px,
                               max(alignment.y_pos, 0) * px),
                              H_post.shape[1] * px, H_post.shape[0] * px,
                              lw=2.5, edgecolor='red', facecolor='none', ls='--')
            ax.add_patch(rect)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='Height (mm)', shrink=0.8)

    ax3 = fig.add_subplot(gs[1, :])
    d_max  = max(abs(np.nanpercentile(delta_map, 1)),
                 abs(np.nanpercentile(delta_map, 99)), 5.0)
    norm_d = TwoSlopeNorm(vmin=-d_max, vcenter=0, vmax=d_max)
    im3 = ax3.imshow(delta_map, cmap='RdBu_r', norm=norm_d, aspect='auto',
                     origin='upper',
                     extent=[0, delta_map.shape[1]*px, delta_map.shape[0]*px, 0])
    mid_row = delta_map.shape[0] // 2
    ax3.axhline(mid_row * px, color='yellow', lw=1.5, ls='--', label='Perfil central')
    ax3.legend(fontsize=9, loc='upper right')
    plt.colorbar(im3, ax=ax3, label='delta POST-PRE (um)', shrink=0.8)
    ax3.text(1.18, 0.98,
             f"Area analisis: {stats.analysis_area_mm2:.2f} mm2\n"
             f"Pts validos  : {stats.n_valid_points:,}\n"
             f"dZ global    : {stats.delta_z_pre_post_um:+.2f} um\n"
             f"  -------\n"
             f"ABLACION\n"
             f"  Area  : {stats.ablation_area_mm2:.3f} mm2\n"
             f"  Max   : {stats.max_ablation_um:.2f} um\n"
             f"  Media : {stats.mean_ablation_um:.2f} um\n"
             f"  Vol.  : {stats.ablation_volume_mm3:.5f} mm3\n"
             f"  -------\n"
             f"DEPOSICION\n"
             f"  Area  : {stats.deposition_area_mm2:.3f} mm2\n"
             f"  Max   : {stats.max_deposition_um:.2f} um\n"
             f"  Vol.  : {stats.deposition_volume_mm3:.5f} mm3",
             transform=ax3.transAxes, fontsize=8.5, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, pad=0.6))
    ax3.set_title(f'MAPA DELTA (POST - PRE)  |  RMS={stats.rms_diff_um:.2f} um  |  '
                  f'Ablacion={stats.ablation_area_mm2:.3f} mm2',
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)')

    ax4a = fig.add_subplot(gs[2, 0])
    dv = delta_map[~np.isnan(delta_map)].flatten()
    ax4a.hist(dv, bins=200, color='steelblue', alpha=0.7, edgecolor='none')
    ax4a.axvline(0, color='k', lw=1.5)
    ax4a.axvline(stats.mean_diff_um, color='red', lw=1.5, ls='--',
                 label=f'Media={stats.mean_diff_um:.2f}um')
    ax4a.set_xlabel('delta POST-PRE (um)'); ax4a.set_ylabel('Pixeles')
    ax4a.set_title('Distribucion de diferencias', fontsize=11)
    ax4a.legend(fontsize=9); ax4a.grid(True, alpha=0.3)

    ax4b = fig.add_subplot(gs[2, 1])
    profile = delta_map[mid_row, :]
    x_prof  = np.arange(len(profile)) * px
    ax4b.plot(x_prof, profile, color='navy', lw=0.8)
    ax4b.axhline(0, color='k', lw=1.0, alpha=0.5)
    ax4b.fill_between(x_prof, profile, 0,
                      where=(np.nan_to_num(profile) < 0), alpha=0.4,
                      color='blue', label='Ablacion')
    ax4b.fill_between(x_prof, profile, 0,
                      where=(np.nan_to_num(profile) > 0), alpha=0.4,
                      color='red', label='Deposicion')
    ax4b.set_xlabel('X (mm)'); ax4b.set_ylabel('delta (um)')
    ax4b.set_title(f'Perfil central (fila {mid_row})', fontsize=11)
    ax4b.legend(fontsize=9); ax4b.grid(True, alpha=0.3)

    fig.suptitle(f'COMPARACION PRE/POST  --  {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=15, fontweight='bold', y=1.01)

    out = Path(output_dir) / f"comparison_{_safe_name(pre_name)}_vs_{_safe_name(post_name)}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {out.name}")
    return str(out)


def plot_delta_closeup(delta_map: np.ndarray, stats: LaserEffectStats,
                       pixel_size_mm: float, output_dir: str,
                       pre_name: str, post_name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    px    = pixel_size_mm
    h_mm  = delta_map.shape[0] * px
    w_mm  = delta_map.shape[1] * px
    fig_w = 14
    fig_h = max(6, min(fig_w / (w_mm / h_mm), 12))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    d_max  = max(abs(np.nanpercentile(delta_map, 1)),
                 abs(np.nanpercentile(delta_map, 99)), 5.0)
    norm_d = TwoSlopeNorm(vmin=-d_max, vcenter=0, vmax=d_max)
    im = ax.imshow(delta_map, cmap='RdBu_r', norm=norm_d, aspect='auto',
                   extent=[0, w_mm, h_mm, 0], origin='upper')

    filled = np.where(np.isnan(delta_map), 0, delta_map)
    levels = np.arange(-int(d_max), int(d_max) + 1, max(1, int(d_max) // 5))
    if len(levels) > 1:
        cs = ax.contour(np.linspace(0, w_mm, delta_map.shape[1]),
                        np.linspace(0, h_mm, delta_map.shape[0]),
                        filled, levels=levels,
                        colors='black', linewidths=0.5, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d um')

    plt.colorbar(im, ax=ax, label='delta POST-PRE (um)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'MAPA DELTA DETALLADO  |  {pre_name} -> {post_name}\n'
                 f'Ablacion max: {stats.max_ablation_um:.1f} um  |  '
                 f'Deposicion max: {stats.max_deposition_um:.1f} um  |  '
                 f'Vol. ablacion: {stats.ablation_volume_mm3:.5f} mm3',
                 fontsize=12, fontweight='bold')

    out = Path(output_dir) / f"delta_{_safe_name(pre_name)}_vs_{_safe_name(post_name)}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {out.name}")


def plot_horizontal_profiles(H_pre_reg: np.ndarray, H_post_final: np.ndarray,
                              delta_map: np.ndarray, delta_z: float,
                              pixel_size_mm: float, pre_name: str, post_name: str,
                              output_dir: str, config: ComparisonConfig) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    px  = pixel_size_mm
    n   = config.n_profiles
    spx = config.profile_smooth_px

    nr = min(H_pre_reg.shape[0], H_post_final.shape[0], delta_map.shape[0])
    nc = min(H_pre_reg.shape[1], H_post_final.shape[1], delta_map.shape[1])

    rows  = np.linspace(int(nr * 0.10), int(nr * 0.90), n, dtype=int)
    x_mm  = np.arange(nc) * px
    w_mm  = nc * px
    h_mm  = nr * px

    fig      = plt.figure(figsize=(20, 3.8 * n))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2],
                                  wspace=0.25, left=0.06, right=0.97,
                                  top=0.93, bottom=0.04)

    ax_map = fig.add_subplot(gs_outer[0, 0])
    d_max  = max(abs(np.nanpercentile(delta_map[:nr, :nc], 1)),
                 abs(np.nanpercentile(delta_map[:nr, :nc], 99)), 5.0)
    ax_map.imshow(delta_map[:nr, :nc], cmap='RdBu_r',
                  norm=TwoSlopeNorm(vmin=-d_max, vcenter=0, vmax=d_max),
                  aspect='auto', extent=[0, w_mm, h_mm, 0], origin='upper')
    ax_map.set_xlabel('X (mm)', fontsize=10)
    ax_map.set_ylabel('Y (mm)', fontsize=10)
    ax_map.set_title('Mapa DELTA\n(lineas = perfiles)', fontsize=11, fontweight='bold')

    colors   = plt.cm.tab10(np.linspace(0, 0.9, n))
    gs_right = gridspec.GridSpecFromSubplotSpec(
        n * 2, 1, subplot_spec=gs_outer[0, 1],
        hspace=0.08, height_ratios=[1.6, 1.0] * n)

    for idx, row in enumerate(rows):
        ax_map.axhline(row * px, color=colors[idx], lw=2.0, ls='--', alpha=0.9)
        ax_map.text(w_mm * 0.01, row * px - h_mm * 0.015,
                    f'P{idx+1}', fontsize=9, fontweight='bold', color=colors[idx])

        pre_um  = (_smooth_profile(H_pre_reg[row, :nc], spx) - np.nanmean(H_pre_reg[row, :nc])) * 1000
        post_um = (_smooth_profile(H_post_final[row, :nc] - delta_z, spx)
                   - np.nanmean(H_pre_reg[row, :nc])) * 1000
        dlt_s   = _smooth_profile(delta_map[row, :nc], spx)

        ax_h = fig.add_subplot(gs_right[idx * 2])
        ax_h.plot(x_mm, pre_um,  color='#1f77b4', lw=1.2, label='PRE',  alpha=0.9)
        ax_h.plot(x_mm, post_um, color='#d62728', lw=1.2, label='POST', alpha=0.9, ls='--')
        ax_h.axhline(0, color='gray', lw=0.6, ls=':', alpha=0.6)
        ax_h.set_ylabel('Altura (um)', fontsize=9)
        ax_h.set_title(f'Perfil {idx+1}  --  fila {row}  (Y = {row*px:.2f} mm)',
                       fontsize=10, fontweight='bold', color=colors[idx])
        ax_h.legend(fontsize=8, loc='upper right', ncol=2)
        ax_h.grid(True, alpha=0.25); ax_h.tick_params(labelbottom=False)

        ax_d = fig.add_subplot(gs_right[idx * 2 + 1], sharex=ax_h)
        ax_d.plot(x_mm, dlt_s, color='#2ca02c', lw=1.0, alpha=0.85)
        ax_d.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax_d.fill_between(x_mm, dlt_s, 0, where=(np.nan_to_num(dlt_s) < 0),
                          alpha=0.35, color='#4575b4', label='Ablacion')
        ax_d.fill_between(x_mm, dlt_s, 0, where=(np.nan_to_num(dlt_s) > 0),
                          alpha=0.35, color='#d73027', label='Deposicion')
        ax_d.set_ylabel('delta (um)', fontsize=9)
        ax_d.legend(fontsize=7, loc='upper right', ncol=2)
        ax_d.grid(True, alpha=0.25)
        if idx == n - 1:
            ax_d.set_xlabel('X (mm)', fontsize=10)
        else:
            ax_d.tick_params(labelbottom=False)

        vdlt = dlt_s[~np.isnan(dlt_s)]
        if vdlt.size > 0:
            ax_d.text(0.995, 0.97,
                      f"mean={vdlt.mean():+.1f} um\nmin ={vdlt.min():+.1f} um\nmax ={vdlt.max():+.1f} um",
                      transform=ax_d.transAxes, fontsize=7.5, ha='right', va='top',
                      family='monospace',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, pad=0.3))

    fig.suptitle(f'PERFILES HORIZONTALES PRE/POST\n{pre_name}  vs  {post_name}  --  '
                 f'{datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=13, fontweight='bold')

    out = Path(output_dir) / f"profiles_{_safe_name(pre_name)}_vs_{_safe_name(post_name)}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {out.name}")
    return str(out)


# ============================================================================
# REPORT CSV
# ============================================================================

def save_report_csv(alignment: AlignmentResult, stats: LaserEffectStats,
                    pre_name: str, post_name: str, output_dir: str) -> str:
    out = Path(output_dir) / f"report_{_safe_name(pre_name)}_vs_{_safe_name(post_name)}.csv"
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Parametro', 'Valor', 'Unidad'])
        w.writerow(['PRE',  pre_name,  ''])
        w.writerow(['POST', post_name, ''])
        w.writerow(['---ALINEACION---', '', ''])
        w.writerow(['Y shift (POST en PRE)', f"{alignment.y_mm:.3f}", 'mm'])
        w.writerow(['X shift (POST en PRE)', f"{alignment.x_mm:.3f}", 'mm'])
        w.writerow(['Confianza alineacion',   f"{alignment.confidence:.3f}", ''])
        w.writerow(['SNR alineacion',         f"{alignment.snr:.2f}", ''])
        w.writerow(['Solapamiento',           f"{alignment.overlap_ratio*100:.1f}", '%'])
        w.writerow(['theta_x',               f"{alignment.theta_x_deg:.4f}", 'deg'])
        w.writerow(['theta_y',               f"{alignment.theta_y_deg:.4f}", 'deg'])
        w.writerow(['R2 rotacion',           f"{alignment.r_squared:.4f}", ''])
        w.writerow(['Correccion rotacion',    'SI' if alignment.rotation_corrected else 'NO', ''])
        w.writerow(['Rotacion en plano',      f"{alignment.inplane_angle_deg:+.3f}", 'deg'])
        w.writerow(['---Z OFFSET---', '', ''])
        w.writerow(['dZ global (POST-PRE)', f"{stats.delta_z_pre_post_um:.3f}", 'um'])
        w.writerow(['---DIFERENCIAS GLOBALES---', '', ''])
        w.writerow(['Puntos validos',  str(stats.n_valid_points), 'px'])
        w.writerow(['Area analisis',   f"{stats.analysis_area_mm2:.4f}", 'mm2'])
        w.writerow(['Media diff',      f"{stats.mean_diff_um:.4f}", 'um'])
        w.writerow(['Std diff',        f"{stats.std_diff_um:.4f}", 'um'])
        w.writerow(['RMS diff',        f"{stats.rms_diff_um:.4f}", 'um'])
        w.writerow(['---ABLACION (POST < PRE)---', '', ''])
        w.writerow(['Area ablacion',      f"{stats.ablation_area_mm2:.4f}", 'mm2'])
        w.writerow(['Prof. max ablacion', f"{stats.max_ablation_um:.4f}", 'um'])
        w.writerow(['Media ablacion',     f"{stats.mean_ablation_um:.4f}", 'um'])
        w.writerow(['Volumen ablacion',   f"{stats.ablation_volume_mm3:.6f}", 'mm3'])
        w.writerow(['---DEPOSICION (POST > PRE)---', '', ''])
        w.writerow(['Area deposicion',    f"{stats.deposition_area_mm2:.4f}", 'mm2'])
        w.writerow(['Alt. max deposicion',f"{stats.max_deposition_um:.4f}", 'um'])
        w.writerow(['Volumen deposicion', f"{stats.deposition_volume_mm3:.6f}", 'mm3'])
    logger.info(f"Guardado: {out.name}")
    return str(out)


# ============================================================================
# PERFIL EXPORTADO  PRE vs POST
# ============================================================================

def _render_profile_comparison(distances: np.ndarray, x_arr: np.ndarray, y_arr: np.ndarray,
                                H_pre_reg: np.ndarray, H_post_final: np.ndarray,
                                delta_z: float, pixel_size_mm: float,
                                pre_r0: int, pre_c0: int,
                                pre_name: str, post_name: str,
                                output_dir: str, out_suffix: str,
                                titulo: str, smooth_px: int = 3) -> str:
    """
    Nucleo comun: dado un set de puntos (distance_mm, x_mm, y_mm) en el sistema
    de coordenadas de la interfaz de calibracion (x_min=0, y_min=-ny*px, y_max=0),
    muestrea PRE/POST alineados y genera el grafico comparativo.
    """
    from scipy.ndimage import map_coordinates as _mc

    px     = pixel_size_mm
    nr, nc = H_pre_reg.shape

    # row = -y_mm/px porque la interfaz usa el eje Y invertido respecto a la fila
    rows_f = -y_arr / px - pre_r0
    cols_f =  x_arr / px - pre_c0

    def _sample(H, rr, cc):
        nan_mask = np.isnan(H)
        filled   = np.where(nan_mask, 0.0, H)
        rr_c     = np.clip(rr, 0, nr - 1)
        cc_c     = np.clip(cc, 0, nc - 1)
        vals     = _mc(filled,   [rr_c, cc_c], order=1, mode='constant', cval=np.nan)
        n_frac   = _mc(nan_mask.astype(np.float32), [rr_c, cc_c],
                       order=1, mode='constant', cval=1.0)
        vals[(n_frac > 0.1) | (rr < 0) | (rr > nr-1) | (cc < 0) | (cc > nc-1)] = np.nan
        return vals

    h_pre_mm  = _sample(H_pre_reg,              rows_f, cols_f)
    h_post_mm = _sample(H_post_final - delta_z, rows_f, cols_f)

    valid = ~np.isnan(h_pre_mm) & ~np.isnan(h_post_mm)
    if valid.sum() < 10:
        logger.warning(f"Perfil ({out_suffix}): <10 puntos validos en la region solapamiento — sin grafico")
        return ""

    ref      = np.nanmean(h_pre_mm[valid])
    pre_um   = (h_pre_mm  - ref) * 1000.0
    post_um  = (h_post_mm - ref) * 1000.0
    delta_um = post_um - pre_um

    pre_s   = _smooth_profile(pre_um,   smooth_px)
    post_s  = _smooth_profile(post_um,  smooth_px)
    delta_s = _smooth_profile(delta_um, smooth_px)

    d = distances[valid]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
    fig.suptitle(f'{titulo} — PRE vs POST\n{pre_name}  vs  {post_name}  '
                 f'({datetime.now().strftime("%Y-%m-%d %H:%M")})',
                 fontsize=12, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(d, pre_s[valid],  color='#1f77b4', lw=1.5, label='PRE',  alpha=0.9)
    ax1.plot(d, post_s[valid], color='#d62728', lw=1.5, label='POST', alpha=0.9, ls='--')
    ax1.set_ylabel('Altura relativa (µm)', fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    ax2 = axes[1]
    dv = delta_s[valid]
    ax2.plot(d, dv, color='#2ca02c', lw=1.2, alpha=0.85)
    ax2.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax2.fill_between(d, dv, 0, where=(np.nan_to_num(dv) < 0),
                     alpha=0.35, color='#4575b4', label='Ablacion')
    ax2.fill_between(d, dv, 0, where=(np.nan_to_num(dv) > 0),
                     alpha=0.35, color='#d73027', label='Deposicion')
    ax2.set_ylabel('Δ altura (µm)', fontsize=11)
    ax2.set_xlabel('Distancia a lo largo del perfil (mm)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    dv_clean = dv[~np.isnan(dv)]
    if len(dv_clean) > 0:
        ax2.text(0.995, 0.97,
                 f"mean={dv_clean.mean():+.1f}  std={dv_clean.std():.1f}  "
                 f"min={dv_clean.min():+.1f}  max={dv_clean.max():+.1f}  µm",
                 transform=ax2.transAxes, fontsize=8, ha='right', va='top',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, pad=0.3))

    out = Path(output_dir) / f"{out_suffix}_{_safe_name(pre_name)}_vs_{_safe_name(post_name)}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {out.name}")
    return str(out)


def plot_profile_from_csv(H_pre_reg: np.ndarray, H_post_final: np.ndarray,
                           delta_z: float, pixel_size_mm: float,
                           pre_r0: int, pre_c0: int,
                           profile_csv_path: str,
                           pre_name: str, post_name: str,
                           output_dir: str,
                           smooth_px: int = 3) -> str:
    """
    Extrae y compara un perfil PRE/POST usando un CSV de trayectoria exportado
    desde la interfaz de calibracion.

    El CSV tiene columnas: distance_mm, height_mm, x_profile_mm, y_profile_mm
    Las coordenadas x/y estan en el sistema de coordenadas de la interfaz
    (x_min=0, y_min=-ny*px, y_max=0 — eje Y invertido respecto a la fila).

    H_pre_reg y H_post_final son la region de solapamiento (ya alineados 1:1).
    pre_r0, pre_c0: fila/col del primer pixel de esa region en el PRE original.
    """
    import csv as _csv

    distances, x_arr, y_arr = [], [], []
    try:
        with open(profile_csv_path, newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                distances.append(float(row['distance_mm']))
                x_arr.append(float(row['x_profile_mm']))
                y_arr.append(float(row['y_profile_mm']))
    except Exception as e:
        logger.warning(f"No se pudo cargar perfil CSV: {e}")
        return ""

    if not distances:
        logger.warning("Perfil CSV vacio")
        return ""

    return _render_profile_comparison(
        np.array(distances), np.array(x_arr), np.array(y_arr),
        H_pre_reg, H_post_final, delta_z, pixel_size_mm, pre_r0, pre_c0,
        pre_name, post_name, output_dir, out_suffix="profile_csv",
        titulo="PERFIL EXPORTADO", smooth_px=smooth_px,
    )


def plot_profile_between_points(H_pre_reg: np.ndarray, H_post_final: np.ndarray,
                                 delta_z: float, pixel_size_mm: float,
                                 pre_r0: int, pre_c0: int,
                                 p1: Tuple[float, float], p2: Tuple[float, float],
                                 pre_name: str, post_name: str,
                                 output_dir: str,
                                 n_samples: int = 400,
                                 smooth_px: int = 3) -> str:
    """
    Genera y compara un perfil PRE/POST a lo largo de la recta que une dos
    puntos (x_mm, y_mm) definidos manualmente por el usuario.

    p1, p2 deben estar en el MISMO sistema de coordenadas que usa la interfaz
    de calibracion / el DXF exportado (x_min=0, y_min=-ny*px, y_max=0), para
    poder reutilizar exactamente los mismos puntos de origen.
    """
    x1, y1 = p1
    x2, y2 = p2
    t = np.linspace(0.0, 1.0, n_samples)
    x_arr     = x1 + t * (x2 - x1)
    y_arr     = y1 + t * (y2 - y1)
    distances = t * float(np.hypot(x2 - x1, y2 - y1))

    return _render_profile_comparison(
        distances, x_arr, y_arr,
        H_pre_reg, H_post_final, delta_z, pixel_size_mm, pre_r0, pre_c0,
        pre_name, post_name, output_dir, out_suffix="profile_pts",
        titulo=f"PERFIL P1({x1:.2f},{y1:.2f}) -> P2({x2:.2f},{y2:.2f})", smooth_px=smooth_px,
    )


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

def compare_pre_post(pre_csv: str, post_csv: str,
                     output_dir: str,
                     config: Optional[ComparisonConfig] = None,
                     profile_csv: Optional[str] = None,
                     profile_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                     ) -> Tuple[AlignmentResult, LaserEffectStats]:
    if config is None:
        config = ComparisonConfig()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pre_name  = Path(pre_csv).stem
    post_name = Path(post_csv).stem

    logger.info("\n" + "="*70)
    logger.info(f"PRE : {pre_name}")
    logger.info(f"POST: {post_name}")
    logger.info("="*70)

    H_pre,  px = load_vr6000_csv(pre_csv)
    H_post, _  = load_vr6000_csv(post_csv)
    config.pixel_size_mm = px

    logger.info(f"PRE  {H_pre.shape}  ({H_pre.shape[1]*px:.2f} x {H_pre.shape[0]*px:.2f} mm)")
    logger.info(f"POST {H_post.shape}  ({H_post.shape[1]*px:.2f} x {H_post.shape[0]*px:.2f} mm)")

    coarse = coarse_search(H_pre, H_post, config)
    if coarse is None or coarse[3] < config.confidence_threshold:
        logger.warning("Confianza baja — usando (0,0) como fallback")
        y_pos, x_pos, snr, conf = 0, 0, 0.0, 0.0
    else:
        y_pos, x_pos, snr, conf = coarse
        y_pos, x_pos, snr, conf = fine_search(H_pre, H_post, y_pos, x_pos, config)

    logger.info(f"\nAlineacion traslacion: Y={y_pos:+d}px ({y_pos*px:+.3f}mm), "
                f"X={x_pos:+d}px ({x_pos*px:+.3f}mm)")
    logger.info(f"SNR={snr:.2f}, Confianza={conf:.3f}")

    # Rotacion en plano (eje Z): un giro de 0.05 deg provoca ~1 px de
    # desfase en el borde de un campo de 20 mm, impidiendo que la textura
    # superficial cancele y generando ruido creciente hacia los bordes.
    inplane_angle = 0.0
    if config.inplane_rotation_search:
        inplane_angle = search_inplane_rotation(H_pre, H_post, y_pos, x_pos, config)
        if abs(inplane_angle) > config.inplane_angle_step_deg * 0.5:
            logger.info(f"Aplicando rotacion en plano: {inplane_angle:+.2f} deg")
            H_post = _rotate_image(H_post, inplane_angle)
        else:
            logger.info("Rotacion en plano negligible — sin correccion")
            inplane_angle = 0.0

    slices = _overlap_region(H_pre.shape, H_post.shape, y_pos, x_pos)
    if slices is None:
        raise RuntimeError(f"Sin solapamiento con shift=({y_pos},{x_pos})")

    pre_sr, pre_sc, post_sr, post_sc = slices
    H_pre_reg  = H_pre[pre_sr, pre_sc]
    H_post_reg = H_post[post_sr, post_sc]

    ov_valid      = int((~np.isnan(H_pre_reg) & ~np.isnan(H_post_reg)).sum())
    overlap_ratio = ov_valid / max(int((~np.isnan(H_post_reg)).sum()), 1)
    logger.info(f"Solapamiento: {H_pre_reg.shape}  ({overlap_ratio:.1%})")

    delta_z = compute_z_offset(H_pre_reg, H_post_reg)
    logger.info(f"Z offset (POST-PRE): {delta_z*1000:.3f} um")

    theta_x, theta_y, r2 = compute_rotation(H_pre_reg, H_post_reg, delta_z, px)
    theta_mag = np.sqrt(theta_x**2 + theta_y**2)
    rotation_corrected = False
    H_post_final = H_post_reg.copy()

    if theta_mag > config.rotation_mag_threshold_deg and r2 > config.rotation_r2_threshold:
        logger.info(f"Correccion rotacion: theta={theta_mag:.4f} deg, R2={r2:.3f}")
        H_post_final = apply_rotation_correction(H_post_reg, theta_x, theta_y, px)
        delta_z = compute_z_offset(H_pre_reg, H_post_final)
        rotation_corrected = True
    else:
        logger.info(f"Rotacion omitida (theta={theta_mag:.4f} deg, R2={r2:.3f})")

    alignment = AlignmentResult(
        y_pos=y_pos, x_pos=x_pos,
        y_mm=y_pos * px, x_mm=x_pos * px,
        confidence=conf, snr=snr,
        overlap_ratio=overlap_ratio,
        delta_z_mm=delta_z,
        theta_x_deg=theta_x, theta_y_deg=theta_y, r_squared=r2,
        rotation_corrected=rotation_corrected,
        inplane_angle_deg=inplane_angle,
    )

    delta_map, stats = compute_difference_map(H_pre_reg, H_post_final, delta_z, config)

    plot_comparison(H_pre, H_post, H_post_final, delta_map,
                    alignment, stats, pre_name, post_name, px, output_dir)
    plot_delta_closeup(delta_map, stats, px, output_dir, pre_name, post_name)
    plot_horizontal_profiles(H_pre_reg, H_post_final, delta_map,
                              delta_z, px, pre_name, post_name, output_dir, config)
    if profile_csv:
        plot_profile_from_csv(H_pre_reg, H_post_final, delta_z, px,
                              pre_sr.start, pre_sc.start,
                              profile_csv, pre_name, post_name, output_dir)
    if profile_points:
        plot_profile_between_points(H_pre_reg, H_post_final, delta_z, px,
                                    pre_sr.start, pre_sc.start,
                                    profile_points[0], profile_points[1],
                                    pre_name, post_name, output_dir)
    save_report_csv(alignment, stats, pre_name, post_name, output_dir)

    logger.info("\n" + "="*70)
    logger.info(f"  Alineacion: Y={alignment.y_mm:+.3f}mm, X={alignment.x_mm:+.3f}mm  "
                f"rot_plano={alignment.inplane_angle_deg:+.2f}deg")
    logger.info(f"  Confianza : {alignment.confidence:.3f}  SNR: {alignment.snr:.2f}")
    logger.info(f"  Solape    : {alignment.overlap_ratio:.1%}")
    logger.info(f"  dZ global : {stats.delta_z_pre_post_um:+.3f} um")
    logger.info(f"  RMS diff  : {stats.rms_diff_um:.3f} um")
    logger.info(f"  Ablacion  : {stats.ablation_area_mm2:.3f}mm2  "
                f"max={stats.max_ablation_um:.2f}um")
    logger.info("="*70)

    return alignment, stats


# ============================================================================
# MAIN  —  anadir aqui nuevos datasets
# ============================================================================

if __name__ == "__main__":

    BASE = Path(__file__).resolve().parent
    OUT  = str(BASE / "resultados")

    cfg = ComparisonConfig(
        pixel_size_mm             = 0.011814,
        smoothing_sigma           = 1.0,
        coarse_downsample         = 4,
        min_snr                   = 1.5,
        confidence_threshold      = 0.30,
        min_overlap_ratio         = 0.40,
        rotation_r2_threshold     = 0.20,
        rotation_mag_threshold_deg= 0.001,
        outlier_sigma             = 3.0,
        laser_effect_threshold_um = 2.0,
        border_erosion_px         = 50,
        min_cluster_area_px       = 25,
        stats_percentile          = 0.5,
        n_profiles                = 5,
        profile_smooth_px         = 5,
        max_shift_px              = 30,
        background_correction_degree = 1,   # plano: elimina tilt reposicionamiento
        background_sigma_clip        = 2.5,
    )

    # ------------------------------------------------------------------ #
    # GRUPOS DE COMPARACION                                                #
    # Cada entrada: (ruta_pre, ruta_post, etiqueta)                        #
    #        o     (ruta_pre, ruta_post, etiqueta, ruta_perfil_csv)        #
    # Rutas relativas a la carpeta de este script.                         #
    # ------------------------------------------------------------------ #
    GRUPOS = [
        {
            "titulo": "parte_2 — Region 2 (17-4 PH)",
            "comparaciones": [
                ("parte_2/17-4 PH Region 2_Height.csv",
                 "parte_2/17-4PH_Region2_Pass1_Height.csv",
                 "Virgen vs Pasada1"),
                ("parte_2/17-4 PH Region 2_Height.csv",
                 "parte_2/17-4PH_Region2_Pass2_Height.csv",
                 "Virgen vs Pasada2"),
                ("parte_2/17-4PH_Region2_Pass1_Height.csv",
                 "parte_2/17-4PH_Region2_Pass2_Height.csv",
                 "Pasada1 vs Pasada2"),
            ],
        },
        {
            "titulo": "steel_parte1 — Steel Pre/Post tratamiento",
            "comparaciones": [
                ("steel_parte1/steel_pretreatment_Height.csv",
                 "steel_parte1/steel_posttreatment_Height.csv",
                 "Pre vs Post tratamiento"),
            ],
        },
        {
            "titulo": "csv_entrada — Prueba1 Steel vs Postlinea1 Steel",
            "comparaciones": [
                ("../csv_entrada/prueba1_steel_Height.csv",
                 "../csv_entrada/postlinea1_steel_Height.csv",
                 "Prueba1 vs Postlinea1 Steel",
                 "../salidas/csv/Perfil_prueba1_steel_Height_P1_P3_20260629_184209.csv"),
                # Mismo par, pero definiendo el perfil a mano con dos puntos
                # (x_mm, y_mm) en el sistema de coordenadas de la interfaz de
                # calibracion / DXF (x_min=0, y_min=-alto_mm, y_max=0).
                # Cambia estos puntos por los que quieras inspeccionar:
                # ("../csv_entrada/prueba1_steel_Height.csv",
                #  "../csv_entrada/postlinea1_steel_Height.csv",
                #  "Perfil manual P1-P2",
                #  ((4.06, -3.14), (4.55, -3.31))),
            ],
        },
        # --- ANADIR NUEVOS DATASETS AQUI ---
        # {
        #     "titulo": "nombre_carpeta — descripcion",
        #     "comparaciones": [
        #         ("nombre_carpeta/archivo_pre.csv",
        #          "nombre_carpeta/archivo_post.csv",
        #          "Etiqueta comparacion"),
        #         # con perfil CSV exportado de la interfaz:
        #         # ("pre.csv", "post.csv", "etiqueta", "ruta/perfil.csv"),
        #         # o con dos puntos definidos a mano ((x1,y1),(x2,y2)):
        #         # ("pre.csv", "post.csv", "etiqueta", ((x1, y1), (x2, y2))),
        #     ],
        # },
    ]

    resumen = []

    for grupo in GRUPOS:
        print(f"\n{'#'*70}")
        print(f"  {grupo['titulo']}")
        print(f"{'#'*70}")

        for comp in grupo["comparaciones"]:
            pre_rel, post_rel, etiqueta = comp[0], comp[1], comp[2]
            profile_arg  = comp[3] if len(comp) > 3 else None
            pre_path     = str(BASE / pre_rel)
            post_path    = str(BASE / post_rel)

            profile_path   = None
            profile_points = None
            if isinstance(profile_arg, str):
                profile_path = str(BASE / profile_arg)
            elif profile_arg is not None:
                profile_points = profile_arg

            print(f"\n{'='*70}")
            print(f"  {etiqueta}")
            print(f"{'='*70}")

            try:
                alignment, stats = compare_pre_post(
                    pre_csv        = pre_path,
                    post_csv       = post_path,
                    output_dir     = OUT,
                    config         = cfg,
                    profile_csv    = profile_path,
                    profile_points = profile_points,
                )
                resumen.append({
                    "grupo": grupo["titulo"],
                    "comparacion": etiqueta,
                    "shift_y_mm": alignment.y_mm,
                    "shift_x_mm": alignment.x_mm,
                    "rms_um": stats.rms_diff_um,
                    "abl_max_um": stats.max_ablation_um,
                    "abl_area_mm2": stats.ablation_area_mm2,
                })
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n\n{'#'*70}")
    print("  RESUMEN GLOBAL")
    print(f"{'#'*70}")
    print(f"{'Grupo':<35} {'Comparacion':<22} {'RMS':>8} {'Abl.max':>9} {'Abl.area':>10}")
    print("-" * 90)
    for r in resumen:
        print(f"{r['grupo'][:34]:<35} {r['comparacion'][:21]:<22} "
              f"{r['rms_um']:>7.2f}um {r['abl_max_um']:>8.1f}um {r['abl_area_mm2']:>9.2f}mm2")

    print(f"\nArchivos en: {OUT}")
