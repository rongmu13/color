# main/color.py —— RGB → 多色彩空间，全部输出真实值 float32 GeoTIFF（No skimage）
# Author: Shinshu Univ. R.Y.  |  Research/Education
import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

import cv2
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# 可选：JPEG EXIF 复制（仅用于可视化导出的 JPEG）
try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False

# ---------------- UI ----------------
st.set_page_config(page_title="RGB → 色空間変換（全真实值）", layout="wide")
st.title("RGB → 色空間変換（全真实值 float32 GeoTIFF）")
st.sidebar.success("Build: all-float32 • opencv-only")

with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間（全部真实值输出）",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0
    )
    # 预览增强：仅影响右侧显示与 PNG/JPEG 预览，不改 GeoTIFF 数值
    p_low = st.slider("预览百分位下界（仅用于 XYZ/LUV/可选增强）", 0.0, 10.0, 2.0, 0.5)
    p_high = st.slider("预览百分位上界（仅用于 XYZ/LUV/可选增强）", 90.0, 100.0, 98.0, 0.5)
    preview_max = st.slider("プレビュー最大辺(px)", 256, 4096, 1024, 128)

uploaded = st.file_uploader(
    "画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif","tiff","jpg","jpeg","png"]
)

# --------------- Utils ---------------
def ensure_hwc(x):
    if x.ndim == 3 and x.shape[0] in (1,3,4) and (x.shape[2] not in (1,3,4)):
        x = np.transpose(x, (1,2,0))
    return x

def to_rgb01(arr):
    """任意位深 RGB → float32 [0,1]（不做百分位拉伸，保持真实相对亮度）"""
    if arr.dtype in (np.float32, np.float64):
        x = np.clip(arr, 0, 1).astype(np.float32)
    elif arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn: mx = mn + 1.0
        x = (arr - mn) / (mx - mn)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def percent_stretch_u8(x, p_lo=2, p_hi=98):
    """用于预览：对 float32 数组做百分位拉伸到 uint8"""
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo: hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255.0 + 0.5).astype(np.uint8)

# ---- 各空间：真实值转换（返回 float32 HWC） ----
def rgb_to_lab_real(rgb01):   # L[0..100], a,b≈[-128..127]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB).astype(np.float32)

def rgb_to_hsv_real(rgb01):   # H[0..360), S,V[0..1]
    out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HSV).astype(np.float32)
    return out

def rgb_to_hls_real(rgb01):   # H[0..360), L,S[0..1]
    out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HLS).astype(np.float32)
    return out

def rgb_to_ycrcb_real(rgb01): # Y,Cr,Cb ∈ [0,1]（OpenCV 对 float 域）
    out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    return out

def rgb_to_xyz_real(rgb01):   # 近似 [0,1]，可能轻微越界，保留
    out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2XYZ).astype(np.float32)
    return out

def rgb_to_luv_real(rgb01):   # L[0..100]，u,v 可为负
    out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Luv).astype(np.float32)
    return out

def rgb_to_gray_real(rgb01):  # [0,1]，单通道
    g = cv2.cvtColor(rgb01, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g[..., None]

# ---- 预览映射（真实值 → uint8 RGB） ----
def preview_from_real(cs_name, arr_f32, p_lo=2, p_hi=98):
    if cs_name == "LAB":
        L = np.clip(arr_f32[:,:,0], 0, 100) * (255.0/100.0)
        a = np.clip(arr_f32[:,:,1] + 127.0, 0, 255)
        b = np.clip(arr_f32[:,:,2] + 127.0, 0, 255)
        img = np.stack([L,a,b], axis=2).astype(np.uint8)
    elif cs_name in ("HSV","HLS"):
        H = np.mod(arr_f32[:,:,0], 360.0) * (255.0/360.0)
        C1 = np.clip(arr_f32[:,:,1], 0, 1) * 255.0
        C2 = np.clip(arr_f32[:,:,2], 0, 1) * 255.0
        img = np.stack([H, C1, C2], axis=2).astype(np.uint8)
    elif cs_name in ("YCrCb","Gray"):
        # 0..1 → 0..255
        if arr_f32.shape[2] == 1:
            g = np.clip(arr_f32[:,:,0], 0, 1)
            img = np.repeat((g*255.0+0.5).astype(np.uint8)[...,None], 3, axis=2)
        else:
            img = np.clip(arr_f32, 0, 1)
            img = (img*255.0 + 0.5).astype(np.uint8)
    elif cs_name in ("XYZ","LUV"):
        # 动态范围不固定：用百分位拉伸
        chs = []
        for c in range(arr_f32.shape[2]):
            chs.append(percent_stretch_u8(arr_f32[:,:,c], p_lo, p_hi))
        img = np.stack(chs, axis=2)
    else:
        raise ValueError("Unsupported preview colorspace")
    return img

def pil_preview(img_u8, max_side=1024):
    h, w = img_u8.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img_u8 = cv2.resize(img_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_u8)

def copy_exif_jpeg(src_bytes, dst_bytes_io):
    if not HAS_PIEXIF: return False
    try:
        exif_dict = piexif.load(src_bytes)
        exif_bytes = piexif.dump(exif_dict)
        img = Image.open(io.BytesIO(dst_bytes_io.getvalue()))
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", exif=exif_bytes, quality=95)
        dst_bytes_io.seek(0); dst_bytes_io.truncate(0); dst_bytes_io.write(out_io.getvalue())
        return True
    except Exception:
        return False

def infer_is_tiff(name): return name.lower().endswith((".tif",".tiff"))
def infer_is_jpeg(name): return name.lower().endswith((".jpg",".jpeg"))

# ---------------- Main ----------------
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# 统一转换函数映射
CONV = {
    "LAB":   rgb_to_lab_real,
    "HSV":   rgb_to_hsv_real,
    "HLS":   rgb_to_hls_real,
    "YCrCb": rgb_to_ycrcb_real,
    "XYZ":   rgb_to_xyz_real,
    "LUV":   rgb_to_luv_real,
    "Gray":  rgb_to_gray_real,
}

# ---- A: GeoTIFF/TIFF ----
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF（全部真实值 float32）")
    try:
        with rasterio.MemoryFile(uploaded.read()) as mem:
            with mem.open() as src:
                h, w, count = src.height, src.width, src.count
                bands_to_read = min(3, count)
                st.write(f"原画像サイズ：{w}×{h}／バンド数：{count}（先頭{bands_to_read}使用）")

                block = 1024
                out_channels = 1 if color_space == "Gray" else 3
                profile = src.profile.copy()
                profile.update({"count": out_channels, "dtype": "float32",
                                "compress":"deflate","predictor":2})

                tags_global = src.tags()
                tags_per_band = [src.tags(i+1) for i in range(min(3,count))]

                out_mem = rasterio.MemoryFile()
                with out_mem.open(**profile) as dst:
                    for y in range(0, h, block):
                        for x in range(0, w, block):
                            win = Window(x, y, min(block, w-x), min(block, h-y))
                            arr = src.read(indexes=list(range(1, bands_to_read+1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:,:,-1]] * (3 - arr.shape[2])
                                arr = np.concatenate([arr] + [p[...,None] for p in pads], axis=2)

                            rgb01 = to_rgb01(arr)
                            out_block = CONV[color_space](rgb01)  # float32

                            for ch in range(out_channels):
                                dst.write(out_block[:,:,ch], ch+1, window=win)

                    dst.update_tags(**tags_global)
                    for ch in range(min(len(tags_per_band), out_channels)):
                        try: dst.update_tags(ch+1, **tags_per_band[ch])
                        except Exception: pass

                    # 带上合理的波段说明
                    try:
                        if color_space == "LAB":
                            dst.set_band_description(1, "L (0-100)")
                            dst.set_band_description(2, "a (~-128..127)")
                            dst.set_band_description(3, "b (~-128..127)")
                        elif color_space == "HSV":
                            for i, nm in enumerate(["H (0..360)","S (0..1)","V (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "HLS":
                            for i, nm in enumerate(["H (0..360)","L (0..1)","S (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "YCrCb":
                            for i, nm in enumerate(["Y (0..1)","Cr (0..1)","Cb (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "XYZ":
                            for i, nm in enumerate(["X","Y","Z"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "LUV":
                            for i, nm in enumerate(["L (0..100)","u","v"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "Gray":
                            dst.set_band_description(1, "Gray (0..1)")
                    except Exception:
                        pass

                # 预览
                with out_mem.open() as prev_ds:
                    scale = max(prev_ds.width, prev_ds.height) / max(preview_max, 1)
                    scale = max(scale, 1.0)
                    prev = prev_ds.read(
                        indexes=list(range(1, out_channels+1)),
                        out_shape=(out_channels,
                                   int(prev_ds.height/scale),
                                   int(prev_ds.width/scale)),
                        resampling=Resampling.average
                    )
                    prev = ensure_hwc(prev).astype(np.float32)
                    prev_u8 = preview_from_real(color_space, prev, p_low, p_high)
                st.image(pil_preview(prev_u8, max_side=preview_max),
                         caption=f"プレビュー（{color_space} / float32 真实值）",
                         width="stretch")

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}_float32.tif"
                st.download_button("⬇️ 変換結果をダウンロード（GeoTIFF: float32）",
                                   data=out_bytes, file_name=out_name, mime="image/tiff")
        st.success("✅ 全部真实值 float32 GeoTIFF 已生成。")
    except Exception as e:
        st.exception(e)

# ---- B: JPEG/PNG ----
else:
    st.subheader("🧭 JPEG/PNG（可视化预览；数值请用 GeoTIFF）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)

        rgb01 = to_rgb01(rgb)
        real = CONV[color_space](rgb01)  # float32
        prev_u8 = preview_from_real(color_space, real, p_low, p_high)

        st.image(pil_preview(prev_u8, max_side=preview_max),
                 caption=f"プレビュー（{color_space} / float32 真实值 → 映射显示）",
                 width="stretch")

        # 提供 float32 GeoTIFF（无地理参照时用单位像素变换）
        profile = {"driver":"GTiff","height":real.shape[0],"width":real.shape[1],
                   "count":real.shape[2],"dtype":"float32","crs":None,
                   "transform":from_origin(0,0,1,1),"compress":"deflate","predictor":2}
        mem = rasterio.io.MemoryFile()
        with mem.open(**profile) as ds:
            for i in range(real.shape[2]):
                ds.write(real[:,:,i], i+1)
        st.download_button("⬇️ GeoTIFF（float32 真实值）をダウンロード",
                           data=mem.read(),
                           file_name=Path(filename).stem + f"_{color_space}_float32.tif",
                           mime="image/tiff")

        # PNG/JPEG 仅作可视化（不含真实值）
        png_io = io.BytesIO(); Image.fromarray(prev_u8).save(png_io, format="PNG", compress_level=6)
        st.download_button("⬇️ PNG（可視化）", data=png_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}_preview.png", mime="image/png")

        jpg_io = io.BytesIO(); Image.fromarray(prev_u8).save(jpg_io, format="JPEG", quality=95)
        exif_ok = False
        if is_jpeg and HAS_PIEXIF: exif_ok = copy_exif_jpeg(src_bytes, jpg_io)
        label = "⬇️ JPEG（可視化；EXIF 尽量保留）"
        if is_jpeg: label += " ✅EXIFコピー済" if exif_ok else " ⚠EXIFコピー不可"
        st.download_button(label, data=jpg_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}_preview.jpg", mime="image/jpeg")

        st.info("⚠️ 数值分析请使用上面的 GeoTIFF（float32）。PNG/JPEG 仅用于预览/分享。")
    except Exception as e:
        st.exception(e)
