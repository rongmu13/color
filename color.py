# app.py —— すべて実数値（float32）で出力する版
# UIは日本語、コメントは中国語
import io
from pathlib import Path
import numpy as np
import streamlit as st

from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine

# skimage: 真实色彩空间转换
from skimage.color import (
    rgb2lab, rgb2hsv, rgb2xyz, rgb2luv, rgb2ycbcr, rgb2gray
)
import colorsys

# ---------------------------
# 基本設定（页面设置）
# ---------------------------
st.set_page_config(page_title="色空間変換 Shinshu Univ. R.Y.", layout="wide")

st.title("RGB → 色空間変換 Shinshu Univ. R.Y.")
st.caption(
    "RGB画像（GeoTIFF/TIFF/JPG/PNG）をアップロードし、右側で変換先の色空間を選択してください。"
    "変換結果は **float32 の GeoTIFF（実数値）** としてダウンロードできます。"
    "GeoTIFF 入力時は投影・アフィン等の地理参照を可能な限り引き継ぎます。"
)

with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間（すべて実数値で出力）",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0,
        help="全モードで float32 の実数値 GeoTIFF を出力。プレビューは表示のために自動スケーリングします。"
    )
    # 仅用于预览的拉伸，不影响下载
    p_low = st.slider("下位パーセンタイル（プレビュー用）", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上位パーセンタイル（プレビュー用）", 90.0, 100.0, 99.0, 0.5)
    preview_max = st.slider("プレビューの最大辺サイズ（px）", 256, 2048, 1024, 128)

uploaded = st.file_uploader(
    "画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    help="アップロード上限にご注意。解析用の出力は常に float32 GeoTIFF です。"
)

# ---------------------------
# ユーティリティ（工具函数）
# ---------------------------
def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    """把任意 dtype/范围的数组按百分位拉伸到 uint8（仅用于预览显示）"""
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255.0 + 0.5).astype(np.uint8)

def ensure_hwc(rgb_like):
    """把 (bands, H, W) 或 (H, W, bands) 统一成 (H, W, C)"""
    if rgb_like.ndim == 3 and rgb_like.shape[0] in (3,4) and (rgb_like.shape[2] not in (3,4)):
        rgb_like = np.transpose(rgb_like, (1, 2, 0))
    return rgb_like

def _to01_float(rgb_any):
    """任意 dtype 的 RGB 转 0..1 的 float32"""
    arr = rgb_any.astype(np.float32)
    if rgb_any.dtype == np.uint16:
        arr /= 65535.0
    elif rgb_any.dtype == np.uint8:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def rgb2hls_np(rgb01):
    """colorsys の HLS を使った逐像素转换，输出 H,L,S ∈ [0..1]"""
    h, w, _ = rgb01.shape
    out = np.empty_like(rgb01, dtype=np.float32)
    flat_in = rgb01.reshape(-1, 3)
    flat_out = out.reshape(-1, 3)
    for i in range(flat_in.shape[0]):
        r, g, b = flat_in[i]
        h_, l_, s_ = colorsys.rgb_to_hls(float(r), float(g), float(b))
        flat_out[i] = (h_, l_, s_)
    return out

def convert_colorspace_real(img_rgb_any, mode):
    """
    真实值转换（所有模式都返回 float32）：
      LAB:  L[0..100], a/b ≈ [-128..127]
      HSV:  H,S,V ∈ [0..1]
      HLS:  H,L,S ∈ [0..1]
      XYZ:  X,Y,Z ∈ [0..1]（相対D65）
      LUV:  L ∈ [0..100], u/v 为浮点（色域依赖）
      YCrCb: skimage 定义为 [0..255] 浮点
      Gray:  [0..1]（线性亮度）
    """
    rgb01 = _to01_float(img_rgb_any)
    if mode == "LAB":
        return rgb2lab(rgb01).astype(np.float32)
    elif mode == "HSV":
        return rgb2hsv(rgb01).astype(np.float32)
    elif mode == "HLS":
        return rgb2hls_np(rgb01).astype(np.float32)
    elif mode == "YCrCb":
        return rgb2ycbcr(rgb01).astype(np.float32)
    elif mode == "XYZ":
        return rgb2xyz(rgb01).astype(np.float32)
    elif mode == "LUV":
        return rgb2luv(rgb01).astype(np.float32)
    elif mode == "Gray":
        g = rgb2gray(rgb01).astype(np.float32)
        return g[..., None]
    else:
        raise ValueError("Unsupported color space")

def infer_is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))

# ---------------------------
# メイン処理（主逻辑）
# ---------------------------
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください（GeoTIFF/TIFF/JPG/PNG に対応）。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)

# ===== A：GeoTIFF/TIFF（地理参照あり） =====
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF 処理（実数値 float32 出力）")
    try:
        with rasterio.MemoryFile(uploaded.read()) as memfile:
            with memfile.open() as src:
                h, w = src.height, src.width
                count = src.count
                profile = src.profile.copy()
                tags_global = src.tags()

                bands_to_read = min(3, count)
                st.write(f"原画像サイズ：{w} × {h}／バンド数：{count}（変換に使用：先頭{bands_to_read}バンド）")

                block_size = 1024
                out_channels = 1 if color_space == "Gray" else 3

                # 输出 float32
                out_profile = profile.copy()
                out_profile.update({
                    "count": out_channels,
                    "dtype": "float32",
                    "compress": "deflate",
                    "predictor": 3  # 浮点预测器
                })

                out_mem = rasterio.MemoryFile()
                with out_mem.open(**out_profile) as dst:
                    for y in range(0, h, block_size):
                        for x in range(0, w, block_size):
                            win = Window(col_off=x, row_off=y,
                                         width=min(block_size, w - x),
                                         height=min(block_size, h - y))
                            arr = src.read(indexes=list(range(1, bands_to_read + 1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:, :, -1]] * (3 - arr.shape[2])
                                arr = np.concatenate([arr] + [p[..., None] for p in pads], axis=2)

                            out_block = convert_colorspace_real(arr, color_space)  # float32 真实值
                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], indexes=ch + 1, window=win)

                    # 标签/地理参照
                    dst.update_tags(**tags_global)
                    # Band description（便于在 QGIS 里识别）
                    if color_space == "LAB":
                        dst.set_band_description(1, "L_0-100")
                        dst.set_band_description(2, "a_-128_127")
                        dst.set_band_description(3, "b_-128_127")
                    elif color_space == "HSV":
                        dst.set_band_description(1, "H_0-1")
                        dst.set_band_description(2, "S_0-1")
                        dst.set_band_description(3, "V_0-1")
                    elif color_space == "HLS":
                        dst.set_band_description(1, "H_0-1")
                        dst.set_band_description(2, "L_0-1")
                        dst.set_band_description(3, "S_0-1")
                    elif color_space == "XYZ":
                        dst.set_band_description(1, "X_0-1")
                        dst.set_band_description(2, "Y_0-1")
                        dst.set_band_description(3, "Z_0-1")
                    elif color_space == "LUV":
                        dst.set_band_description(1, "L_0-100")
                        dst.set_band_description(2, "u_float")
                        dst.set_band_description(3, "v_float")
                    elif color_space == "YCrCb":
                        dst.set_band_description(1, "Y_0-255")
                        dst.set_band_description(2, "Cb_0-255")
                        dst.set_band_description(3, "Cr_0-255")
                    elif color_space == "Gray":
                        dst.set_band_description(1, "Gray_0-1")

                # 预览（仅显示用：百分位拉伸到 8bit）
                with out_mem.open() as preview_ds:
                    prev = preview_ds.read(indexes=list(range(1, out_channels + 1)))
                    prev = ensure_hwc(prev)
                    prev = percentile_scale_to_uint8(prev, p_low, p_high)
                    if prev.shape[2] == 1:
                        prev = np.repeat(prev, 3, axis=2)
                    h0, w0 = prev.shape[:2]
                    scale = min(preview_max / max(h0, w0), 1.0)
                    if scale < 1.0:
                        prev = (prev if prev.dtype == np.uint8 else prev.astype(np.uint8))
                        prev = cv2.resize(prev, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
                    st.image(prev, caption=f"プレビュー（{color_space} 実数値 → 表示用スケーリング）", use_container_width=True)

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}_float32.tif"
                st.download_button(
                    "⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
                    data=out_bytes,
                    file_name=out_name,
                    mime="image/tiff"
                )
        st.success("✅ 変換が完了しました。出力はすべて実数値（float32）の GeoTIFF です。")
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

# ===== B：JPEG/PNG（EXIF/GPSは保持不能。実数値GeoTIFFで出力） =====
else:
    st.subheader("🧭 JPEG/PNG 処理（実数値 float32 GeoTIFF で出力）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)  # uint8

        out_img = convert_colorspace_real(rgb, color_space)  # float32

        # 预览（仅显示用）
        prev = percentile_scale_to_uint8(out_img, p_low, p_high)
        if prev.ndim == 2:
            prev = prev[..., None]
        if prev.shape[2] == 1:
            prev = np.repeat(prev, 3, axis=2)
        h0, w0 = prev.shape[:2]
        scale = min(preview_max / max(h0, w0), 1.0)
        if scale < 1.0:
            prev = cv2.resize(prev, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
        st.image(prev, caption=f"プレビュー（{color_space} 実数値 → 表示用スケーリング）", use_container_width=True)

        # 写成 float32 GeoTIFF（无地理参照）
        h, w = out_img.shape[:2]
        out_channels = 1 if out_img.ndim == 2 else out_img.shape[2]
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": out_channels,
            "dtype": "float32",
            "crs": None,
            "transform": Affine.translation(0, 0) * Affine.scale(1, 1),
            "compress": "deflate",
            "predictor": 3
        }
        mem = rasterio.MemoryFile()
        with mem.open(**profile) as dst:
            if out_channels == 1:
                dst.write(out_img.astype(np.float32), 1)
                if color_space == "Gray":
                    dst.set_band_description(1, "Gray_0-1")
            else:
                for ch in range(out_channels):
                    dst.write(out_img[:, :, ch].astype(np.float32), ch+1)
                # Band 描述
                if color_space == "LAB":
                    dst.set_band_description(1, "L_0-100")
                    dst.set_band_description(2, "a_-128_127")
                    dst.set_band_description(3, "b_-128_127")
                elif color_space == "HSV":
                    dst.set_band_description(1, "H_0-1")
                    dst.set_band_description(2, "S_0-1")
                    dst.set_band_description(3, "V_0-1")
                elif color_space == "HLS":
                    dst.set_band_description(1, "H_0-1")
                    dst.set_band_description(2, "L_0-1")
                    dst.set_band_description(3, "S_0-1")
                elif color_space == "XYZ":
                    dst.set_band_description(1, "X_0-1")
                    dst.set_band_description(2, "Y_0-1")
                    dst.set_band_description(3, "Z_0-1")
                elif color_space == "LUV":
                    dst.set_band_description(1, "L_0-100")
                    dst.set_band_description(2, "u_float")
                    dst.set_band_description(3, "v_float")
                elif color_space == "YCrCb":
                    dst.set_band_description(1, "Y_0-255")
                    dst.set_band_description(2, "Cb_0-255")
                    dst.set_band_description(3, "Cr_0-255")

        out_bytes = mem.read()
        out_name = Path(filename).stem + f"_{color_space}_float32.tif"
        st.download_button("⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
                           data=out_bytes, file_name=out_name, mime="image/tiff")
        st.info("注：JPEG/PNG は EXIF/GPS を保持しません。本アプリは解析用に float32 GeoTIFF を出力します。")
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

st.caption("本アプリは研究・教育用です。")
