# app.py —— すべて実数値（float32）で出力 / OpenCV版（scikit-image不要）
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

import cv2

# ---------------------------
# 基本設定（页面设置）
# ---------------------------
st.set_page_config(page_title="RGB → 色空間変換（実数値出力）Shinshu Univ. R.Y.", layout="wide")

st.title("RGB → 色空間変換（実数値出力）Shinshu Univ. R.Y.")
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

def infer_is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))

def to_float01(rgb_any):
    """
    任意 dtype 的 RGB → float32 的 [0,1].
    支持 uint8 / uint16 / float.
    """
    arr = rgb_any.astype(np.float32)
    if rgb_any.dtype == np.uint16:
        arr /= 65535.0
    elif rgb_any.dtype == np.uint8:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def convert_colorspace_real_opencv(rgb_any, mode):
    """
    使用 OpenCV，在 float32 输入下输出“真实值”（全部为 float32）：
      输入：任意 dtype 的 RGB(HWC)，内部会归一到 [0,1]
      输出（OpenCV 约定）：
        LAB:  L[0..100], a/b ≈ [-128..127]
        HSV:  H[0..360], S[0..1], V[0..1]
        HLS:  H[0..360], L[0..1], S[0..1]
        XYZ:  0..1
        LUV:  L[0..100], u/v：浮点
        YCrCb: 0..255（浮点）
        Gray:  0..1
    """
    rgb01 = to_float01(rgb_any)  # HWC, float32, [0,1]
    # OpenCV 需要 0..1 的 float32 输入
    if mode == "LAB":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Lab)
    elif mode == "HSV":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HSV)  # H:0..360
    elif mode == "HLS":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HLS)  # H:0..360
    elif mode == "YCrCb":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb)  # 0..255 浮点
    elif mode == "XYZ":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2XYZ)
    elif mode == "LUV":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Luv)
    elif mode == "Gray":
        out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2GRAY)[..., None]
    else:
        raise ValueError("Unsupported color space")
    return out.astype(np.float32)

def write_float32_geotiff_from_array(arr_float, out_name, crs, transform):
    """
    把 (H,W,C或1) 的 float32 数组写成 GeoTIFF。
    crs/transform 可为 None（无地理参照）。
    """
    if arr_float.ndim == 2:
        arr_float = arr_float[..., None]
    h, w, c = arr_float.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "crs": crs,
        "transform": transform if transform is not None else Affine.translation(0, 0) * Affine.scale(1, 1),
        "compress": "deflate",
        "predictor": 3
    }
    mem = rasterio.MemoryFile()
    with mem.open(**profile) as dst:
        for i in range(c):
            dst.write(arr_float[:, :, i], i+1)
    return mem

def set_band_descriptions(ds, mode):
    try:
        if mode == "LAB":
            ds.set_band_description(1, "L_0-100")
            ds.set_band_description(2, "a_-128_127")
            ds.set_band_description(3, "b_-128_127")
        elif mode == "HSV":
            ds.set_band_description(1, "H_0-360_deg")
            ds.set_band_description(2, "S_0-1")
            ds.set_band_description(3, "V_0-1")
        elif mode == "HLS":
            ds.set_band_description(1, "H_0-360_deg")
            ds.set_band_description(2, "L_0-1")
            ds.set_band_description(3, "S_0-1")
        elif mode == "XYZ":
            ds.set_band_description(1, "X_0-1")
            ds.set_band_description(2, "Y_0-1")
            ds.set_band_description(3, "Z_0-1")
        elif mode == "LUV":
            ds.set_band_description(1, "L_0-100")
            ds.set_band_description(2, "u_float")
            ds.set_band_description(3, "v_float")
        elif mode == "YCrCb":
            ds.set_band_description(1, "Y_0-255")
            ds.set_band_description(2, "Cr_0-255")  # 注意 OpenCV 顺序为 YCrCb
            ds.set_band_description(3, "Cb_0-255")
        elif mode == "Gray":
            ds.set_band_description(1, "Gray_0-1")
    except Exception:
        pass

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
                crs = src.crs
                transform = src.transform
                tags_global = src.tags()
                bands_to_read = min(3, count)

                st.write(f"原画像サイズ：{w} × {h}／バンド数：{count}（変換に使用：先頭{bands_to_read}バンド）")

                block = 1024
                out_channels = 1 if color_space == "Gray" else 3

                # 先写一个空的内存 GeoTIFF
                out_profile = src.profile.copy()
                out_profile.update({
                    "count": out_channels,
                    "dtype": "float32",
                    "compress": "deflate",
                    "predictor": 3
                })
                out_mem = rasterio.MemoryFile()
                with out_mem.open(**out_profile) as dst:
                    for y in range(0, h, block):
                        for x in range(0, w, block):
                            win = Window(x, y, min(block, w-x), min(block, h-y))
                            arr = src.read(indexes=list(range(1, bands_to_read+1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:, :, -1]] * (3 - arr.shape[2])
                                arr = np.concatenate([arr] + [p[..., None] for p in pads], axis=2)

                            out_block = convert_colorspace_real_opencv(arr, color_space)  # float32
                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], ch+1, window=win)

                    dst.update_tags(**tags_global)
                    set_band_descriptions(dst, color_space)

                # 预览：仅显示用的拉伸
                with out_mem.open() as prev_ds:
                    prev = prev_ds.read(indexes=list(range(1, out_channels+1)))
                    prev = ensure_hwc(prev)
                    prev = percentile_scale_to_uint8(prev, p_low, p_high)
                    if prev.shape[2] == 1:
                        prev = np.repeat(prev, 3, axis=2)
                    h0, w0 = prev.shape[:2]
                    scale = min(preview_max / max(h0, w0), 1.0)
                    if scale < 1.0:
                        prev = cv2.resize(prev, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
                    st.image(prev, caption=f"プレビュー（{color_space} 実数値 → 表示用スケーリング）", use_container_width=True)

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}_float32.tif"
                st.download_button("⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
                                   data=out_bytes, file_name=out_name, mime="image/tiff")
        st.success("✅ 変換が完了しました。出力はすべて実数値（float32）の GeoTIFF です。")
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

# ===== B：JPEG/PNG（無地理参照。実数値 GeoTIFF で出力） =====
else:
    st.subheader("🧭 JPEG/PNG 処理（実数値 float32 GeoTIFF で出力）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)  # uint8

        out_img = convert_colorspace_real_opencv(rgb, color_space)  # float32

        # 预览（仅显示）
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
        mem = write_float32_geotiff_from_array(out_img, "out.tif", crs=None, transform=None)
        with mem.open() as ds:
            set_band_descriptions(ds, color_space)
        out_bytes = mem.read()
        out_name = Path(filename).stem + f"_{color_space}_float32.tif"
        st.download_button("⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
                           data=out_bytes, file_name=out_name, mime="image/tiff")
        st.info("注：JPEG/PNG は EXIF/GPS を保持しません。本アプリは解析用に float32 GeoTIFF を出力します。")
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

st.caption("本アプリは研究・教育用です。")
