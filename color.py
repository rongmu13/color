# app.py —— 実数値(float32)出力 + 堅牢な依存性チェック版
import io, sys, platform
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

# ── 依赖自检：OpenCV / rasterio 都可能在云端导入失败 ─────────────────
CV2_OK = True; CV2_ERR = ""
try:
    import cv2
except Exception as e:
    CV2_OK = False; CV2_ERR = f"{e.__class__.__name__}: {e}"

RAST_OK = True; RAST_ERR = ""
try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
except Exception as e:
    RAST_OK = False; RAST_ERR = f"{e.__class__.__name__}: {e}"

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RGB → 色空間変換（実数値出力）", layout="wide")
st.title("RGB → 色空間変換（実数値出力 / float32 GeoTIFF）")

with st.sidebar:
    st.subheader("🔎 ランタイム情報")
    st.write(f"Python: {sys.version.split()[0]}")
    try:
        import numpy as np
        st.write(f"NumPy: {np.__version__}")
    except Exception as e:
        st.write(f"NumPy: エラー {e}")
    st.write(f"OpenCV: {'OK' if CV2_OK else 'NG'}")
    if not CV2_OK: st.error(f"cv2 import 失敗: {CV2_ERR}")
    st.write(f"rasterio: {'OK' if RAST_OK else 'NG'}")
    if not RAST_OK: st.warning(f"rasterio import 失敗: {RAST_ERR}")

    color_space = st.selectbox(
        "変換先の色空間（全て 実数値 出力）",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"], index=0
    )
    p_low  = st.slider("下位パーセンタイル（プレビュー）", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上位パーセンタイル（プレビュー）", 90.0, 100.0, 99.0, 0.5)
    preview_max = st.slider("プレビュー最大辺(px)", 256, 2048, 1024, 128)

st.caption("※ 出力は常に float32 の GeoTIFF（実数値）。プレビューは表示用に8bitへスケーリング。")

# ── ユーティリティ ────────────────────────────────────────────────────────
def infer_is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))

def ensure_hwc(arr):
    if arr.ndim == 3 and arr.shape[0] in (3,4) and (arr.shape[2] not in (3,4)):
        arr = np.transpose(arr, (1,2,0))
    return arr

def to_float01(rgb_any):
    arr = rgb_any.astype(np.float32)
    if rgb_any.dtype == np.uint16: arr /= 65535.0
    elif rgb_any.dtype == np.uint8: arr /= 255.0
    return np.clip(arr, 0.0, 1.0)

def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo: hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255.0 + 0.5).astype(np.uint8)

def convert_colorspace_real_opencv(rgb_any, mode):
    if not CV2_OK:
        raise RuntimeError(f"OpenCV 未导入成功：{CV2_ERR}")
    rgb01 = to_float01(rgb_any)
    if mode == "LAB":  out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Lab)       # L[0..100], a/b≈[-128..127]
    elif mode == "HSV": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HSV)      # H[0..360], S,V[0..1]
    elif mode == "HLS": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2HLS)      # H[0..360], L,S[0..1]
    elif mode == "YCrCb": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb)  # 0..255 浮点
    elif mode == "XYZ": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2XYZ)      # 0..1
    elif mode == "LUV": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Luv)      # L[0..100], u,v 浮点
    elif mode == "Gray": out = cv2.cvtColor(rgb01, cv2.COLOR_RGB2GRAY)[...,None] # 0..1
    else: raise ValueError("Unsupported color space")
    return out.astype(np.float32)

def set_band_descriptions(ds, mode):
    try:
        if mode == "LAB":
            ds.set_band_description(1, "L_0-100"); ds.set_band_description(2, "a_-128_127"); ds.set_band_description(3, "b_-128_127")
        elif mode == "HSV":
            ds.set_band_description(1, "H_0-360_deg"); ds.set_band_description(2, "S_0-1"); ds.set_band_description(3, "V_0-1")
        elif mode == "HLS":
            ds.set_band_description(1, "H_0-360_deg"); ds.set_band_description(2, "L_0-1"); ds.set_band_description(3, "S_0-1")
        elif mode == "XYZ":
            ds.set_band_description(1, "X_0-1"); ds.set_band_description(2, "Y_0-1"); ds.set_band_description(3, "Z_0-1")
        elif mode == "LUV":
            ds.set_band_description(1, "L_0-100"); ds.set_band_description(2, "u_float"); ds.set_band_description(3, "v_float")
        elif mode == "YCrCb":
            ds.set_band_description(1, "Y_0-255"); ds.set_band_description(2, "Cr_0-255"); ds.set_band_description(3, "Cb_0-255")
        elif mode == "Gray":
            ds.set_band_description(1, "Gray_0-1")
    except Exception:
        pass

def write_float32_geotiff_from_array(arr_float, crs=None, transform=None, tags=None):
    if not RAST_OK:
        raise RuntimeError(f"rasterio 未导入成功：{RAST_ERR}")
    if arr_float.ndim == 2: arr_float = arr_float[...,None]
    h,w,c = arr_float.shape
    profile = {
        "driver":"GTiff","height":h,"width":w,"count":c,"dtype":"float32",
        "crs": crs, "transform": transform if transform is not None else Affine.translation(0,0)*Affine.scale(1,1),
        "compress":"deflate","predictor":3
    }
    mem = rasterio.MemoryFile()
    with mem.open(**profile) as dst:
        for i in range(c): dst.write(arr_float[:,:,i], i+1)
        if tags: dst.update_tags(**tags)
    return mem

# ── 输入 ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
                            type=["tif","tiff","jpg","jpeg","png"])
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください。")
    st.stop()

name = uploaded.name
is_tif = infer_is_tiff(name)

# ── GeoTIFF/TIFF ────────────────────────────────────────────────────────────
if is_tif:
    st.subheader("🗺 GeoTIFF/TIFF 処理（float32 実数値）")
    if not RAST_OK:
        st.error("rasterio が読み込めず、GeoTIFF の処理ができません。側欄のエラー内容をご確認ください。")
        st.stop()
    try:
        with rasterio.MemoryFile(uploaded.read()) as mf:
            with mf.open() as src:
                h,w,count = src.height, src.width, src.count
                tags = src.tags()
                bands_to_read = min(3, count)
                st.write(f"原画像：{w}×{h}／バンド数：{count}（先頭{bands_to_read}バンド使用）")

                block=1024
                out_ch = 1 if color_space=="Gray" else 3
                out_profile = src.profile.copy()
                out_profile.update({"count":out_ch,"dtype":"float32","compress":"deflate","predictor":3})

                mem = rasterio.MemoryFile()
                with mem.open(**out_profile) as dst:
                    for y in range(0,h,block):
                        for x in range(0,w,block):
                            win = Window(x,y,min(block,w-x),min(block,h-y))
                            arr = src.read(indexes=list(range(1,bands_to_read+1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:,:, -1]]*(3-arr.shape[2])
                                arr = np.concatenate([arr]+[p[...,None] for p in pads], axis=2)
                            out_block = convert_colorspace_real_opencv(arr, color_space)
                            for ch in range(out_ch):
                                dst.write(out_block[:,:,ch], ch+1, window=win)
                    dst.update_tags(**tags)
                    set_band_descriptions(dst, color_space)

                # 预览（仅显示）
                with mem.open() as ds:
                    prev = ds.read(indexes=list(range(1,out_ch+1)))
                    prev = ensure_hwc(prev)
                    prev = percentile_scale_to_uint8(prev, p_low, p_high)
                    if prev.shape[2]==1: prev = np.repeat(prev,3,axis=2)
                    H,W = prev.shape[:2]
                    scale = min(preview_max / max(H,W), 1.0)
                    if scale<1.0:
                        prev = cv2.resize(prev, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if CV2_OK else np.array(Image.fromarray(prev).resize((int(W*scale), int(H*scale))))
                    st.image(prev, caption=f"プレビュー（{color_space} 実数値 → 表示用）", use_container_width=True)

                st.download_button(
                    "⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
                    data=mem.read(),
                    file_name=Path(name).stem + f"_{color_space}_float32.tif",
                    mime="image/tiff"
                )
        st.success("✅ 完了：float32 実数値 GeoTIFF を出力しました。")
    except Exception as e:
        st.error(f"エラー：{e}")

# ── JPEG/PNG ───────────────────────────────────────────────────────────────
else:
    st.subheader("🧭 JPEG/PNG 処理（float32 実数値 GeoTIFF 出力）")
    try:
        rgb = np.array(Image.open(io.BytesIO(uploaded.read())).convert("RGB"))
        out_img = convert_colorspace_real_opencv(rgb, color_space)
        # 预览
        prev = percentile_scale_to_uint8(out_img, p_low, p_high)
        if prev.ndim==2: prev = prev[...,None]
        if prev.shape[2]==1: prev = np.repeat(prev,3,axis=2)
        H,W = prev.shape[:2]
        scale = min(preview_max / max(H,W), 1.0)
        if scale<1.0:
            prev = cv2.resize(prev, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if CV2_OK else np.array(Image.fromarray(prev).resize((int(W*scale), int(H*scale))))
        st.image(prev, caption=f"プレビュー（{color_space} 実数値 → 表示用）", use_container_width=True)

        if not RAST_OK:
            st.error("rasterio が無いため、GeoTIFF への保存はできません。側欄エラーを確認し、requirements を修正してください。")
            st.stop()

        mem = write_float32_geotiff_from_array(out_img, crs=None, transform=None, tags=None)
        with mem.open() as ds: set_band_descriptions(ds, color_space)
        st.download_button(
            "⬇️ 変換結果をダウンロード（float32 GeoTIFF）",
            data=mem.read(),
            file_name=Path(name).stem + f"_{color_space}_float32.tif",
            mime="image/tiff"
        )
        st.success("✅ 完了：float32 実数値 GeoTIFF を出力しました。")
    except Exception as e:
        st.error(f"エラー：{e}")

st.caption("本アプリは研究・教育用です。")
