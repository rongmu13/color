# color.py —— RGB → 色空間変換（LAB は実数 float32 GeoTIFF 対応 / No skimage）
# Author: Shinshu Univ. R.Y.  |  研究・教育用途
# 変更点:
#  - LAB 実数出力は OpenCV を使用（入力を0..1のfloat32にすればL[0..100], a/b≈[-127,127]）
#  - rasterio.read の引数を indexes に修正
#  - Streamlit の use_container_width を width="stretch" に変更

import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

# GeoTIFF
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# OpenCV（色空間変換）
import cv2

# JPEG EXIF コピー（任意）
try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False

# =========================
# UI
# =========================
st.set_page_config(page_title="RGB → 色空間変換 Shinshu Univ. R.Y.", layout="wide")
st.title("RGB → 色空間変換 Shinshu Univ. R.Y.")
st.caption(
    "RGB画像（GeoTIFF/TIFF/JPG/PNG）をアップロードし、右側で変換先の色空間を選択してください。"
    " 地理参照（投影・アフィン）や EXIF/GPS 情報の保持に対応します。"
)

with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0,
        help="一般的に用いられる色空間を選択できます。Grayは1チャンネルになります。"
    )
    lab_real = st.checkbox(
        "LAB を実数（float32: L[0..100], a/b[-127..127]）で出力する",
        value=True,
        help="オンにすると、LAB を GeoTIFF(float32) で保存します。"
    )
    enable_8bit_scale = st.checkbox(
        "高ビット深度画像を8bitへスケーリングしてから変換する（LAB実数には無関係）",
        value=True,
        help="ドローンGeoTIFFは10/12/16bitが多いので、OpenCVの8bit変換が安定します。"
    )
    p_low = st.slider("下位パーセンタイル（8bitスケーリング）", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上位パーセンタイル（8bitスケーリング）", 90.0, 100.0, 99.0, 0.5)
    preview_max = st.slider(
        "プレビューの最大辺サイズ（px）", 256, 4096, 1024, 128,
        help="プレビュー表示のみ影響します。ダウンロード結果には影響しません。"
    )

uploaded = st.file_uploader(
    "画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    help="Streamlit Cloud では単一ファイルの上限は約200MBです。"
)

# =========================
# ユーティリティ
# =========================
def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    """任意dtypeをパーセンタイルでuint8[0,255]へ（非・実数LABで使用）"""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)

def ensure_hwc(x):
    """(C,H,W) or (H,W,C) -> (H,W,C)"""
    if x.ndim == 3 and x.shape[0] in (1,3,4) and (x.shape[2] not in (1,3,4)):
        x = np.transpose(x, (1,2,0))
    return x

def to_rgb01(arr):
    """RGB任意dtypeを float32 の[0,1]へ（実数LAB用）"""
    if arr.dtype in (np.float32, np.float64):
        x = np.clip(arr, 0, 1).astype(np.float32)
    elif arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            mx = mn + 1.0
        x = (arr - mn) / (mx - mn)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def rgb_to_lab_real_opencv(rgb_any_dtype):
    """
    OpenCVで実数LABへ。入力を0..1 float32にすれば、
    出力は L[0..100], a/b≈[-127,127] の float32。
    """
    rgb01 = to_rgb01(rgb_any_dtype)
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab

def lab_preview_u8(lab_f32):
    """実数LAB → 可視化用の疑似カラー(uint8)"""
    L = np.clip(lab_f32[:, :, 0], 0, 100) * (255.0/100.0)
    a = np.clip(lab_f32[:, :, 1] + 127.0, 0, 255)
    b = np.clip(lab_f32[:, :, 2] + 127.0, 0, 255)
    return np.stack([L, a, b], axis=2).astype(np.uint8)

def convert_colorspace_uint8(rgb_u8, mode):
    """RGB uint8 -> 指定色空間（uint8返し。GrayはHxWx1）"""
    if mode == "LAB":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    elif mode == "HSV":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    elif mode == "HLS":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HLS)
    elif mode == "YCrCb":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
    elif mode == "XYZ":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2XYZ)
    elif mode == "LUV":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2Luv)
    elif mode == "Gray":
        out = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)[..., None]
    else:
        raise ValueError("Unsupported color space")
    return out

def pil_preview(img_u8, max_side=1024):
    """uint8画像のプレビュー縮小"""
    h, w = img_u8.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img_u8 = cv2.resize(img_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_u8)

def infer_is_tiff(name): return name.lower().endswith((".tif",".tiff"))
def infer_is_jpeg(name): return name.lower().endswith((".jpg",".jpeg"))

def copy_exif_jpeg(src_bytes, dst_bytes_io):
    """JPEGのEXIFをコピー（できる場合のみ）"""
    if not HAS_PIEXIF:
        return False
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

# =========================
# メイン
# =========================
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください（GeoTIFF/TIFF/JPG/PNG に対応）。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# ----- A: GeoTIFF/TIFF -----
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF 処理")
    try:
        with rasterio.MemoryFile(uploaded.read()) as mem:
            with mem.open() as src:
                h, w, count = src.height, src.width, src.count
                bands_to_read = min(3, count)
                st.write(f"原画像サイズ：{w} × {h}／バンド数：{count}（変換に使用：先頭{bands_to_read}バンド）")

                block = 1024
                out_channels = 1 if color_space == "Gray" else 3
                out_dtype = "float32" if (color_space == "LAB" and lab_real) else "uint8"

                profile = src.profile.copy()
                profile.update({
                    "count": out_channels,
                    "dtype": out_dtype,
                    "compress": "deflate",
                    "predictor": 2
                })

                tags_global = src.tags()
                tags_per_band = [src.tags(i+1) for i in range(min(3, count))]

                out_mem = rasterio.MemoryFile()
                with out_mem.open(**profile) as dst:
                    for y in range(0, h, block):
                        for x in range(0, w, block):
                            win = Window(x, y, min(block, w-x), min(block, h-y))
                            arr = src.read(indexes=list(range(1, bands_to_read+1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:, :, -1]] * (3 - arr.shape[2])
                                arr = np.concatenate([arr] + [p[..., None] for p in pads], axis=2)

                            if color_space == "LAB" and lab_real:
                                out_block = rgb_to_lab_real_opencv(arr)  # float32
                            else:
                                if enable_8bit_scale:
                                    arr_u8 = percentile_scale_to_uint8(arr, p_low, p_high)
                                else:
                                    arr_u8 = arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
                                out_block = convert_colorspace_uint8(arr_u8, color_space)

                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], ch+1, window=win)

                    dst.update_tags(**tags_global)
                    for ch in range(min(len(tags_per_band), out_channels)):
                        try:
                            dst.update_tags(ch+1, **tags_per_band[ch])
                        except Exception:
                            pass
                    if color_space == "LAB" and lab_real:
                        try:
                            dst.set_band_description(1, "L (0-100)")
                            dst.set_band_description(2, "a (-127..127)")
                            dst.set_band_description(3, "b (-127..127)")
                        except Exception:
                            pass

                # プレビュー
                with out_mem.open() as prev_ds:
                    scale = max(prev_ds.width, prev_ds.height) / max(preview_max, 1)
                    scale = max(scale, 1.0)
                    preview = prev_ds.read(
                        indexes=list(range(1, out_channels+1)),
                        out_shape=(out_channels,
                                   int(prev_ds.height/scale),
                                   int(prev_ds.width/scale)),
                        resampling=Resampling.average
                    )
                    preview = ensure_hwc(preview)
                    if color_space == "LAB" and lab_real:
                        if preview.dtype != np.float32:
                            preview = preview.astype(np.float32)
                        prev_u8 = lab_preview_u8(preview)
                    else:
                        prev_u8 = preview.astype(np.uint8)
                        if prev_u8.shape[2] == 1:
                            prev_u8 = np.repeat(prev_u8, 3, axis=2)

                st.image(prev_u8, caption=f"プレビュー（{color_space}{' / 実数LAB' if (color_space=='LAB' and lab_real) else ''}）", width="stretch")

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}{'_real' if (color_space=='LAB' and lab_real) else ''}.tif"
                st.download_button("⬇️ 変換結果をダウンロード（GeoTIFF）", data=out_bytes, file_name=out_name, mime="image/tiff")

        st.success("✅ 変換が完了しました。GeoTIFF の投影・座標・タグは可能な範囲で引き継ぎ済み。")
    except Exception as e:
        st.exception(e)

# ----- B: JPEG/PNG -----
else:
    st.subheader("🧭 JPEG/PNG 処理（EXIF/GPS を可能ならコピー）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)

        lab_geotiff_bytes = None
        lab_f32 = None

        if color_space == "LAB" and lab_real:
            lab_f32 = rgb_to_lab_real_opencv(rgb)  # float32
            profile = {
                "driver": "GTiff",
                "height": lab_f32.shape[0],
                "width":  lab_f32.shape[1],
                "count":  3,
                "dtype":  "float32",
                "crs":    None,
                "transform": from_origin(0, 0, 1, 1),
                "compress": "deflate",
                "predictor": 2
            }
            mem = rasterio.io.MemoryFile()
            with mem.open(**profile) as ds:
                ds.write(lab_f32[:, :, 0], 1)
                ds.write(lab_f32[:, :, 1], 2)
                ds.write(lab_f32[:, :, 2], 3)
                try:
                    ds.set_band_description(1, "L (0-100)")
                    ds.set_band_description(2, "a (-127..127)")
                    ds.set_band_description(3, "b (-127..127)")
                except Exception:
                    pass
            lab_geotiff_bytes = mem.read()
            prev_u8 = lab_preview_u8(lab_f32)
        else:
            # 可視化ルート（uint8）
            if enable_8bit_scale and rgb.dtype != np.uint8:
                rgb_u8 = percentile_scale_to_uint8(rgb, p_low, p_high)
            else:
                rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
            out_img = convert_colorspace_uint8(rgb_u8, color_space)
            prev_u8 = out_img if out_img.shape[2] == 3 else np.repeat(out_img, 3, axis=2)

        st.image(pil_preview(prev_u8, max_side=preview_max), caption=f"プレビュー（{color_space}{' / 実数LAB' if (color_space=='LAB' and lab_real) else ''}）", width="stretch")

        if lab_geotiff_bytes is not None:
            gt_name = Path(filename).stem + "_LAB_real_float32.tif"
            st.download_button("⬇️ GeoTIFF（実数LAB: float32）をダウンロード", data=lab_geotiff_bytes, file_name=gt_name, mime="image/tiff")

        # PNG 可視化
        png_io = io.BytesIO()
        Image.fromarray(prev_u8).save(png_io, format="PNG", compress_level=6)
        st.download_button("⬇️ PNG をダウンロード（可視化）", data=png_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}.png", mime="image/png")

        # JPEG 可視化（可能ならEXIFコピー）
        jpg_io = io.BytesIO()
        Image.fromarray(prev_u8).save(jpg_io, format="JPEG", quality=95)
        exif_ok = False
        if is_jpeg and HAS_PIEXIF:
            exif_ok = copy_exif_jpeg(src_bytes, jpg_io)
        btn = "⬇️ JPEG をダウンロード（EXIF/GPS を可能なら保持）"
        if is_jpeg:
            btn += " ✅EXIFコピー済" if exif_ok else " ⚠EXIFコピー不可"
        st.download_button(btn, data=jpg_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}.jpg", mime="image/jpeg")

        st.info("解析に用いる数値は GeoTIFF（特に LAB 実数 float32）を使用してください。PNG/JPEG は可視化用途です。")
    except Exception as e:
        st.exception(e)

st.caption("本アプリは研究・教育用です。")
