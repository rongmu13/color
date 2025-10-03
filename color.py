
import io
import gc
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

import cv2
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# JPEG の EXIF コピー（任意）
try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False

# ---------------- UI ----------------
st.set_page_config(page_title="色空間変換 Shinshu Univ. R.Y.", layout="wide")
st.title("RGB → 色空間変換 Shinshu Univ. R.Y.")

with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間（すべて実数で出力）",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0
    )
    # プレビューの見やすさのみ（出力数値は変えない）
    p_low = st.slider("プレビュー下位パーセンタイル", 0.0, 10.0, 2.0, 0.5)
    p_high = st.slider("プレビュー上位パーセンタイル", 90.0, 100.0, 98.0, 0.5)
    preview_max = st.slider("プレビュー最大辺(px)", 256, 4096, 1024, 128)
    block = st.selectbox("タイル書き出しブロックサイズ", [256, 512, 1024], index=1)

uploaded = st.file_uploader(
    "画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif", "tiff", "jpg", "jpeg", "png"]
)

# --------------- Utils ---------------
def ensure_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and (x.shape[2] not in (1, 3, 4)):
        x = np.transpose(x, (1, 2, 0))
    return x

def to_rgb01(arr: np.ndarray) -> np.ndarray:
    """任意ビット深度RGB → float32 [0,1]"""
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

def percent_stretch_u8(x: np.ndarray, p_lo=2, p_hi=98) -> np.ndarray:
    """プレビュー専用：float32 → 0..255 に正規化"""
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255.0 + 0.5).astype(np.uint8)

# ---- 各色空間：実数へ変換（float32 HWC） ----
def rgb_to_lab_real(rgb01):   # L[0..100], a,b≈[-128..127]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB).astype(np.float32)

def rgb_to_hsv_real(rgb01):   # H[0..360), S,V[0..1]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2HSV).astype(np.float32)

def rgb_to_hls_real(rgb01):   # H[0..360), L,S[0..1]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2HLS).astype(np.float32)

def rgb_to_ycrcb_real(rgb01): # Y,Cr,Cb ∈ [0,1]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb).astype(np.float32)

def rgb_to_xyz_real(rgb01):   # おおむね [0,1]
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2XYZ).astype(np.float32)

def rgb_to_luv_real(rgb01):   # L[0..100]、u,v は負あり得る
    return cv2.cvtColor(rgb01, cv2.COLOR_RGB2Luv).astype(np.float32)

def rgb_to_gray_real(rgb01):  # [0,1]、1ch
    g = cv2.cvtColor(rgb01, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g[..., None]

CONV = {
    "LAB":   rgb_to_lab_real,
    "HSV":   rgb_to_hsv_real,
    "HLS":   rgb_to_hls_real,
    "YCrCb": rgb_to_ycrcb_real,
    "XYZ":   rgb_to_xyz_real,
    "LUV":   rgb_to_luv_real,
    "Gray":  rgb_to_gray_real,
}

# ---- プレビュー変換（実数 → 画面表示用の uint8 RGB） ----
def preview_from_real(cs_name: str, arr_f32: np.ndarray, p_lo=2, p_hi=98) -> np.ndarray:
    if cs_name == "LAB":
        L = np.clip(arr_f32[:, :, 0], 0, 100) * (255.0 / 100.0)
        a = np.clip(arr_f32[:, :, 1] + 127.0, 0, 255)
        b = np.clip(arr_f32[:, :, 2] + 127.0, 0, 255)
        img = np.stack([L, a, b], axis=2).astype(np.uint8)
    elif cs_name in ("HSV", "HLS"):
        H = np.mod(arr_f32[:, :, 0], 360.0) * (255.0 / 360.0)
        C1 = np.clip(arr_f32[:, :, 1], 0, 1) * 255.0
        C2 = np.clip(arr_f32[:, :, 2], 0, 1) * 255.0
        img = np.stack([H, C1, C2], axis=2).astype(np.uint8)
    elif cs_name in ("YCrCb", "Gray"):
        if arr_f32.shape[2] == 1:
            g = np.clip(arr_f32[:, :, 0], 0, 1)
            img = np.repeat((g * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=2)
        else:
            img = np.clip(arr_f32, 0, 1)
            img = (img * 255.0 + 0.5).astype(np.uint8)
    elif cs_name in ("XYZ", "LUV"):
        chs = [percent_stretch_u8(arr_f32[:, :, c], p_lo, p_hi) for c in range(arr_f32.shape[2])]
        img = np.stack(chs, axis=2)
    else:
        raise ValueError("未対応の色空間です")
    return img

def pil_preview(img_u8: np.ndarray, max_side=1024) -> Image.Image:
    h, w = img_u8.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img_u8 = cv2.resize(img_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_u8)

def copy_exif_jpeg(src_bytes: bytes, dst_bytes_io: io.BytesIO) -> bool:
    if not HAS_PIEXIF:
        return False
    try:
        exif_dict = piexif.load(src_bytes)
        exif_bytes = piexif.dump(exif_dict)
        img = Image.open(io.BytesIO(dst_bytes_io.getvalue()))
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", exif=exif_bytes, quality=95)
        dst_bytes_io.seek(0)
        dst_bytes_io.truncate(0)
        dst_bytes_io.write(out_io.getvalue())
        return True
    except Exception:
        return False

def infer_is_tiff(name: str) -> bool: return name.lower().endswith((".tif", ".tiff"))
def infer_is_jpeg(name: str) -> bool: return name.lower().endswith((".jpg", ".jpeg"))

# ---------------- Main ----------------
if uploaded is None:
    st.info("👆 最初に画像をアップロードしてください。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# ---- A: GeoTIFF/TIFF（地理参照） ----
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF")
    try:
        # 入力はバイトから開く（読み取りはウィンドウ単位なのでOK）
        with rasterio.MemoryFile(uploaded.read()) as mem:
            with mem.open() as src:
                h, w, count = src.height, src.width, src.count
                bands_to_read = min(3, count)
                st.write(f"元画像サイズ：{w}×{h}／バンド数：{count}（先頭{bands_to_read}バンドを使用）")

                out_channels = 1 if color_space == "Gray" else 3
                profile = src.profile.copy()
                profile.update({"count": out_channels, "dtype": "float32",
                                "compress": "deflate", "predictor": 2})

                tags_global = src.tags()
                tags_per_band = [src.tags(i + 1) for i in range(min(3, count))]

                # 出力は一時ファイルにストリーミング書き出し（大画像でもメモリ安全）
                tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                tmp_out_path = tmp_out.name
                tmp_out.close()

                with rasterio.open(tmp_out_path, "w", **profile) as dst:
                    for y in range(0, h, block):
                        for x in range(0, w, block):
                            win = Window(x, y, min(block, w - x), min(block, h - y))
                            arr = src.read(indexes=list(range(1, bands_to_read + 1)), window=win)
                            arr = ensure_hwc(arr)
                            if arr.shape[2] < 3:
                                pads = [arr[:, :, -1]] * (3 - arr.shape[2])
                                arr = np.concatenate([arr] + [p[..., None] for p in pads], axis=2)

                            rgb01 = to_rgb01(arr)
                            out_block = CONV[color_space](rgb01)   # float32

                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], ch + 1, window=win)

                            # 逐次解放
                            del arr, rgb01, out_block
                            gc.collect()

                    # タグ継承
                    dst.update_tags(**tags_global)
                    for ch in range(min(len(tags_per_band), out_channels)):
                        try:
                            dst.update_tags(ch + 1, **tags_per_band[ch])
                        except Exception:
                            pass

                    # バンド説明
                    try:
                        if color_space == "LAB":
                            dst.set_band_description(1, "L (0..100)")
                            dst.set_band_description(2, "a (~-128..127)")
                            dst.set_band_description(3, "b (~-128..127)")
                        elif color_space == "HSV":
                            for i, nm in enumerate(["H (0..360)", "S (0..1)", "V (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "HLS":
                            for i, nm in enumerate(["H (0..360)", "L (0..1)", "S (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "YCrCb":
                            for i, nm in enumerate(["Y (0..1)", "Cr (0..1)", "Cb (0..1)"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "XYZ":
                            for i, nm in enumerate(["X", "Y", "Z"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "LUV":
                            for i, nm in enumerate(["L (0..100)", "u", "v"], 1):
                                dst.set_band_description(i, nm)
                        elif color_space == "Gray":
                            dst.set_band_description(1, "Gray (0..1)")
                    except Exception:
                        pass

                # プレビューは縮小のみ読み込み
                with rasterio.open(tmp_out_path) as prev_ds:
                    scale = max(prev_ds.width, prev_ds.height) / max(preview_max, 1)
                    scale = max(scale, 1.0)
                    prev = prev_ds.read(
                        indexes=list(range(1, out_channels + 1)),
                        out_shape=(out_channels,
                                   int(prev_ds.height / scale),
                                   int(prev_ds.width / scale)),
                        resampling=Resampling.average
                    )
                prev = ensure_hwc(prev).astype(np.float32)
                prev_u8 = preview_from_real(color_space, prev, p_low, p_high)
                st.image(pil_preview(prev_u8, max_side=preview_max),
                         caption=f"プレビュー（{color_space} / 実数 float32）",
                         width="stretch")

                # ダウンロード：ファイルを読み込んで渡す（巨大配列を常駐させない）
                with open(tmp_out_path, "rb") as f:
                    st.download_button("⬇️ 変換結果をダウンロード（GeoTIFF: float32）",
                                       data=f.read(),
                                       file_name=Path(filename).stem + f"_{color_space}_float32.tif",
                                       mime="image/tiff")

    except Exception as e:
        st.exception(e)

# ---- B: JPEG/PNG ----
else:
    st.subheader("🧭 ")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)

        rgb01 = to_rgb01(rgb)
        real = CONV[color_space](rgb01)       # float32 (H, W, C)

        # プレビュー生成（可視化のみ）
        prev_u8 = preview_from_real(color_space, real, p_low, p_high)
        st.image(pil_preview(prev_u8, max_side=preview_max),
                 caption=f"プレビュー（{color_space} / 実数 float32 → 可視化）",
                 width="stretch")

        # GeoTIFF（float32）も一時ファイルへ書き出し
        tmp_gt = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp_gt_path = tmp_gt.name
        tmp_gt.close()

        profile = {
            "driver": "GTiff",
            "height": real.shape[0],
            "width":  real.shape[1],
            "count":  real.shape[2],
            "dtype":  "float32",
            "crs":    None,
            "transform": from_origin(0, 0, 1, 1),
            "compress": "deflate",
            "predictor": 2
        }
        with rasterio.open(tmp_gt_path, "w", **profile) as ds:
            for i in range(real.shape[2]):
                ds.write(real[:, :, i], i + 1)

        # すぐに数値配列を解放
        del real, rgb, rgb01
        gc.collect()

        with open(tmp_gt_path, "rb") as f:
            st.download_button("⬇️ ダウンロード",
                               data=f.read(),
                               file_name=Path(filename).stem + f"_{color_space}_float32.tif",
                               mime="image/tiff")

        # PNG / JPEG（プレビュー画像）
        png_io = io.BytesIO()
        Image.fromarray(prev_u8).save(png_io, format="PNG", compress_level=6)
        st.download_button("⬇️ ダウンロード",
                           data=png_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}_preview.png",
                           mime="image/png")

        jpg_io = io.BytesIO()
        Image.fromarray(prev_u8).save(jpg_io, format="JPEG", quality=95)
        exif_ok = False
        if is_jpeg and HAS_PIEXIF:
            exif_ok = copy_exif_jpeg(src_bytes, jpg_io)
        label = "⬇️ JPEG（プレビュー；EXIF は可能なら保持）"
        if is_jpeg:
            label += " ✅EXIFコピー済" if exif_ok else " ⚠EXIFコピー不可"
        st.download_button(label,
                           data=jpg_io.getvalue(),
                           file_name=Path(filename).stem + f"_{color_space}_preview.jpg",
                           mime="image/jpeg")

        st.info("ℹ️ 数値解析・GIS 連携は GeoTIFF（float32）をお使いください。")

    except Exception as e:
        st.exception(e)




