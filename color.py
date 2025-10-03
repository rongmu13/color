# app.py —— RGB → 各色彩空间转换（含：实数 LAB float32 GeoTIFF 导出）
# Author: Shinshu Univ. R.Y.  |  用于研究/教育
# 说明：
# - 若选择 LAB 且勾选“实数输出”，将以 float32 GeoTIFF 输出真实范围：
#     L ∈ [0,100], a,b ∈ [-128,127]
# - 其他色彩空间仍按 uint8(0–255) 可视化导出（便于预览/分享）
# - GeoTIFF 分支保留投影/变换/标签；JPEG 分支尽可能复制 EXIF/GPS

import io
from pathlib import Path
import numpy as np
import streamlit as st

from PIL import Image
import cv2

# 可选：JPEG EXIF 复制
try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False

# GeoTIFF 支持
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin

def rgb_to_lab_real_opencv(rgb):
    """
    输入: HxWx3 的 RGB，任意 dtype
    输出: float32 的 LAB（L[0..100], a/b[-127..127]）
    """
    # 归一化到 float32 的 0..1（你已有 to_rgb01，可直接复用）
    rgb01 = to_rgb01(rgb).astype(np.float32)
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab


# ---------------------------
# 基本設定（页面设置）
# ---------------------------
st.set_page_config(page_title="RGB → 色空間変換 Shinshu Univ. R.Y.", layout="wide")

st.title("RGB → 色空間変換 Shinshu Univ. R.Y.")
st.caption(
    "RGB画像（GeoTIFF/TIFF/JPG/PNG）をアップロードし、右側で変換先の色空間を選択してください。"
    " 変換後の画像をダウンロードできます。地理参照（投影・アフィン）や EXIF/GPS 情報の保持に対応します。"
)

# ---------------------------
# 侧边栏设置
# ---------------------------
with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0,
        help="一般的に用いられる色空間を選択できます。Grayは1チャンネルになります。"
    )
    # 仅对 LAB 生效：是否以真实值 float32 输出
    lab_real = st.checkbox(
        "LAB を実数（float32: L[0..100], a/b[-128..127]）で出力する",
        value=True,
        help="オンの場合、LAB は skimage により実数範囲で GeoTIFF(float32) に保存されます。"
    )
    enable_8bit_scale = st.checkbox(
        "高ビット深度画像を8bitへスケーリングしてから変換する（推奨・LAB実数には無関係）",
        value=True,
        help="ドローンのGeoTIFFは10/12/16bitが多いです。OpenCVの色空間変換は8bitが安定です。"
    )
    p_low = st.slider("下位パーセンタイル（8bitスケーリング）", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上位パーセンタイル（8bitスケーリング）", 90.0, 100.0, 99.0, 0.5)

    preview_max = st.slider(
        "プレビューの最大辺サイズ（px）", 256, 2048, 1024, 128,
        help="画面上のプレビュー表示のみ影響し、ダウンロード結果には影響しません。"
    )

uploaded = st.file_uploader(
    "画像をアップロード（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    help="Streamlit Cloud では単一ファイルのアップロード上限は約200MBです。超える場合はローカル実行または自前サーバーをご利用ください。"
)

# ---------------------------
# ユーティリティ
# ---------------------------
def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    """把任意 dtype 的数组按百分位缩放到 uint8 [0,255]（用于非实数LAB路径）"""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr

def ensure_hwc(rgb_like):
    """把 (bands, H, W) 或 (H, W, bands) 统一成 (H, W, C)"""
    if rgb_like.ndim == 3 and rgb_like.shape[0] in (3, 4) and (rgb_like.shape[2] not in (3, 4)):
        rgb_like = np.transpose(rgb_like, (1, 2, 0))
    return rgb_like

def to_rgb01(arr):
    """把任意RGB数组转换到 float32 的 [0,1]，不做百分位拉伸（用于真实 LAB）"""
    if arr.dtype in (np.float32, np.float64):
        out = np.clip(arr, 0, 1).astype(np.float32)
    elif arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        mn, mx = np.min(arr), np.max(arr)
        if mx <= mn:
            mx = mn + 1.0
        out = (arr - mn) / (mx - mn)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def lab_preview_u8(lab_f32):
    """
    把真实 LAB(float32) 做成 uint8 预览（仅显示）：
      R=L, G=a, B=b 的伪彩
      L:  [0,100]   → *255/100
      a,b:[-128,127]→ +128 再裁剪
    """
    L = np.clip(lab_f32[:, :, 0], 0, 100) * (255.0 / 100.0)
    a = np.clip(lab_f32[:, :, 1] + 128.0, 0, 255)
    b = np.clip(lab_f32[:, :, 2] + 128.0, 0, 255)
    prev = np.stack([L, a, b], axis=2).astype(np.uint8)
    return prev

def convert_colorspace(img_rgb_uint8, mode):
    """RGB uint8(HWC) → 目标色彩空间，输出 uint8 HWC（Gray 为 HxWx1）"""
    if mode == "LAB":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    elif mode == "HSV":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV)
    elif mode == "HLS":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HLS)
    elif mode == "YCrCb":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2YCrCb)
    elif mode == "XYZ":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2XYZ)
    elif mode == "LUV":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Luv)
    elif mode == "Gray":
        out = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        out = out[..., None]
    else:
        raise ValueError("Unsupported color space")
    return out

def pil_preview(img_hwc_uint8, max_side=1024):
    """缩放预览图（uint8）"""
    h, w = img_hwc_uint8.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    preview = img_hwc_uint8
    if scale < 1.0:
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(preview)

def copy_exif_jpeg(src_bytes, dst_bytes_io):
    """将 JPEG 的 EXIF 原样复制到新 JPEG（如果有）"""
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

def infer_is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))

def infer_is_jpeg(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg"))

# ---------------------------
# メイン処理
# ---------------------------
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください（GeoTIFF/TIFF/JPG/PNG に対応）。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# ===== A：GeoTIFF/TIFF（地理参照あり） =====
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF 処理")
    try:
        with rasterio.MemoryFile(uploaded.read()) as memfile:
            with memfile.open() as src:
                # 基本情報
                h, w = src.height, src.width
                count = src.count
                profile = src.profile.copy()
                tags_global = src.tags()
                tags_per_band = [src.tags(i + 1) for i in range(min(3, count))]

                bands_to_read = min(3, count)
                st.write(f"原画像サイズ：{w} × {h}／バンド数：{count}（変換に使用：先頭{bands_to_read}バンド）")

                block_size = 1024
                out_channels = 1 if color_space == "Gray" else 3

                # 输出数据类型：LAB(实数)→float32；否则 uint8
                out_profile = profile.copy()
                out_profile.update({
                    "count": out_channels,
                    "dtype": "float32" if (color_space == "LAB" and lab_real) else "uint8",
                    "compress": "deflate",
                    "predictor": 2
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

                            if color_space == "LAB" and lab_real:
                                # 真实 LAB：RGB→[0,1]→rgb2lab→float32
                                rgb01 = to_rgb01(arr)
                                out_block = rgb_to_lab_real_opencv(arr)  .astype(np.float32)
                            else:
                                # 原有 uint8 流程（用于非实数LAB或其他色彩空间）
                                if enable_8bit_scale:
                                    arr_u8 = percentile_scale_to_uint8(arr, p_low, p_high)
                                else:
                                    arr_u8 = arr
                                    if arr_u8.dtype != np.uint8:
                                        arr_u8 = np.clip(arr_u8, 0, 255).astype(np.uint8)
                                out_block = convert_colorspace(arr_u8, color_space)

                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], indexes=ch + 1, window=win)

                    # 标签/地理参照
                    dst.update_tags(**tags_global)
                    for ch in range(min(len(tags_per_band), out_channels)):
                        try:
                            dst.update_tags(ch + 1, **tags_per_band[ch])
                        except Exception:
                            pass

                    # 可选：写 band 描述（QGIS 里更清楚）
                    try:
                        if color_space == "LAB" and lab_real:
                            dst.set_band_description(1, "L (0-100)")
                            dst.set_band_description(2, "a (-128..127)")
                            dst.set_band_description(3, "b (-128..127)")
                    except Exception:
                        pass

                # 预览
                with out_mem.open() as preview_ds:
                    scale = max(preview_ds.width, preview_ds.height) / max(preview_max, 1)
                    scale = max(scale, 1.0)
                    preview = preview_ds.read(
                        indexes=list(range(1, out_channels + 1)),
                        out_shape=(
                            out_channels,
                            int(preview_ds.height / scale),
                            int(preview_ds.width / scale),
                        ),
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

                    st.image(prev_u8, caption=f"プレビュー（{color_space}{' / 実数LAB' if (color_space=='LAB' and lab_real) else ''}）", use_container_width=True)

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}{'_real' if (color_space=='LAB' and lab_real) else ''}.tif"
                st.download_button(
                    "⬇️ 変換結果をダウンロード（GeoTIFF）",
                    data=out_bytes,
                    file_name=out_name,
                    mime="image/tiff"
                )
        st.success("✅ 変換が完了しました。GeoTIFF の投影・座標・タグは可能な限り引き継がれています。"
                   " 色空間変換により画素値が変化するのは正常な挙動です。")
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

# ===== B：JPEG/PNG（EXIF/GPS の保持は可能な範囲で） =====
else:
    st.subheader("🧭 JPEG/PNG 処理（可能なら EXIF/GPS をコピー）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)

        # —— 实数 LAB 专用路径（另生成 float32 GeoTIFF 下载）——
        lab_geotiff_bytes = None
        lab_f32 = None
        if color_space == "LAB" and lab_real:
            rgb01 = to_rgb01(rgb)
            lab_f32 = rgb_to_lab_real_opencv(rgb).astype(np.float32)

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
                    ds.set_band_description(2, "a (-128..127)")
                    ds.set_band_description(3, "b (-128..127)")
                except Exception:
                    pass
            lab_geotiff_bytes = mem.read()

        # —— 预览（实数LAB→伪彩；否则走原 uint8 转换）——
        if color_space == "LAB" and lab_real:
            prev_u8 = lab_preview_u8(lab_f32)
        else:
            # 非实数LAB路径：可选择 8bit 拉伸后再转色彩空间
            if enable_8bit_scale and rgb.dtype != np.uint8:
                rgb_u8 = percentile_scale_to_uint8(rgb, p_low, p_high)
            else:
                rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)

            out_img = convert_colorspace(rgb_u8, color_space)
            prev = out_img
            if prev.shape[2] == 1:
                prev = np.repeat(prev, 3, axis=2)
            prev_u8 = prev

        st.image(pil_preview(prev_u8, max_side=preview_max), caption=f"プレビュー（{color_space}{' / 実数LAB' if (color_space=='LAB' and lab_real) else ''}）", use_container_width=True)

        # —— 下载：实数 LAB 的 GeoTIFF（float32）
        if lab_geotiff_bytes is not None:
            gt_name = Path(filename).stem + "_LAB_real_float32.tif"
            st.download_button(
                "⬇️ GeoTIFF（実数LAB: float32, L[0..100], a/b[-128..127]）をダウンロード",
                data=lab_geotiff_bytes,
                file_name=gt_name,
                mime="image/tiff"
            )

        # —— PNG（uint8，可视化/分享；不含EXIF）
        png_io = io.BytesIO()
        if color_space == "LAB" and lab_real:
            save_png = prev_u8  # 伪彩预览图
        else:
            save_png = out_img.squeeze() if 'out_img' in locals() and out_img.shape[2] == 1 else out_img
        Image.fromarray(save_png).save(png_io, format="PNG", compress_level=6)
        png_name = Path(filename).stem + f"_{color_space}.png"
        st.download_button("⬇️ PNG をダウンロード（可視化）", data=png_io.getvalue(), file_name=png_name, mime="image/png")

        # —— JPEG（uint8，尽量复制EXIF/GPS）
        jpg_io = io.BytesIO()
        Image.fromarray(save_png).save(jpg_io, format="JPEG", quality=95)
        exif_ok = False
        if is_jpeg and HAS_PIEXIF:
            exif_ok = copy_exif_jpeg(src_bytes, jpg_io)

        jpeg_name = Path(filename).stem + f"_{color_space}.jpg"
        btn_label = "⬇️ JPEG をダウンロード（EXIF/GPS を可能なら保持）"
        if is_jpeg:
            btn_label += " ✅EXIFコピー済" if exif_ok else " ⚠EXIFコピー不可の場合あり"
        st.download_button(btn_label, data=jpg_io.getvalue(), file_name=jpeg_name, mime="image/jpeg")

        st.info(
            "注：PNG は EXIF を保持しません。GPS 情報が必要な場合は、元が JPEG のときに上の JPEG ダウンロードをご利用ください。"
            " 解析用の数値として正しく扱いたい場合、GeoTIFF（特に LAB 実数 float32）をご利用ください。"
        )
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")

st.caption("本アプリは研究・教育用です。")

