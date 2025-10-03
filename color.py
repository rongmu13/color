# app.py  —— 全面日语化 UI（#注释保留中文）
import io
from pathlib import Path
import numpy as np
import streamlit as st

from PIL import Image
import piexif  # 仅用于 JPEG EXIF 复制
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine

import cv2

# ---------------------------
# 基本設定（页面设置）
# ---------------------------
st.set_page_config(page_title="RGB → 色空間変換 Shinshu Univ. R.Y.", layout="wide")

st.title("RGB → 色空間変換 Shinshu Univ. R.Y.")
st.caption(
    "RGB画像（GeoTIFF/TIFF/JPG/PNG）をアップロードし、右側で変換先の色空間を選択してください。"
    "変換後の画像をダウンロードできます。地理参照（投影・アフィン）や EXIF/GPS 情報の保持に対応します。"
)

with st.sidebar:
    st.header("⚙️ 設定")
    color_space = st.selectbox(
        "変換先の色空間",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0,
        help="一般的に用いられる色空間を選択できます。Grayは1チャンネルになります。"
    )
    enable_8bit_scale = st.checkbox(
        "高ビット深度画像を8bitへスケーリングしてから変換する（推奨）",
        value=True,
        help="ドローンのGeoTIFFは10/12/16bitが多いです。OpenCVの色空間変換は8bitが安定です。"
    )
    p_low = st.slider("下位パーセンタイル（スケーリング）", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上位パーセンタイル（スケーリング）", 90.0, 100.0, 99.0, 0.5)

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
# ユーティリティ（工具函数）
# ---------------------------
def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    """把任意 dtype 的数组按百分位缩放到 uint8 [0,255]"""
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
    if rgb_like.ndim == 3 and rgb_like.shape[0] in (3,4) and (rgb_like.shape[2] not in (3,4)):
        rgb_like = np.transpose(rgb_like, (1, 2, 0))
    return rgb_like

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
    """缩放预览图"""
    h, w = img_hwc_uint8.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    preview = img_hwc_uint8
    if scale < 1.0:
        preview = cv2.resize(preview, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(preview)

def copy_exif_jpeg(src_bytes, dst_bytes_io):
    """将 JPEG 的 EXIF 原样复制到新 JPEG（如果有）"""
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
# メイン処理（主逻辑）
# ---------------------------
if uploaded is None:
    st.info("👆 まず画像をアップロードしてください（GeoTIFF/TIFF/JPG/PNG に対応）。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# ===== A：GeoTIFF/TIFF（地理参照あり） =====
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF 処理（ブロック単位でメモリ節約）")
    try:
        with rasterio.MemoryFile(uploaded.read()) as memfile:
            with memfile.open() as src:
                # 基本情報の取得
                h, w = src.height, src.width
                count = src.count
                crs = src.crs
                transform: Affine = src.transform
                profile = src.profile.copy()
                tags_global = src.tags()
                tags_per_band = [src.tags(i+1) for i in range(min(3, count))]

                bands_to_read = min(3, count)
                st.write(f"原画像サイズ：{w} × {h}／バンド数：{count}（変換に使用：先頭{bands_to_read}バンド）")

                block_size = 1024
                out_channels = 1 if color_space == "Gray" else 3

                out_profile = profile.copy()
                out_profile.update({
                    "count": out_channels,
                    "dtype": "uint8",
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

                            if enable_8bit_scale:
                                arr_u8 = percentile_scale_to_uint8(arr, p_low, p_high)
                            else:
                                arr_u8 = arr
                                if arr_u8.dtype != np.uint8:
                                    arr_u8 = np.clip(arr_u8, 0, 255).astype(np.uint8)

                            out_block = convert_colorspace(arr_u8, color_space)
                            for ch in range(out_channels):
                                dst.write(out_block[:, :, ch], indexes=ch + 1, window=win)

                    # タグ・地理参照の引き継ぎ
                    dst.update_tags(**tags_global)
                    for ch in range(min(len(tags_per_band), out_channels)):
                        try:
                            dst.update_tags(ch + 1, **tags_per_band[ch])
                        except Exception:
                            pass

                # プレビュー生成（ここは indexes= が正しい）
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
                    preview = ensure_hwc(preview).astype(np.uint8)
                    if preview.shape[2] == 1:
                        preview = np.repeat(preview, 3, axis=2)
                    st.image(preview, caption=f"プレビュー（{color_space}）", use_container_width=True)

                out_bytes = out_mem.read()
                out_name = Path(filename).stem + f"_{color_space}.tif"
                st.download_button(
                    "⬇️ 変換結果をダウンロード（GeoTIFF／投影・タグ保持）",
                    data=out_bytes,
                    file_name=out_name,
                    mime="image/tiff"
                )
        st.success("✅ 変換が完了しました。GeoTIFF の投影・座標・タグは可能な限り引き継がれています。"
                   "色空間変換により画素値が変化するのは正常な挙動です。")
    except Exception as e:
        st.error("エラーが発生しました。GeoTIFF の読み込みまたは書き出しに失敗しました。"
                 "ファイル形式・破損・メモリ不足などをご確認ください。")

# ===== B：JPEG/PNG（EXIF/GPS の保持は可能な範囲で） =====
else:
    st.subheader("🧭 JPEG/PNG 処理（可能なら EXIF/GPS をコピー）")
    try:
        src_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        rgb = np.array(pil)

        if enable_8bit_scale and rgb.dtype != np.uint8:
            rgb = percentile_scale_to_uint8(rgb, p_low, p_high)

        out_img = convert_colorspace(rgb, color_space)

        prev = out_img
        if prev.shape[2] == 1:
            prev = np.repeat(prev, 3, axis=2)
        st.image(pil_preview(prev, max_side=preview_max), caption=f"プレビュー（{color_space}）", use_container_width=True)

        # PNG 出力（EXIFなし・解析向け）
        png_io = io.BytesIO()
        save_for_png = out_img.squeeze() if out_img.shape[2] == 1 else out_img
        Image.fromarray(save_for_png).save(png_io, format="PNG", compress_level=6)
        png_name = Path(filename).stem + f"_{color_space}.png"
        st.download_button("⬇️ PNG をダウンロード（解析向け）", data=png_io.getvalue(), file_name=png_name, mime="image/png")

        # JPEG 出力（EXIF/GPS を可能ならコピー）
        jpg_io = io.BytesIO()
        Image.fromarray(save_for_png).save(jpg_io, format="JPEG", quality=95)
        exif_ok = False
        if is_jpeg:
            exif_ok = copy_exif_jpeg(src_bytes, jpg_io)

        jpeg_name = Path(filename).stem + f"_{color_space}.jpg"
        btn_label = "⬇️ JPEG をダウンロード（EXIF/GPS を可能なら保持）"
        if is_jpeg:
            btn_label += " ✅EXIFコピー済" if exif_ok else " ⚠EXIFコピー不可の場合あり"
        st.download_button(btn_label, data=jpg_io.getvalue(), file_name=jpeg_name, mime="image/jpeg")

        st.info(
            "注：PNG は EXIF を保持しません。GPS 情報が必要な場合は、元が JPEG のときに上の JPEG ダウンロードをご利用ください。"
            "地理情報を正しく保持できる標準的な形式が必要な場合、GeoTIFF のご利用を推奨します。"
        )
    except Exception as e:
        st.error("エラーが発生しました。画像の読み込みまたは変換に失敗しました。ファイル形式や破損状態をご確認ください。")

st.caption(
    "本アプリは研究・教育用です。"
)


