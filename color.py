import io
import os
from pathlib import Path
import numpy as np
import streamlit as st

# 图像 & 元数据相关库
from PIL import Image
import piexif  # 仅用于 JPEG EXIF 复制
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine

# 颜色空间转换（OpenCV-Headless）
import cv2

# ---------------------------
# 基本设置
# ---------------------------
st.set_page_config(page_title="RGB → 多色彩空间转换（含GPS/投影保留）", layout="wide")

st.title("📷 RGB → LAB/HSV/XYZ… 转换（尽量保留 GPS/投影/像素信息）")
st.caption("上传 RGB 图像（支持 GeoTIFF / TIFF / JPG / PNG），选择右侧色彩空间，下载结果。尽量保留原有地理参考或 EXIF GPS。")

with st.sidebar:
    st.header("⚙️ 转换设置")
    color_space = st.selectbox(
        "目标色彩空间",
        ["LAB", "HSV", "HLS", "YCrCb", "XYZ", "LUV", "Gray"],
        index=0,
        help="常用色彩空间可选。注意：非 RGB 空间保存为普通 3 通道（或 Gray 1 通道）影像数据，用于分析；普通看图软件显示可能“怪异”。"
    )
    # 对 16-bit/浮点数据缩放到 8-bit 供 OpenCV 转换
    enable_8bit_scale = st.checkbox(
        "将高位深影像缩放到 8-bit 后再转换（推荐）",
        value=True,
        help="无人机 GeoTIFF 常为 10/12/16-bit。OpenCV 的多数颜色转换对 8-bit 最稳。开启后按 1～99 百分位做线性缩放。"
    )
    p_low = st.slider("下百分位(用于缩放)", 0.0, 10.0, 1.0, 0.5)
    p_high = st.slider("上百分位(用于缩放)", 90.0, 100.0, 99.0, 0.5)

    preview_max = st.slider("预览最大边尺寸(px)", 256, 2048, 1024, 128,
                            help="仅影响页面预览，不影响下载结果。")

uploaded = st.file_uploader(
    "上传图像（.tif/.tiff/.jpg/.jpeg/.png）",
    type=["tif", "tiff", "jpg", "jpeg", "png"]
)

# ---------------------------
# 工具函数
# ---------------------------
def percentile_scale_to_uint8(arr, p_low=1, p_high=99):
    """把任意 dtype 的 3D/HWC 或 2D 数组按百分位缩放到 uint8 [0,255]"""
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
    """把 (bands, H, W) 或 (H, W, bands) 统一成 (H, W, 3)"""
    if rgb_like.ndim == 3 and rgb_like.shape[0] in (3,4) and rgb_like.shape[2] not in (3,4):
        # (bands, H, W) -> (H, W, bands)
        rgb_like = np.transpose(rgb_like, (1, 2, 0))
    return rgb_like

def to_bgr(img_hwc_uint8):
    """OpenCV 大多以 BGR 输入；我们标准化输入为 RGB，再转 BGR"""
    return cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2BGR)

def convert_colorspace(img_rgb_uint8, mode):
    """RGB uint8(HWC) -> 目标色彩空间，输出 uint8 HWC（Gray 为 HxWx1）"""
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
        out = out[..., None]  # 保持三维
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
    """
    将 JPEG 的 EXIF 原样复制到新 JPEG（如果有）。
    src_bytes: 原图字节
    dst_bytes_io: 已写好 JPEG 图像数据的 BytesIO
    """
    try:
        exif_dict = piexif.load(src_bytes)
        exif_bytes = piexif.dump(exif_dict)
        # 把 exif 写入已存在的 JPEG 字节（需要重写）
        img = Image.open(io.BytesIO(dst_bytes_io.getvalue()))
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", exif=exif_bytes, quality=95)
        dst_bytes_io.seek(0)
        dst_bytes_io.truncate(0)
        dst_bytes_io.write(out_io.getvalue())
        return True
    except Exception:
        return False

def infer_is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))

def infer_is_jpeg(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg"))

# ---------------------------
# 处理主逻辑
# ---------------------------
if uploaded is None:
    st.info("👆 请先上传一张图片（GeoTIFF/TIFF/JPG/PNG）。")
    st.stop()

filename = uploaded.name
is_tiff = infer_is_tiff(filename)
is_jpeg = infer_is_jpeg(filename)

# ========== 分支 A：GeoTIFF/TIFF（含地理参考） ==========
if is_tiff:
    st.subheader("🗺 GeoTIFF/TIFF 处理（分块不爆内存）")
    with rasterio.MemoryFile(uploaded.read()) as memfile:
        with memfile.open() as src:
            # 读取基本信息
            h, w = src.height, src.width
            count = src.count
            crs = src.crs
            transform: Affine = src.transform
            profile = src.profile.copy()
            tags_global = src.tags()  # 全局标签
            tags_per_band = [src.tags(i+1) for i in range(min(3, count))]

            # 只取前三个波段作为 RGB（若不足 3，则填充）
            bands_to_read = min(3, count)
            st.write(f"原图尺寸：{w}×{h}，波段数：{count}（用于转换的前三个波段：{bands_to_read}）")

            # 分块读取 + 可选缩放到 8-bit，再做颜色转换
            block_size = 1024  # 可根据机器性能调整
            out_channels = 1 if color_space == "Gray" else 3

            # 输出文件（GeoTIFF）写入到内存，再提供下载
            out_profile = profile.copy()
            out_profile.update({
                "count": out_channels,
                "dtype": "uint8",  # 转换后统一为 8-bit，便于可视化与互操作
                "compress": "deflate",
                "predictor": 2
            })

            out_mem = rasterio.MemoryFile()
            with out_mem.open(**out_profile) as dst:
                # 逐块处理
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        win = Window(col_off=x, row_off=y,
                                     width=min(block_size, w - x),
                                     height=min(block_size, h - y))
                        # 读前三个波段
                        arr = src.read(indexes=list(range(1, bands_to_read + 1)), window=win)
                        arr = ensure_hwc(arr)  # -> HWC
                        if arr.shape[2] < 3:
                            # 不足 3 通道则补齐
                            pads = [arr[:,:, -1]] * (3 - arr.shape[2])
                            arr = np.concatenate([arr] + [p[..., None] for p in pads], axis=2)

                        # 缩放到 uint8（若开启）
                        if enable_8bit_scale:
                            arr_u8 = percentile_scale_to_uint8(arr, p_low, p_high)
                        else:
                            # 若不是 uint8，做安全裁剪转换
                            arr_u8 = arr
                            if arr_u8.dtype != np.uint8:
                                arr_u8 = np.clip(arr_u8, 0, 255).astype(np.uint8)

                        # 颜色空间转换（要求 RGB 顺序）
                        out_block = convert_colorspace(arr_u8, color_space)  # HWC
                        # 写出（分通道）
                        for ch in range(out_channels):
                            dst.write(out_block[:, :, ch], indexes=ch + 1, window=win)

                # 复制地理参考和标签
                dst.update_tags(**tags_global)
                # 尽量复制每个波段的标签（若通道数对应得上）
                for ch in range(min(len(tags_per_band), out_channels)):
                    try:
                        dst.update_tags(ch + 1, **tags_per_band[ch])
                    except Exception:
                        pass

                # 复制投影/仿射（rasterio 写入 profile 已包含）
                # 提示：rasterio 会保留 transform、crs、width、height 等核心地理信息

            # 生成预览（缩略图）
            with out_mem.open() as preview_ds:
                # 采样读取小图用于显示
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
                st.image(preview, caption=f"预览（{color_space}）", use_container_width=True)

            # 准备下载
            out_bytes = out_mem.read()
            out_name = Path(filename).stem + f"_{color_space}.tif"
            st.download_button(
                "⬇️ 下载转换结果（GeoTIFF，保留投影/标签）",
                data=out_bytes,
                file_name=out_name,
                mime="image/tiff"
            )
    st.success("✅ 转换完成。GeoTIFF 的投影/坐标/标签等已尽量保留。注意：像素值因色彩空间转换而改变是正常现象。")

# ========== 分支 B：JPEG/PNG（尽量保留 EXIF，特别是 JPEG 的 GPS） ==========
else:
    st.subheader("🧭 JPEG/PNG 处理（复制 EXIF/GPS 到结果 JPEG）")

    src_bytes = uploaded.read()
    pil = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    rgb = np.array(pil)  # HWC, uint8

    # 预处理（一般相机/无人机 JPG 已是 8-bit，不需缩放）
    if enable_8bit_scale and rgb.dtype != np.uint8:
        rgb = percentile_scale_to_uint8(rgb, p_low, p_high)

    out_img = convert_colorspace(rgb, color_space)  # HWC（gray -> HxWx1）

    # 供页面预览
    prev = out_img
    if prev.shape[2] == 1:
        prev = np.repeat(prev, 3, axis=2)
    st.image(pil_preview(prev, max_side=preview_max), caption=f"预览（{color_space}）", use_container_width=True)

    # 导出：两个版本
    # 1) PNG（无 EXIF，通用）
    png_io = io.BytesIO()
    save_for_png = out_img.squeeze() if out_img.shape[2] == 1 else out_img
    Image.fromarray(save_for_png).save(png_io, format="PNG", compress_level=6)
    png_name = Path(filename).stem + f"_{color_space}.png"
    st.download_button("⬇️ 下载 PNG（推荐做分析用）", data=png_io.getvalue(), file_name=png_name, mime="image/png")

    # 2) JPEG（尝试复制 EXIF/GPS；注意部分色彩空间+JPEG非标准，主要为“携带元数据”的容器）
    jpg_io = io.BytesIO()
    Image.fromarray(save_for_png).save(jpg_io, format="JPEG", quality=95)
    exif_ok = False
    if is_jpeg:
        exif_ok = copy_exif_jpeg(src_bytes, jpg_io)

    jpeg_name = Path(filename).stem + f"_{color_space}.jpg"
    btn_label = "⬇️ 下载 JPEG（尝试保留 EXIF/GPS）" + (" ✅ EXIF已复制" if exif_ok else " ⚠️ EXIF可能无法复制")
    st.download_button(btn_label, data=jpg_io.getvalue(), file_name=jpeg_name, mime="image/jpeg")

    st.info("提示：PNG 不携带 EXIF；若你需要带 GPS 的结果，请使用上方 JPEG 下载（若原图为 JPEG）。"
            "如果需要“带 GPS 的标准地理影像容器”，建议把源数据保存为 GeoTIFF 再处理。")

st.caption("小贴士：色彩空间的像素值与人眼直观颜色不是一回事；转换后图像用于分析/特征提取最合适。")

