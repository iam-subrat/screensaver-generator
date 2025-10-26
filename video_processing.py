import cv2
import numpy as np
from typing import Tuple, Optional


def fit_and_pad_image_to_size(
    img: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """
    Resize `img` to fit inside (target_w, target_h) while preserving aspect ratio.
    If the resized image doesn't fill the target, pad with the image border pixels
    (replicating the edge pixel values) so the result exactly equals (target_h, target_w).

    Inputs:
    - img: HxWxC BGR image
    - target_w, target_h: desired output width and height (pixels)

    Output: padded image of shape (target_h, target_w, C)
    """
    if img is None:
        raise ValueError("img is None")

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid image with zero width or height")

    # scale to fit inside the rectangle
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # choose interpolation method: downscale -> INTER_AREA, upscale -> INTER_LINEAR
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # compute padding to center the resized image
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # replicate the edge pixels to fill padding (uses the images border pixel value)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    # As a safety, crop or pad if rounding caused off-by-one
    padded = padded[:target_h, :target_w]
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        # if still wrong, create blank and paste centered
        out = np.zeros((target_h, target_w, resized.shape[2]), dtype=resized.dtype)
        y0 = (target_h - resized.shape[0]) // 2
        x0 = (target_w - resized.shape[1]) // 2
        out[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        # replicate border for any remaining pixels
        out = cv2.copyMakeBorder(
            out,
            0,
            max(0, target_h - out.shape[0]),
            0,
            max(0, target_w - out.shape[1]),
            borderType=cv2.BORDER_REPLICATE,
        )
        return out

    return padded


def apply_image_to_video(
    video_path: str, image_path: str, rect: Tuple[int, int, int, int], out_path: str
):
    """
    For every frame in `video_path`, place the `image_path` resized to the rectangle `rect`.
    `rect` is (x_min, y_min, x_max, y_max) inclusive coordinates in the video's frame space.
    The image is resized to fit while preserving aspect ratio and padded (replicated edge)
    so it exactly fills the rectangle. The modified frames are saved to `out_path`.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or cannot be opened: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    x_min, y_min, x_max, y_max = rect
    rect_w = x_max - x_min + 1
    rect_h = y_max - y_min + 1

    # Precompute the padded image once (same for all frames). If you want to vary per-frame, move inside loop.
    patch = fit_and_pad_image_to_size(img, rect_w, rect_h)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure patch channels match frame (both BGR)
        if patch.shape[2] == 3 and frame.shape[2] == 3:
            frame[y_min : y_max + 1, x_min : x_max + 1] = patch
        else:
            # fallback: convert patch if needed
            patch_conv = patch
            if patch.shape[2] != frame.shape[2]:
                patch_conv = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            frame[y_min : y_max + 1, x_min : x_max + 1] = patch_conv

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# Example usage (commented out):
# if __name__ == '__main__':
#     apply_image_to_video('generated-video.mp4', 'some_image.png', (494, 236, 783, 424), 'output.mp4')


def find_largest_color_rectangle(
    frame: np.ndarray,
    target_color: np.ndarray,
    tolerance: int = 40,
    min_area: int = 1000,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the largest axis-aligned bounding rectangle for pixels close to target_color (BGR) in `frame`.
    Returns (x_min, y_min, x_max, y_max) or None.
    """
    # Convert to HSV and perform hue-based matching for robustness to lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Convert the target BGR color to HSV
    tc_bgr = (
        np.uint8([[target_color[::-1]]])
        if isinstance(target_color, (tuple, list))
        else np.uint8([[target_color[::-1]]])
    )
    tc_hsv = cv2.cvtColor(tc_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    th, ts, tv = int(tc_hsv[0]), int(tc_hsv[1]), int(tc_hsv[2])
    # tolerance controls hue window; set reasonable sat/value thresholds
    hue_tol = max(8, int(tolerance * 0.4))
    sat_tol = max(40, int(tolerance * 0.6))
    val_tol = max(40, int(tolerance * 0.6))

    low1 = np.array(
        [max(0, th - hue_tol), max(30, ts - sat_tol), max(30, tv - val_tol)]
    )
    high1 = np.array([min(179, th + hue_tol), 255, 255])
    # handle hue wrap-around
    if th - hue_tol < 0:
        low2 = np.array(
            [179 + (th - hue_tol), max(30, ts - sat_tol), max(30, tv - val_tol)]
        )
        high2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, low1, high1)
        mask2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif th + hue_tol > 179:
        low2 = np.array([0, max(30, ts - sat_tol), max(30, tv - val_tol)])
        high2 = np.array([th + hue_tol - 180, 255, 255])
        mask1 = cv2.inRange(hsv, low1, high1)
        mask2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, low1, high1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    largest_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area and area > max_area:
            max_area = area
            largest_rect = (x, y, x + w - 1, y + h - 1)
    return largest_rect


def apply_image_to_video_per_frame(
    video_path: str,
    image_path: str,
    target_color: Tuple[int, int, int],
    out_path: str,
    tolerance: int = 40,
    min_area: int = 1000,
):
    """
    For every frame in `video_path`, detect the largest region matching `target_color` and place
    `image_path` resized & padded into that rectangle. The detection runs per-frame, so moving
    or changing rectangles will be followed.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or cannot be opened: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    # last_rect_float stores smoothed center_x, center_y, width, height as floats
    last_rect_float = None
    last_patch = None
    last_overlay_patch = None
    missing_decay = 1.0

    # Temporal smoothing and blending parameters
    # smoothing_alpha: how fast to track new detections (1.0 = instant, lower = smoother)
    # blend_alpha: when overlaying, weight of the current patch (1.0 = no blending, 0.0 = keep previous)
    smoothing_alpha = 0.6
    blend_alpha = 0.9
    hold_when_missing = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rect = find_largest_color_rectangle(
            frame,
            np.array(target_color, dtype=np.uint8),
            tolerance=tolerance,
            min_area=min_area,
        )

        if rect is not None:
            # reset missing decay when we have a detection
            missing_decay = 1.0

            x_min, y_min, x_max, y_max = rect
            rect_w = x_max - x_min + 1
            rect_h = y_max - y_min + 1
            center_x = x_min + rect_w / 2.0
            center_y = y_min + rect_h / 2.0

            new_rect = np.array(
                [center_x, center_y, float(rect_w), float(rect_h)], dtype=np.float32
            )

            if last_rect_float is None:
                last_rect_float = new_rect.copy()
            else:
                # exponential smoothing
                last_rect_float = (
                    smoothing_alpha * new_rect
                    + (1.0 - smoothing_alpha) * last_rect_float
                )

            # compute integer rectangle from smoothed float rect
            cx, cy, w_f, h_f = last_rect_float
            w_i = max(1, int(round(w_f)))
            h_i = max(1, int(round(h_f)))
            x_min_i = int(round(cx - w_i / 2.0))
            y_min_i = int(round(cy - h_i / 2.0))
            x_max_i = x_min_i + w_i - 1
            y_max_i = y_min_i + h_i - 1

            # ensure inside frame bounds
            x_min_i = max(0, x_min_i)
            y_min_i = max(0, y_min_i)
            x_max_i = min(frame.shape[1] - 1, x_max_i)
            y_max_i = min(frame.shape[0] - 1, y_max_i)

            # recompute final width/height after clipping
            final_w = x_max_i - x_min_i + 1
            final_h = y_max_i - y_min_i + 1

            # reuse patch if same size
            if (
                last_patch is not None
                and last_patch.shape[1] == final_w
                and last_patch.shape[0] == final_h
            ):
                patch = last_patch
            else:
                patch = fit_and_pad_image_to_size(img, final_w, final_h)
                last_patch = patch

            # blending with previous overlay to reduce flicker
            effective_blend = blend_alpha
            if last_overlay_patch is not None:
                # if sizes mismatch, reset previous overlay
                if last_overlay_patch.shape[:2] != patch.shape[:2]:
                    last_overlay_patch = None

            if last_overlay_patch is not None:
                # convert to float for blending
                cur = patch.astype(np.float32)
                prev = last_overlay_patch.astype(np.float32)
                blended = (
                    effective_blend * cur + (1.0 - effective_blend) * prev
                ).astype(patch.dtype)
                overlay = blended
            else:
                overlay = patch

            # apply overlay to the frame region
            frame[y_min_i : y_max_i + 1, x_min_i : x_max_i + 1] = overlay
            last_overlay_patch = overlay
        else:
            # no detection in this frame
            if (
                hold_when_missing
                and last_rect_float is not None
                and last_patch is not None
            ):
                # gradually decay how strongly we keep showing the last patch
                missing_decay *= 0.95
                if missing_decay < 0.05:
                    # stop showing after long absence
                    last_overlay_patch = None
                    # optionally: clear last_rect_float to require a fresh detection later
                    # last_rect_float = None
                else:
                    # compute integer rectangle from last_rect_float and place last_overlay_patch with reduced alpha
                    cx, cy, w_f, h_f = last_rect_float
                    w_i = max(1, int(round(w_f)))
                    h_i = max(1, int(round(h_f)))
                    x_min_i = int(round(cx - w_i / 2.0))
                    y_min_i = int(round(cy - h_i / 2.0))
                    x_max_i = x_min_i + w_i - 1
                    y_max_i = y_min_i + h_i - 1
                    x_min_i = max(0, x_min_i)
                    y_min_i = max(0, y_min_i)
                    x_max_i = min(frame.shape[1] - 1, x_max_i)
                    y_max_i = min(frame.shape[0] - 1, y_max_i)

                    if last_overlay_patch is not None:
                        # blend last overlay with the current frame using decay as alpha
                        alpha = missing_decay
                        over = last_overlay_patch.astype(np.float32)
                        region = frame[
                            y_min_i : y_max_i + 1, x_min_i : x_max_i + 1
                        ].astype(np.float32)
                        composited = (alpha * over + (1.0 - alpha) * region).astype(
                            frame.dtype
                        )
                        frame[y_min_i : y_max_i + 1, x_min_i : x_max_i + 1] = composited

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# End of file
