import cv2
import numpy as np
from typing import Tuple


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
