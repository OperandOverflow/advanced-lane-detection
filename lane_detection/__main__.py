"""Command line entry point: `python -m lane_detection {image,video} ...`."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .lane_finding import draw_poly_lines
from .pipeline import LaneFindingPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'output_images'


def run_image(args, pipeline):
    img = cv2.imread(str(args.input))
    if img is None:
        sys.exit(f'Could not read image: {args.input}')

    detection = pipeline.detect(img)

    output = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f'{Path(args.input).stem}_lanes.jpg'
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), detection.overlay)
    print(f'Wrote {output}')

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        stages = {
            'binary': detection.binary * 255,
            'binary_warped': detection.binary_warped * 255,
            'poly_fit': draw_poly_lines(detection.binary_warped, detection.left_fitx,
                                        detection.right_fitx, detection.ploty),
        }
        for name, stage_img in stages.items():
            path = debug_dir / f'{stem}_{name}.jpg'
            cv2.imwrite(str(path), stage_img)
            print(f'Wrote {path}')

    print(f'Left curve radius [m]:  {detection.left_curverad:.1f}')
    print(f'Right curve radius [m]: {detection.right_curverad:.1f}')
    print(f'Center offset [m]:      {detection.veh_pos:.3f}')
    print(f'Left fit (px):  {np.array2string(detection.left_fit, precision=6)}')
    print(f'Right fit (px): {np.array2string(detection.right_fit, precision=6)}')


def run_video(args, pipeline):
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        sys.exit(f'Could not open video: {args.input}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = Path(args.output) if args.output else REPO_ROOT / f'{Path(args.input).stem}_output.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    start = time.perf_counter()
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(pipeline.process(frame))
            frames += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start
    print(f'Wrote {output} ({frames} frames in {elapsed:.1f} s)')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lane_detection', description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)

    image_parser = subparsers.add_parser('image', help='detect lanes on a single image')
    image_parser.add_argument('input', type=Path)
    image_parser.add_argument('-o', '--output', type=Path,
                              help='output image (default: output_images/<name>_lanes.jpg)')
    image_parser.add_argument('--debug-dir', type=Path,
                              help='also save the binary, warped and polynomial fit stages here')

    video_parser = subparsers.add_parser('video', help='detect lanes on every frame of a video')
    video_parser.add_argument('input', type=Path)
    video_parser.add_argument('-o', '--output', type=Path, help='output video (default: <name>_output.mp4)')

    args = parser.parse_args(argv)

    pipeline = LaneFindingPipeline()

    if args.mode == 'image':
        run_image(args, pipeline)
    else:
        run_video(args, pipeline)


if __name__ == '__main__':
    main()
