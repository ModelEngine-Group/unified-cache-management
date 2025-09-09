import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_video(videofile, results_dir, fps, overwrite_existing_results):
    videoframe_dir = os.path.join(results_dir, os.path.basename(videofile).replace(".mp4", ""))
    if os.path.exists(videoframe_dir):
        if not overwrite_existing_results:
            print(f"Skipping {videofile} as results already exist.")
            return
        os.system(f'rm -rf {videoframe_dir}')
    try:
        os.makedirs(videoframe_dir)
    except:
        # MLVU has duplicated videos in differnet subclasses
        return
    framefile_tpl = os.path.join(videoframe_dir, "%06d.jpg")
    print(f"{videofile} -> {framefile_tpl}")
    os.system(f'ffmpeg -i {videofile} -vf fps={fps} -vsync vfr {framefile_tpl} -hide_banner -loglevel error')
    # NOTE: You can add `-q:v 1` to achieve the highest frame quality, but it will be slower.
    # We did not add `-q:v 1`. If is useful for those models trained with high quality image only.

def main(args):
    videofiles = glob.glob(args.videofile_tpl)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_video, videofiles, [args.results_dir]*len(videofiles), 
                              [args.fps]*len(videofiles), [args.overwrite_existing_results]*len(videofiles)), 
                  total=len(videofiles)))

if __name__ == "__main__":
    """
    For each video `*/videoname.mp4`, this script creats a directory
    {results_dir}/videoname and puts extracted frames in it.
    """
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--videofile_tpl', type=str, required=True, 
                        help='Template for video files, e.g., /path/to/videos/*.mp4')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Directory where extracted frames will be stored')
    parser.add_argument('--num_workers', type=int, default=32, 
                        help='Number of parallel workers for processing videos')
    parser.add_argument('--fps', type=int, default=25, 
                        help='Frames per second for extracted frames')
    parser.add_argument('--overwrite_existing_results', action='store_true', 
                        help='Overwrite existing results if they exist')

    args = parser.parse_args()
    main(args)
