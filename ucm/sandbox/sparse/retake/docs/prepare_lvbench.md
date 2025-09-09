## Prepare LVBench Dataset


### Step 1: Download LVBench data from [huggingface](https://huggingface.co/datasets/THUDM/LVBench/tree/main)
```bash
git clone https://huggingface.co/datasets/THUDM/LVBench # Contain annotations only
git clone https://huggingface.co/datasets/AIWinter/LVBench # Contain videos only
```
Move all_files in `AIWinter/LVBench` into `THUDM/LVBench`.

Denote the root directory of download LVBench dataset as `lvbench_root`, it should has the following structure:
```
${lvbench_root}/
├── docs/
├── video_info.meta.jsonl
├── all_videos_split.zip.001
├── all_videos_split.zip.002
├── ...
└── all_videos_split.zip.014
```


### Step 2: Unzip everything
```bash
cd ${lvbench_root}
cat all_videos_split.zip.* > all_videos.zip
unzip all_videos.zip
```


### Step 3: Extract frames of all videos
```bash
cd ${retake_root}
python scripts/utils/frame_extraction.py \
--videofile_tpl ${lvbench_root}/all_videos/'*.mp4' \
--results_dir ${lvbench_root}/video_25fps \
--fps 25 \
--num_workers 32
```


### Step 4: Build LVBench dataset
```bash
cd ${retake_root}
python scripts/utils/build_lvbench_dataset.py --hf_root ${lvbench_root}
```
Note that you can NOT modify folder `${lvbench_root}/video_25fps` after this step, since the absolute path of extracted frames are written into annotation files `lvbench.json`:
```
retake_root/
├── dataset/
    ├── lvbench/
        ├── lvbench.json
├── ...
```