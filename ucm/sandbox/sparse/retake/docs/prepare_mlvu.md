## Prepare MLVU Dataset


### Step 1: Download MLVU dataset from [huggingface](https://huggingface.co/datasets/MLVU/MVLU)
```bash
git clone https://huggingface.co/datasets/MLVU/MVLU
git clone https://huggingface.co/datasets/MLVU/MLVU_Test
```

Denote the root directory of download MLVU dataset as `mlvu_root`, it should has the following structure:
```
${mlvu_root}/
├── MLVU/
    ├── json
        ...
    ├── video
        ...
├── figs/
```

Denote the root directory of download MLVU-Test dataset as `mlvu_test_root`, it should has the following structure:
```
${mlvu_test_root}/
├── MLVU_Test/
    ├── test_question.json
    ├── test_video.tar.gz.part-aa
    ├── test_video.tar.gz.part-ab
    ...
├── figs/
├── test_generation_tasks.json
├── test_multi_choice_tasks.json
```

Unzip MLVU-Test videos:
```bash
cd MLVU_Test
cat test_video.tar.gz.part-* | tar -xzvf -
```


### Step 2: Extract frames of all videos
```bash
cd ${retake_root}
python scripts/utils/frame_extraction.py \
--videofile_tpl ${mlvu_root}/MLVU/video/'*/*.mp4' \
--results_dir ${mlvu_root}/MLVU/video_25fps \
--fps 25 \
--num_workers 32
python scripts/utils/frame_extraction.py \
--videofile_tpl ${mlvu_test_root}/MLVU_Test/video/'*/*.mp4' \
--results_dir ${mlvu_test_root}/MLVU_Test/video_25fps \
--fps 25 \
--num_workers 32
```


### Step 3: Build MLVU dataset
```bash
cd ${retake_root}
python scripts/utils/build_mlvu_dataset.py --hf_root ${mlvu_root}
python scripts/utils/build_mlvu_test_dataset.py --hf_root ${mlvu_test_root}
```
Note that you can NOT modify folder `${mlvu_root}/MLVU/video_25fps` and `${mlvu_test_root}/MLVU_Test/video_25fps` after this step, since the absolute path of extracted frames are written into annotation files `mlvu.json` and `mlvu_test.json`:
```
retake_root/
├── dataset/
    ├── mlvu/
        ├── mlvu.json
        ├── mlvu_test.json
├── ...
```