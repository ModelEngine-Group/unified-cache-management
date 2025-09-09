ckpt_path=$1
config_path=$2
num_gpus=$3
frame_extraction_fps=$4

PYTHONPATH=$PYTHONPATH:./ python retake/infer_eval.py \
--hf_qwen2vl7b_path $ckpt_path \
--config_path $config_path \
--n_gpus $num_gpus \
--video_frame_extraction_fps $frame_extraction_fps \
"${@:5}"