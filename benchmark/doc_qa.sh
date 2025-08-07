#!/bin/bash

model=""
results_dir=""
sizes=""
ip=""
port=""
connector=""
device=""

while getopts ":m:d:l:i:p:c:b:" opt; do
    case $opt in
        m)
            model=$OPTARG
            ;;
        d)
            results_dir=$OPTARG
            ;;
        l)
            sizes=$OPTARG
            ;;
        i)
            ip=$OPTARG
            ;;
        p)
            port=$OPTARG
            ;;
        c)
            connector=$OPTARG
            ;;
        b)
            device=$OPTARG
            ;;
        \?)
            echo "Invalid argument: -$OPTARG" >&2
            echo "Usage: $0 [-m <models] [-d <result_output_directory>] [-l <context_lengths>] [-i <LLM server ip>] [-p <LLM server port>] [-c <connector to use in UCM>] [-b <backend compute device>]"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG needs an argument" >&2
            echo "Usage: $0 [-m <models] [-d <result_output_directory>] [-l <context_lengths>] [-i <LLM server ip>] [-p <LLM server port>] [-c <connector to use in UCM>] [-b <backend compute device>]"
            exit 1
            ;;
    esac
done

if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    if [ $? -ne 0 ]; then
        echo "mkdir $results_dir failed"
    fi
fi

IFS=',' read -r -a lengths <<< "$sizes"

allowed_values=(2 4 8 16 32 64)

for length in "${lengths[@]}"; do
    is_valid=false
    for allowed in "${allowed_values[@]}"; do
        if [ "$length" -eq "$allowed" ]; then
            is_valid=true
            break
        fi
    done
    if [ "$is_valid" == false ]; then
        echo "context length $length is invalid. Supported lengths: 2/4/8/16/32/64. Use comma to separate different lengths, e.g. -l 2,4,8,16"
        exit 1
    fi
done

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://${ip}:${port}/v1"

python warm_up.py --model ${model} --ip ${ip} --port ${port}

model_name=$(basename "$model")

for length in "${lengths[@]}"; do

    doc_qa_result_file_path="$results_dir/docqa_TTFT_${length}k_${model_name}_${connector}_connector_${device}.jsonl"
    if [ -f "$doc_qa_result_file_path" ]; then
        rm "$doc_qa_result_file_path"
    fi

    python token_benchmark_ray.py --model $model --mean-output-tokens 10 --stddev-output-tokens 1 --timeout 600 --num-concurrent-requests 1 --results-dir $results_dir --llm-api openai --context-length $length --scenario doc-qa --connector $connector --device $device
    python token_benchmark_ray.py --model $model --mean-output-tokens 10 --stddev-output-tokens 1 --timeout 600 --num-concurrent-requests 1 --results-dir $results_dir --llm-api openai --context-length $length --scenario doc-qa --connector $connector --device $device

    echo "TTFT results with length ${length}k have been saved into file $doc_qa_result_file_path"

done

python plot_doc_qa.py --context-lengths $sizes --result-path $results_dir --model $model --connector $connector --device $device