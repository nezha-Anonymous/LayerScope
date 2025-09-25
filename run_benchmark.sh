#!/bin/bash


MODELS=(
    "/home/share/bz/model/Mixtral-8x7B-v0.1"
    "/home/share/bz/model/Qwen3-30B-A3B" 
    "/home/share/bz/model/DeepSeek-V2-Lite"
    "/home/share/bz/model/Moonlight-16B-A3B-Instruct"
)
# BATCH_SIZES=(4 8 16 32 64)
BATCH_SIZES=(4 )

for model in "${MODELS[@]}"; do

    for bs in "${BATCH_SIZES[@]}"; do
        echo "============================================"
        echo "Running benchmark for model: $model"
        echo "With batch_size: $bs"
        echo "============================================"
        

        # python latency.py --model "$model" --batch_size "$bs"
        python microbench.py --model "$model"                
        echo ""
        echo "Benchmark for $model with batch_size=$bs completed"
        echo ""
    done
done

echo "All benchmarks completed!"