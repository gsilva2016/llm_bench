#!/bin/bash
#

python3 main.py --model_id llama-8b-int8 --device GPU.1 --save_file bench-llm-llama-8b-int8-a770.csv --enable_kvcache
python3 main.py --model_id llama-8b-int8 --device GPU.1 --save_file bench-llm-llama-8b-int8-a770.csv

python3 main.py --model_id llama-8b-int4 --device GPU.1 --save_file bench-llm-llama-8b-int4-a770.csv --enable_kvcache
python3 main.py --model_id llama-8b-int4 --device GPU.1 --save_file bench-llm-llama-8b-int4-a770.csv

python3 main.py --model_id llama-8b-fp16 --device GPU.1 --save_file bench-llm-llama-8b-fp16-a770.csv
python3 main.py --model_id llama-8b-fp16 --device GPU.1 --save_file bench-llm-llama-8b-fp16-a770.csv --enable_kvcache

python3 main.py --model_id llama-8b-int4 --device GPU.0 --save_file bench-llm-llama-8b-int4-b580.csv
python3 main.py --model_id llama-8b-int4 --device GPU.0 --save_file bench-llm-llama-8b-int4-b580.csv --enable_kvcache

python3 main.py --model_id llama-8b-int8 --device GPU.0 --save_file bench-llm-llama-8b-int8-b580.csv
python3 main.py --model_id llama-8b-int8 --device GPU.0 --save_file bench-llm-llama-8b-int8-b580.csv --enable_kvcache

python3 main.py --model_id llama-8b-int4 --device GPU.0 --save_file bench-llm-llama-8b-int4-258v.csv
python3 main.py --model_id llama-8b-int4 --device GPU.0 --save_file bench-llm-llama-8b-int4-258v.csv --enable_kvcache

