# LLM Bench
Benchmark LLMs using OpenVINO GenAI

## Install needed software
```
sudo apt install python3-venv
#python3 -m venv genai
#source genai/bin/activate
conda create --name genai python=3.11 -y
conda activate genai
python -m pip install --upgrade-strategy eager "optimum-intel[openvino]==1.27.0"
#pip install optimum-intel==1.27.0
#pip install nncf
pip install openvino-genai
```

## Authenticate with HF for downloading models

```
huggingface-cli login --token <YOUR TOKEN>
```

## Download models and convert to OpenVINO format

```
optimum-cli export openvino --model meta-llama/Llama-3.1-8B-Instruct --weight-format fp16 --trust-remote-code llama-8b-fp16
optimum-cli export openvino --model meta-llama/Llama-3.1-8B-Instruct --weight-format int8 --trust-remote-code llama-8b-int8
optimum-cli export openvino --model meta-llama/Llama-3.1-8B-Instruct --weight-format int4 --trust-remote-code llama-8b-int4
```

## Run benchmark 

Edit the run-bench.sh with your devices. The example run-bench.sh is configured for accessing discrete Intel GPUs e.g. GPU.0 and GPU.1 but can be set to GPU for using an integrated GPU instead.

```
./run-bench.sh
```
