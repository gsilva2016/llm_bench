import argparse
import time
import sys
import openvino_genai as ov_genai
import questions

# https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.PerfMetrics.html

parser = argparse.ArgumentParser()
parser.add_argument("--model_id")
parser.add_argument("--device")
parser.add_argument("--save_file")
parser.add_argument("--enable_kvcache", default=False, action="store_true")
parser.add_argument("--enable_streaming", default=False, action="store_true")
args = parser.parse_args()

model_id = args.model_id
device = args.device
bench_results_file = args.save_file
enable_kvcache = args.enable_kvcache

print("Model_ID: ", model_id)
print("Device: ", device)
print("Results File: ", bench_results_file)
print("Enable KVCache: ", enable_kvcache)

all_q = questions.get_all_questions()
#scheduler_config = ov_genai.SchedulerConfig()
#scheduler_config.enable_prefix_caching = False
#scheduler_config.max_num_batched_tokens = sys.maxsize
pipe = ov_genai.LLMPipeline(model_id, device)
#pipe = ov_genai.LLMPipeline(model_id, device, scheduler_config=scheduler_config)
config = pipe.get_generation_config()

print("Start latency benchmark...")

MAX_NEW_TOKENS = 250
NUM_RUNS = 3
total_t = 0
total_runs_t = 0
num_questions = 0
total_inferences = 0

config.max_new_tokens = MAX_NEW_TOKENS
config.temperature = 0.2
#config.top_k = 50
#config.top_p = 0.9
#config.repetition_penalty = 1.2

with open(bench_results_file, "w") as file:
    file.write("sep=|\n")
    file.write(f"Question #| Run #| Latency(msec)| Output| TTFT Avg Latency(msec)| TPOT Avg Latency(msec)| Throughput Avg. TPS| {NUM_RUNS} Runs Avg. Latency(msec)| All Runs Avg. Latency(msec)\n")

    perf_metrics = None
    for q in all_q:

        total_runs_t = 0
        num_questions = num_questions + 1
        quotes = '"'
        newline = '\n'

        if enable_kvcache:
            pipe.start_chat()
        for i in range(NUM_RUNS):
            total_inferences = total_inferences + 1
            start = time.time()
            result = pipe.generate([ q['question'] ], config, max_new_tokens=MAX_NEW_TOKENS)
            if perf_metrics is None:
                perf_metrics = result.perf_metrics
            else:
                perf_metrics = perf_metrics + result.perf_metrics
            
            end = time.time()
            latency = (end-start) * 1000
            total_runs_t = total_runs_t + (end-start)
            total_t = total_t + latency
            
            file.write(f"{num_questions}| {i+1}| {latency}| {quotes}{result.texts[0].replace('{newline}', '')}{quotes}| {result.perf_metrics.get_ttft().mean}| {result.perf_metrics.get_tpot().mean}| {result.perf_metrics.get_throughput().mean}|")
            if i < NUM_RUNS-1:
                file.write("N/A|N/A\n")
            
            #print(f"Question #{num_questions} Run #{i+1} latency: {latency}(ms), output: {result}")
            print(f"Question #{num_questions} Run #{i+1} latency: {latency}(ms) , ttft avg. latency {result.perf_metrics.get_ttft().mean}(ms) running ttft {perf_metrics.get_ttft().mean}, tpot avg. latency {result.perf_metrics.get_tpot().mean}(ms) running tpot {perf_metrics.get_tpot().mean}")

        n_runs_avg_latency = (total_runs_t / NUM_RUNS) * 1000
        print(f"{NUM_RUNS} runs avg. latency: {n_runs_avg_latency}(ms)")
        total_runs_t = 0
        if num_questions < len(all_q):
            file.write(f"{n_runs_avg_latency}| N/A\n")
        else:
            file.write(f"{n_runs_avg_latency}| ")
        print("Running Average Latency (msec): " , total_t / total_inferences)
    
    if enable_kvcache:
        pipe.finish_chat()
    all_runs_avg_latency = total_t / total_inferences
    file.write(f"{all_runs_avg_latency}\n")
    print(f"All runs avg. latency: {all_runs_avg_latency}(ms)")
    #print(f"Average Latency (all {total_runs_t}  questions): ", total_t / num_questions)
    print(f"Benchmark data saved to {bench_results_file}")
