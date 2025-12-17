import argparse
import time
import openvino_genai as ov_genai
import questions

parser = argparse.ArgumentParser()
parser.add_argument("--model_id")
parser.add_argument("--device")
parser.add_argument("--save_file")
args = parser.parse_args()


# GPU.0 B580
# GPU.1 A770
device = "GPU.1"
#model_id = "llama-8b-fp16"
model_id = "llama-8b-int4"
#model_id = "llama-8b-int8"
model_id = args.model_id
device = args.device
bench_results_file = args.save_file


all_q = questions.get_all_questions()
pipe = ov_genai.LLMPipeline(model_id, device)  # Use CPU or GPU as devices without any other code change
config = pipe.get_generation_config()

print("Start latency benchmark...")

MAX_NEW_TOKENS = 250
NUM_RUNS = 3
total_t = 0
total_runs_t = 0
num_questions = 0

config.max_new_tokens = MAX_NEW_TOKENS
config.temperature = 0.2
#config.top_k = 50
#config.top_p = 0.9
#config.repetition_penalty = 1.2

with open(bench_results_file, "w") as file:
    file.write("sep=|\n")
    file.write(f"Question #| Run #| Latency(sec)| Output| {NUM_RUNS} Runs Avg. Latency(sec)| All Runs Avg. Latency(sec)\n")

    for q in all_q:

        total_runs_t = 0
        num_questions = num_questions + 1

        for i in range(NUM_RUNS):
            start = time.time()
            result = pipe.generate(q['question'], config, max_new_tokens=MAX_NEW_TOKENS)
            end = time.time()
            latency = (end-start) * 1000
            total_runs_t = total_runs_t + (end-start)  
            total_t = total_t + (end-start)

            file.write(f"{num_questions}| {i+1}| {latency}| \"{result.replace('\n', '')}\"|")
            if i < NUM_RUNS-1:
                file.write("N/A|N/A\n")
            #else:
            #    file.write(f"{}, N/A\n")
            #print("Result: ", result)
            #print(f"Question #{num_questions} Run #{i+1} latency: {latency}(ms), output: {result}")
            print(f"Question #{num_questions} Run #{i+1} latency: {latency}(ms)")

        n_runs_avg_latency = (total_runs_t / NUM_RUNS) * 1000
        print(f"{NUM_RUNS} runs avg. latency: {n_runs_avg_latency}(ms)")
        total_runs_t = 0
        if num_questions < len(all_q):
            file.write(f"{n_runs_avg_latency}| N/A\n")
        else:
            file.write(f"{n_runs_avg_latency}| ")
        #print("Average Latency (question): ", total_runs_t / NUM_RUNS)

    all_runs_avg_latency = total_t / num_questions
    # BUG with all runs avg latency below. Compute this manually for now with Excel...
    file.write(f"{all_runs_avg_latency}\n")
    print(f"All runs avg. latency: {all_runs_avg_latency}(ms)")
    #print(f"Average Latency (all {total_runs_t}  questions): ", total_t / num_questions)
    print(f"Benchmark data saved to {bench_results_file}")
