# Llama-3.1-8B
# profiling phase
CUDA_VISIBLE_DEVICES=0 python main.py --display_results --active --model meta-llama/Llama-3.1-8B-Instruct --cpu_memory_limit 160 --gpu_memory_limit 12 
# running phase
CUDA_VISIBLE_DEVICES=0 python main.py --total_questions 2000 --batch_size 8 --dyn_pipeline --arrival_rates 2 4 6 8 10 --rate_change_interval 600 --model meta-llama/Llama-3.1-8B-Instruct --cpu_memory_limit 160 --gpu_memory_limit 12
# vllm
CUDA_VISIBLE_DEVICES=0 python main.py --total_questions 2000 --batch_size 16 --vllm --arrival_rates 2 4 6 8 10  --rate_change_interval 600 --resident_partitions 5 --model meta-llama/Llama-3.1-8B-Instruct --cpu_memory_limit 160 --gpu_memory_limit 12 


# Llama-3.1-70B
# profiling phase
CUDA_VISIBLE_DEVICES=0 python main.py --display_results --active --model meta-llama/Llama-3.1-70B-Instruct --cpu_memory_limit 256 --gpu_memory_limit 24 

# running phase
CUDA_VISIBLE_DEVICES=0 python main.py --total_questions 2000 --batch_size 8 --dyn_pipeline --arrival_rates 10 15 20 --rate_change_interval 300  --model meta-llama/Llama-3.1-70B-Instruct --cpu_memory_limit 256 --gpu_memory_limit 24
# vllm
CUDA_VISIBLE_DEVICES=0 python main.py --total_questions 2000 --batch_size 16 --vllm --arrival_rates 10 15 20 --model meta-llama/Llama-3.1-70B-Instruct --rate_change_interval 300 --resident_partitions 2 --cpu_memory_limit 256 --gpu_memory_limit 24 

