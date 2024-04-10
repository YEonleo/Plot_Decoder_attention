export CUDA_VISIBLE_DEVICES=1

python CCde.py --model-name huggyllama/llama-7b --plot_layer 'all' --context 'true'

python CCde.py --model-name huggyllama/llama-7b --plot_layer 'all' --context 'false'