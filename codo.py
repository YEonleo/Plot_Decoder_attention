import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class codo:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=24):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    
    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    
    def generate(self, input_text,input_text2,plot_layer,context, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1,  **kwargs):
        with torch.no_grad():
            all_ids = self.tokenizer(input_text+input_text2,  return_tensors="pt").input_ids.to(self.device)
            input_ids = self.tokenizer(input_text,  return_tensors="pt").input_ids.to(self.device)

            if mode == 'baseline':
                outputs = self.model(all_ids, output_attentions=True,output_hidden_states=True)
                lm_logits = []
                for i in outputs.hidden_states:
                    head = self.model.lm_head(i)
                    lm_logits.append(head)

            save_path = os.path.join("./attention_plots", f"results_{input_text2[-3:]}_context_{context}")
            os.makedirs(save_path, exist_ok=True)

            if plot_layer == 'all':
                visualize_attention_for_generated_tokens_across_layers(outputs.attentions, self.tokenizer, all_ids, input_ids, save_path)
                visualize_all_tokens_across_all_layers(outputs.attentions, self.tokenizer, all_ids, input_ids, lm_logits, save_path)
            elif plot_layer == 'tokens_across':
                visualize_attention_for_generated_tokens_across_layers(outputs.attentions, self.tokenizer, all_ids, input_ids, save_path)
            elif plot_layer == 'all_tokens_across':
                visualize_all_tokens_across_all_layers(outputs.attentions, self.tokenizer, all_ids, input_ids, lm_logits, save_path)


        output_str = ''
        
        return output_str

# 각 Layer에서 attention score 나오게

def visualize_all_tokens_across_all_layers(attentions, tokenizer, all_ids, input_ids, lm_logits, save_path):
    try:
        # 현재 시간을 기준으로 결과 폴더 이름 설정
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과를 저장할 기본 경로 설정
        base_save_path = os.path.join(save_path, "detailed_results", current_time_str)
        
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        output_tokens = tokenizer.convert_ids_to_tokens(all_ids.squeeze().tolist())[len(input_tokens):]

        for layer_index, layer_attention in enumerate(attentions):
            # 각 레이어별로 결과 폴더 생성
            layer_save_path = os.path.join(base_save_path, f"Layer_{layer_index + 1}")
            os.makedirs(layer_save_path, exist_ok=True)
            
            for token_index, _ in enumerate(output_tokens):
                # 결과 이미지 파일 경로 설정
                image_save_path = os.path.join(layer_save_path, f"Token_{token_index + 1}_Combined.png")
                
                fig, axs = plt.subplots(1, 2, figsize=(70, 8))
                # 어텐션 스코어 시각화 부분
                attention_scores = layer_attention[0, :, input_ids.size(-1) + token_index, :].mean(0).cpu().numpy()
                token_labels = input_tokens + output_tokens[:token_index + 1]
                axs[0].bar(range(len(token_labels)), attention_scores[:len(token_labels)])
                axs[0].set_xticks(range(len(token_labels)))
                axs[0].set_xticklabels(token_labels, rotation=90)
                axs[0].set_title(f"Layer {layer_index + 1} Token {token_index + 1} Attention Scores")

                # 상위 토큰 확률 시각화 부분
                if lm_logits:
                    logits = lm_logits[layer_index][0, input_ids.size(-1) + token_index, :]
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, 10, dim=-1)
                    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.cpu().numpy().flatten())
                    top_probs = top_probs.cpu().numpy().flatten()
                    sns.barplot(x=top_probs, y=top_tokens, ax=axs[1])
                    axs[1].set_title(f"Layer {layer_index + 1} Token {token_index + 1} Top Tokens by Probability")

                plt.tight_layout()
                plt.savefig(image_save_path)
                plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# 모든 Layer에서 attention token 나오게

def visualize_attention_for_generated_tokens_across_layers(attentions, tokenizer, all_ids, input_ids, save_path):
    # 현재 시간을 기반으로 저장 경로 설정
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_save_path = os.path.join(save_path, "attention_for_generated_tokens", current_time_str)
    os.makedirs(detailed_save_path, exist_ok=True)

    # 입력 및 생성 토큰 준비
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    all_tokens = tokenizer.convert_ids_to_tokens(all_ids.squeeze().tolist())
    # <s> 토큰의 인덱스 찾기
    s_token_index = all_tokens.index('<s>') if '<s>' in all_tokens else -1
    generated_token_indices = range(len(input_tokens), len(all_tokens))  # 생성된 토큰의 인덱스

    for gen_token_index in generated_token_indices:
        generated_token_attention_across_layers = []

        # 각 레이어에서 생성된 특정 토큰에 대한 어텐션 스코어 추출
        for layer_attention in attentions:
            attention_scores = layer_attention[0, :, gen_token_index, :].mean(0).cpu().numpy()
            generated_token_attention_across_layers.append(attention_scores)

        # 레이어별 어텐션 스코어 시각화 (역순으로 레이어 표시)
        plt.figure(figsize=(60, 8))
        # <s> 토큰 제외하고 시각화, 자기 자신에 대한 어텐션 점수도 제외
        exclude_indices = [gen_token_index, s_token_index] if s_token_index != -1 else [gen_token_index]
        filtered_attention_scores = np.array(generated_token_attention_across_layers)[:, [i for i in range(gen_token_index+1) if i not in exclude_indices]]
        sns.heatmap(
            filtered_attention_scores[::-1],  # 레이어 순서를 역순으로
            cmap="viridis",
            xticklabels=[token for i, token in enumerate(all_tokens[:gen_token_index+1]) if i not in exclude_indices],  # <s> 토큰과 자기 자신 제외
            yticklabels=[f"Layer {len(attentions) - i}" for i in range(len(attentions))][::-1],  # 레이어 번호 역순으로
        )
        plt.title(f"Attention Across All Layers for Generated Token: '{all_tokens[gen_token_index]}' (Excluding self-attention and <s> token)")
        plt.xlabel("Tokens (Input + Generated up to current, excluding self and <s>)")
        plt.ylabel("Layer")

        plt.tight_layout()
        plt.savefig(os.path.join(detailed_save_path, f"token_{gen_token_index}.png"))
        plt.close()


