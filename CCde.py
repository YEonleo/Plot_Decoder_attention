import torch
import pandas as pd
import argparse
from tqdm import tqdm
from codo import codo
from datasets import load_dataset
import os

def create_demo_text():
    question, answer = [], []
    
    question.append("Which magazine was started first Arthur's Magazine or First for Women?")
    answer.append("Arthur's Magazine..")

    question.append("The Oberoi family is part of a hotel company that has a head office in what city?")
    answer.append("Delhi.")

    question.append("Musician and satirist Allie Goertz wrote a song about the The Simpsons character Milhouse, who Matt Groening named after who?")
    answer.append("President Richard Nixon.")

    question.append("What nationality was James Henry Miller's wife?")
    answer.append("American.")

    question.append("Cadmium Chloride is slightly soluble in this chemical, it is also called what?")
    answer.append("alcohol.")

    question.append("Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?")
    answer.append("Jonathan Stark.")

    # Concatenate demonstration examples ...
    demo_text = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.And you can get information from context.' + '\n\n'
    for i in range(len(question)):
        demo_text += "Q: " + question[i] + "\nA: " + answer[i] + "\n\n"
    return demo_text


def create_prompt_from_dataset(dataset, include_context):

    
    prompts, answers = [], []
    for item in dataset:
        question = item['question']
        answer = item['answer']
        context = item.get('context', {})  # 안전하게 context를 가져옵니다. 'context' 키가 없다면 빈 사전을 반환합니다.
        supporting_facts = item.get('supporting_facts', {})  # 마찬가지로 안전하게 가져옵니다.

        extracted_facts = []
        # 컨텍스트와 지원 사실을 포함하는 경우에만 추출 로직을 실행합니다.
        if include_context and context and supporting_facts:
            for title, sent_ids in zip(supporting_facts.get('title', []), supporting_facts.get('sent_id', [])):
                if title in context.get('title', []):
                    title_index = context['title'].index(title)
                    if isinstance(sent_ids, list):
                        for sent_id in sent_ids:
                            if sent_id < len(context.get('sentences', [])[title_index]):
                                sentence = context['sentences'][title_index][sent_id]
                                extracted_facts.append(f'context : {sentence}')
                    elif isinstance(sent_ids, int):
                        if sent_ids < len(context.get('sentences', [])[title_index]):
                            sentence = context['sentences'][title_index][sent_ids]
                            extracted_facts.append(f'context : {sentence}')

        # 컨텍스트를 포함시키지 않는 경우, 지시문만을 추가합니다.
        instruction = create_demo_text()  # 적절한 지시문을 여기에 추가합니다.
        
        if extracted_facts:
            extracted_facts_str = '\n'.join(extracted_facts)
            prompt = f"{instruction}{extracted_facts_str}\n\nQ: {question}\nA: "
        else:
            # 컨텍스트가 없거나 포함하지 않기로 결정된 경우 질문만을 사용합니다.
            prompt = f"{instruction}\n\nQ: {question}\nA: "
        

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def load_csv(file_path):
    """Load and preprocess data from CSV file."""
    df = pd.read_csv(file_path)
    list_data = df.to_dict('records')
    return list_data

def main(args):
    text_data = load_dataset("hotpot_qa","fullwiki")

    train_dataset = text_data['train']
    validation_dataset = text_data['validation']
    test_dataset = text_data['test']

    list_data_dict = load_csv(args.data_path)

    llm = codo(args.model_name, args.device, args.num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:","context :"])

    #Layer에서 early exit
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        
    if args.context.lower() in ['true', '1', 't', 'y', 'yes']:
        include_context = True
    else:
        include_context = False
    prompts, answers = create_prompt_from_dataset(train_dataset, include_context=include_context)
    prompts = prompts[10:20]
    answers = answers[10:20]

    with torch.no_grad():
        for prompt, true_answer in tqdm(zip(prompts, answers), total=len(prompts)):
            # reference answers

            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value, post_softmax=False)

            generated_answer = llm.generate(prompt,true_answer,args.plot_layer,args.context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--data_path", type=str, default="./tfqa/TruthfulQA.csv")
    parser.add_argument("--max_gpu_memory", type=int, default=24)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    #plot 방식
    parser.add_argument("--plot_layer", type=str, default="all")
    parser.add_argument("--context", type=str, default='False')
    args = parser.parse_args()

    

    main(args)
