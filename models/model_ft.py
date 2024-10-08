from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

import os
import sys
path_project_folder = os.path.dirname(os.path.dirname(__file__)) #get path of file's directory two levels up (get path of project folder)
sys.path.append(path_project_folder) #add project folder to path to sys.path to be able to load modules
from models.access import hf_repo_ft

def create_model_ft(ft_model_name, do_sample, max_new_tokens):
    
    #load base model
    if ft_model_name.startswith('meditron-7b_'):
        model_name = 'epfl-llm/meditron-7b'
    elif ft_model_name.startswith('medalpaca-7b_'):
        model_name = 'medalpaca/medalpaca-7b'
    elif ft_model_name.startswith('medalpaca-13b_'):
        model_name = 'medalpaca/medalpaca-13b'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    #load adapters
    adapters_name = f"{hf_repo_ft}/{ft_model_name}"
    model = PeftModel.from_pretrained(model, adapters_name)

    #merge to create ft model
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name) #ft model has same tokenizer as base model

    #create text generator that returns clean output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def generate_text_clean(prompt):
        #tokenize input
        prompt_tokens = tokenizer(prompt, return_tensors='pt')["input_ids"].to(device)

        #generate
        model_output = model.generate(
            inputs=prompt_tokens, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.pad_token_id, 
            do_sample=do_sample, 
            max_new_tokens=max_new_tokens)
        
        return tokenizer.decode(model_output[0]) #decoded model output
    
    #return clean text generator function --> output = input prompt + output
    return generate_text_clean






if __name__ == "__main__":
    print('--- check GPU info')
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        print(f"Running on {current_device_properties}")
        print(f"torch.version.cuda = {torch.version.cuda}")


    print('--- example')

    #create text_generator
    ft_model_name = "meditron-7b_gen_n600"
    # ft_model_name = "meditron-7b_med_n600"
    # ft_model_name = "medalpaca-7b_med_n800"
    # ft_model_name = "medalpaca-13b_both_n800"

    print(f'model: {ft_model_name}')

    do_sample = False
    max_new_tokens = 512
    text_generator = create_model_ft(ft_model_name, do_sample, max_new_tokens)

    #test it out an example
    prompt = "Question: What is a diabetes? Answer:"
    response = text_generator(prompt)
    print('prompt: ')
    print(prompt)
    print('response: ')
    print(response)

    print("Complete!")