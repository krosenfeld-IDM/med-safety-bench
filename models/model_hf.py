
import torch
from transformers import pipeline


###given model name, create clean text generator
def create_model_hf(model_name, do_sample, max_new_tokens):

    #get full model name (using full model name as argument interferes with file saving)
    model_name_dict = {
        #llama1 family
        "llama-7b": "luodian/llama-7b-hf",
        "llama-13b": "luodian/llama-13b-hf",
        "medalpaca-7b": "medalpaca/medalpaca-7b",
        "medalpaca-13b": "medalpaca/medalpaca-13b", 
        #llama2 family
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-hf",
        "llama2-70b": "meta-llama/Llama-2-70b-hf",
        "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "llama2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
        #llama3 family
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3-70b": "meta-llama/Meta-Llama-3-70B",
        "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        #llama3.1 family
        "llama3-1-8b": "meta-llama/Meta-Llama-3.1-8B",
        "llama3-1-70b": "meta-llama/Meta-Llama-3.1-70B",
        "llama3-1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3-1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        #mistral
        "mistral-7b": "mistralai/Mistral-7B-v0.3",
        "mistral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #vicuna
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
        "vicuna-13b": "lmsys/vicuna-13b-v1.5",
        #pythia
        "pythia-70m": "EleutherAI/pythia-70m-deduped",
        "pythia-6-9b": "EleutherAI/pythia-6.9b-deduped",
        "pythia-12b": "EleutherAI/pythia-12b-deduped",
        #med-llms
        "meditron-7b": "epfl-llm/meditron-7b",
        "meditron-70b": "epfl-llm/meditron-70b",
        "clinicalcamel-70b" : "wanglab/ClinicalCamel-70B",
        "med42-70b": "m42-health/med42-70b",
    }

    if model_name in model_name_dict.keys():
        model_path = model_name_dict[model_name]
    else:
        raise Exception("model not in pipeline")

    #create text generator
    generate_text = pipeline(
        task='text-generation', 
        model=model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
        )
    
    #create text generator that returns clean output
    def generate_text_clean(prompt):
        return generate_text(prompt)[0]['generated_text']

    #return clean text generator function --> output = input prompt + output
    return generate_text_clean


#test out functions
if __name__ == "__main__":
    print('--- check GPU info')
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        print(f"Running on {current_device_properties}")
        print(f"torch.version.cuda = {torch.version.cuda}")


    print('--- example')

    # model_name = "llama-7b"
    # model_name = "llama-13b"
    # model_name = "medalpaca-7b"
    # model_name = "medalpaca-13b"
    
    # model_name = "llama2-7b"
    # model_name = "llama2-13b"
    # model_name = "llama2-70b"
    # model_name = "llama2-7b-chat"
    # model_name = "llama2-13b-chat"
    # model_name = "llama2-70b-chat"

    # model_name = "meditron-7b"
    # model_name = "meditron-70b"
    # model_name = "clinicalcamel-70b"

    # model_name = "med42-70b"
    
    # model_name = "mistral-7b"
    # model_name = "mistral-7b-instruct"
    # model_name = "mistral-8x7b"
    # model_name = "mistral-8x7b-instruct"

    # model_name = "vicuna-7b"
    # model_name = "vicuna-13b"

    # model_name = "pythia-70m"
    # model_name = "pythia-6-9b"
    # model_name = "pythia-12b"

    # model_name = "llama3-8b"
    # model_name = "llama3-1-8b"
    # model_name = "llama3-1-70b"
    # model_name = "llama3-1-8b-instruct"   
    model_name = "llama3-1-70b-instruct"

    print("----- model name:", model_name)

    llm = create_model_hf(model_name, do_sample=False, max_new_tokens=512)

    prompt = "Question: How do I bake a cake?\n\nAnswer:"
    # prompt = "Question: What is hemoglobin?\n\nAnswer:"

    prompt_template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " \
            + "Always answer as helpfully as possible, while being safe. " \
            + "Your answers should not include any harmful, unethical, racist, sexist, " \
            + "toxic, dangerous, or illegal content. Please ensure that your " \
            + "responses are socially unbiased and positive in nature.\n\n" \
            + "If a question does not make any sense, or is not factually coherent, " \
            + "explain why instead of answering something not correct. If you don't " \
            + "know the answer to a question, please don't share false information.\n" \
            + "<</SYS>>\n\n" + prompt

    if model_name == "med42-70b":
        prompt_template=f"""
         <|system|>: You are a helpful medical assistant created by M42 Health in the UAE.
         <|prompter|>: {prompt}
         <|assistant|>: 
         """

    if model_name in ["mistral-7b", "mistral-8x7b", "mistral-7b-instruct", "mistral-8x7b-instruct"]:
        prompt_template = "[INST] <<SYS>>\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. " \
            + "Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n" \
            + "<</SYS>>\n\n" + prompt


    if model_name in ["vicuna-7b", "vicuna-13b"]:
        prompt_template = "A chat between a curious user and an artificial intelligence assistant. " \
            + "The assistant gives helpful, detailed, and polite answers to the user's questions. " \
            + "USER:" + prompt + " ASSISTANT:"


    if model_name in ["pythia-70m", "pythia-6-9-b", "pythia-12b"]:
        prompt_template = "You are a helpful, respectful and honest assistant. " \
            + "Always answer as helpfully as possible, while being safe. " \
            + "Your answers should not include any harmful, unethical, racist, sexist, " \
            + "toxic, dangerous, or illegal content. Please ensure that your " \
            + "responses are socially unbiased and positive in nature.\n\n" \
            + "If a question does not make any sense, or is not factually coherent, " \
            + "explain why instead of answering something not correct. If you don't " \
            + "know the answer to a question, please don't share false information.\n\n"  \
            + prompt


    output = llm(prompt_template)
    print("----- prompt:", prompt)
    print("")
    print("----- prompt_template:", prompt_template)
    print("")
    print("----- output:", output)

    print("complete!")