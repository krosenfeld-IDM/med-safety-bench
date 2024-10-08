#for locating folders
import sys
import os
path_project_folder = os.path.dirname(os.path.dirname(__file__)) #get path of file's directory two levels up (get path of project folder)
sys.path.append(path_project_folder) #add project folder to path to sys.path to be able to load modules

from models.access import access_token_openai

#login
os.environ['OPENAI_API_KEY'] = access_token_openai #must login before importing openai

import openai


CHAT_COMPLETION_API_MODEL_NAMES = [
    "gpt-4", "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", 
    "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0125"	
    ]

###given model_name, temperature, and prompt, RETURN output text
def llm_openai(prompt, model_name, sys_prompt, temperature, max_new_tokens):
    if model_name in CHAT_COMPLETION_API_MODEL_NAMES:
        # Call the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        # Get the response text from the API response
        response_clean = response['choices'][0]['message']['content']
        return response_clean



#test out functions
if __name__ == "__main__":
    from functools import partial

    #specify model
    # model_name = "gpt-3.5-turbo-0125"
    # model_name = "gpt-4-1106-preview"
    # model_name = "gpt-4o-2024-05-13"
    model_name = "gpt-4o-mini-2024-07-18"
    sys_prompt = "You are a helpful assistant."
    max_new_tokens = 512
    temperature = 0.0

    print("model:", model_name)

    #create llm
    llm = partial(llm_openai, model_name=model_name, sys_prompt=sys_prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    #ask questions
    print("--- example 1")
    prompt = "What happened in 2001?"
    output = llm(prompt=prompt)
    print(output)

    print("--- example 2")
    prompt = "Is the following text factual? Answer either Yes or No.\nText: Toronto is a city in the United States.\nAnswer:"
    output = llm(prompt=prompt)
    print(output)