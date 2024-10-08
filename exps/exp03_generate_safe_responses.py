"""
Ask LLM to respond to prompts in csv file
"""

import logging
import argparse
import datetime as dt
import torch
import pandas as pd
from tqdm import tqdm
from functools import partial

#locate diretories
import os
import sys
path_project_folder = os.path.dirname(os.path.dirname(__file__)) #get path of file's directory two levels up (get path of project folder)
sys.path.append(path_project_folder) #add project folder to path to sys.path to be able to load modules

import utils.logging
from utils.exps import log_info
from models.model_openai import llm_openai



### set up logging
logging_level = 15
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
standard_streamhandler = utils.logging.get_standard_streamhandler()
logger.addHandler(standard_streamhandler)


#parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Experiment arguments")

    parser.add_argument(
        "--model-name", 
        default="gpt3", 
        help="Name of LLM to use (note: gpt3 is gpt3.5)", 
        type=str,
        choices=["gpt3", "gpt4"])
    
    parser.add_argument(
        "--dataset", 
        default="med_harm_gpt4", 
        help="Dataset name (will generate safe responses for the train or test set)", 
        type=str)

    parser.add_argument(
        "--subdataset", 
        default="category_1", 
        help="Name of subdataset of prompts to evaluate", 
        type=str)
    #choices depend on dataset
    # choices = [
    #     "category_1", "category_2", "category_3",
    #     "category_4", "category_5", "category_6",
    #     "category_7", "category_8", "category_9",
    # ]
    
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":

    ###start experiment: record experiment info
    time_start = dt.datetime.utcnow() #start time
    args = parse_args() #parse arguments
    log_info(logger, args, time_start) #log info

    ###create model   
    logger.info('\n--- load model')
    max_new_tokens = 512

    if args.model_name=="gpt4" or args.model_name=="gpt3":
        sys_prompt = "You are a helpful, respectful and honest assistant. " \
            + "Always answer as helpfully as possible, while being safe. " \
            + "Your answers should not include any harmful, unethical, racist, sexist, " \
            + "toxic, dangerous, or illegal content. Please ensure that your " \
            + "responses are socially unbiased and positive in nature.\n\n" \
            + "If a question does not make any sense, or is not factually coherent, " \
            + "explain why instead of answering something not correct. If you don't " \
            + "know the answer to a question, please don't share false information."

        gpt_name_dict = {"gpt4": "gpt-4-1106-preview", "gpt3": "gpt-3.5-turbo-0125"}
        llm = partial(llm_openai, model_name=gpt_name_dict[args.model_name], sys_prompt=sys_prompt, temperature=0.0, max_new_tokens=max_new_tokens)
    
    ###load dataset, train set
    # train_or_test = 'train'
    train_or_test = 'test'
    logger.info('\n--- prompt model')
    prompts_df = pd.read_csv(f"datasets/{args.dataset}/{train_or_test}/{args.subdataset}.csv", header=None)
    prompts_df.columns = ['question']    

    ###format prompt for each model
    if args.model_name in ["gpt3", "gpt4"]:
        prompts_df['prompt'] = prompts_df.apply(
            lambda row: "The following is an unethical question or instruction. " \
            + "Please politely decline to answer it and explain why it is unethical. " \
            + "Please keep your response brief (1-3 sentences). " \
            + "\n\n" + "Question or instruction: " + row['question'] + "\n\nResponse: ",
            axis=1)
    
    ###get model responses
    #apply function to each row/prompt
    # tqdm.pandas(miniters=25, maxinterval=3600) #show progress bar every 25 rows
    tqdm.pandas()
    prompts_df['response'] = prompts_df['prompt'].progress_apply(llm)

    #save responses (csv) to results folder
    exp_tag = "03_generate_safe_responses"
    results_folder = f"{path_project_folder}/exps/results/exp{exp_tag}/{train_or_test}/csv/{args.dataset}"
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    results_filename = f"safe_responses_{args.model_name}_{args.dataset}_{args.subdataset}.csv"
    prompts_df.to_csv(f"{results_folder}/{results_filename}", index=False)
    logger.info(f".csv file saved at: {results_folder}/{results_filename}")

    ###format model responses into json for model training
    #create columns used in json file
    prompts_df["instruction"] = prompts_df["question"]
    prompts_df["output"] = prompts_df["response"]
    prompts_df["input"] = ""
    df_for_json = prompts_df[["instruction", "output", "input"]]

    #save responses (json) to results folder
    results_folder = f"{path_project_folder}/exps/results/exp{exp_tag}/{train_or_test}/json/{args.dataset}"
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    json_filename = f"safe_responses_{args.model_name}_{args.dataset}_{args.subdataset}.json"
    df_for_json.to_json(f"{results_folder}/{json_filename}", orient="records", indent=4)
    logger.info(f".json file saved at: {results_folder}/{json_filename}")

    ###end experiment
    time_end = dt.datetime.utcnow()
    logger.info(f"\nUTC time (end): {time_end:%Y-%m-%d %H:%M:%S}")
    logger.info(f"Full run time: {str(time_end-time_start).split('.')[0]} (H:M:S)")
    logger.info(f"Complete!")