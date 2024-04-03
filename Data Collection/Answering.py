from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
import time
from LLMConnection import getLLMResponse

load_dotenv()

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def get_final_answers(context, subproblems, final_question):
    systemInstructions = "You are a helpful bot that can answer reasoning questions based off board game sitations"
    responses = []
    prompt = f"Here is the context of the reasoning question: \n{context}"
    # Send the context to ChatGPT and get an initial response
    getLLMResponse(systemInstructions, prompt)

    for subproblem in subproblems:
        prompt = f"Answer the following subproblem based on the context provided to you earlier: \n{subproblem}"
        chat_response = getLLMResponse(systemInstructions, prompt)
        responses.append(chat_response)

    final_prompt = f"Answer the final question below based on what we have discussed till now. Finally, also state wheather it is proved , disproved or unkown. \n{final_question}"
    # Finally, ask the main question and get the response
    final_answer = getLLMResponse(systemInstructions, final_prompt)

    return responses, final_answer

def main():

    dataPath = './Data Processing/output-with-subproblems.json'
    df = pd.read_json(dataPath, dtype=str)

    FINAL_ANS_DIR = './final_answers_output.json'
    saveDf = None
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        context = row['context']
        subproblems = row['llama-subproblems'].split('||')
        final_question = row['question']
        
        subproblem_responses, final_answer = get_final_answers(context, subproblems, final_question)
        
        if saveDf is None:
            saveDf = pd.DataFrame(columns=df.columns.tolist() + ['subproblem_responses', 'final_answer'])
            saveDf = saveDf.astype(str)
        else:
            saveDf = pd.read_json(FINAL_ANS_DIR, dtype=str)

        new_record = row.to_dict()
        new_record['subproblem_responses'] = "||".join(subproblem_responses)
        new_record['final_answer'] = final_answer

        temp_df = pd.DataFrame([new_record])
        saveDf = pd.concat([saveDf, temp_df], ignore_index=True)
        saveDf.to_json(FINAL_ANS_DIR, orient='records')



if __name__ == "__main__":
    main()

    # Load data
    # get initial response
    # break down questions using reasoning (custom model, LLAMA)
# Iteratively answer the subquestions based on the model, context, and subquestions
# get a final answer
# compare