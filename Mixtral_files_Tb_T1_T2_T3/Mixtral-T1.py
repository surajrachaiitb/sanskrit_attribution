from llama_cpp import Llama
import numpy as np
import pandas as pd 

#------------------------------------------------------------


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
    n_ctx=5000,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35  # The number of layers to offload to GPU, if you have GPU acceleration available
)


#------------------------------------------------------------

def word_string(i):
    string = """"""

    for key, values in shr_meaning[i].items():
        #print(key, values)
        string += f'{key}'
        string += ', '

    return string


#--------------------------------------------------------------

results = {"id": [], "sloka": [], "translation": []}

data = pd.read_csv("your data file path")
shr_meaning = np.load('shr word file path', allow_pickle=True).item()


#--------------------------------------------------------------


# Iterate over the rows in the CSV file
for _, row in data.iterrows():
    #id_val = i+1
    #i = id_val
    s = row["sloka"]
    sloka = s
    id = row['id']
    words = word_string(id)
    #t1 = row['translation']


    print(f"Processing ID: {id}")
    
    # Construct the prompt for Mistral
    my_prompt = f'''[INST]
    Your task is to translate Sanskrit verse into English.
    You are provided with the Sanskrit verse and its corresponding word-split (Sanadhi-vichchhed). Which means you have the individual sanskrit words of which the verse is composed of. Leverage the english meanings of the word splits to enhance the translation by adding more accurate and relevant details which you learn from the word split meanings.

    Examples of good translated sentences:
    1. A Dvija or a learnt person should carefully recite the Vedas everyday. For, this is the prime duty; all other duties are called secondary
    2. Not knowing others that Brahman, namely Rishyasringa, the best, will always be abiding his father, lest his renowned celibacy always praised by the Brahmans, will be hindered.
    3. O son of Kuntī, I am the taste of water, the light of the sun and the moon, the syllable oṁ in the Vedic mantras; I am the sound in ether and ability in man.
    
    Example:
    Sanskrit verse: konvasmin sāmprataṃ loke guṇavān kaśca vīryavān
    Word level split: kon, asmin, sampratah, lokah, gunavan, kasca, viryah, van
    Enhaced translation: Who in this world is currently endowed with all good qualities and is brave?

    Please generate accurate, sensible, and contextually correct translations. Strictly provide only the translation and no other filler text. Use Sanskrit words in translation only if relevant. Don't repeat the Sanskrit words unnecessarily in the translations.

    Sanskrit Verse:
    {sloka}

    Word level split:
    {words}

    Enhaced translation:
    [/INST]'''



    # Simple inference example
    output = llm(
        my_prompt,       # Use your custom prompt
        max_tokens=512,  # Generate up to 512 tokens
        stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
        echo=True        # Whether to echo the prompt
    )

    # Print the output
    #print(output)
    # Extract the last output translation
    last_output_translation = output['choices'][-1]['text']

    # Print the last output translation
    #rint(last_output_translation)
    # Find the index where the translation begins
    inst_end_index = last_output_translation.find("[/INST]")

    # Extract the text after the [/INST] tag
    translation = last_output_translation[inst_end_index + len("[/INST]"):].strip()

    # Print the extracted translation text
    print("----------------------")
    print(sloka)
    print("----------------------")
    print(words)
    print("----------------------")
    #print(t1)
    #print("----------------------")
    print(translation)
    #print(translation

    # Add results to dictionary
    results["id"].append(id)
    results["sloka"].append(sloka)
    results["translation"].append(translation)


#---------------------------------------------------------
    

# Convert result dictionary to DataFrame
result_df = pd.DataFrame(results)
result_df.to_csv('save your file.csv', index = False)