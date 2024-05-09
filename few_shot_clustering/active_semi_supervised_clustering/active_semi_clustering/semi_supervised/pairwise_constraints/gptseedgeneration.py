from collections import Counter, defaultdict
import json
import jsonlines
import numpy as np
from openai import OpenAI
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import time
from tqdm import tqdm
import pickle


from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.pairwise_constraints.pckmeans import PCKMeans
from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.semi_supervised.pairwise_constraints.gptclustering_prompts import select_keyphrase_expansion_prompt
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

"""
Object for Data Seed Generation in preparation for a clustering task
"""

# Get the ChatGPT Key from env variable
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

class GPTSeedGeneration():
    def __init__(self, labels, features, documents, encoder_model=None, dataset_name=None, key_phrase_prompt=None, seed_words_prompt=None, text_type=None, prompt_for_encoder=None, keep_original_entity=True, split=None, n_clusters=3, side_information=None, read_only=False, instruction_only=False, demonstration_only=False, cache_file_name="gpt_paraphrase_cache.jsonl"):
        self.X = features
        self.labels=labels
        self.dataset_name = dataset_name
        self.documents = documents
        self.seed_words_prompt=seed_words_prompt
        self.encoder_model = encoder_model
        # If a dataset is specified, then we'll automatically infer the correct encoder model to use.
        # Otherwise, a model object must be provided.
        if self.dataset_name is None:
            assert self.encoder_model is not None, "Provide an encoder model to use for keyphrase clustering"
        self.prompt = key_phrase_prompt
        self.text_type = text_type
        self.prompt_for_encoder = prompt_for_encoder
        self.keep_original_entity = keep_original_entity
        self.n_clusters = n_clusters
        self.side_information = side_information
        cache_file = cache_file_name
        self.instruction_only = instruction_only
        self.demonstration_only = demonstration_only
        if instruction_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_instruction_only" + "_cache.jsonl"
        elif demonstration_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_demonstration_only" + "_cache.jsonl"
        if os.path.exists(cache_file):
            self.cache_rows = list(jsonlines.open(cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(cache_file, mode='r')
        self.NUM_RETRIES = 1
        self.read_only = read_only
        self.centroids = []

        split_str = f"_{split}" if split else ""

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set

    def create_template_block(self, entity_idx, text_type):
        filled_template = f"""{text_type}: "{self.documents[entity_idx]}"

Keyphrases:"""
        return filled_template

    def construct_gpt3_template(self, doc_idx, instruction_only=False, demonstration_only=False):
        if self.dataset_name is not None:
            prompt_prefix = select_keyphrase_expansion_prompt(self.dataset_name)
        else:
            assert self.prompt is not None
            prompt_prefix = self.prompt

        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
            text_type = "Entity"
        elif self.dataset_name == "clinc" or self.dataset_name == "bank77":
            text_type = "Query"
        elif self.dataset_name == "tweet":
            text_type = "Tweet"
        else:
            assert self.text_type is not None
            text_type = self.text_type
        completion_block = self.create_template_block(doc_idx, text_type)
        return f"{prompt_prefix}\n\n{completion_block}"

    

    def generate(self):

        # 1- first part is to generate keyphrases for each document 

        document_expansion_mapping = {}
        for row in self.cache_rows:
            document_expansion_mapping[row["entity"]] = row["expansion"]

        print("nb of documents to generate keyphrases from: ",len(document_expansion_mapping))

        # For each document, generate the keyphrases from chatgpt request 
        for doc_idx, document in tqdm(enumerate(self.documents)):
            if document not in document_expansion_mapping:
                if self.read_only:
                    continue

                # Construct the gpt prompt for current document
                template_to_fill = self.construct_gpt3_template(doc_idx, instruction_only=self.instruction_only, demonstration_only=self.demonstration_only)
                print(f"PROMPT:\n{template_to_fill}")

                failure = True
                num_retries = 0
                while failure and num_retries < self.NUM_RETRIES:
                    cache_row = None
                    try:
                        start = time.perf_counter()

                        deployment_name="GPT-3-5-turbo-chat"

                        # GPT Request
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": template_to_fill},
                            ],
                        )

                        # GPT Response
                        message = response.choices[0].message.content
                        if message.startswith("Keywords:"):
                            message = message[len("Keywords:"):].strip()
                        try:
                            entity_expansions = json.loads(message)
                            print("-",message)
                            if not isinstance(entity_expansions, list) or not isinstance(entity_expansions[0], str):
                                failure = True
                            document_expansion_mapping[document] = entity_expansions
                            cache_row = {"entity": document, "expansion": entity_expansions}
                            self.cache_writer.write(cache_row)
                            failure = False
                        except:
                            time.sleep(0.8)
                        num_retries += 1
                        end = time.perf_counter()
                        if end - start < 1:
                            time.sleep(1 - (end - start))
                    except Exception as e:
                        print(e)
                        time.sleep(3)
        if not self.read_only:
            self.cache_writer.close()

        all_expansions = []
        for doc in self.documents:
            if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
                doc_expansions = [doc]
            else:
                doc_expansions = []
            if doc in document_expansion_mapping:
                doc_expansions.extend(document_expansion_mapping[doc])
            all_expansions.append(", ".join(doc_expansions))

        

        # 2- Second part is to generate the seed words from the keyphrases of step 1

        cluster_seed_words = []
        # For each cluster (i) generate seed words from the keyphrases of documents in cluster (i)
        for i in set(self.labels):
            c_indices = [x for x, v in enumerate(self.labels) if v == i]
            c_elements = [all_expansions[i] for i in c_indices]
            c_joined_keyphrases = ", ".join(c_elements)
            c_keyphrase_prompt = self.seed_words_prompt + f"""
                                                    keyphrases : {c_joined_keyphrases}

                                                    Seed words : 
                                                    """
        
            failure = True
            num_retries = 0
            while failure and num_retries < self.NUM_RETRIES:
                cache_row = None
                try:
                    start = time.perf_counter()

                    deployment_name="GPT-3-5-turbo-chat"
                    # CHATGPT Request
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": c_keyphrase_prompt},
                        ],
                    )

                    # CHATGPT Response
                    message = response.choices[0].message.content

                    seed_words_index = message.find('Seed words:')
    
                    if seed_words_index != -1:
                        # Extract the substring after 'Seed words:'
                        message = message[seed_words_index + len('Seed words:'):].strip()

                    try:
                        # Get each of the seed words
                        seed_words = message.replace("[","").replace("]","").replace('"','')
                        
                        if not isinstance(seed_words, str):
                            failure = True
                            print("failure of type")
                        cluster_seed_words.append(seed_words)
                    except Exception as e:
                        print("Exception: ", e)
                        time.sleep(0.8)
                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except Exception as e:
                    print(e)
                    time.sleep(3)
            
        # Dump cluster seed words in pickle file
        with open('cluster-seed-words/'+self.dataset_name+'_clusters_seed_words.pkl', 'wb') as f:
            pickle.dump(cluster_seed_words, f)
            
        print(self.dataset_name," Seed words and keyphrases generation finished.")


        
