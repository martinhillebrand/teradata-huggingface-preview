import onnxruntime as rt

from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

import transformers

import gc
import json


with open('conversion_config.json') as json_file:
    conversion_config = json.load(json_file)


    model_id = conversion_config["model_id"]
    number_of_generated_embeddings = conversion_config["number_of_generated_embeddings"]
    precision_to_filename_map = conversion_config["precision_to_filename_map"]
    
    sentences_1 = 'How is the weather today?'
    sentences_2 = 'What is the current weather like today?'
    
    print(f"Testing on cosine similiarity between sentences: \n'{sentences_1}'\n'{sentences_2}'\n\n\n")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("./")
    enc1 = tokenizer(sentences_1)
    enc2 = tokenizer(sentences_2)
    
    for precision, file_name in precision_to_filename_map.items():
    
        
        onnx_session = rt.InferenceSession(file_name)
        embeddings_1_onnx = onnx_session.run(None,     {"input_ids": [enc1.input_ids], 
             "attention_mask": [enc1.attention_mask]})[1][0]
    
        embeddings_2_onnx = onnx_session.run(None,     {"input_ids": [enc2.input_ids], 
             "attention_mask": [enc2.attention_mask]})[1][0]
    
        del onnx_session
        gc.collect()
        print(f'Cosine similiarity for ONNX model with precision "{precision}" is {str(cos_sim(embeddings_1_onnx, embeddings_2_onnx))}')
    
    
    
    
    model = SentenceTransformer(model_id, trust_remote_code=True)
    embeddings_1_sentence_transformer = model.encode(sentences_1, normalize_embeddings=True, trust_remote_code=True)
    embeddings_2_sentence_transformer = model.encode(sentences_2, normalize_embeddings=True, trust_remote_code=True)
    print('Cosine similiarity for original sentence transformer model is '+str(cos_sim(embeddings_1_sentence_transformer, embeddings_2_sentence_transformer)))