---
language:
- multilingual
- ar
- zh
- nl
- en
- fr
- de
- it
- ko
- pl
- pt
- ru
- es
- tr
library_name: sentence-transformers
license: apache-2.0
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- onnx
- teradata

---
# A Teradata Vantage compatible Embeddings Model

# sentence-transformers/distiluse-base-multilingual-cased-v1

## Overview of this Model

An Embedding Model which maps text (sentence/ paragraphs) into a vector.  The [sentence-transformers/distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1) model well known for its effectiveness in capturing semantic meanings in text data. It's a state-of-the-art model trained on a large corpus, capable of generating high-quality text embeddings.

- 134.73M params (Sizes in ONNX format - "fp32": 515.61MB, "fp16": 257.93MB, "int8": 129.29MB, "uint8": 129.29MB)
- 512 maximum input tokens 
- 512 dimensions of output vector
- Licence: apache-2.0. The released models can be used for commercial purposes free of charge.
- Reference to Original Model: https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1


## Quickstart: Deploying this Model in Teradata Vantage

We have pre-converted the model into the ONNX format compatible with BYOM 6.0, eliminating the need for manual conversion. 

**Note:** Ensure you have access to a Teradata Database with BYOM 6.0 installed.

For detailed information, refer to the ONNXEmbeddings documentation: TODO

To get started, clone the pre-converted model directly from the Teradata HuggingFace repository.


```python

import teradataml as tdml
import getpass
from huggingface_hub import hf_hub_download

model_name = "distiluse-base-multilingual-cased-v1"
number_dimensions_output = 512
model_file_name = "model.onnx"

# Step 1: Download Model from Teradata HuggingFace Page

hf_hub_download(repo_id=f"Teradata/{model_name}", filename=f"onnx/{model_file_name}", local_dir="./")
hf_hub_download(repo_id=f"Teradata/{model_name}", filename=f"tokenizer.json", local_dir="./")

# Step 2: Create Connection to Vantage

tdml.create_context(host = input('enter your hostname'), 
                    username=input('enter your username'), 
                    password = getpass.getpass("enter your password"))

# Step 3: Load Models into Vantage
# a) Embedding model
tdml.save_byom(model_id = model_name, # must be unique in the models table
               model_file = model_file_name,
               table_name = 'embeddings_models' )
# b) Tokenizer
tdml.save_byom(model_id = model_name, # must be unique in the models table
              model_file = 'tokenizer.json',
              table_name = 'embeddings_tokenizers') 

# Step 4: Test ONNXEmbeddings Function
# Note that ONNXEmbeddings expects the 'payload' column to be 'txt'. 
# If it has got a different name, just rename it in a subquery/CTE.
input_table = "emails.emails"
embeddings_query = f"""
SELECT 
        *
from mldb.ONNXEmbeddings(
        on {input_table} as InputTable
        on (select * from embeddings_models where model_id = '{model_name}') as ModelTable DIMENSION
        on (select model as tokenizer from embeddings_tokenizers where model_id = '{model_name}') as TokenizerTable DIMENSION
        using
            Accumulate('id', 'txt') 
            ModelOutputTensor('sentence_embedding')
            EnableMemoryCheck('false')
            OutputFormat('FLOAT32({number_dimensions_output})')
            OverwriteCachedModel('true')
    ) a 
"""
DF_embeddings = tdml.DataFrame.from_query(embeddings_query)
DF_embeddings
```



## What Can I Do with the Embeddings?

Teradata Vantage includes pre-built in-database functions to process embeddings further. Explore the following examples:

- **Semantic Clustering with TD_KMeans:** [Semantic Clustering Python Notebook](https://github.com/Teradata/jupyter-demos/blob/main/UseCases/Language_Models_InVantage/Semantic_Clustering_Python.ipynb)
- **Semantic Distance with TD_VectorDistance:** [Semantic Similarity Python Notebook](https://github.com/Teradata/jupyter-demos/blob/main/UseCases/Language_Models_InVantage/Semantic_Similarity_Python.ipynb)
- **RAG-Based Application with TD_VectorDistance:** [RAG and Bedrock Query PDF Notebook](https://github.com/Teradata/jupyter-demos/blob/main/UseCases/Language_Models_InVantage/RAG_and_Bedrock_QueryPDF.ipynb)


## Deep Dive into Model Conversion to ONNX

**The steps below outline how we converted the open-source Hugging Face model into an ONNX file compatible with the in-database ONNXEmbeddings function.** 

You do not need to perform these steps—they are provided solely for documentation and transparency. However, they may be helpful if you wish to convert another model to the required format.


### Part 1. Importing and Converting Model using optimum

We start by importing the pre-trained [sentence-transformers/distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1) model from Hugging Face.

To enhance performance and ensure compatibility with various execution environments, we'll use the [Optimum](https://github.com/huggingface/optimum) utility to convert the model into the ONNX (Open Neural Network Exchange) format. 

After conversion to ONNX, we are fixing the opset in the ONNX file for compatibility with ONNX runtime used in Teradata Vantage

We are generating ONNX files for multiple different precisions: fp32, fp16, int8, uint8

You can find the detailed conversion steps in the file [convert.py](./convert.py)

### Part 2. Running the model in Python with onnxruntime & compare results

Once the fixes are applied, we proceed to test the correctness of the ONNX model by calculating cosine similarity between two texts using native SentenceTransformers and ONNX runtime, comparing the results.

If the results are identical, it confirms that the ONNX model gives the same result as the native models, validating its correctness and suitability for further use in the database.


```python
import onnxruntime as rt

from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

import transformers


sentences_1 = 'How is the weather today?'
sentences_2 = 'What is the current weather like today?'

# Calculate ONNX result
tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v1")
predef_sess = rt.InferenceSession("onnx/model.onnx")

enc1 = tokenizer(sentences_1)
embeddings_1_onnx = predef_sess.run(None,     {"input_ids": [enc1.input_ids], 
     "attention_mask": [enc1.attention_mask]})

enc2 = tokenizer(sentences_2)
embeddings_2_onnx = predef_sess.run(None,     {"input_ids": [enc2.input_ids], 
     "attention_mask": [enc2.attention_mask]})


# Calculate embeddings with SentenceTransformer
model = SentenceTransformer(model_id, trust_remote_code=True)
embeddings_1_sentence_transformer = model.encode(sentences_1, normalize_embeddings=True, trust_remote_code=True)
embeddings_2_sentence_transformer = model.encode(sentences_2, normalize_embeddings=True, trust_remote_code=True)

# Compare results
print("Cosine similiarity for embeddings calculated with ONNX:" + str(cos_sim(embeddings_1_onnx[1][0], embeddings_2_onnx[1][0])))
print("Cosine similiarity for embeddings calculated with SentenceTransformer:" + str(cos_sim(embeddings_1_sentence_transformer, embeddings_2_sentence_transformer)))
```

You can find the detailed ONNX vs. SentenceTransformer result comparison steps in the file [test_local.py](./test_local.py)

