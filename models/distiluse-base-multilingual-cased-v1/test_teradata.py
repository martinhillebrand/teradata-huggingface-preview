import sys
import teradataml as tdml
from tabulate import tabulate

import json


with open('conversion_config.json') as json_file:
    conversion_config = json.load(json_file)


    model_id = conversion_config["model_id"]
    number_of_generated_embeddings = conversion_config["number_of_generated_embeddings"]
    precision_to_filename_map = conversion_config["precision_to_filename_map"]
    
    host = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    
    print("Setting up connection to teradata...")
    tdml.create_context(host = host, username = username, password = password)
    print("Done\n\n")
    
    
    print("Deploying tokenizer...")
    try:
        tdml.db_drop_table('tokenizer_table')
    except:
        print("Can't drop tokenizers table - it's not existing")
    tdml.save_byom('tokenizer',
                  'tokenizer.json',
                  'tokenizer_table')
    print("Done\n\n")
    
    print("Testing models...")
    try:
        tdml.db_drop_table('model_table')
    except:
        print("Can't drop models table - it's not existing")
    
    for precision, file_name in precision_to_filename_map.items():
        print(f"Deploying {precision} model...")
        tdml.save_byom(precision,
                      file_name,
                      'model_table')
        print(f"Model {precision} is deployed\n")
    
        print(f"Calculating embeddings with {precision} model...")
        try:
            tdml.db_drop_table('emails_embeddings_store')
        except:
            print("Can't drop embeddings table - it's not existing")
        
        tdml.execute_sql(f"""
            create volatile table emails_embeddings_store as (
                select 
                    *
            from mldb.ONNXEmbeddings(
                    on emails.emails as InputTable
                    on (select * from model_table where model_id = '{precision}') as ModelTable DIMENSION
                    on (select model as tokenizer from tokenizer_table where model_id = 'tokenizer') as TokenizerTable DIMENSION
               
                    using
                        Accumulate('id', 'txt') 
                        ModelOutputTensor('sentence_embedding')
                        EnableMemoryCheck('false')
                        OutputFormat('FLOAT32({number_of_generated_embeddings})')
                        OverwriteCachedModel('true')
                ) a 
        ) with data on commit preserve rows
        
        """)
        print("Embeddings calculated")
        print(f"Testing semantic search with cosine similiarity on the output of the model with precision '{precision}'...")
        tdf_embeddings_store = tdml.DataFrame('emails_embeddings_store')
        tdf_embeddings_store_tgt = tdf_embeddings_store[tdf_embeddings_store.id == 3]
        
        tdf_embeddings_store_ref = tdf_embeddings_store[tdf_embeddings_store.id != 3]
        
        cos_sim_pd = tdml.DataFrame.from_query(f"""
            SELECT 
                dt.target_id, 
                dt.reference_id,
                e_tgt.txt as target_txt,
                e_ref.txt as reference_txt,
                (1.0 - dt.distance) as similiarity 
            FROM
                TD_VECTORDISTANCE (
                    ON ({tdf_embeddings_store_tgt.show_query()}) AS TargetTable
                    ON ({tdf_embeddings_store_ref.show_query()}) AS ReferenceTable DIMENSION
                    USING
                        TargetIDColumn('id')
                        TargetFeatureColumns('[emb_0:emb_{number_of_generated_embeddings - 1}]')
                        RefIDColumn('id')
                        RefFeatureColumns('[emb_0:emb_{number_of_generated_embeddings - 1}]')
                        DistanceMeasure('cosine')
                        topk(3)
                ) AS dt
            JOIN emails.emails e_tgt on e_tgt.id = dt.target_id
            JOIN emails.emails e_ref on e_ref.id = dt.reference_id;
            """).to_pandas()
        print(tabulate(cos_sim_pd, headers='keys', tablefmt='fancy_grid'))
        print("Done\n\n")
    
    
    tdml.remove_context()