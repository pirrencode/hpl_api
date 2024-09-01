# def egtl_quantative_data_experiment(model):
#     criterion_table = "STAGING_STORE.CALC_CR_SCL_STAGING"
#     experiment_table = "ALLIANCE_STORE.EGTL_QUANTATIVE_DATA_EXPERIMENT"
#     experiment_number = get_record_count_for_model(model, experiment_table) + 1
#     experiment_id = get_largest_record_id(experiment_table) + 1
#     st.write(f"Starting quantative experiment for {model} number {experiment_number}, ID {experiment_id}")

#     start_date = datetime.now(pytz.utc).strftime('%Y-%B-%d %H:%M:%S')
    
#     errors_encountered = False
#     error_type = None
#     error_message = None
    
#     try:
#         start_time = time.time()
#         normalized_data, genai_response_time, input_df_size, prompt_volume, output_volume, df_correctness_check, normalized_data_volume = normalize_data_for_egtl_experiment(model)
        
#         save_start_time = time.time()
#         save_data_to_snowflake(normalized_data, criterion_table)
#         save_data_to_snowflake_time = time.time() - save_start_time
        
#         total_time = time.time() - start_time
    
#     except Exception as e:
#         errors_encountered = True
#         error_type = type(e).__name__
#         error_message = str(e)
#         genai_response_time = 0
#         prompt_volume = 0
#         output_volume = 0
#         total_time = 0
#         normalized_data_volume = 0
#         save_data_to_snowflake_time = 0
    
#     end_date = datetime.now(pytz.utc).strftime('%Y-%B-%d %H:%M:%S')
#     rows_processed = get_table_row_count(criterion_table)
#     correctness = df_correctness_check

#     insert_data_in_quantative_experiment_table(experiment_id, 
#                                                model,                                               
#                                                start_date, 
#                                                end_date, 
#                                                genai_response_time, 
#                                                save_data_to_snowflake_time,                                               
#                                                total_time,
#                                                rows_processed,  
#                                                input_df_size,
#                                                prompt_volume,    
#                                                output_volume,
#                                                normalized_data_volume,                                                                                                                                        
#                                                correctness,
#                                                errors_encountered, 
#                                                error_type, 
#                                                error_message)
#     st.write(f"System has completed quantative experiment for {model} number {experiment_number}. Experiment ID {experiment_id}.")    