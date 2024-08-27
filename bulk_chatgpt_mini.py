import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom CSS for styling and external stylesheet
st.markdown(
    """
    <style>
        p,.appview-container,h1,.stHeadingWithActionElements,.stWidgetLabel,.stMarkdown,.st-ae,.st-bd,.st-be,.st-bf,.st-bg,.st-bh,.st-bi,.st-bj,.st-bk,.st-bl,.st-bm,.st-ah,.st-bn,.st-bo,.st-bp,.st-bq,.st-br,.st-bs,.st-bt,.st-bu,.st-ax,.st-ay,.st-az,.st-bv,.st-b1,.st-b2,.st-bc,.st-bw,.st-bx,.st-by{
        color: black !important;
        font-family: "Raleway", Sans-serif;
        }

        .appview-container,h1,.stHeadingWithActionElements,.stWidgetLabel,.stMarkdown,.st-ae,.st-bd,.st-be,.st-bf,.st-bg,.st-bh,.st-bi,.st-bj,.st-bk,.st-bl,.st-bm,.st-ah,.st-bn,.st-bo,.st-bp,.st-bq,.st-br,.st-bs,.st-bt,.st-bu,.st-ax,.st-ay,.st-az,.st-bv,.st-b1,.st-b2,.st-bc,.st-bw,.st-bx,.st-by{
        background-color: white !important;
        }
        
        button{
        background-color: #1098A7 !important;
        border: none;
        outline: none;
        font-family: "Raleway", Sans-serif;
        font-size: 16px;
        font-weight: 500;
        border-radius: 0px 0px 0px 0px;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Setup
st.title('Bulk ChatGPT v3')

# Subtitle
st.markdown(
    """
    by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
    """,
    unsafe_allow_html=True
)

# Input for the OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

if uploaded_file and api_key:
    # Read the uploaded file into a DataFrame to get column names
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    # Allow user to map columns to variable names
    st.write("Map each column to a variable name that will be used in the prompts:")
    column_to_variable = {}
    for column in columns:
        variable_name = st.text_input(f"Enter a variable name for {column}", value=column)
        column_to_variable[column] = variable_name

    # System and User Prompts customization
    system_prompt = st.text_area("Edit the system prompt", value="Edit the system prompt. You can include any of the variable names defined above surrounded by curly braces, like {variable_name}.")
    user_prompt_template = st.text_area("Edit the user prompt", value="Edit the user prompt. You can include any of the variable names defined above surrounded by curly braces, like {variable_name}.")

    # Placeholder for progress updates
    progress_text = st.empty()

    # Button to download responses as CSV
    if st.button("Generate Responses"):
        client = OpenAI(api_key=api_key)
        all_responses = []

        # Function to generate responses using the OpenAI client
        def generate_response(row):
            formatted_user_prompt = user_prompt_template.format(**{var: row[col] for col, var in column_to_variable.items()})
            formatted_system_prompt = system_prompt.format(**{var: row[col] for col, var in column_to_variable.items()})
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": formatted_system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ],
                model="gpt-4o-mini"
            )
            return response.choices[0].message.content.strip()

        # Batch processing
        batch_size = 10  # Adjust the batch size as needed
        num_batches = len(df) // batch_size + 1
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_df = df.iloc[start_idx:end_idx]

            # Iterate over each row in the batch and collect responses
            for index, row in batch_df.iterrows():
                try:
                    response = generate_response(row)
                    response_data = [row[col] for col in columns] + [response]  # Appends response to data
                    all_responses.append(response_data)
                except Exception as e:
                    logging.error(f"Error processing row {index}: {e}")
                # Respect API rate limits
                #time.sleep(0.5)

            # Update progress
            progress_text.text(f"Processed batch {batch_num + 1} of {num_batches}")

        # Create the DataFrame
        response_df = pd.DataFrame(all_responses, columns=columns + ['Response'])
        csv = response_df.to_csv(index=False).encode('utf-8')

        # Provide the download button for the CSV
        st.download_button(label="Download as CSV", data=csv, file_name="responses.csv", mime="text/csv")
