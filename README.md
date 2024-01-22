# Information_Retrival_Application

## Problem Statement
In the ever-evolving world, staying informed is crucial for making decisions. However, accessing relevant and up-to-date information can be time-consuming and overwhelming. This Tool aims to address this problem by providing users with an effortless way to retrieve valuable insights from the articles.

## Data Preparation
To ensure accurate and insightful results, the tool leverages a combination of web scraping and API integration to gather information from reputable news websites.

## Approach

###  User Input
#### User provides URLs:
Users input article URLs.

Data Splitting and Storage

Data Splitting:
The retrieved data is split into manageable chunks for efficient storage and retrieval.

Vector Database:
Chunks of data are stored in a vector database, allowing for quick and organized retrieval.
Interactive Queries and LLM Integration

User Prompts:
Users provide prompts or questions related to articles

Retrieve Related Chunks:
Relevant chunks from the vector database are retrieved based on user prompts.

Large Language Models (LLM):
Utilize LLM to process the retrieved chunks and generate detailed answers or summaries.

User Output
User Receives Answers:
Users receive answers to their queries, generated by LLM and enriched with information from the relevant data chunks.

## Final Mockup Application

### User Interface:

Input Section:
Users provide article URLs.

Data Splitting and Storage:
The tool automatically splits the data into chunks for efficient processing and storage in the vector database.

User Prompts:
Users input prompts or questions.

Retrieve and Process:
Relevant chunks are retrieved from the vector database based on user prompts.
LLM processes the chunks and generates context-aware answers.

User Output:
Users receive detailed answers or summaries enriched with information from the relevant data chunks.