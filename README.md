# Graph RAG
<p align="center">
  <img src="graph_rag\images\graph-rag-ui.png" alt="graph-rag" width="500"/>
</p>

The following repository supports the article written [here](https://medium.com/@markmikeobrien/build-graphrag-using-streamlit-langchain-neo4j-gpt-4o-b0d1e938016e) & provides an implementation of GraphRAG which does the following:
- Extracts content from a series of YouTube videos, Wikipedia articles & text files. It is initially setup with resources on Garmin watches but you can swap this out for anything you like
- Once extracted it will build a knowledge graph for all the extracted entities in the content
- A custom retriever will combine a hybrid approach using both vector nearest neighbor & cipher queries to extract relevant content to the users query
- Finally,, we will bring this all together in a basic Stramlit chat interface so you can talk with the knowledge graph

# Getting Started
There are two components here:
1. A Neo4J instance running locally within a docker container
2. The Python app which is managing graph construction and query via GPT-4o

## Neo4J Docker Setup
Download the following JAR to your $/user/plugins directory - [APOC](https://github.com/neo4j/apoc/releases/tag/5.20.0)

Now run the following docker command (assuming you have Docker Desktop installed)
```
docker run `
    -p 7474:7474 -p 7687:7687 `
    -v ${PWD}/data:/data -v ${PWD}/plugins:/plugins `
    --name neo4j-v5-apoc `
    -e NEO4J_apoc_export_file_enabled=true `
    -e NEO4J_apoc_import_file_enabled=true `
    -e NEO4J_apoc_import_file_use_neo4j_config=true `
    -e NEO4J_PLUGINS='["apoc"]' `
    -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" `
    neo4j:5.20.0
```

Navigate to the URL exposed by the container and set the password, ensure the app has this before starting it up. 

## Run App
**Important** - Add you OpenAI API key before proceeding. 

First ensure you have installed the dependencies. I use [Poetry](https://python-poetry.org/) for deps management. 

`poetry install`

Next start up the streamlit app

`streamlit run .\graph_rag\app.py`

## Setup the Knowledge Graph
<p align="center">
  <img src="graph_rag\images\graph-rag-build.png" alt="graph-rag" width="500"/>
</p>

The first time you run the app you'll need to build the graph. This can be achieved with the default resources right away by hitting the "Populate Graph" button. This will take a few minutes but will result in a fully formed graph that you can visualize in Neo4j. 

You can now chat with the LLM which has a graph based knowledge base to feed from. 

Resetting the graph will remove all nodes and edges as well as any metadata, useful if you want to start again. 

## Liked This?
If you liked this consider supporting me for free by [joining my twice weekly newsletter](https://bit.ly/45lG2pR) on all things AI or by treating me to a cup of coffee, if you think I deserve it. 

Checkout my blog where I deep dive in AI and the latest innovations [Bright Journey AI](https://brightjourneyai.com/)

<a href="https://buymeacoffee.com/brightjourneyai" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>