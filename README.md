# AI Models

## Getting Started

### Installation Instructions - Local development installation

First, install Python (3.9 or higher)

**1. Prerequisites**

* Install Python 3.11 or higher.
* Acquire your OpenAI API Key.
* Install Git

**2. Fork and clone the GitHub repository**

* One member of your group should fork this repository to his or her account.
* Then, you can git clone this repository.
* We assume for the remaining installation steps that you opened a shell inside of the cloned repository.

**3. Setup local environment and install Python dependencies**

Create environment

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

**4. Setup OpenAI API Key**

Create a `.env` file in the project root and paste your API Key. The file should have this content:

```
OPENAI_API_KEY=your_api_key_here
```

**5. Start the FastAPI server:**

Add the --reload flag for easier debugging

```
cd src
uvicorn ner_api:app --reload --port 8001
```

The HTTP API will be available at http://localhost:8001

### Build and run the docker container


```
# build the docker container
docker build -t ai-models .
```

```
docker run \
    -p 8001:80 \
    -e OPENAI_API_KEY=... \
    ai-models
```

## Architecture

### AI Models Overview

<img src="https://github.com/potatobot-rwanda/ai-models/blob/main/images/NER-NEC-NEL.drawio.png" width="400">

The detection of locations, temporal expressions and potatoes works in three stages:

1. Named Entity Detection to detect spans in the text that contain named entities.
2. Named Entitiy Classification to classify the span as potato, location or named temporal expression.
3. Named Entitiy Normalization to normalize temporal expressions or Named Entity Linking to link a named entity to the knowledge base

