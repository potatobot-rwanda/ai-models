# AI Models

## Getting Started

### Installation Instructions - Local development installation

First, install Python (3.9 or higher)

**1. Prerequisites**

* Install Python 3.9 or higher.
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

The HTTP API will be available at http://localhost:8002

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

### Run unit tests

The folder `src/tests` contains unit tests. You can run all unit tests together:

```
cd src
python -m unittest discover -s tests
```

Or you can run an individual test:

```
python -m unittest tests.test_temporal_normalizer_english
```

## Architecture

### System architecture

Please refer to the [PotatoBot documentation](https://github.com/potatobot-rwanda/potatobot?tab=readme-ov-file#architecture-high-level-overview) for an overview over the system architecture.

### AI Models Overview

<img src="https://github.com/potatobot-rwanda/ai-models/blob/main/images/NER-NEC-NEL.drawio.png" width="400">

The image shows the information flow:

1. Entity Detection to detect spans in the text that contain named entities. Currently, the ER system implemented using LLMs. If time permits, we can also train a second Entity Recognition that uses transformers.
2. Temporal Expression Analysis transforms the detected temporal expressions (e.g., "Last month") to a machine readable format (e.g., 1.8.2025-31.8.2025).
3. Link potato varieties accepts a potato name as an input and return the ID of that potato variety from the database. It also takes spelling variations into account. 
4. Link locations accepts as input the name of the location in the database and returns the ID of the location from the database, taking spelling variations into account. It can also retrieve multiple locations. To disambiguate between multiple matches, one can provide an additional context, e.g., the name of the sector, to link locations.

