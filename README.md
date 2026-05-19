# Project Nebula

Speed up your stock investment research using agentic automation.


## What is Nebula?

Nebula lets you define an **Investor Profile**.  
This is your investing strategy in a structured format.

It then applies this profile to stock analysis using AI agents.

You don’t manually research stocks.  
You define your logic once.  
Nebula reuses it at scale.

**Core idea:** Investors follow repeatable frameworks to evaluate stocks.  
Nebula turns that framework into an automated system for analysis and screening.


## How it works

- Define an Investor Profile (risk, strategy, preferences)
- Select data sources (financial websites, market data, etc.)
- Run analysis or stock screening using agents

## Setup & Usage

### 1. Start MongoDB
Nebula uses MongoDB for storage. Start the database using Docker:
```bash
docker-compose up -d db
```
*(Optional)* View your data via Mongo Express at `http://localhost:8081`.

### 2. Run Voyager API
⚠️ You must run the [Voyager API](https://github.com/relativityai/voyager) to access external data.



## CLI Usage

### 1. Create a Profile
Generate a template and then save it after editing:
```bash
python cli.py create-template --name my-strategy --sources screener
# Edit templates/my-strategy.json
python cli.py save-profile --file templates/my-strategy.json
```

### 2. Run Correlation Analysis
```bash
python cli.py correlate-share --share-name "HDFC Bank" --symbol HDFCBANK --profile-name my-strategy
```

### 3. Read Scores
```bash
python cli.py read-scores --corr-id <ID_FROM_PREVIOUS_STEP>
```

## API Usage

Start the API:
```bash
python api.py
```
Documentation is automatically available at `http://localhost:8002/docs`.

### Key Endpoints
- `GET /profiles`: List all profiles.
- `POST /profiles`: Create/Update a profile (JSON body).
- `POST /correlate`: Start a new analysis (JSON body).
- `GET /analysis/{corr_id}`: Get results of an analysis.