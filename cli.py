import typer
from pprint import pprint
from typing import List
import os, time
from rich import print
from rich.console import Console
from rich.text import Text
from rich.table import Table
from datetime import datetime
import re
import uuid
import json
import pandas as pd

from src.utils import (
    printmd,
    write_json, 
    read_json, 
    write_file,
    read_file
    )


console = Console()

import logging
logging.basicConfig(level=logging.INFO)

app = typer.Typer()
config = read_json('src/config.json')


@app.command()
def generate_profile(
        docs_dir: str, 
        name: str,
        # llm :str = "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        max_rpm = 5,
        max_iter : int = 5,
        max_retry_limit : int = 4,
        llm :str = "gemini/gemini-2.5-flash"
        # "openrouter/mistralai/mistral-7b-instruct:free"
        ):

    start = time.time()

    if not os.path.exists(docs_dir):
        print("Directory not found !")
        return


    os.makedirs(config['paths']['executions'], exist_ok=True)
    os.makedirs(config['paths']['profiles'], exist_ok=True)


    save_dir = os.path.join(config['paths']['profiles'], name)
    metadata_path = os.path.join(save_dir, "metadata.json")
    parameters_path = os.path.join(save_dir, 'parameters.md')
    parameters_dir = os.path.join(save_dir, "parameters")
    os.makedirs(save_dir)
    os.makedirs(parameters_dir, exist_ok=True)


    from src.crews import ProfilerCrew


    response = ProfilerCrew(llm=llm, max_rpm=max_rpm).crew().kickoff(
        inputs={
            "docs_dir" : docs_dir,
            "output_dir" : save_dir
        }
    )

    end = time.time()

    metadata = {
        "name" : name,
        "created_at" : datetime.utcnow().isoformat(),
        "llm": llm,
        "max_rpm" :max_rpm,
        "max_iter" :max_iter,
        "max_retry_limit" :max_retry_limit,
        "docs_dir" : docs_dir,
        "documents" : os.listdir(docs_dir),
        "duration" : (end-start) 
    }


    write_json(metadata, metadata_path)

    logging.info("Profile generation complete")
    logging.info(f"Saved profile at : {save_dir}")

@app.command()
def available_profiles():
    base_path = config['paths']['profiles']

    table = Table(
        title="Available Profiles",
        header_style="bold cyan",
        show_lines=False,
    )

    table.add_column("Profile ID / Folder", style="yellow", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Created")
    table.add_column("Docs Dir")
    table.add_column("# Docs", justify="right")
    table.add_column("# Params", justify="right")

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        params_dir = os.path.join(folder_path, "parameters")
        metadata_path = os.path.join(folder_path, "metadata.json")

        if not os.path.isdir(folder_path) or not os.path.exists(metadata_path):
            continue

        try:
            d = read_json(metadata_path)

            created_at = datetime.fromisoformat(
                d["created_at"].replace("Z", "+00:00")
            ).strftime("%d %b %Y %H:%M")

            documents = d.get("documents", [])
            doc_count = len(documents) if isinstance(documents, list) else "N/A"
            num_params = None
            if os.path.exists(params_dir):
                num_params = len(os.listdir(params_dir))
            table.add_row(
                folder,
                d.get("name", "N/A"),
                created_at,
                d.get("docs_dir", "N/A"),
                str(doc_count),
                # f"{d.get('duration', 0)/60:.2f} min",
                str(num_params)
            )

        except Exception as e:
            table.add_row(
                folder,
                f"[red]{e}[/]",
                "-", "-", "-"
            )

    console.print(table)


@app.command()
def read_profile(profile: str):
    profiles = os.listdir(config['paths']['profiles'])

    if profile in profiles:
        md = read_file(os.path.join(config['paths']['profiles'], profile,f"profile.md"))
        printmd(md)
    else:
        print("[bold cyan]Cant find {profile}. Please choose from the following profiles:[/]")
        for p in profiles:
            print(f"  • [green]{p.split('.')[0]}[/]")


@app.command()
def available_analysis():
    base_path = config['paths']['analysis']

    table = Table(
        title="Available Analysis",
        show_lines=False,
        header_style="bold cyan"
    )

    table.add_column("ID / Folder", style="yellow", no_wrap=True)
    table.add_column("Symbol", style="green")
    table.add_column("Share", style="green")
    table.add_column("Profile")
    table.add_column("Model")
    table.add_column("Created")
    table.add_column("RPM", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Duration (min)", justify="right")

    rows = []  # ← collect rows first

    for runid in os.listdir(base_path):
        run_dir = os.path.join(base_path, runid)
        run_path = os.path.join(run_dir, "run.json")

        if not os.path.isdir(run_dir) or not os.path.exists(run_path):
            continue

        try:
            run = read_json(run_path)

            created_dt = datetime.fromisoformat(
                run["created_at"].replace("Z", "+00:00")
            )

            created_str = created_dt.strftime("%d %b %Y %H:%M")
            duration = run.get("duration", 0)

            rows.append((
                created_dt,  # ← keep raw datetime for sorting
                (
                    runid,
                    run.get("symbol", "N/A"),
                    run.get("share_name", "N/A"),
                    run.get("profile", "N/A"),
                    run.get("model", "N/A"),
                    created_str,
                    str(run.get("rpm", "N/A")),
                    str(run.get("iterations", "N/A")),
                    f"{duration/60:.0f} m {duration%60:.0f} s",
                )
            ))

        except Exception as e:
            rows.append((
                datetime.min,
                (
                    runid,
                    f"[red]{e}[/]",
                    "-", "-", "-", "-", "-", "-", "-"
                )
            ))

    # ✅ sort: oldest → newest (latest at bottom)
    rows.sort(key=lambda x: x[0])

    # render table
    for _, row in rows:
        table.add_row(*row)

    console.print(table)


@app.command()
def available_tools():

    from src.tools import CUSTOM_TOOLS_MAP

    table = Table(
        title="Available Tools",
        header_style="bold cyan",
        show_lines=False,
    )

    table.add_column("Tool Name", style="yellow", no_wrap=True)
    table.add_column("Description", style="green")

    for tool in CUSTOM_TOOLS_MAP:
        table.add_row(tool, CUSTOM_TOOLS_MAP[tool])

    console.print(table)

from src.voyager import nse_financials_analysis

@app.command()
def financial_check(
    symbol:str, 
    profile:str,  
    ):

    profile = read_json(os.path.join(config['paths']['profiles'], profile, 'profile.json'))
    qualitative = profile['parameters']['qualitative']
    quantitative = profile['parameters']['quantitative']


    logging.info(f"Quantitative analysis | Fundamental Analysis")
    fundamental_analysis = nse_financials_analysis(symbol, thresholds=quantitative['financials'])

    score = fundamental_analysis['score']
    pprint(score)


@app.command()
def correlate_share(
    share_name:str, 
    symbol:str, 
    profile:str,  
    model :str = "cerebras/qwen-3-32b",
    corr_id : str = None,
    iters : int = 1,
    rpm :int = 2,
    max_retry : int = 3
    ):

    start = time.time()


    from src.technicals import get_price_data, technical_analysis_talib
    from src.agents import NebulAgent


    from datetime import datetime
    import random
    import string

    def short_id(n_random=2):
        ts = datetime.utcnow().strftime("%y%m%d%H%M%S")  # YYMMDDHHMMSS
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n_random))
        return f"{ts}{rand}"

    def extract_score(text):
        m = re.search(r'FINAL_SCORE:\s*(\d{1,3})', text)
        score = int(m.group(1)) if m else None
        return score


    corr_id = corr_id if corr_id else short_id()
    save_dir = os.path.join(config['paths']['analysis'], corr_id)
    run_path = os.path.join(save_dir, 'run.json')
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(run_path):
        logging.info("Loading previous run metadata...")
        run = read_json(run_path)
        run['rpm'] = rpm
        run['max_retry'] = max_retry
        run['iterations'] = max(run['iterations'], iters)
    else:
        run = {
            "symbol" : symbol,
            "share_name" : share_name,
            "created_at" : datetime.utcnow().isoformat(),
            "profile" : profile,
            "model" : model,
            "corr_id" : corr_id,
            "iterations" : iters,
            "rpm" : rpm,
            "max_retry" : max_retry,
            "runs" : {}
        }
        write_json(run, run_path)
        

    # loading profile
    profile = read_json(os.path.join(config['paths']['profiles'], profile, 'profile.json'))
    qualitative = profile['parameters']['qualitative']
    quantitative = profile['parameters']['quantitative']


    logging.info(f"Initiating correlation analysis ({corr_id} | {symbol} | {share_name})")

    logging.info(f"Quantitative analysis | Fundamental Analysis")
    fundamental_analysis = nse_financials_analysis(symbol, thresholds=quantitative['financials'])
    run['fundamental_analysis'] = fundamental_analysis
    write_json(run, run_path)

    logging.info(f"Quantitative analysis | Technical Analysis")
    price_df = get_price_data(f"{symbol}.NS", "3y")
    technical_analysis = technical_analysis_talib(symbol, price_df, thresholds=quantitative['technicals'])
    run['technical_analysis'] = technical_analysis
    write_json(run, run_path)

    
    from collections import deque
    q = deque()


    logging.info(f"Qualitative analysis | Initializing Agent...")
    agent = NebulAgent(symbol=symbol, profile=profile, model=model)

    for i in range(iters):
        iteration = str(i + 1)
        
        if not iteration in run['runs'].keys():
            run["runs"][iteration] = {
                "iteration" : iteration,
                "start_time" : time.time(),
                "parameters" : {}
            }
    

        for param in qualitative:

            if param in run['runs'][iteration]['parameters'].keys() and run['runs'][iteration]['parameters'][param]:
                logging.info(f"Qualitative analysis | Exists | {param}")
                continue

            logging.info(f"Qualitative analysis | {param}")

            title = qualitative[param]['title']
            desc = qualitative[param]['description']
            tools = qualitative[param]['tools']
            weight = float(qualitative[param]['weight'])

            # rate limiting check

            now = time.time()
            window_start = now - 60

            while q and q[0] < window_start:
                q.popleft()

            if len(q) >= rpm:
                print("Rate limit hit")
                sleep_time = 60 - (now - q[0])
                time.sleep(sleep_time)

            q.append(time.time())

            response = agent.run(title, desc, tools)

            score = extract_score(response.choices[0].message.content)

            run["runs"][iteration]['parameters'][param] = {}
            run["runs"][iteration]['parameters'][param]['score'] = score
            run["runs"][iteration]['parameters'][param]['weight'] = weight
            run['runs'][iteration]['parameters'][param]['response'] = response.model_dump()
            run['duration'] = time.time() - start
            run['end_time'] = time.time()
            run['runs'][iteration]['end_time'] = time.time()
    
            write_json(run, run_path)
            logging.info(f"Saved : {run_path}")

        # console.print(f"Iteration")
        console.print(f"ITERATION {iteration} COMPLETE", style="green")

    logging.info("Correlation analysis complete")


@app.command()
def read_scores(
    corr_id : str
    ):

    save_dir = os.path.join(config['paths']['analysis'], corr_id)
    run_path = os.path.join(save_dir, 'run.json')
    run = read_json(run_path)

    profile = read_json(os.path.join(config['paths']['profiles'], run['profile'], 'profile.json'))


    fundamental_score = run['fundamental_analysis']['score']['composite_score']
    technical_score = run['technical_analysis']['score']['composite_score']

    import pandas as pd
    # import contextlib
    # from rich.errors import NotRenderableError

    dfs = []

    for iter_id, iteration in run["runs"].items():

        params = []
        scores = []

        for p in iteration['parameters']:
            params.append(p)
            scores.append(iteration['parameters'][p]['score'])

        dfs.append(pd.DataFrame({'parameter': params, iter_id : scores}).set_index('parameter'))

    df = pd.concat(dfs, axis = 1, join='outer', ignore_index=False)
    df['average'] = df.mean(axis=1).astype(int)
    # df.loc['mean'] = df.mean()

    avgs = df.mean(axis=1)
    weights = { p: profile['parameters']['qualitative'][p]['weight'] for p in profile['parameters']['qualitative'] }
    qual_score = sum([ weights[p] * avgs.loc[p] for p in avgs.index ]) / sum(weights.values())
    qual_score = round(qual_score, 2)


    print(df)
    print()
    console.print(f"Composite Fundamental Score: {fundamental_score}")
    console.print(f"Composite Technical Score: {technical_score}")
    console.print(f"Weighted Average Qualitative Score: {qual_score}")




if __name__ == "__main__":
    app()
