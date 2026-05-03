from datetime import datetime, timedelta
import json
import os
import re, json

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import track

console = Console() # Import anywhere

# __________read_write_helpers________

def write_file(content, file_path):
    with open(file_path, "w") as f:
        f.write(content)
        f.close()

def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

def write_json(content, file_path):
    with open(file_path, "w") as f:
        json.dump(content, f, indent=4)


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        return data


# _________more_helpers_____________

def printmd(text: str):
    markdown = Markdown(text)
    console.print(markdown)

def k_days_before(date_str: str, k: int, date_format: str = "%Y-%m-%d") -> str:
    date_obj = datetime.strptime(date_str, date_format)
    new_date = date_obj - timedelta(days=k)
    return new_date.strftime(date_format)

def panel_print(
    text: str,
    title: str = "Output",
    subtitle: str = "",
    border_style: str = "cyan",
    expand: bool = True
):
    panel = Panel(
        text,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        expand=expand
    )
    console.print(panel)



def extract_json_block(response: str) -> str:
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if match:
        json_str = match.group()
        # Remove control characters except for newline and tab escape sequences
        safe_json = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        return json.loads(safe_json)
    return None

    
def estimate_prompts(
    prompts_per_share: int,
    num_shares : int,
    rerun_every_k_months : int,
    batch_size = 64
):

    net_prompts_per_run = prompts_per_share * num_shares
    num_iterations = net_prompts_per_run / 64

    prompts_1_y = (12/rerun_every_k_months) * net_prompts_per_run
    prompts_3_y = prompts_1_y * 3
    prompts_5_y = prompts_1_y * 5
    prompts_10_y = prompts_1_y * 10

    print("\n")
    console.log("[bold blue]Nebula Backtesting Prompt Calculator")

    console.log("[bold yellow]Parameters")
    console.log(f"Prompts per share: {prompts_per_share}")
    console.log(f"Total number of shares: {num_shares}")
    console.log(f"Re-rerun every k months: {rerun_every_k_months}")
    
    console.log("[bold yellow]Output")
    console.log(f"Net prompts per run: {net_prompts_per_run:.2f}")
    console.log(f"1 Year prompts: {prompts_1_y:,}")
    console.log(f"3 Years prompts: {prompts_3_y:,}")
    console.log(f"5 Years prompts: {prompts_5_y:,}")
    console.log(f"10 Years prompts: {prompts_10_y:,}")
    
    
    print("\n")

    

    