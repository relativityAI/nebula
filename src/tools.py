from pprint import pprint
import os
from functools import wraps
from typing import List, Union, Optional


# from src.voyager import (
#     nse_annual_reports_list,
#     nse_annual_report_section_download,
#     nse_list_annual_report_sections,
#     nse_financials,
#     nse_financials_analysis,
#     nse_shareholdings,
#     nse_preprocess_financials,
#     nse_announcements_search,
#     nse_announcement_extract,
# )


# from src.technicals import technical_analysis_talib, get_price_data

##############################################


from pydantic import create_model
import inspect


def func_to_model(func):
    sig = inspect.signature(func)

    fields = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != inspect._empty else str
        default = ... if param.default == inspect._empty else param.default

        fields[name] = (annotation, default)

    return create_model(f"{func.__name__}_Input", **fields)


##############################################


class ToolSpec:
    def __init__(self, func, name: str, description: Optional[str]):
        self.name = name
        self.description = description
        self.handler = func
        self.input_model = func_to_model(func)


class Tools:
    def __init__(self):
        self.tools = {}

    def register(self, tools: Union[ToolSpec, List[ToolSpec]]):

        if type(tools) == ToolSpec:
            tools = List[tools]

        for tool in tools:
            self.tools[tool.name] = tool

    def list_tools(self):
        return list(self.tools.keys())

    def model(self, name: str):
        tool = self.tools[name]
        return tool.input_model

    def execute(self, name: str, **kwargs):
        tool = self.tools[name]
        validated = tool.input_model(**kwargs)
        return tool.handler(**validated.model_dump())


##############################################

# tools primary ones to start with


def read_expenses(symbol: str):
    """
    Fetch and read the expenses data
    from a company's financials
    including materials cost, employee expenses, tax etc
    """

    df = nse_financials(symbol)

    quarterly_df = df[(df["contextRef"] == "OneD") | (df["contextRef"] == "OneI")]

    annual_df = df[(df["contextRef"] == "FourD") | (df["contextRef"] == "OneI")]

    q_pt = quarterly_df.pivot_table(
        index="date",  # rows
        columns="tag",  # new columns
        values="value",  # cell values
        aggfunc="first",  # in case of duplicates
    )

    a_pt = annual_df.pivot_table(
        index="date",  # rows
        columns="tag",  # new columns
        values="value",  # cell values
        aggfunc="first",  # in case of duplicates
    )

    q_pt = nse_preprocess_financials(q_pt)
    a_pt = nse_preprocess_financials(a_pt)

    final_columns = [
        "material_cost_percentage",
        "employee_cost_percentage",
        "tax_percentage",
    ]

    text = f"""
    Quarterly Expenses

    {q_pt[final_columns].round(2).to_string()}

    ---
    Annual Expenses
    
    {a_pt[final_columns].round(2).to_string()}
    """

    return text


def read_latest_transcript(symbol: str):
    """
    Tool that returns the entire text content
    from the Latest earnings call or conference call transcript of a company.
    """

    results = nse_announcements_search(symbol, "transcript")
    url = results[0]["attchmntFile"]

    text = nse_announcement_extract(url)
    return text



def read_shareholdings(symbol: str):
    """
    Tool that returns the historical shareholdings data
    of the company
    including promoter (owners/ management) shareholdings
    DII - domestic institutional sharesholding
    FII - foreign institutional sharesholding

    The shareholding values are either in percentages
    or in decimal form of percentage (Eg - 23 or .23 - both mean 23% )
    """

    df = nse_shareholdings(symbol)
    pt = df.pivot_table(
        index="date",  # rows
        columns=["tag", "contextRef"],  # new columns
        values="value",  # cell values
        aggfunc="first",  # in case of duplicates
    )
    final = pt[
        [
            [
                "ShareholdingAsAPercentageOfTotalNumberOfShares",
                "ShareholdingOfPromoterAndPromoterGroupI",
            ],
            [
                "ShareholdingAsAPercentageOfTotalNumberOfShares",
                "ShareholdingOfPromoterAndPromoterGroup_ContextI",
            ],
            ["ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsForeignI"],
            ["ShareholdingAsAPercentageOfTotalNumberOfShares", "InstitutionsDomesticI"],
            ["ShareholdingAsAPercentageOfTotalNumberOfShares", "NonInstitutionsI"],
        ]
    ]

    # print(final)

    return final.to_string()

    # pprint(final.index)
    # pprint(final.columns[0])

##############################################


def score_financials(symbol: str, thresholds: dict):
    """
    Applies financials check and return a score
    """

    score = nse_financials_analysis(symbol, thresholds)["score"]["composite_score"]
    return score


def score_technicals(symbol: str, thresholds: dict):
    """
    Applies technical analysis and return a score
    """

    symbol = "NS." + symbol
    price_df = get_price_data(symbol)
    analysis = technical_analysis_talib(symbol, price_df)

    print(analysis)



def list_available_annual_report_sections(symbol: str):
    """
    List out the available sections or table of contents in
    the LATEST annual report of a company
    """

    data = nse_annual_reports_list(symbol)
    url = data[0]["fileName"]

    return nse_list_annual_report_sections(url)


def search_annual_report_sections(symbol: str, query: str, top_k=5):
    """
    An LLM finds and returns the correct section
    from the table of contents
    of the annual report
    """

    from thefuzz import fuzz
    import pandas as pd

    sections = list_available_annual_report_sections(symbol)

    end = 9999
    for i, s in enumerate(sections[::-1]):
        s["score"] = fuzz.token_sort_ratio(
            s["section"].strip().lower(), query.strip().lower()
        )
        s["start"] = s["page"]
        s["end"] = max(end - 1, 1)
        end = s["page"]

    matches = sorted(sections, key=lambda x: x["score"], reverse=True)

    df = pd.DataFrame(matches)
    # pprint(df.head())

    return df.head(top_k).to_dict("records")


def read_annual_report_section(
    symbol: str, section_keywords: List[str] | str, lag: int = 0
):
    """
    Extract the content of a specific section of an annual report
    section from the LATEST annual report of a company.

    Arguments:
    symbol (str):
    section_keywords (str):
        Keyword or phrase used to identify and extract a specific
        section from the annual report (e.g. "management discussion analysis",
    """

    # apply some kinda concensus, if we're not sure what the keyword is

    if isinstance(section_keywords, str):
        matches = search_annual_report_sections(symbol, section_keywords)
        keyword = matches[0]["section"]
    else:

        matches = []
        for k in section_keywords:
            matches.extend(search_annual_report_sections(symbol, k))

        matches = sorted(matches, key=lambda x: x["score"], reverse=True)
        keyword = matches[0]["section"]

    data = nse_annual_reports_list(symbol)
    url = data[0]["fileName"]

    text = nse_annual_report_section_download(url, keyword, lag)
    return text


def read_annual_report_governance(symbol: str):
    """
    Read the Governance section
    Information about number of executive and non executive and independent directors
    """
    return read_annual_report_section(
        symbol, ["Governance", "Board of Directors"], lag=0
    )


def read_annual_report_mda(symbol: str):
    """
    Read the Management disc and anal section
    """
    return read_annual_report_section(
        symbol, "Management discussion and analysis", lag=2
    )


# def pdf_reader(file_path: str) -> str:
#     """
#     PDF Reader
#     Reads pdf files.
#     """
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text


# def directory_creator(directory_path: str):
#     """
#     Directory Creator Tool
#     Create a directory at a specific path.
#     """
#     os.makedirs(directory_path, exist_ok=True)


# def read_file(file_path: str) -> str:
#     """
#     Read File Tool
#     Read contents of a file like txt, csv, or md.
#     """
#     with open(file_path, "r") as f:
#         return f.read()

##############################################

funcs = [read_latest_transcript]

tools = [
    ToolSpec(func=func, name=func.__name__, description=func.__doc__) for func in funcs
]

toolmanager = Tools()
toolmanager.register(tools=tools)


##############################################


# =====================================================================
# Async Tools for LLM Agent Tool Calling
# =====================================================================

import asyncio
import httpx
import re
import os
from io import BytesIO


async def tavily_web_search(query: str) -> str:
    """Search the web via Tavily API for recent information."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 5,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return f"Error: Tavily search returned status {resp.status_code}"
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "No web search results found."
            parts = []
            for r in results:
                parts.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', 'N/A')}\n"
                )
            return "\n---\n".join(parts)
        except Exception as e:
            return f"Error during web search: {str(e)}"


async def fetch_web_source(source: str, symbol: str) -> str:
    """Fetch raw data from a stock analysis web source via Voyager.
    Supported sources: screener, trendlyne."""
    voyager_base = os.getenv("VOYAGER_URL", "http://localhost:8001")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{voyager_base}/web-source",
                params={"id": source, "symbol": symbol},
                timeout=30,
            )
            if resp.status_code != 200:
                return f"Error: {source} returned status {resp.status_code}"
            text = resp.text
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:50000]
        except Exception as e:
            return f"Error fetching {source}: {str(e)}"


async def read_document_from_url(url: str) -> str:
    """Fetch a document from a URL (SEC filing, PDF, etc.) and extract its text content."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True, timeout=60)
            if resp.status_code != 200:
                return f"Error: failed to fetch document with status {resp.status_code}"
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(BytesIO(resp.content))
                    text = ""
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
                    return text[:50000] or "PDF contained no extractable text."
                except Exception as e:
                    return f"Error extracting PDF: {str(e)}"
            else:
                text = resp.text
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:50000]
        except Exception as e:
            return f"Error reading document URL: {str(e)}"


async def read_company_transcript(symbol: str) -> str:
    """Get the latest earnings call transcript for a company (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_latest_transcript, symbol)


async def read_annual_report_section_async(symbol: str, section_keywords: str) -> str:
    """Read a specific section from the latest annual report (async).
    section_keywords: e.g. 'management discussion analysis', 'corporate governance', 'business overview'."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: read_annual_report_section(symbol, section_keywords)
    )


async def read_shareholdings_async(symbol: str) -> str:
    """Get the shareholding pattern (promoter, FII, DII) for a company (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_shareholdings, symbol)


if __name__ == "__main__":

    import json

    available_tools = toolmanager.list_tools()
    model = toolmanager.model(available_tools[0])
    print(json.dumps(model.model_json_schema(), indent=4))

    pass
