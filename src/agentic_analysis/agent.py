import os
import re
import json
import functools
import asyncio
from loguru import logger
import litellm
import src.tools as tools_module

QUALITATIVE_PROMPT = """You are an institutional-grade financial analyst analyzing {symbol}.

PARAMETER: {parameter}
GUIDELINES: {guidelines}

INSTRUCTIONS:
1. Use the available tools to gather data. You MUST call at least one tool before concluding.
2. Be concise and specific — avoid generic statements.
3. Every factual claim MUST cite its source in brackets, e.g. [transcript], [annual report - governance section], [web search: source name], [document URL], [screener data], [shareholding pattern].
4. If a claim cannot be supported by tool output, say so explicitly.

Once you have sufficient information, output your assessment in this exact format:

SCORE JUSTIFICATION: (1-2 sentences explaining why this score was chosen, citing key sources)

FINDINGS:
- <finding with source citation>
- <finding with source citation>
- <finding with source citation>

RISKS:
- <risk with source citation>
- <risk with source citation>

CONCLUSION: <one sentence>

FINAL_SCORE: X

Where X is an integer between 0 and 100."""


TOOL_DEFS = {
    "read_company_transcript": {
        "type": "function",
        "function": {
            "name": "read_company_transcript",
            "description": "Get the latest earnings call transcript of the company. Contains management commentary on performance, strategy, and outlook.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "read_annual_report_section": {
        "type": "function",
        "function": {
            "name": "read_annual_report_section",
            "description": "Read a specific section from the company's latest annual report. Use for: Management Discussion & Analysis, Corporate Governance, Business Overview, Risk Factors, Director Reports, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_keywords": {
                        "type": "string",
                        "description": "Keywords to identify the section (e.g. 'management discussion analysis', 'corporate governance', 'business overview', 'risk factors', 'director report')",
                    }
                },
                "required": ["section_keywords"],
            },
        },
    },
    "read_shareholdings": {
        "type": "function",
        "function": {
            "name": "read_shareholdings",
            "description": "Get the historical shareholding pattern including promoter holdings, FII (foreign institutional investors), DII (domestic institutional investors), and public holdings.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "tavily_web_search": {
        "type": "function",
        "function": {
            "name": "tavily_web_search",
            "description": "Search the web for recent news, analyst opinions, controversies, or any information relevant to the parameter. Use when you need up-to-date information beyond company filings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'TCS management quality controversies 2025'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    "fetch_web_source": {
        "type": "function",
        "function": {
            "name": "fetch_web_source",
            "description": "Fetch structured data from stock analysis websites (screener or trendlyne). Can provide financial ratios, momentum scores, durability scores, and valuation metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["screener", "trendlyne"],
                        "description": "The web source to fetch data from. 'screener' for financial ratios, 'trendlyne' for momentum/durability/valuation scores.",
                    }
                },
                "required": ["source"],
            },
        },
    },
    "read_document_url": {
        "type": "function",
        "function": {
            "name": "read_document_url",
            "description": "Read the content of a provided document URL (SEC filing, PDF report, etc.). Use this to analyze documents supplied alongside the analysis request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The document URL to read",
                    }
                },
                "required": ["url"],
            },
        },
    },
}


class NebulaAgent:
    def __init__(self, model="cerebras/qwen-3-32b", api_key=None):
        self.model = model
        self.api_key = api_key

    async def analyze_parameter(
        self,
        symbol: str,
        parameter: str,
        guidelines: str,
        documents=None,
        web_search=False,
        web_sources=None,
    ):
        logger.info(f"Analyzing '{parameter}' for {symbol} (web_search={web_search}, web_sources={web_sources})")

        tools = [
            TOOL_DEFS["read_company_transcript"],
            TOOL_DEFS["read_annual_report_section"],
            TOOL_DEFS["read_shareholdings"],
        ]
        if web_search:
            tools.append(TOOL_DEFS["tavily_web_search"])
        if web_sources:
            tools.append(TOOL_DEFS["fetch_web_source"])
        if documents:
            tools.append(TOOL_DEFS["read_document_url"])

        handlers = {
            "read_company_transcript": functools.partial(
                tools_module.read_company_transcript, symbol
            ),
            "read_annual_report_section": functools.partial(
                tools_module.read_annual_report_section_async, symbol
            ),
            "read_shareholdings": functools.partial(
                tools_module.read_shareholdings_async, symbol
            ),
        }
        if web_search:
            handlers["tavily_web_search"] = tools_module.tavily_web_search
        if web_sources:
            handlers["fetch_web_source"] = functools.partial(
                tools_module.fetch_web_source, symbol=symbol
            )
        if documents:
            handlers["read_document_url"] = tools_module.read_document_from_url

        prompt = QUALITATIVE_PROMPT.format(
            symbol=symbol,
            parameter=parameter,
            guidelines=guidelines,
        )

        messages = [{"role": "user", "content": prompt}]

        max_turns = 10
        kwargs = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0.1}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        for turn in range(max_turns):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                logger.error(f"LLM call failed on turn {turn}: {e}")
                return {"analysis": f"LLM analysis failed: {str(e)}", "score": 50.0, "error": str(e)}

            msg = response.choices[0].message
            finish = response.choices[0].finish_reason

            if finish == "tool_calls" and msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    handler = handlers.get(func_name)
                    if not handler:
                        result = f"Unknown tool: {func_name}"
                    else:
                        try:
                            coro_or_val = handler(**args)
                            if asyncio.iscoroutine(coro_or_val):
                                result = await coro_or_val
                            else:
                                result = coro_or_val
                        except Exception as e:
                            logger.warning(f"Tool {func_name} failed: {e}")
                            result = f"Error executing {func_name}: {str(e)}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result)[:30000],
                    })
            else:
                content = msg.content or ""
                score_match = re.search(r"FINAL_SCORE:\s*(\d+)", content)
                if score_match:
                    score = float(score_match.group(1))
                    return {"analysis": content, "score": min(score, 100.0)}
                return {"analysis": content, "score": 50.0, "error": "FINAL_SCORE not found in LLM response"}

        return {
            "analysis": "Analysis did not complete within the maximum number of tool call turns.",
            "score": 50.0,
            "error": "Max tool-call turns reached",
        }
