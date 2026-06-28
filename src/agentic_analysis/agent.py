import os
import re
import json
import time
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
5. TOOL FAILURES: If a tool call fails (indicated by a `[TOOL FAILED: ...]` prefix in its response), you MUST explicitly state in your assessment that the data source was unavailable and that your analysis is limited as a result. Do not fabricate data or pretend the tool succeeded.

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
    "read_nse_document_url": {
        "type": "function",
        "function": {
            "name": "read_nse_document_url",
            "description": "Read the content of an NSE document URL (e.g. earnings transcript PDF, annual report PDF from nsearchives.nseindia.com). Use this to extract text from NSE corporate documents supplied alongside the analysis request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The NSE document URL to read (typically starts with https://nsearchives.nseindia.com/)",
                    }
                },
                "required": ["url"],
            },
        },
    },
}


class NebulaAgent:
    def __init__(self, model="cerebras/qwen-3-32b", api_key=None, rpm=10):
        self.model = model
        self.api_key = api_key
        self.rpm = rpm
        self._last_call_time = 0.0

    async def _rate_limit(self):
        if self.rpm <= 0:
            return
        min_interval = 60.0 / self.rpm
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {wait:.2f}s (rpm={self.rpm})")
            await asyncio.sleep(wait)
        self._last_call_time = time.time()

    async def analyze_parameter(
        self,
        symbol: str,
        parameter: str,
        guidelines: str,
        documents=None,
        web_search=False,
        web_sources=None,
        source="",
    ):
        logger.info(f"Analyzing '{parameter}' for {symbol} (web_search={web_search}, web_sources={web_sources})")

        tools = [
            TOOL_DEFS["read_shareholdings"],
        ]
        if web_search:
            tools.append(TOOL_DEFS["tavily_web_search"])
        if web_sources:
            tools.append(TOOL_DEFS["fetch_web_source"])
        if documents:
            tools.append(TOOL_DEFS["read_nse_document_url"])

        handlers = {
            "read_shareholdings": functools.partial(
                tools_module.read_shareholdings_async, symbol=symbol, source=source
            ),
        }
        if web_search:
            handlers["tavily_web_search"] = tools_module.tavily_web_search
        if web_sources:
            handlers["fetch_web_source"] = functools.partial(
                tools_module.fetch_web_source, symbol=symbol
            )
        if documents:
            handlers["read_nse_document_url"] = functools.partial(
                tools_module.read_nse_document_url, symbol=symbol
            )

        prompt = QUALITATIVE_PROMPT.format(
            symbol=symbol,
            parameter=parameter,
            guidelines=guidelines,
        )

        if documents:
            docs_str = "\n".join(f"- {d}" for d in documents)
            prompt += (
                f"\n\nAVAILABLE DOCUMENTS:\n{docs_str}\n"
                f"Use the `read_nse_document_url` tool with the exact URL from this list. "
                f"Do not fabricate or guess document URLs."
            )

        messages = [{"role": "user", "content": prompt}]
        tool_calls_record = []

        max_turns = 10
        kwargs = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0.1}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        for turn in range(max_turns):
            try:
                await self._rate_limit()
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                logger.error(f"LLM call failed on turn {turn}: {e}")
                return {"analysis": f"LLM analysis failed: {str(e)}", "score": 50.0, "error": str(e), "tool_calls": tool_calls_record}

            msg = response.choices[0].message
            finish = response.choices[0].finish_reason

            if finish == "tool_calls" and msg.tool_calls:
                messages.append(msg.model_dump())
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    handler = handlers.get(func_name)
                    success = False
                    elapsed = 0.0
                    if not handler:
                        result = f"Unknown tool: {func_name}"
                        logger.warning(f"[ToolCall] {func_name} — not found in handler map")
                    else:
                        t0 = time.time()
                        try:
                            coro_or_val = handler(**args)
                            if asyncio.iscoroutine(coro_or_val):
                                result = await coro_or_val
                            else:
                                result = coro_or_val
                            success = True
                        except Exception as e:
                            logger.warning(f"[ToolCall] {func_name} — exception: {e}")
                            result = f"Error executing {func_name}: {str(e)}"

                        elapsed = time.time() - t0
                        data_len = len(str(result))
                        status = "OK" if success else "FAIL"
                        logger.info(
                            f"[ToolCall] {func_name} args={args} → {status} "
                            f"({data_len} chars, {elapsed:.2f}s)"
                        )

                    if success and isinstance(result, str) and result.startswith("Error:"):
                        success = False

                    tool_calls_record.append({
                        "tool_name": func_name,
                        "args": args,
                        "result": str(result)[:5000],
                        "status": "OK" if success else "FAIL",
                        "duration": round(elapsed, 2),
                        "error": None if success else result,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (f"[TOOL FAILED: {func_name}]\n" if not success else "") + str(result)[:30000],
                    })
            else:
                content = msg.content or ""
                score_match = re.search(r"FINAL_SCORE:\s*(\d+)", content)
                if score_match:
                    score = float(score_match.group(1))
                    return {"analysis": content, "score": min(score, 100.0), "tool_calls": tool_calls_record}
                return {"analysis": content, "score": 50.0, "error": "FINAL_SCORE not found in LLM response", "tool_calls": tool_calls_record}

        return {
            "analysis": "Analysis did not complete within the maximum number of tool call turns.",
            "score": 50.0,
            "error": "Max tool-call turns reached",
            "tool_calls": tool_calls_record,
        }
