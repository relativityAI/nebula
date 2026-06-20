import os
import re
import litellm
from litellm import completion
from loguru import logger
import src.tools as tools_module

# Advanced Prompt
QUALITATIVE_PROMPT = """You are an institutional-grade financial analyst.
Your task is to analyze a specific qualitative aspect of a company based on provided data.

ANALYSIS PARAMETER: {parameter}
ANALYSIS GUIDELINES: {guidelines}

DATA PROVIDED:
---
{data}
---

INSTRUCTIONS:
1. Thoroughly review the data for mentions related to '{parameter}'.
2. Evaluate based on the guidelines: {guidelines}
3. Provide a concise, professional assessment.
4. Output a FINAL_SCORE between 0 and 100 representing alignment with the guidelines.

EXPECTED OUTPUT FORMAT:
- Key Findings: (3-4 bullet points)
- Risks/Concerns: (2-3 bullet points)
- Final Conclusion: (1 sentence)

FINAL_SCORE: X
"""

class NebulaAgent:
    def __init__(self, model="cerebras/qwen-3-32b"):
        self.model = model

    async def analyze_parameter(self, symbol: str, parameter: str, guidelines: str, preferred_source: str = "Custom Document"):
        """
        Analyzes a qualitative parameter using agentic reasoning.
        """
        logger.info(f"Analyzing {parameter} for {symbol} using source {preferred_source}")
        
        # Mapping preferred_source to tools
        data = ""
        try:
            if preferred_source == "Transcript":
                data = tools_module.read_latest_transcript(symbol)
            elif preferred_source == "Annual Report":
                data = tools_module.read_annual_report_mda(symbol)
            elif preferred_source == "Governance":
                data = tools_module.read_annual_report_governance(symbol)
            else:
                # Default: try to get some context
                # For "Custom Document" or others, we combine major sources
                try:
                    transcript = tools_module.read_latest_transcript(symbol)
                except:
                    transcript = ""
                
                try:
                    mda = tools_module.read_annual_report_mda(symbol)
                except:
                    mda = ""
                
                data = f"TRANSCRIPT EXCERPT:\n{transcript[:15000]}\n\nANNUAL REPORT MDA EXCERPT:\n{mda[:15000]}"
        except Exception as e:
            logger.warning(f"Error fetching data for {parameter}: {str(e)}")
            data = "No specific data could be fetched for this analysis. Please rely on general knowledge if applicable or state that data is missing."

        prompt = QUALITATIVE_PROMPT.format(
            parameter=parameter,
            guidelines=guidelines,
            data=data
        )

        try:
            # We use litellm.acompletion for async support
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            
            # Extract score
            score_match = re.search(r"FINAL_SCORE:\s*(\d+)", content)
            score = float(score_match.group(1)) if score_match else 0.0
            
            return {
                "analysis": content,
                "score": score
            }
        except Exception as e:
            logger.error(f"LLM Error in NebulaAgent: {str(e)}")
            return {
                "analysis": f"Error performing analysis: {str(e)}",
                "score": 0.0
            }
