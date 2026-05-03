import os
from src.utils import read_json
from pprint import pprint
import src.tools
from dotenv import load_dotenv

load_dotenv()

import litellm
from litellm import completion

litellm.set_verbose = False


base_prompt = """You are an expert business analyst.
You are given a information on a particular aspect of a company
along with the exact instructions on how to analyze that aspect.
Your task is to assess that information based on the given instructions
and then output a correlation score - representing how well the company's
particular aspects align with the investment framework given in the instructions.

---

FRAMEWORK
{title}
{framework}

---

COMPANY DETAILS
{data}

---

EXPECTED OUTPUT
Return the response in the following STRICT format:

- Alignment highlights (max 3 bullet points)
- Misalignment / risk factors (max 3 bullet points)
- Brief overall assessment (1 bullet point)

FINAL_SCORE: X

Where:
- X is an integer between 0 and 100
- FINAL_SCORE must be on its own line
- Do not include any text after the FINAL_SCORE line
- Do not include percentages, symbols, or explanations after the number

"""


class NebulAgent(object):
    def __init__(
        self,
        symbol,
        profile,
        model="cerebras/qwen-3-32b",
        # model="cerebras/llama-3.3-70b",
    ):

        self.model = model
        self.symbol = symbol
        self.profile_name = profile

    def run(self, title, desc, tools):

        data = """"""

        for tool in tools:
            func = getattr(src.tools, tool)
            text = func(symbol=self.symbol)
            data += text
            data += "\n\n"

        prompt = base_prompt.format(title=title, framework=desc, data=data)

        # print(prompt)

        response = completion(
            model=self.model, messages=[{"content": prompt, "role": "user"}]
        )

        return response
