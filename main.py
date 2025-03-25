#!/bin/python

import logging
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from pathlib import Path
from argparse import ArgumentParser
from openai import OpenAI
from llm_guard.output_scanners import NoRefusalLight

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
cur_dir = Path(__file__).parent

class PromptsConfig:
    def __init__(self, config_file):
        self.file = Path(config_file)
        self.full_text = ""
        self.prompts_dict = {}

    def fetch(self):
        text = self.file.read_text()
        if text != self.full_text:
            self.full_text = text
            self.parse_full_text()

    def parse_full_text(self):
        sections = None
        cur_section_id = None
        cur_section_content = ""

        self.prompts_dict = {}
        log.info("Parsing prompts config content into separate prompt templates")
        for i, l in enumerate(self.full_text.splitlines(keepends=True)):
            if i == 0:
                sections = l.strip().split(",")
                sections[-1] = sections[-1].strip()
            elif not l.isspace() and len(l) > 0 and ((not cur_section_id) or l.strip() in sections):
                if len(cur_section_content) > 0:
                    self.prompts_dict[cur_section_id] = cur_section_content
                cur_section_id = l.strip()
                cur_section_content = ""
            elif cur_section_id:
                cur_section_content += l
        self.prompts_dict[sections[-1]] = cur_section_content
        
    def get(self, prompt_id, fetch=False):
        if fetch:
            self.fetch()
        return self.prompts_dict[prompt_id]

parser = ArgumentParser()
parser.add_argument("-e", "--endpoint", required=False, help="The type of endpoint serving the model", choices=["bedrock", "localhost"], default="bedrock")
parser.add_argument("-m", "--model", required=False, help="The model identifier")
parser.add_argument("-a", "--aws-profile", required=False, help="The AWS profile to use in your credentials file")
parser.add_argument("-f", "--prompt-file", required=True, help="The flat file containing all the prompts that might be used (the system prompt must be tagged SYSTEM)")
parser.add_argument("-p", "--prompt", required=True, help="This is the name of the section in the prompts flat file to use for the prompt. Can be a comma-delimited list, in which case the prompts alternate between human and assistant, for few-shot engineering.")
parser.add_argument("-k", "--key", action="append", help="Can be used to pass in arguments to the prompt template. Can be specified multiple times.")
parser.add_argument("-v", "--value", action="append", help="Can be used to pass in arguments to the prompt template. Can be specified multiple times. Prepend '@' to indicate a file location whose contents will be read into the template variable value.")
parser.add_argument("-t", "--temperature", required=False, type=float, default=0.0, help="The model temperature (defaults to zero)")
parser.add_argument("-n", "--num", required=False, type=int, default=1, help="The number of choices to generate (defaults to one)")
parser.add_argument("-s", "--exclude-sys", required=False, action="store_true", help="Flag indicating whether to ignore the system prompt in the prompt file (defaults to False)")
parser.add_argument("-r", "--rebuff-prompt", required=False, help="When provided, scan the LLM output for refusals, and if a refusal is detected use the indicated prompt to reply.")
parser.add_argument("-y", "--api-key", required=False, help="When using a locally-hosted model, this is the OpenAI key to pass in.", default="token-abc123")

args = parser.parse_args()

pconfig = PromptsConfig(args.prompt_file)
template_args = {}
if args.key and len(args.key) > 0:
    for i, k in enumerate(args.key):
        val = args.value[i]
        if val.startswith("@"):
            val = Path(val[1:]).read_text()
        template_args[k] = val
print(template_args)

def get_prompt_templates(prompt_titles):
    for i, p in enumerate(prompt_titles.split(",")):
        if i % 2 == 0:
            yield HumanMessagePromptTemplate.from_template(pconfig.get(p))
        else:
            yield AIMessagePromptTemplate.from_template(pconfig.get(p))

def invoke_bedrock(prompt, llm, template_args):
    runnable = prompt | llm
    resp = runnable.stream(template_args)
    ttl_resp = ""
    for chunk in resp:
        print(chunk.content, end="", flush=True)
        ttl_resp += chunk.content
    return ttl_resp

if args.exclude_sys:
    pconfig.fetch()
    p = ChatPromptTemplate.from_messages([p for p in get_prompt_templates(args.prompt)])
else:
    p = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(pconfig.get("SYSTEM", True))] + [p for p in get_prompt_templates(args.prompt)]
    )

if args.endpoint == "bedrock":
    profile_name = None
    if args.aws_profile:
        profile_name = args.aws_profile
    model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    if args.model:
        model = args.model
    llm = ChatBedrock(
        model_id=model,
        model_kwargs={
            "max_tokens": 8192,
            "temperature": args.temperature
        },
        config=Config(connect_timeout=120, read_timeout=120, retries={"mode": "adaptive"}),
        streaming=True,
        credentials_profile_name=profile_name
    )

    ttl_resp = invoke_bedrock(p, llm, template_args)
    
    if args.rebuff_prompt:
        scanner = NoRefusalLight()
        last_human_prompt = p.format_messages(**template_args)[-1].content
        sanitized_output, is_valid, risk_score = scanner.scan(last_human_prompt, ttl_resp)
        if not is_valid:
            print("\nDetected refusal, rebuffing with rebuff prompt...\n")
            p.append(HumanMessage(content=pconfig.get(args.rebuff_prompt)))
            invoke_bedrock(p, llm, template_args)
        
else:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=args.api_key
    )
    roles = { SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant" }
    messages = [{"role": roles[type(m)], "content": m.content} for m in p.format_messages(**template_args)]
    model = "local"
    if args.model:
        model = args.model
    resp = client.chat.completions.create(
        model=model,
        temperature=args.temperature,
        messages=messages,
        n=args.num
    )
    for choice in resp.choices:
        if args.num > 1:
            print("*** CHOICE ***")
        print(choice.message.content)
        print()
