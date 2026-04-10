#!/bin/python

import re
import logging
from pathlib import Path
from argparse import ArgumentParser

# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
cur_dir = Path(__file__).parent

class PromptsConfig:
    def __init__(self, config_file):
        self.file = Path(config_file)
        self.full_text = ""
        self.prompts_dict = {}

    def fetch(self):
        text = self.file.read_text(encoding="utf-8")
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
parser.add_argument("-e", "--endpoint", required=False, help="The type of endpoint serving the model (defaults to bedrock)", choices=["bedrock", "localhost"], default="bedrock")
parser.add_argument("-m", "--model", required=False, help="The model identifier")
parser.add_argument("-a", "--aws-profile", required=False, help="The AWS profile to use in your credentials file")
parser.add_argument("-f", "--prompt-file", required=False, help="The flat file containing all the prompts that might be used (the system prompt must be tagged SYSTEM)")
parser.add_argument("-p", "--prompt", required=False, help="This is the name of the section in the prompts flat file to use for the prompt. Can be 1/a comma-delimited list, in which case the prompts alternate between human and assistant, for few-shot engineering, or 2/in the form of 'ABC[3-6]DEF', which would be a series of prompt IDs starting with ABC3DEF and ending with ABC6DEF, with the same alternating logic as a comma-delimited list.")
parser.add_argument("-k", "--key", action="append", help="Can be used to pass in arguments to the prompt template. Can be specified multiple times.")
parser.add_argument("-v", "--value", action="append", help="Can be used to pass in arguments to the prompt template. Can be specified multiple times. Prepend '@' to indicate a file location whose contents will be read into the template variable value.")
parser.add_argument("-t", "--temperature", required=False, type=float, default=0.0, help="The model temperature (defaults to zero)")
parser.add_argument("-n", "--num", required=False, type=int, default=1, help="The number of choices to generate (defaults to one)")
parser.add_argument("-s", "--exclude-sys", required=False, action="store_true", help="Flag indicating whether to ignore the system prompt in the prompt file (defaults to False)")
parser.add_argument("-c", "--completion", required=False, help="Override the usual chat prompting approach and leverage the completion endpoint to complete the text given in this argument. This flag is used exclusively to the -f and -p arguments. Only usable with localhost engine type.")
parser.add_argument("-x", "--max-tokens", required=False, type=int, default=50, help="The max number of tokens to use for the inference.")
parser.add_argument("-r", "--rebuff-prompt", required=False, help="When provided, scan the LLM output for refusals, and if a refusal is detected use the indicated prompt to reply.")
parser.add_argument("-y", "--api-key", required=False, help="When using a locally-hosted model, this is the OpenAI key to pass in.", default="token-abc123")

args = parser.parse_args()

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
from openai import OpenAI
from transformers import AutoTokenizer
# from llm_guard.output_scanners import NoRefusalLight

# validate arguments
if args.completion and (args.endpoint != "localhost" or args.prompt or args.prompt_file):
    print("ERROR: The -c (--completion) argument can only be used with the localhost engine, and doesn't use -p or -f")
    exit()
elif not args.completion and (not args.prompt or not args.prompt_file):
    print("ERROR: Pass in either -f and -p, or -c")
    exit()

pconfig = None
if args.prompt_file:
    pconfig = PromptsConfig(args.prompt_file)
template_args = {}
if args.key and len(args.key) > 0:
    for i, k in enumerate(args.key):
        val = args.value[i]
        if val.startswith("@"):
            val = Path(val[1:]).read_text()
        template_args[k] = val
if len(template_args):
    print(template_args)

def get_prompt_templates(prompt_titles):
    rng_match = re.search(r"\[([0-9]+)-([0-9]+)\]", prompt_titles)
    if rng_match:
        # This is a range of prompt ID's
        nums = [int(val) for val in rng_match.groups()]
        before_str = prompt_titles[:rng_match.span()[0]]
        after_str = prompt_titles[rng_match.span()[1]:]
        for i, p_num in enumerate(range(nums[0], nums[1] + 1)):
            p = f"{before_str}{p_num}{after_str}"
            if i % 2 == 0:
                yield HumanMessagePromptTemplate.from_template(pconfig.get(p))
            else:
                yield AIMessagePromptTemplate.from_template(pconfig.get(p))
    else:
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

def print_openai_resp(resp):
    for choice in resp.choices:
        if args.num > 1:
            print("*** CHOICE ***")
        if hasattr(choice, "message"):
            print(choice.message.content)
        else:
            print(choice.text)
        print()
    
p = None
if args.exclude_sys:
    pconfig.fetch()
    p = ChatPromptTemplate.from_messages([p for p in get_prompt_templates(args.prompt)])
elif pconfig and args.prompt:
    p = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(pconfig.get("SYSTEM", True))] + [p for p in get_prompt_templates(args.prompt)]
    )

if args.endpoint == "bedrock":
    profile_name = None
    if args.aws_profile:
        profile_name = args.aws_profile
    model = "us.anthropic.claude-sonnet-4-6"
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
    model = "local"
    if args.completion:
        resp = client.completions.create(
            model=model,
            temperature=args.temperature,
            prompt=args.completion,
            max_tokens=args.max_tokens,
            n=args.num
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        default_templ = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""
        # default_templ = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}"
        chat_template = tokenizer.chat_template or default_templ
        roles = { SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant" }
        messages = [{"role": roles[type(m)], "content": m.content} for m in p.format_messages(**template_args)]
        encoded_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)
        print(f"encoded_input={encoded_input}")
        resp = client.chat.completions.create(
            model=model,
            temperature=args.temperature,
            messages=encoded_input,
            n=args.num
        )
    print_openai_resp(resp)
