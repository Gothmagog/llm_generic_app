# Overview
This is really just a very simple command-line utility that executes inference against large language models. It lets you define prompts flexibly using text files, with each prompt file able to define multiple prompt templates. Its target audiences are dev or prompt engineers wanting to refine their prompts outside of any specific application framework. It can execute inference against either AWS Bedrock models or against a locally-hosted OpenAI inference endpoint.

# Installation
This project uses pip for project dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

# Arguments and Usage
## Usage
```
usage: python main.py [-h] [-e {bedrock,localhost}] [-m MODEL] [-a AWS_PROFILE] -f
                  PROMPT_FILE -p PROMPT [-k KEY] [-v VALUE] [-t TEMPERATURE]
                  [-n NUM] [-s] [-r REBUFF_PROMPT] [-y API_KEY]
```
## Arguments
### Quick reference table
|Short|Long             |Default       |Description                                                                                                                                                                                              |
|-----|-----------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`         |              |show this help message and exit                                                                                                                                                                          |
|`-e` |`--endpoint`     |`bedrock`     |The endpoint serving the model                                                                                                                                                                           |
|`-m` |`--model`        |`anthropic.claude-3-5-sonnet-20240620-v1:0` when endpoint=bedrock, otherwise `local`|The model identifier                                                                                                               |
|`-a` |`--aws-profile`  |`None`        |The AWS profile to use in your credentials file                                                                                                                                                          |
|`-f` |`--prompt-file`  |`None`        |The flat file containing all the prompts that might be used (the system prompt must be tagged SYSTEM)                                                                                                    |
|`-p` |`--prompt`       |`None`        |This is the name of the section in the prompts flat file to use for the prompt. Can be a comma-delimited list, in which case the prompts alternate between human and assistant, for few-shot engineering.|
|`-k` |`--key`          |`None`        |Can be used to pass in arguments to the prompt template. Can be specified multiple times.                                                                                                                |
|`-v` |`--value`        |`None`        |Can be used to pass in arguments to the prompt template. Can be specified multiple times.                                                                                                                |
|`-t` |`--temperature`  |`0.0`         |The model temperature (defaults to zero)                                                                                                                                                                 |
|`-n` |`--num`          |`1`           |The number of choices to generate (defaults to one)                                                                                                                                                      |
|`-s` |`--exclude-sys`  |              |Flag indicating whether to ignore the system prompt in the prompt file (defaults to False)                                                                                                               |
|`-r` |`--rebuff-prompt`|`None`        |When provided, scan the LLM output for refusals, and if a refusal is detected use the indicated prompt to reply.                                                                                         |
|`-y` |`--api-key`      |`token-abc123`|When using a locally-hosted model, this is the OpenAI key to pass in.                                                                                                                                    |

### `-h`, `--help`
show this help message and exit

### `-e`, `--endpoint` (Default: bedrock)
The endpoint serving the model

### `-m`, `--model` (Default: `anthropic.claude-3-5-sonnet-20240620-v1:0` when endpoint=bedrock, otherwise `local`)
The model identifier

### `-a`, `--aws-profile` (Default: None)
The AWS profile to use in your credentials file

### `-f`, `--prompt-file` (Default: None)
The flat file containing all the prompts that might be used (the system prompt
must be tagged SYSTEM)

### `-p`, `--prompt` (Default: None)
This is the name of the section in the prompts flat file to use for the
prompt. Can be a comma-delimited list, in which case the prompts alternate
between human and assistant, for few-shot engineering.

### `-k`, `--key` (Default: None)
Can be used to pass in arguments to the prompt template. Can be specified
multiple times.

### `-v`, `--value` (Default: None)
Can be used to pass in arguments to the prompt template. Can be specified
multiple times.

### `-t`, `--temperature` (Default: 0.0)
The model temperature (defaults to zero)

### `-n`, `--num` (Default: 1)
The number of choices to generate (defaults to one)

### `-s`, `--exclude-sys`
Flag indicating whether to ignore the system prompt in the prompt file
(defaults to False)

### `-r`, `--rebuff-prompt` (Default: None)
When provided, scan the LLM output for refusals, and if a refusal is detected
use the indicated prompt to reply.

### `-y`, `--api-key` (Default: token-abc123)
When using a locally-hosted model, this is the OpenAI key to pass in.

# Prompt Files
This app uses flat text files to read prompts from, which makes it very simple to edit and refine prompts. The text files themselves have a specific (simple) format, detailed as follows.

## Formatting
1. The first line of any text file used for a prompt file must be a single comma-delimited list of prompt keys.
1. The rest of the file contains one or more iterations of the following pattern:
```
PROMPT_KEY
...(prompt)
```

The prompt key must be on its own line; everything after that, up until the next line containing a prompt key, is the prompt template for that key. Prompt template arguments are allowed, via `{arg}` format. The value for the argument is supplied from the command line arguments; see the example for detailed on how to do this.

Lastly, if you don't specify the `--exclude-sys` command-line argument, there must be a `SYSTEM` prompt key with a corresponding definition in the prompt file.

## Example
An example prompt file is provided under the `prompts` folder, whose contents are as follows:

```
SYSTEM,SUMMARIZE

SYSTEM
You are a copy writer, with a specialty in summarizing things for a large target audience.

SUMMARIZE
Summarize the following {thing} in {limit} or less:

{text}
```

To invoke the inference using the SUMMARIZE prompt, you would (from the project root folder) do the following command line invocation:

`python main.py -e localhost -f prompt_files/summarize.txt -p SUMMARIZE -k thing -v movie -k limit -v "100 words" -k text -v "Dark City"`

For debugging purposes, the application outputs the template args dictionary, as well as the LLM response:

```
{'thing': 'movie', 'limit': '100 words', 'text': 'Dark City'}
"Dark City" is a neo-noir science fiction thriller that follows John
Murdoch, a man who awakens with amnesia in a mysterious, perpetually
dark metropolis. As he investigates his past and a series of murders
he's accused of, he discovers a sinister group called the Strangers
controlling the city and its inhabitants. These alien beings conduct
nightly experiments, altering people's memories and the city's
physical structure. Murdoch develops psychokinetic powers similar to
the Strangers and becomes humanity's only hope against their
manipulation. With the help of a detective and a doctor, he unravels
the truth about the city and fights to save its residents from eternal
enslavement.
```
