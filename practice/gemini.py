import time 
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from datetime import datetime

client = genai.Client()
MODEL = "gemini-3.8-flash"

calculator = types.FunctionDeclaration(
    name="calculator",
    description="Evaluate a basic arithmetic expression, e.g. '23 * 7 + 4'.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
date=types.FunctionDeclaration(
    name="date",
    description="Get the current date and time.",
   
)
read_file_decl= types.FunctionDeclaration(
    name="read_file",
    description="Read the contents of a file.",
    parameters={
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
    },
)

config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[calculator, date, read_file_decl])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
def get_time():
    return datetime.now().isoformat()

def run_tool(name, args):
    if name == "calculator":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}))
        except Exception as e:
            return f"Error evaluating expression: {e}"
    elif name == "date":
        return get_time()
    elif name == "read_file":
        filename = args.get("filename")
        if filename:
            return read_file_decl(filename)
        else:
            return "Error: 'filename' argument is required."
    return f"unknown tool: {name}"
     


content = [types.Content(role="user", parts=[types.Part(text="What is (1234 * 56) + 789?")])]

for step in range(10):
    response = client.models.generate_content(model=MODEL, contents=content, config=config)
    content.append(response.candidates[0].content)

    calls = response.function_calls
    if not calls:
        print(response.text)
        break

    result_parts = []
    for call in calls:
        output = run_tool(call.name, dict(call.args))        # 1. run the tool
        result_parts.append(                                  # 2. wrap and collect the result
            types.Part.from_function_response(
                name=call.name, response={"result": output}
            )
        )

    content.append(types.Content(role="user", parts=result_parts))
for attempt in range(5):
    try:
        response = client.models.generate_content(model=MODEL, contents=content, config=config)
        break
    except Exception as e:
        print(f"attempt {attempt + 1} failed: {str(e)[:80]}")
        time.sleep(2 ** attempt)   # wait 1s, 2s, 4s, 8s, 16s
else:
    raise SystemExit("API kept failing, try again later")    