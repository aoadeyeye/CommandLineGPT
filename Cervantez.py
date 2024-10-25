import os
import openai
import anthropic
import json

# Load configuration from the provided JSON file
def load_config():
    try:
        with open('config.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: The config file was not found.")
        return {}
    except json.JSONDecodeError:
        print("Error: Failed to parse the config file.")
        return {}

class OpenAIChatbot:
    def __init__(self, model, config):
        self.model = model
        self.instructions = config.get('instructions', 'default_instructions')
        self.temperature = config.get('temperature', 1.0)

        # Initialize the OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("API key is not set. Please set the OPENAI_API_KEY environment variable.")
            exit(1)

        # Initialize client and assistant creation
        self.client = openai
        self.assistant = self.client.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": self.instructions}]
        )

    def get_response(self, prompt):
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Error: {e}"

class ClaudeAgent:
    def __init__(self, model, config):
        self.model = model
        self.temperature = config.get('temperature', 1.0)
        self.instructions = config.get('instructions', 'default instructions')

        # Initialize the Anthropic API client
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_response(self, prompt):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error: {e}"

def main():
    config = load_config()
    models = config.get("MODELS", [])
    tasks = config.get("TASKS", [])
    instructions = config.get("CONFIG", {}).get("instructions", "You are a helpful assistant.")

    agents = []
    for model in models:
        if "claude" in model["model_name"].lower():
            agent = ClaudeAgent(model, config)
        else:
            agent = OpenAIChatbot(model, config)
        agents.append(agent)

    for task in tasks:
        request = task["request"]
        initial_responses = []
        for agent in agents:
            response = agent.get_response(request + "\n\n" + instructions)
            initial_responses.append(response)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Initial response from agent:\n{response}\n\n")

        critiqued_responses = []
        for i, agent in enumerate(agents):
            other_responses = "\n\n".join(
                [f"Response from another agent:\n{resp}" for j, resp in enumerate(initial_responses) if j != i]
            )
            critique_prompt = "Another LLM responded to the same question as follows. Find the flaws:\n\n" + other_responses
            critique_response = agent.get_response(critique_prompt)
            critiqued_responses.append(critique_response)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Critique by agent {i}:\n{critique_response}\n\n")

        for i, agent in enumerate(agents):
            other_critiques = "\n\n".join(
                [f"Criticism from another agent:\n{resp}" for j, resp in enumerate(critiqued_responses) if j != i]
            )
            refine_prompt = "Other agents criticized your response as follows. Validate criticism and refine as needed:\n\n" + other_critiques
            refined_response = agent.get_response(refine_prompt)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Refined response by agent {i}:\n{refined_response}\n\n")

if __name__ == "__main__":
    main()
