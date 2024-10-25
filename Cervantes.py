#import os
import openai
import anthropic
import json

with open('config.json') as f:
    config = json.load(f)
class OpenAIChatbot:
    def __init__(self):
        # Load the configuration file
        try:
            with open('/Users/aareilorioosha/CommandLineGPT/config.json', 'r') as file:
                config = json.load(file)
                print(config)  # Display the content of the config for debugging
        except FileNotFoundError:
            print("Error: The config file was not found.")
            config = {}
        except json.JSONDecodeError:
            print("Error: Failed to parse the config file.")
            config = {}

        # Safely access the instructions and model keys with default values
        self.instructions = config.get('instructions', 'default_instructions')
        self.model = config.get('model', 'default_model')
#import json

# OpenAIChatbot uses the latest OpenAI API version based on Helper.py
class OpenAIChatbot:
    def __init__(self, model, config):
        self.model = model
        self.instructions = config.get('instructions', 'default_value')
        self.temperature = config.get('temperature', 1.0)
        # Initialize other attributes as needed

def main():
    # Define your configuration dictionary
    config = {
        'instructions': 'You are very helpful',
        'temperature': 1.0,  # Include the temperature key with a default value
        # Add other configuration settings as needed
    }

    # Define the model (adjust based on your use case)
    model = "gpt-4o"  # For OpenAI API

    # Pass the config to the OpenAIChatbot
    agent = OpenAIChatbot(model, config)
#class OpenAIChatbot:
   # def __init__(self, model, config):
    #def __init__(self, config):
       # self.model = model
        #self.instructions = config.get('instructions','default_value')
        #self.temperature = config.get('temperature', 1.0)
        #self.model = model
        #self.model = config['model']
        
        # Check for 'temperature' key and provide a default if missing
        #if 'temperature' not in config:
           # print("Warning: 'temperature' key not found in config, using default value of 1.0")
        

        # Initialize the OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
            print("API key is not set. Please set the OPENAI_API_KEY environment variable.")
            exit(1)

        # Initialize client and thread, assuming the assistant creation part is correct based on Helper.py
            self.client = openai.OpenAI()

        # Create an Assistant and a Thread for interactions
            self.assistant = self.client.beta.assistants.create(
            model=self.model,
            instructions="You are a helpful assistant.",  # You can update this or read from the config if needed
            name="OpenAI Assistant",
            tools=[{"type": "file_search"}]
        )
            self.thread = self.client.beta.threads.create()
    
    #def main():
    # Define your configuration dictionary
        #config = {
           # 'instructions': 'You are a helpful assistance',
            #'temperature': 1.0,  # Include the temperature key with a default value
            # Add other configuration settings as needed
   # }

    # Ensure model is defined before using it
    #model = "gpt-4o"  # Replace with your actual model initialization

    # Pass the config to the OpenAIChatbot
    #agent = OpenAIChatbot(model, config)

    def get_response(self, prompt):
        try:
            # Add a message to the thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=prompt,
            )

            # Run the Assistant to process the response
            my_run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )

            # Wait for the response
            while my_run.status in ["queued", "in_progress"]:
                my_run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=my_run.id
                )

            # Retrieve and return the assistant's response
            if my_run.status == "completed":
                all_messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id
                )
                for message in all_messages.data:
                    if message.role == "assistant":
                        return message.content[0].text.value.strip()

            return "Error: Could not complete the request."
        except Exception as e:
            return f"Error: {e}"

# ClaudeAgent uses the updated syntax for Anthropic's Claude API, based on ClaudeChatUL.py
class ClaudeAgent:
    def __init__(self, model, config):
        self.model = model
        self.temperature = config.get('temperature', 1.0)  # Default value if not found
        self.instructions = config.get('instructions', 'default instructions')

def main():
    # Define your configuration dictionary
    config = {
        'instructions': 'You are very hepful',
        'temperature': 1.0,  # Ensure this key exists
    }

    # Set the OpenAI API key
    openai.api_key = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    #openai.api_key = os.getenv("OPENAI_API_KEY")

    # Define the model (adjust based on your use case)
    model = "gpt-3.5-turbo"  # For OpenAI API

    # Create an instance of ClaudeAgent, passing config
    agent = ClaudeAgent(model, config)
    
#class ClaudeAgent:
    #def __init__(self, config):
    #self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
       # self.model = config['model_code']
        #self.temperature = config['temperature']
#
    def get_response(self, prompt):
        try:
            # Send the message to Claude and get the response
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error: {e}"

# Load configuration from the provided JSON file
def load_config():
    with open('FOO.json', 'r') as file:
        return json.load(file)

# Main function to handle the interaction logic
def main():
    config_data = load_config()
    models = config_data["MODELS"]
    tasks = config_data["TASKS"]
    instructions = config_data["CONFIG"]["instructions"]

    # Initialize agents based on the models in the config
    agents = []
    for model in models:
        if "claude" in model["model_name"].lower():
            agent = ClaudeAgent(model)
        else:
            agent = OpenAIChatbot(model, config)
        agents.append(agent)

    # Loop through each task
    for task in tasks:
        request = task["request"]

        # Step 1: Each agent responds to the original question
        initial_responses = []
        for agent in agents:
            response = agent.get_response(request + "\n\n" + instructions)
            initial_responses.append(response)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Initial response from agent:\n{response}\n\n")

        # Step 2: Each agent critiques other agents' responses
        critiqued_responses = []
        for i, agent in enumerate(agents):
            other_responses = "\n\n".join([f"Response from another agent:\n{resp}" for j, resp in enumerate(initial_responses) if j != i])
            critique_prompt = "Another LLM responded to the same question as follows. Find the flaws:\n\n" + other_responses
            critique_response = agent.get_response(critique_prompt)
            critiqued_responses.append(critique_response)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Critique by agent {i}:\n{critique_response}\n\n")

        # Step 3: Each agent refines its response based on the criticism
        for i, agent in enumerate(agents):
            other_critiques = "\n\n".join([f"Criticism from another agent:\n{resp}" for j, resp in enumerate(critiqued_responses) if j != i])
            refine_prompt = "Other agents criticized your response as follows. Validate criticism and refine as needed:\n\n" + other_critiques
            refined_response = agent.get_response(refine_prompt)
            with open('log_' + task["file_name"], 'a') as f:
                f.write(f"Refined response by agent {i}:\n{refined_response}\n\n")

if __name__ == "__main__":
    main()
