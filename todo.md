1. ~~Make the project work with OpenAI as a model provider.
1.1. 
2. Add an integration test that runs with MODEL_PROVIDER=openai (skipped unless OPENAI_API_KEY is available) or a CI matrix item for OpenAI if you want to test both providers automatically.
3. Add an integration test that runs only when OPENAI_API_KEY is present (skip otherwise) to verify OpenAI end-to-end.
4. Create a script that sets up the environment, installs dependencies, downloads necessary models, and starts the application.
5. Generate a web interface to interact with the agent.