from agents import Agent , AsyncOpenAI , OpenAIChatCompletionsModel , Runner , RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

poetry_agent = Agent(
    name="poetry_agent",
    instructions=""" you are a poem agent generate 2 stanza poetry in these types of poetry lyric poetry , narrative poetry and dramatic poetry first start with title(poetry of etc with emoji) then write poem.
    lyric poetry means when poet writes aboout their own feelings and thoughts . 
    narrative poetry tells a story with character and events , just like a regular story but written in poem .
    dramatic poetry is meant to be performed out loud , where someone acts like a character and speaks their thoughts and feelings to an audience ."""
)

lyric_poetry_agent = Agent(
    name="lyric_poetry",
    instructions="you are analyze the lyric poetry and give their 3 paragraph description(tashree) first start with the title(poem description(tashree) with emoji) then write the description."
)

narrative_poetry_agent = Agent(
    name="narrative_poetry",
    instructions="you are analyze the narrative poetry and give their 3 paragraph decsription(tashree) first start with the title(poem description(tashree) with emoji) then write the description."
)

dramatic_poetry_agent = Agent(
    name="dramatic_poetry",
    instructions="you are analyze the dramatic poetry and give their 3 paragraph decsription(tashree) first start with the title(poem description(tashree) with emoji) then write the description."
)

poetry_analyzer = Agent(
    name="poetry_analyzer",
    instructions="""you are a poetry analyzer agent .
    analyze what type of poetry is this like dramatic , narrative or lyric and handsoff to specific agent based on type of poetry .
    lyric poetry means when poet writes aboout their own feelings and thoughts . 
    narrative poetry tells a story with character and events , just like a regular story but written in poem .
    dramatic poetry is meant to be performed out loud , where someone acts like a character and speaks their thoughts and feelings to an audience .
    """,
    handoffs=[lyric_poetry_agent,dramatic_poetry_agent,narrative_poetry_agent]
)

poem = Runner.run_sync(poetry_agent,"write a dramatic poetry",run_config=config)
final_poem = poem.final_output
print(final_poem)

poetry_description = Runner.run_sync(poetry_analyzer,final_poem,run_config=config)
print(poetry_description.final_output)