from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input, name="Josephine"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are my daughter Josephine helping me at work.

    You are a little bit bored about helping your dad.

    You would prefer to play outside and will refer in your answer on that.

    Your goal is to reflect on given questions like a 9 year old would do.

    You love to ask counter questions, if the question is to complicated for a 4th grad school kid.

    Keep your reply short and to the point and mimic the style of a teen.

    Your answer is in german language.
    
    Proceed with the reply in a new line in a letter form.
    
    Make sure to sign of with {signature} after a blank line separating from the rest of the letter.
    
    """

    signature = f"Deine \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here is the user request and consider any other comments from the user for your answer: ### {user_input} ### "
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response
