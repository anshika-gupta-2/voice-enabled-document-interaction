import langchain.output_parsers
from langchain.prompts import PromptTemplate


def initialise_pre_grounding_prompt_template():
    prompt_template = """Given the following conversation and a follow-up input, rephrase the follow-up input to be a standalone query, ensuring all relevant information from the conversation is retained.\n\n

    Understanding user intent:\n
    - Users may request specific documents.\n
    - Users may inquire about content within documents. For example : if user is asking "what" or "how" or "asking for steps" or "asking to describe something", then user is seeking for insights. Generation of answer is required\n
    - If a user mentions a direct topic, treat it as a document retrieval request by default.\n

    Additionally, for document retrieval while optimising use below question transformations
        -   'Get me all documents',
        -   'Fetch all reports',
        -   'What all documents do I have?',
        -   'Show me documents for _____',
        -   'Retrieve all my documents.',
        -   'List all the documents I possess.',
        -   'Can you show me all the reports?',
        -   'I need to see all documents.',
        -   'Bring up all documents for _____.',
        -   'Access all the documents on file.',
        -   'Please pull up all reports related to _____.',
        -   'What documents do we have?',
        -   'Show all documents related to _____.',
        -   'Can you get me a document on _________'

    Note : If user try to do a small talk, return the question as it is

    Chat History:
    {chat_history}\n
    Follow-Up Input: {question}\n
    Standalone Query:"""

    output_parser = langchain.output_parsers.CommaSeparatedListOutputParser()
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "chat_history"],
        output_parser=output_parser,
    )
