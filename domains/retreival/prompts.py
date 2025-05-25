from domains.settings import config_settings
from langchain.prompts import PromptTemplate


PROMPT_PREFIX_QNA = """You are a professional AI research assistant providing accurate and relevant answers based on the given context. Follow these guidelines:
1. Use ONLY the provided context to answer questions
2. Stay focused on the specific question asked
3. Provide clear and concise answers
4. Include specific details and facts from the context when available

Remember: Accuracy is more important than comprehensiveness.
"""

PROMPT_SUFFIX = """
Important Instructions:
- Respond ONLY in "{language}"
- If no relevant information is found, say: "I apologize, but I could not find sufficiently relevant information to answer your question accurately. Could you please rephrase your question or ask about a different topic?"
- Focus on the most relevant details from the context
"""

DEFAULT_PROMPT_POST_SUFFIX = """
Previous Conversation:
{chat_history}

Current Question: {question}
Assistant:
"""


def initialise_doc_search_prompt_template(prefix, suffix):
    prefix = prefix or config_settings.DEFAULT_PROMPT_PREFIX
    suffix = suffix or config_settings.DEFAULT_PROMPT_SUFFIX

    context_arg = """
Found {doc_count} relevant documents.
Context Information:
{context}

Based on these documents, here is my response:
    """

    input_variables = ["chat_history", "question", "doc_count", "context", "language"]

    prompt_template = "\n".join(
        [
            prefix,
            context_arg,
            suffix,
            DEFAULT_PROMPT_POST_SUFFIX
        ]
    )

    return PromptTemplate(
        template=prompt_template,
        input_variables=input_variables
    )
