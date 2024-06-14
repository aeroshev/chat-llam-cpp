import os
from pathlib import Path
from typing import Final

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate

MODEL_NAME: Final[str] = os.environ['MODEL_NAME']

MODEL_PATH: Final[Path] =  Path(__file__) / "models" / MODEL_NAME

ASK_USER: Final[str] = "Задайте системный промпт для модели LlaMa 3 или введите пустое поле, если хотите использовать стандартный"

TEMPLATE: Final[str] = """### System Prompt
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только на русском языке и ничего не придумывай.

### Current conversation:
{history}

### User Message
{input}

### Assistant"""


@cl.cache
def instantiate_llm() -> LlamaCpp:
    """
    Сохранить в кэше приложения инстанцированную модель.
    :param:
    :return: объект LLM.
    """
    n_batch: int = (
        4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )
    # Make sure the model path is correct for your system!
    llm: LlamaCpp = LlamaCpp(
        model_path=MODEL_PATH.as_posix(),
        n_batch=n_batch,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=64,
        verbose=True,  # Verbose is required to pass to the callback manager
        streaming=True,
    )
    return llm


llm: LlamaCpp = instantiate_llm()


add_llm_provider(
    LangchainGenericProvider(
        id=llm._llm_type,
        name="Llama-cpp",
        llm=llm,
        is_chat=False
    )
)


@cl.on_chat_start
async def prepare_chat() -> None:
    """
    Подготовить чат для работы с пользователем.
    :param:
    :return:
    """
    res: dict = await cl.AskUserMessage(content=ASK_USER).send()
    if res and len(res['output']) > 1:
        template: str = res['output']
    else:
        template: str = TEMPLATE
        

    prompt: PromptTemplate = PromptTemplate(
        template=template,
        input_variables=["history", "input"]
    )

    conversation: ConversationChain = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=ConversationBufferWindowMemory(k=10)
    )

    cl.user_session.set("conv_chain", conversation)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Ответить на вопрос пользователя в чате.
    :param message: объект запрос с приложения chainlit.
    :return:
    """
    conversation: ConversationChain = cl.user_session.get("conv_chain")

    cb: cl.LangchainCallbackHandler = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Assistant"]
    )

    cb.answer_reached = True

    await cl.make_async(conversation)(message.content, callbacks=[cb])
