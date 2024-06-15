import os
from pathlib import Path
from typing import Final

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.llms import LlamaCpp
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

MODEL_NAME: Final[str] = os.environ['MODEL_NAME']
WINDOW_SIZE: Final[int] = int(os.environ.get('WINDOW_SIZE', 10))

MODEL_PATH: Final[Path] =  Path(__file__).parent / "models" / MODEL_NAME

ASK_USER: Final[str] = "Задайте системный промпт для модели LlaMa 3 или введите пустое поле, если хотите использовать стандартный"

DEFAULT_SYSTEM_PROMPT: Final[str] = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только на русском языке и ничего не придумывай."


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
        system: str = res['output']
    else:
        system: str = DEFAULT_SYSTEM_PROMPT
    
    await cl.Message(
        content=f"Для работы выбран системный промпт:\n{system}"
    ).send()
        

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    conversation: ConversationChain = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=WINDOW_SIZE,
            return_messages=True
        )
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
        stream_final_answer=True
    )

    cb.answer_reached = True

    await cl.make_async(conversation)(message.content, callbacks=[cb])
