from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider


# Download the file here https://huggingface.co/TheBloke/Yarn-Llama-2-7B-128K-GGUF/tree/main
# and update the path
MODEL_PATH = "./models/model-q4_K.gguf"


@cl.cache
def instantiate_llm():
    n_batch = (
        4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_batch=n_batch,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=64,
        verbose=True,  # Verbose is required to pass to the callback manager
        streaming=True,
    )
    return llm


llm = instantiate_llm()

add_llm_provider(
    LangchainGenericProvider(
        id=llm._llm_type,
        name="Llama-cpp",
        llm=llm,
        is_chat=False
    )
)


@cl.on_chat_start
def main():
    template = """### System Prompt
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только на русском языке и ничего не придумывай.

### Current conversation:
{history}

### User Message
{input}

### Assistant"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["history", "input"]
    )

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=ConversationBufferWindowMemory(k=10),
        verbose=True
    )

    cl.user_session.set("conv_chain", conversation)


@cl.on_message
async def main(message: cl.Message):
    conversation = cl.user_session.get("conv_chain")

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Assistant"]
    )

    cb.answer_reached = True

    res = await cl.make_async(conversation)(message.content, callbacks=[cb])
