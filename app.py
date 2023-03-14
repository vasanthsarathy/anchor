import streamlit as st
import os
from pipelines import get_pipeline
import logging
from json import JSONDecodeError
from utils import find_substring_indices
from annotated_text import annotation
from markdown import markdown


# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def query(concept, filters={}, top_k_retriever=5):
    params ={"Retriever": {"top_k": top_k_retriever}}
    pipe = get_pipeline("data/narratives/processed")

    prediction = pipe.run(
        query=concept,
        params={"Retriever": {"top_k": top_k_retriever}
        }
    )

    # Format results
    results = []
    spans = prediction['results']
    for idx, span in enumerate(spans):
        context = prediction["documents"][idx].to_dict()['content']
        span_indices = find_substring_indices(context, span)

        if span_indices:
            result = {"context": context,
                      "span": span,
                      "span_start": span_indices[0],
                      "span_end": span_indices[1]}
            results.append(result)
    return results




def main():

    st.set_page_config(page_title="Anchor")

    # Persistent state
    set_state_if_absent("question", "husband's permission")
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("""
    # ⚓ ANCHOR

    #### Grounding Abstract Concepts in Text
    """)



    # Sidebar
    st.sidebar.header("Options")

    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=20,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )


    # Search bar
    question = st.text_input(
        value=st.session_state.question,
        max_chars=100,
        on_change=reset_results,
        label="Concept",
        label_visibility="visible",
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    run_query = (run_pressed or question != st.session_state.question)


    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question

        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
        ):
            try:
                st.session_state.results = query(
                    question, top_k_retriever=top_k_retriever
                )
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return


    if st.session_state.results:

        st.write("## Results:")

        for count, result in enumerate(st.session_state.results):
            if result['span']:
                st.write(
                    markdown(result['context'][:result['span_start']] +
                             str(annotation(result['span'], "anchor", "#fad6a5")) +
                             result['context'][result['span_end']+1:]),
                    unsafe_allow_html=True
                )

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["relevance"])


main()


