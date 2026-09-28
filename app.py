import os
import tempfile
import streamlit as st

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K
from ingestion import SUPPORTED_EXTENSIONS, build_chunks
from rag_engine import RAGEngine

st.set_page_config(page_title='RAG Knowledge Assistant', page_icon='🧠', layout='wide')
st.title('🧠 RAG-Based Knowledge Assistant')
st.caption('Upload documents, retrieve semantic context, and generate grounded answers.')

@st.cache_resource(show_spinner='Loading embedding model...')
def get_engine():
    return RAGEngine(EMBEDDING_MODEL)

engine = get_engine()

with st.sidebar:
    st.header('Knowledge Base')
    uploads = st.file_uploader('Upload documents', type=[ext.lstrip('.') for ext in sorted(SUPPORTED_EXTENSIONS)], accept_multiple_files=True)
    top_k = st.slider('Retrieved chunks (Top-K)', 1, 10, TOP_K)
    chunk_size = st.number_input('Chunk size (words)', 200, 1600, CHUNK_SIZE, 100)
    overlap = st.number_input('Chunk overlap', 20, 400, CHUNK_OVERLAP, 20)
    build = st.button('Build Knowledge Index', type='primary', use_container_width=True)

if build:
    if not uploads:
        st.warning('Upload at least one document first.')
    elif overlap >= chunk_size:
        st.error('Chunk overlap must be smaller than chunk size.')
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for upload in uploads:
                path = os.path.join(temp_dir, upload.name)
                with open(path, 'wb') as file:
                    file.write(upload.getbuffer())
                paths.append(path)
            chunks = build_chunks(paths, int(chunk_size), int(overlap))
            engine.build_index(chunks)
        st.session_state['indexed'] = True
        st.session_state['document_count'] = len(uploads)
        st.session_state['chunk_count'] = len(chunks)
        st.success(f'Indexed {len(uploads)} document(s) into {len(chunks)} chunks.')

if 'indexed' not in st.session_state:
    st.info('Upload documents and click Build Knowledge Index to begin.')
else:
    st.success(f"Knowledge base ready · {st.session_state['document_count']} documents · {st.session_state['chunk_count']} chunks")
    question = st.chat_input('Ask a question about your documents')
    if question:
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            with st.spinner('Retrieving context and generating answer...'):
                results = engine.retrieve(question, top_k)
                answer = engine.answer(question, results)
            st.write(answer)
            if results:
                with st.expander('Retrieved sources'):
                    for item in results:
                        preview = item['text'][:700] + ('...' if len(item['text']) > 700 else '')
                        st.markdown('**' + item['source'] + '** · similarity ' + str(round(item['score'], 3)) + ' · chunk ' + str(item['chunk_id']))
                        st.caption(preview)
