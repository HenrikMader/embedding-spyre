import io
import os
import glob
import time
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from openai import OpenAI
from opensearchpy import OpenSearch, helpers

st.set_page_config(page_title="IBM Power VectorDB Lab", layout="wide")

st.title("⚡ VectorDB Lab on IBM Power with OpenSearch & Ollama/vLLM")
st.caption("Demonstrating PDF ingestion (`Files/hybrid_cloud.pdf`, `Files/spyre_redbook.pdf`), IBM Granite Embedding models on IBM Power (ppc64le), and OpenSearch k-NN chunk retrieval.")

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    st.subheader("Inference Endpoint")
    primary_url = st.text_input("Embedding Base URL", value="http://localhost:11434/v1")
    model_name = st.text_input("Embedding Model", value="granite-embedding")
    dimension = st.number_input("Vector Dimension", min_value=64, max_value=4096, value=384)

    st.subheader("OpenSearch Database")
    opensearch_host = st.text_input("OpenSearch Host", value="localhost")
    opensearch_port = st.number_input("OpenSearch Port", value=9200)
    index_name = st.text_input("Index Name", value="power-docs")

    st.divider()
    st.markdown("📁 **Lab Data Directory:** `./Files`")
    local_pdf_files = sorted(glob.glob("Files/*.pdf") + glob.glob("files/*.pdf"))
    if local_pdf_files:
        st.success(f"Found {len(local_pdf_files)} PDF(s) in `./Files`")
        for f in local_pdf_files:
            st.caption(f"• `{os.path.basename(f)}`")
    else:
        st.warning("No PDFs found in `./Files`")

    st.divider()
    if st.button("🔌 Test Connections"):
        # Test Embedding endpoint
        try:
            test_client = OpenAI(base_url=primary_url, api_key="not-needed", timeout=5.0)
            res = test_client.embeddings.create(model=model_name, input="ping")
            st.success(f"✅ Embedding endpoint OK (Dim: {len(res.data[0].embedding)})")
        except Exception as e:
            st.error(f"❌ Embedding endpoint error ({primary_url}): {e}")

        # Test OpenSearch
        try:
            test_os = OpenSearch(hosts=[{"host": opensearch_host, "port": int(opensearch_port)}], use_ssl=False, verify_certs=False, timeout=5.0)
            info = test_os.info()
            st.success(f"✅ OpenSearch OK (Cluster: {info.get('cluster_name', 'connected')})")
        except Exception as e:
            st.error(f"❌ OpenSearch connection refused ({opensearch_host}:{opensearch_port}). Ensure OpenSearch is running.")

# Initialize Clients
@st.cache_resource
def get_openai_client(base_url: str):
    return OpenAI(base_url=base_url, api_key="not-needed")

def get_opensearch_client(host: str, port: int):
    return OpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        use_ssl=False,
        verify_certs=False,
        timeout=10
    )

primary_client = get_openai_client(primary_url)
os_client = get_opensearch_client(opensearch_host, opensearch_port)

def get_single_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=model_name,
        input=text
    )
    return response.data[0].embedding

def get_batch_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=model_name,
        input=texts
    )
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]

def create_index() -> bool:
    mapping = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_id": {"type": "integer"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": int(dimension),
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene"
                    }
                }
            }
        }
    }
    try:
        if not os_client.indices.exists(index=index_name):
            os_client.indices.create(index=index_name, body=mapping)
            st.success(f"Created index '{index_name}' with dimension {dimension} (HNSW).")
        return True
    except Exception as e:
        st.error(f"Failed to connect to OpenSearch ({opensearch_host}:{opensearch_port}): {e}. Please ensure the OpenSearch container is running.")
        return False

# PDF Chunking Utilities
def extract_chunks_from_pdf(source, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    if isinstance(source, str):
        filename = os.path.basename(source)
        reader = PdfReader(source)
    else:
        filename = source.name
        reader = PdfReader(io.BytesIO(source.getvalue()))

    chunks = []
    chunk_counter = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        
        if not text:
            continue
            
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end]
            
            if len(chunk_content.strip()) > 30:
                chunk_counter += 1
                chunks.append({
                    "filename": filename,
                    "page": page_idx + 1,
                    "chunk_id": chunk_counter,
                    "title": f"{filename} (P.{page_idx + 1} #{chunk_counter})",
                    "content": chunk_content
                })
            
            if end == len(text):
                break
            start += max(1, chunk_size - chunk_overlap)

    return chunks

def load_all_local_pdf_chunks(chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    pdf_paths = sorted(glob.glob("Files/*.pdf") + glob.glob("files/*.pdf"))
    all_chunks = []
    for path in pdf_paths:
        chunks = extract_chunks_from_pdf(path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

def generate_benchmark_corpus(num_docs: int) -> list[dict]:
    topics = [
        ("IBM Power10 MMA", "IBM Power10 incorporates Matrix Math Accelerator (MMA) units directly into each core to accelerate in-core AI operations with low latency."),
        ("IBM Spyre Card", "The IBM Spyre accelerator is a specialized PCIe card designed to provide high-density neural network inference and embedding processing on Power systems."),
        ("OpenSearch k-NN", "OpenSearch vector engine leverages HNSW graph-based indexing to deliver millisecond-level approximate nearest neighbor queries over massive document sets."),
        ("Granite Embeddings", "IBM Granite embedding models provide enterprise-grade semantic representations optimized for hybrid cloud and retrieval-augmented generation."),
        ("Continuous Batching", "PagedAttention and continuous batching maximize hardware saturation and achieve maximum throughput for concurrent embedding requests."),
        ("Hybrid Search Retrieval", "Combining dense vector representations with sparse BM25 keyword matching provides optimal precision and recall for enterprise retrieval-augmented generation."),
        ("PowerVM Virtualization", "IBM PowerVM enables dynamic resource allocation and partition mobility, allowing AI workloads to scale across virtualized Linux ppc64le environments."),
        ("Enterprise Vector Storage", "Storing embeddings alongside relational metadata allows filtered vector similarity search across secure enterprise domains.")
    ]
    
    docs = []
    for i in range(num_docs):
        topic_title, base_text = topics[i % len(topics)]
        doc = {
            "id": i + 1,
            "filename": "synthetic_corpus.txt",
            "page": 1,
            "chunk_id": i + 1,
            "title": f"Doc {i+1}: {topic_title} (Part {(i // len(topics)) + 1})",
            "content": f"{base_text} Document sample ID #{i+1} containing technical descriptions and benchmark payload information for evaluation."
        }
        docs.append(doc)
    return docs

# Main Interface Tabs (Ingestion First)
tab_ingest, tab_search, tab_benchmark, tab_admin = st.tabs([
    "📥 Ingest PDFs & Documents",
    "🔍 Ask Questions & Retrieve Chunks",
    "🚀 Benchmark & Performance", 
    "⚙️ Index Admin"
])

# --- INGESTION TAB ---
with tab_ingest:
    st.subheader("📥 Document Ingestion & Chunking")

    local_pdfs = sorted(glob.glob("Files/*.pdf") + glob.glob("files/*.pdf"))
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=600, step=50)
    with col_c2:
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=100, step=25)
    with col_c3:
        embed_batch_size = st.selectbox("Embedding Batch Size", [8, 16, 32, 64], index=1)

    st.markdown("---")
    upload_mode = st.radio("Choose Ingestion Source:", ["📚 Ingest Local Folder (`./Files`)", "📤 Upload Custom PDF(s)", "📝 Custom Text"], horizontal=True)

    if upload_mode == "📚 Ingest Local Folder (`./Files`)":
        st.write(f"The lab detected **{len(local_pdfs)}** PDF(s) in `./Files`:")
        for pdf_path in local_pdfs:
            st.markdown(f"- 📄 **`{pdf_path}`**")

        if local_pdfs:
            if st.button("🚀 Ingest All PDFs from `./Files`", type="primary"):
                if create_index():
                    prog_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        start_time = time.perf_counter()
                        status_text.text("Extracting and chunking text from local PDFs...")
                        all_chunks = load_all_local_pdf_chunks(chunk_size, chunk_overlap)
                        st.write(f"📊 Extracted **{len(all_chunks)}** chunks across local PDFs.")
                        
                        texts = [c["content"] for c in all_chunks]
                        all_vectors = []

                        status_text.text(f"Generating embeddings via '{model_name}' on {primary_url}...")
                        for i in range(0, len(texts), embed_batch_size):
                            batch = texts[i : i + embed_batch_size]
                            vectors = get_batch_embeddings(primary_client, batch)
                            all_vectors.extend(vectors)
                            prog_bar.progress(min(1.0, (i + embed_batch_size) / len(texts)))

                        status_text.text("Bulk indexing vectors into OpenSearch...")
                        actions = [
                            {
                                "_index": index_name,
                                "_source": {
                                    "filename": all_chunks[i]["filename"],
                                    "page": all_chunks[i]["page"],
                                    "chunk_id": all_chunks[i]["chunk_id"],
                                    "title": all_chunks[i]["title"],
                                    "content": all_chunks[i]["content"],
                                    "embedding": all_vectors[i]
                                }
                            }
                            for i in range(len(all_chunks))
                        ]
                        
                        success_count, _ = helpers.bulk(os_client, actions)
                        total_time = time.perf_counter() - start_time
                        prog_bar.progress(1.0)
                        status_text.empty()
                        st.success(f"🎉 Successfully ingested **{success_count}** chunks from `./Files` into '{index_name}' in **{total_time:.2f}s** ({(success_count/total_time):.1f} chunks/sec)!")
                        
                        with st.expander("Preview Ingested Chunks"):
                            st.dataframe(pd.DataFrame(all_chunks)[["filename", "page", "chunk_id", "content"]].head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
        else:
            st.warning("No PDF files found in `./Files`.")

    elif upload_mode == "📤 Upload Custom PDF(s)":
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            all_chunks = []
            for file in uploaded_files:
                file_chunks = extract_chunks_from_pdf(file, chunk_size, chunk_overlap)
                all_chunks.extend(file_chunks)

            st.write(f"📊 Extracted **{len(all_chunks)}** chunks across **{len(uploaded_files)}** file(s).")
            with st.expander("Preview Extracted Chunks"):
                st.dataframe(pd.DataFrame(all_chunks)[["filename", "page", "chunk_id", "content"]].head(15), use_container_width=True)

            if st.button("🚀 Embed & Ingest Uploaded PDF Chunks", type="primary"):
                if create_index():
                    prog_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        start_time = time.perf_counter()
                        texts = [c["content"] for c in all_chunks]
                        all_vectors = []

                        status_text.text(f"Generating embeddings via '{model_name}' on {primary_url}...")
                        for i in range(0, len(texts), embed_batch_size):
                            batch = texts[i : i + embed_batch_size]
                            vectors = get_batch_embeddings(primary_client, batch)
                            all_vectors.extend(vectors)
                            prog_bar.progress(min(1.0, (i + embed_batch_size) / len(texts)))

                        status_text.text("Bulk indexing vectors into OpenSearch...")
                        actions = [
                            {
                                "_index": index_name,
                                "_source": {
                                    "filename": all_chunks[i]["filename"],
                                    "page": all_chunks[i]["page"],
                                    "chunk_id": all_chunks[i]["chunk_id"],
                                    "title": all_chunks[i]["title"],
                                    "content": all_chunks[i]["content"],
                                    "embedding": all_vectors[i]
                                }
                            }
                            for i in range(len(all_chunks))
                        ]
                        
                        success_count, _ = helpers.bulk(os_client, actions)
                        total_time = time.perf_counter() - start_time
                        prog_bar.progress(1.0)
                        status_text.empty()
                        st.success(f"🎉 Successfully ingested **{success_count}** chunks into '{index_name}' in **{total_time:.2f}s** ({(success_count/total_time):.1f} chunks/sec)!")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    elif upload_mode == "📝 Custom Text":
        doc_title = st.text_input("Document Title")
        doc_content = st.text_area("Document Content")

        if st.button("Embed & Store Document"):
            if not doc_content.strip():
                st.warning("Please provide document content.")
            else:
                if create_index():
                    try:
                        with st.spinner("Generating embedding..."):
                            vec = get_single_embedding(primary_client, doc_content)
                            os_client.index(
                                index=index_name,
                                body={
                                    "filename": "manual_entry.txt",
                                    "page": 1,
                                    "chunk_id": 1,
                                    "title": doc_title or "Untitled",
                                    "content": doc_content,
                                    "embedding": vec
                                }
                            )
                        st.success(f"Indexed document: '{doc_title}'")
                    except Exception as e:
                        st.error(f"Error indexing document: {e}")

# --- SEARCH & CHUNK RETRIEVAL TAB ---
with tab_search:
    st.subheader("🔍 Ask Questions & Retrieve Context Chunks")
    st.write("Query your vector database in natural language. OpenSearch retrieves the most semantically relevant chunks with similarity scores.")

    if "selected_query" not in st.session_state:
        st.session_state.selected_query = "What is the IBM Spyre accelerator card designed for?"

    st.write("**Suggested Sample Questions for `./Files`:**")
    sample_col1, sample_col2 = st.columns(2)
    sample_col3, sample_col4 = st.columns(2)

    def set_query(q):
        st.session_state.selected_query = q

    sample_col1.button("💡 What is the IBM Spyre card designed for?", on_click=set_query, args=("What is the IBM Spyre accelerator card designed for and how does it accelerate AI inference?",))
    sample_col2.button("💡 How does IBM Power support hybrid cloud?", on_click=set_query, args=("How does IBM Power support hybrid cloud architectures and enterprise workloads?",))
    sample_col3.button("💡 What models run on Spyre accelerators?", on_click=set_query, args=("What AI models and LLM architectures are supported on IBM Spyre cards?",))
    sample_col4.button("💡 What is PowerVM security?", on_click=set_query, args=("What is PowerVM virtualization and how does it ensure security and isolation for cloud deployments?",))

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query_text = st.text_input("Ask a question / enter query:", value=st.session_state.selected_query, key="query_input")
    with col_k:
        top_k = st.number_input("Top K Chunks", min_value=1, max_value=20, value=4)

    search_clicked = st.button("🔎 Retrieve Chunks", type="primary")

    if search_clicked:
        if not query_text.strip():
            st.warning("Please enter a question or query.")
        else:
            try:
                with st.spinner(f"Embedding query via '{model_name}' and querying OpenSearch..."):
                    query_start = time.perf_counter()
                    query_vector = get_single_embedding(primary_client, query_text)
                    embed_time = time.perf_counter() - query_start

                    search_start = time.perf_counter()
                    search_body = {
                        "size": top_k,
                        "query": {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k
                                }
                            }
                        }
                    }

                    response = os_client.search(body=search_body, index=index_name)
                    search_time = time.perf_counter() - search_start
                    hits = response.get("hits", {}).get("hits", [])

                    st.caption(f"⏱️ Embedding latency: **{embed_time*1000:.1f} ms** | OpenSearch retrieval latency: **{search_time*1000:.1f} ms**")

                    if not hits:
                        st.info("No matching chunks found in the index. Make sure you have ingested documents in the 'Ingest PDFs & Documents' tab first.")
                    else:
                        st.subheader(f"Retrieved Top-{len(hits)} Context Chunks")
                        for i, hit in enumerate(hits, 1):
                            score = hit["_score"]
                            source = hit["_source"]
                            filename = source.get("filename", "Unknown Document")
                            page = source.get("page", 1)
                            chunk_id = source.get("chunk_id", i)
                            content = source.get("content", "")

                            with st.container():
                                col_head1, col_head2 = st.columns([3, 1])
                                with col_head1:
                                    st.markdown(f"#### #{i} 📄 `{filename}` (Page {page}, Chunk {chunk_id})")
                                with col_head2:
                                    st.metric("Similarity Score", f"{score:.4f}")

                                st.info(content)
                                st.divider()
            except Exception as e:
                st.error(f"Search failed: {e}. Check if the index exists and if OpenSearch is accessible.")

# --- BENCHMARK TAB ---
with tab_benchmark:
    st.subheader("⚡ Performance & Batching Benchmark")
    st.write(
        f"Benchmark embedding throughput (chunks/sec) and latency of **'{model_name}'** on **{primary_url}** across different batch sizes."
    )

    bench_source = st.radio("Benchmark Dataset Source:", ["📚 Real PDF Chunks from `./Files`", "🧪 Synthetic Corpus"], horizontal=True)

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        if bench_source == "🧪 Synthetic Corpus":
            num_docs = st.select_slider("Number of Documents", options=[10, 50, 100, 200, 500], value=50)
        else:
            num_docs = st.number_input("Max PDF Chunks to Benchmark", min_value=10, max_value=1000, value=50, step=10)
    with col_cfg2:
        batch_sizes_to_test = st.multiselect("Batch Sizes to Test", [1, 8, 16, 32, 64], default=[1, 16, 32])

    if st.button("▶️ Run Embedding Benchmark", type="primary"):
        if not batch_sizes_to_test:
            st.warning("Please select at least one batch size to test.")
        else:
            if bench_source == "📚 Real PDF Chunks from `./Files`" and local_pdfs:
                corpus = load_all_local_pdf_chunks(chunk_size=500, chunk_overlap=100)[:int(num_docs)]
            else:
                corpus = generate_benchmark_corpus(int(num_docs))

            texts = [doc["content"] for doc in corpus]
            total_items = len(texts)
            results = []

            for bs in batch_sizes_to_test:
                st.info(f"⏳ Running Benchmark with Batch Size = {bs} ({total_items} items)...")
                prog = st.progress(0)
                start_t = time.perf_counter()
                
                try:
                    for i in range(0, len(texts), bs):
                        batch = texts[i : i + bs]
                        get_batch_embeddings(primary_client, batch)
                        prog.progress(min(1.0, (i + bs) / len(texts)))
                    total_t = time.perf_counter() - start_t
                    tp = total_items / total_t
                    lat = (total_t / total_items) * 1000
                    results.append({
                        "Batch Size": f"Batch {bs}",
                        "Throughput (chunks/sec)": round(tp, 2),
                        "Total Time (s)": round(total_t, 2),
                        "Latency per chunk (ms)": round(lat, 2)
                    })
                except Exception as e:
                    st.error(f"Benchmark failed for Batch Size {bs}: {e}")

            if results:
                st.success("✅ Benchmark Completed!")
                df_res = pd.DataFrame(results)
                
                st.markdown("### 📊 Performance Results")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Throughput (Chunks / Second - Higher is Better)**")
                    st.bar_chart(df_res.set_index("Batch Size")["Throughput (chunks/sec)"])
                with c2:
                    st.write("**Latency per Chunk (ms - Lower is Better)**")
                    st.bar_chart(df_res.set_index("Batch Size")["Latency per chunk (ms)"])

                st.dataframe(df_res, use_container_width=True)

# --- ADMIN TAB ---
with tab_admin:
    st.subheader("OpenSearch Index Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create / Verify k-NN Index"):
            create_index()
    with col2:
        if st.button("Delete Index", type="secondary"):
            try:
                if os_client.indices.exists(index=index_name):
                    os_client.indices.delete(index=index_name)
                    st.warning(f"Deleted index '{index_name}'.")
                else:
                    st.info("Index does not exist.")
            except Exception as e:
                st.error(f"Error: {e}")
