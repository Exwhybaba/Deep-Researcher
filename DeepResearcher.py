import streamlit as st
import asyncio
import aiohttp
import json
from ignoreenv import OPE, SER, JIN

# =======================
# Configuration Constants
# =======================



OPENROUTER_API_KEY= OPE
SERPAPI_API_KEY = SER
JINA_API_KEY = JIN


# Endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"

# Default LLM model (can be changed if desired)
DEFAULT_MODEL = "anthropic/claude-3.5-haiku"

# ============================
# Asynchronous Helper Functions
# ============================
async def call_openrouter_async(session, messages, model=DEFAULT_MODEL):
    """Asynchronously call the OpenRouter chat completion API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "OpenDeepResearcher, by Matt Shumer",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError) as e:
                    print("Unexpected OpenRouter response structure:", result)
                    return None
            else:
                text = await resp.text()
                print(f"OpenRouter API error: {resp.status} - {text}")
                return None
    except Exception as e:
        print("Error calling OpenRouter:", e)
        return None


async def generate_search_queries_async(session, user_query):
    """Generate search queries using the LLM."""
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        try:
            search_queries = eval(response)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print("LLM did not return a list. Response:", response)
                return []
        except Exception as e:
            print("Error parsing search queries:", e, "\nResponse:", response)
            return []
    return []


async def perform_search_async(session, query):
    """Perform a search using SERPAPI."""
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"
    }
    try:
        async with session.get(SERPAPI_URL, params=params) as resp:
            if resp.status == 200:
                results = await resp.json()
                if "organic_results" in results:
                    links = [item.get("link") for item in results["organic_results"] if "link" in item]
                    return links
                else:
                    print("No organic results in SERPAPI response.")
                    return []
            else:
                text = await resp.text()
                print(f"SERPAPI error: {resp.status} - {text}")
                return []
    except Exception as e:
        print("Error performing SERPAPI search:", e)
        return []


async def fetch_webpage_text_async(session, url):
    """Fetch webpage text using Jina."""
    full_url = f"{JINA_BASE_URL}{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    try:
        async with session.get(full_url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                text = await resp.text()
                print(f"Jina fetch error for {url}: {resp.status} - {text}")
                return ""
    except Exception as e:
        print("Error fetching webpage text with Jina:", e)
        return ""


async def is_page_useful_async(session, user_query, page_text):
    """Determine if a webpage is useful for the query."""
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"


async def extract_relevant_context_async(session, user_query, search_query, page_text):
    """Extract relevant context from a webpage."""
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        return response.strip()
    return ""


async def get_new_search_queries_async(session, user_query, previous_search_queries, all_contexts):
    """Determine if additional search queries are needed."""
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly <done>."
        "\nOutput only a Python list or the token <done> without any additional text."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        cleaned = response.strip()
        if cleaned == "<done>":
            return "<done>"
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
            else:
                print("LLM did not return a list for new search queries. Response:", response)
                return []
        except Exception as e:
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []


async def generate_final_report_async(session, user_query, all_contexts):
    """Generate the final report."""
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary."
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    report = await call_openrouter_async(session, messages)
    return report


async def process_link(session, link, user_query, search_query):
    """Process a single link: fetch, judge, and extract context."""
    print(f"Fetching content from: {link}")
    page_text = await fetch_webpage_text_async(session, link)
    if not page_text:
        return None
    usefulness = await is_page_useful_async(session, user_query, page_text)
    print(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = await extract_relevant_context_async(session, user_query, search_query, page_text)
        if context:
            print(f"Extracted context from {link} (first 200 chars): {context[:200]}")
            return context
    return None


# =========================
# Main Asynchronous Routine
# =========================
async def async_main(user_query, iteration_limit):
    """Main asynchronous routine."""
    aggregated_contexts = []  # All useful contexts from every iteration
    all_search_queries = []  # Every search query used across iterations
    iteration = 0

    async with aiohttp.ClientSession() as session:
        # ----- INITIAL SEARCH QUERIES -----
        new_search_queries = await generate_search_queries_async(session, user_query)
        if not new_search_queries:
            st.error("No search queries were generated by the LLM. Exiting.")
            return
        all_search_queries.extend(new_search_queries)

        # ----- ITERATIVE RESEARCH LOOP -----
        while iteration < iteration_limit:
            st.write(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []

            # Perform SERPAPI searches concurrently.
            search_tasks = [perform_search_async(session, query) for query in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)

            # Aggregate all unique links from all search queries of this iteration.
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            st.write(f"Aggregated {len(unique_links)} unique links from this iteration.")

            # Process each link concurrently: fetch, judge, and extract context.
            link_tasks = [
                process_link(session, link, user_query, unique_links[link])
                for link in unique_links
            ]
            link_results = await asyncio.gather(*link_tasks)

            # Collect non-None contexts.
            for res in link_results:
                if res:
                    iteration_contexts.append(res)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                st.warning("No useful contexts were found in this iteration.")

            # ----- ASK THE LLM IF MORE SEARCHES ARE NEEDED -----
            new_search_queries = await get_new_search_queries_async(session, user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                st.success("LLM indicated that no further research is needed.")
                break
            elif new_search_queries:
                st.write("LLM provided new search queries:", new_search_queries)
                all_search_queries.extend(new_search_queries)
            else:
                st.warning("LLM did not provide any new search queries. Ending the loop.")
                break

            iteration += 1

        # ----- FINAL REPORT -----
        st.write("\nGenerating final report...")
        final_report = await generate_final_report_async(session, user_query, aggregated_contexts)
        st.write("\n==== FINAL REPORT ====\n")
        st.write(final_report)


# ======================
# Streamlit Application
# ======================
def main():
    """Streamlit application."""
    st.set_page_config(page_title="Deep Researcher", page_icon="🔍", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border-radius: 8px;
                border: none;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            .stTextInput input {
                font-size: 16px;
                padding: 10px;
            }
            .stMarkdown {
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Authentication
    st.sidebar.title("Authentication")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if username == "exwhybaba55555" and password == "exwhybaba55555":
        st.sidebar.success("Logged in successfully!")
        st.title("🔍 Deep Researcher")
        st.markdown("Welcome to Deep Researcher! Enter your query below to get started.")

        # User input
        user_query = st.text_input("Enter your research query/topic:")
        iteration_limit = st.number_input("Enter maximum number of iterations (default 10):", min_value=1, value=10)

        if st.button("Start Research"):
            if user_query:
                with st.spinner("Processing your query..."):
                    asyncio.run(async_main(user_query, iteration_limit))
            else:
                st.error("Please enter a valid query.")
    else:
        st.sidebar.error("Invalid username or password.")


if __name__ == "__main__":
    main()