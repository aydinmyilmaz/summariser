import os
import sys
from pathlib import Path

# Şu anki dosyanın bulunduğu dizini al
current_dir = Path(__file__).resolve().parent

# Proje kök dizinini Python yoluna ekle
root_dir = current_dir
sys.path.append(str(root_dir))

# Doğru modülden import et
from config.logger_config import setup_logging, get_logger

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from dotenv import load_dotenv
import traceback


def update_edited_review():
    """Callback to update edited review in session state"""
    st.session_state.edited_review_result = st.session_state.review_editor
    logger.debug("Updated edited review via callback")

# Set up logging
setup_logging()
logger = get_logger('app')
review_logger = get_logger('app.review')
optimization_logger = get_logger('app.optimization')

# Constants
MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4": "gpt-4"
}

st.set_page_config(
        page_title="🧑‍💼 📊 Executive Summary Optimizer",
        page_icon="🧑‍💼",
        layout="wide"
    )

use_cache = st.sidebar.toggle("Use caching", value=True)

def conditional_cache_data(func):
    return st.cache_data(func) if use_cache else func

def conditional_cache_resource(func):
    return st.cache_data(func) if use_cache else func

@conditional_cache_data
def read_file_content(file_path: str) -> str:
    """Read content from a file with error handling"""
    logger.debug(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            logger.info(f"Successfully read file: {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        st.error(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error reading file {file_path}: {str(e)}")
        return ""

@conditional_cache_data
def load_initial_prompts() -> Dict[str, str]:
    """Load all prompt files and return their contents"""
    logger.info("Loading initial prompts")
    base_dir = os.path.dirname(os.path.abspath('__file__'))

    prompts = {
        'review_prompt': read_file_content(os.path.join(base_dir, 'prompts', 'review_prompt.txt')),
        'optimization_prompt': read_file_content(os.path.join(base_dir, 'prompts', 'optimization_prompt.txt')),
        'general_guideline': read_file_content(os.path.join(base_dir, 'guidelines', '0_general_guideline.txt')),
        'description_guideline': read_file_content(os.path.join(base_dir, 'guidelines', '1_description_guideline.txt')),
        'objective_guideline': read_file_content(os.path.join(base_dir, 'guidelines', '2_objective_guideline.txt'))
    }

    # Print the keys that were successfully loaded for debugging
    logger.debug(f"Loaded prompt keys: {list(prompts.keys())}")

    failed_loads = [key for key, value in prompts.items() if not value]
    if failed_loads:
        logger.warning(f"Failed to load prompts: {', '.join(failed_loads)}")
        st.warning(f"Failed to load the following prompts: {', '.join(failed_loads)}")

    logger.info("Completed loading initial prompts")
    return prompts

def initialize_session_state(initial_prompts: Dict[str, str]):
    """Initialize session state variables with loaded prompts"""
    logger.debug("Initializing session state")

    if 'prompts' not in st.session_state:
        st.session_state.prompts = initial_prompts
        logger.debug("Initialized prompts in session state")

    if 'edit_modes' not in st.session_state:
        st.session_state.edit_modes = {key: False for key in initial_prompts.keys()}
        logger.debug("Initialized edit modes in session state")

    if 'review_result' not in st.session_state:
        st.session_state.review_result = None
        logger.debug("Initialized review result in session state")

    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
        logger.debug("Initialized optimization result in session state")

    if 'edited_review_result' not in st.session_state:
        st.session_state.edited_review_result = None
        logger.debug("Initialized edited review result in session state")

    if 'is_review_approved' not in st.session_state:
        st.session_state.is_review_approved = False
        logger.debug("Initialized review approval state in session state")

    logger.info("Session state initialization complete")


def create_sidebar(api_key):
    """Create sidebar with settings"""
    logger.debug("Creating sidebar")
    with st.sidebar:
        st.header("⚙️ Settings")

        # Add new toggle for showing/hiding advanced settings and store in session state
        st.session_state.show_advanced = st.toggle("Show Advanced Settings",
                                                 value=st.session_state.get('show_advanced', False),
                                                 key='advanced_settings_toggle')

        # Show API key status
        if api_key:
            st.success("API Key loaded from .env file")
            logger.info("API key loaded from environment")
        else:
            st.error("API Key not found in .env file")
            logger.warning("API key not found in environment")
            api_key = st.text_input("Enter OpenAI API Key manually:", type="password")

        st.subheader("Model Configuration")
        model = st.selectbox("Model", list(MODELS.keys()))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        logger.debug(f"Selected model: {model}, temperature: {temperature}")

        st.divider()
        st.markdown("### About")
        st.markdown("""
        This app performs Executive Summary review and optimization using LLM models.

        Current Models Available:
        - gpt-4o
        - gpt-4o-mini
        """)

        return model, temperature, api_key

def create_prompt_editor(prompt_key: str, prompt_title: str) -> None:
    """Create a prompt editor section"""
    logger.debug(f"Creating prompt editor for {prompt_key}")
    edit_mode = st.checkbox(f"Edit {prompt_title}", key=f"edit_{prompt_key}")
    st.session_state.edit_modes[prompt_key] = edit_mode

    if edit_mode:
        logger.debug(f"Editing prompt: {prompt_key}")
        st.session_state.prompts[prompt_key] = st.text_area(
            f"Edit {prompt_title}",
            value=st.session_state.prompts[prompt_key],
            height=400,
            key=f"edit_area_{prompt_key}"
        )
    else:
        st.text_area(
            f"Current {prompt_title}",
            value=st.session_state.prompts[prompt_key],
            height=400,
            key=f"view_area_{prompt_key}",
            disabled=True
        )
    if st.checkbox("Show Prompt", key=prompt_key):
        st.markdown(st.session_state.prompts[prompt_key])

@conditional_cache_resource
def run_review_chain(
    _llm: ChatOpenAI,
    executive_summary_section: str,
    prompts: Dict[str, str]
) -> str:
    """Run the review chain using LangChain"""
    review_logger.info("Starting review chain")
    review_logger.debug(f"Input text length: {len(executive_summary_section)}")

    try:
        review_prompt_template = PromptTemplate(
            input_variables=[
                "review_prompt",
                "general_guideline",
                "description_guideline",
                "executive_summary_section"
            ],
            template="""
{review_prompt}

Guidelines:
-----------
{general_guideline}

Section-Specific Guidelines:
---------------------------
{description_guideline}

Execution Summary to review:
-----------
{executive_summary_section}
"""
        )

        chain = review_prompt_template | _llm | StrOutputParser()
        review_logger.info("Review chain created, executing...")

        result = chain.invoke({
            "review_prompt": prompts["review_prompt"],
            "general_guideline": prompts["general_guideline"],
            "description_guideline": prompts["description_guideline"],
            "executive_summary_section": executive_summary_section
        })

        review_logger.info("Review completed successfully")
        review_logger.debug(f"Review result length: {len(result)}")
        return result

    except Exception as e:
        review_logger.error(f"Error in review chain: {str(e)}\n{traceback.format_exc()}")
        raise

@conditional_cache_resource
def run_optimization_chain(
    _llm: ChatOpenAI,
    review_result: str,
    executive_summary_section: str,
    prompts: Dict[str, str]
) -> str:
    """Run the optimization chain using LangChain"""
    optimization_logger.info("Starting optimization chain")
    optimization_logger.debug(f"Review result length: {len(review_result)}")

    try:
        optimization_prompt_template = PromptTemplate(
            input_variables=[
                "optimization_prompt",
                "review_result",
                "general_guideline",
                "executive_summary_section",
            ],
            template="""
## Optimization Instructions:
---------------------------
{optimization_prompt}

## Review Report:
---------------------------
{review_result}

## General Guidelines:
-----------
{general_guideline}

## Executive Summary to Make optimizations:
-----------
{executive_summary_section}

## Optimized Executive Summary:
-----------
"""
        )

        chain = optimization_prompt_template | _llm | StrOutputParser()
        optimization_logger.debug("optimization chain created, executing...")

        result = chain.invoke({
            "optimization_prompt": prompts["optimization_prompt"],
            "review_result": review_result,
            "general_guideline": prompts["general_guideline"],
            "executive_summary_section": executive_summary_section,
        })

        optimization_logger.info("optimization completed successfully")
        optimization_logger.debug(f"optimization result length: {len(result)}")
        return result

    except Exception as e:
        optimization_logger.error(f"Error in optimization chain: {str(e)}\n{traceback.format_exc()}")
        raise

def main():
    logger.info("Application started")

    # Clear session state if needed
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session state cleared!")

    st.title("🧑‍💼 📊 Executive Summary Optimizer")
    logger.info("Application started")

    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        logger.info("Environment variables loaded")

        # Load initial prompts
        initial_prompts = load_initial_prompts()

        # Initialize session state
        initialize_session_state(initial_prompts)
        logger.debug("Session state initialized")

        # Create sidebar and get settings
        model, temperature, api_key = create_sidebar(api_key)
        logger.debug(f"Sidebar created with model: {model}")

        # Create main tabs - modified to be conditional
        if st.session_state.show_advanced:
            prompt_tab, guidelines_tab, review_tab = st.tabs([
                "Prompt Management",
                "Guidelines",
                "Review & Optimization"
            ])

            # Prompt Management Tab
            with prompt_tab:
                with st.expander("Review Prompt 📝"):
                    create_prompt_editor("review_prompt", "Review Prompt")

                with st.expander("Optimization Prompt 📝"):
                    create_prompt_editor("optimization_prompt", "Optimization Prompt")

            # Guidelines Tab
            with guidelines_tab:
                guideline_tabs = st.tabs(["General", "Description", "Objective"])

                with guideline_tabs[0]:
                    create_prompt_editor("general_guideline", "General Guidelines")

                with guideline_tabs[1]:
                    create_prompt_editor("description_guideline", "Description Guidelines")

                with guideline_tabs[2]:
                    create_prompt_editor("objective_guideline", "Objective Guidelines")
        else:
            review_tab = st.tabs(["Review & Optimization"])[0]

        # Review & optimization Tab
        with review_tab:
            # Add section selector
            selected_section = st.radio(
                "Select Section to review",
                ["Description", "Objective"],
                horizontal=True,
                key="section_selector"
            )

            # Show relevant guidelines for selected section
            with st.expander(f"📖  Current {selected_section} Guidelines"):
                # Always show general guidelines
                # st.markdown("### General Guidelines")
                st.markdown(st.session_state.prompts["general_guideline"])

                # Show section specific guidelines
                # st.markdown(f"### {selected_section} Guidelines")
                if selected_section == "Description":
                    st.markdown(st.session_state.prompts["description_guideline"])
                else:  # Objective
                    st.markdown(st.session_state.prompts["objective_guideline"])

            # Review Expander
            with st.expander("🔍 Review", expanded=True):
                # Create tabs for different input methods
                input_method = st.radio(
                    "Choose input method",
                    ["📂 Sample Files", "✏️ Paste Text"],
                    horizontal=True
                )

                if input_method == "📂 Sample Files":
                    # Create sample files directory if it doesn't exist
                    sample_dir = "sample_texts"
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                        # Create sample files if they don't exist
                        for prefix in ["description_", "objective_"]:
                            sample_path = os.path.join(sample_dir, f"{prefix}sample.txt")
                            if not os.path.exists(sample_path):
                                with open(sample_path, "w") as f:
                                    f.write(f"Sample {prefix.replace('_', ' ')}text")

                    # Filter files based on selected section
                    prefix_filter = "description_" if selected_section == "Description" else "objective_"
                    sample_files = [f for f in os.listdir(sample_dir)
                                  if f.endswith('.txt') and f.startswith(prefix_filter)]

                    # Sort files numerically based on the number in their name
                    def get_file_number(filename):
                        try:
                            return int(filename.split('_')[1].split('.')[0])
                        except (IndexError, ValueError):
                            return float('inf')

                    sample_files = sorted(sample_files, key=get_file_number)

                    if sample_files:
                        selected_file = st.selectbox(
                            f"Select a {selected_section.lower()} file",
                            sample_files
                        )
                        with open(os.path.join(sample_dir, selected_file), 'r') as f:
                            st.session_state.input_text = f.read()

                        # Text area for editing file content
                        edited_content = st.text_area(
                            "📝 Edit File Content",
                            value=st.session_state.input_text,
                            height=400,
                            key="file_content_editor"
                        )

                        # Button to save changes
                        if st.button("💾 Save Changes"):
                            with open(os.path.join(sample_dir, selected_file), 'w') as f:
                                f.write(edited_content)
                            st.success(f"✅ Changes saved to {selected_file}")

                    else:
                        st.warning(f"⚠️ No {selected_section.lower()} sample files found in the samples directory")
                        st.session_state.input_text = ""

                else:
                    # Direct text input
                    st.session_state.input_text = st.text_area(
                        f"✏️ Input {selected_section} Text",
                        height=400,
                        key="input_text_direct"
                    )

                run_review = st.button("🚀 Run Review", type="primary")

                if run_review:
                    if not api_key:
                        logger.error(f"API key missing for {selected_section} review")
                        st.error("Please enter your OpenAI API key")
                    elif not st.session_state.input_text.strip():
                        logger.warning(f"Empty input text for {selected_section} review")
                        st.error("Please enter some text to review")
                    else:
                        try:
                            with st.spinner(f"Running {selected_section} review..."):
                                logger.info(f"Initializing ChatOpenAI for {selected_section} review")
                                llm = ChatOpenAI(
                                    api_key=api_key,
                                    model=MODELS[model],
                                    temperature=temperature
                                )

                                # Prepare prompts based on selected section
                                section_prompts = st.session_state.prompts.copy()
                                if selected_section == "Description":
                                    section_guideline = section_prompts["description_guideline"]
                                else:  # Objective
                                    section_guideline = section_prompts["objective_guideline"]

                                review_result = run_review_chain(
                                    _llm=llm,
                                    executive_summary_section=st.session_state.input_text,
                                    prompts={
                                        "review_prompt": section_prompts["review_prompt"],
                                        "general_guideline": section_prompts["general_guideline"],
                                        "description_guideline": section_guideline
                                    }
                                )

                                st.session_state.review_result = review_result
                                st.session_state.edited_review_result = review_result  # Initialize edited version
                                st.session_state.is_review_approved = False  # Reset approval status
                                logger.info(f"{selected_section} review completed and stored in session state")
                                st.success(f"{selected_section} review complete!")

                        except Exception as e:
                            logger.error(f"{selected_section} review error: {str(e)}\n{traceback.format_exc()}")
                            st.error(f"An error occurred: {str(e)}")

                # Display review results if they exist
                if st.session_state.review_result and st.checkbox('🖥️ Show Review Results'):
                    st.subheader("Review Results")
                    st.markdown(st.session_state.review_result)
                    st.download_button(
                        label=f"📥 Download {selected_section} Review Results",
                        data=st.session_state.review_result,
                        file_name=f"{selected_section.lower()}_review_results.txt",
                        mime="text/plain"
                    )
                    logger.debug(f"{selected_section} review results displayed and download button created")

        # optimization Expander

            from redlines import Redlines

            with st.expander("⚡ Summary Optimization", expanded=st.session_state.review_result is not None):
                if st.session_state.review_result is None:
                    st.info("Please complete the review first before proceeding to optimization.")
                else:
                    # Initialize text area with the original review result if edited version is None
                    initial_value = (st.session_state.edited_review_result
                                if st.session_state.edited_review_result is not None
                                else st.session_state.review_result)


                    edited_review = st.text_area(
                        "📝 Edit Review Results if needed",
                        value=initial_value,
                        height=400,
                        key="review_editor",
                        on_change=update_edited_review
                    )

                    # Approve button under the text area
                    if st.button("✅ Approve Review", type="primary", key="approve_review"):
                        st.session_state.edited_review_result = edited_review
                        st.session_state.is_review_approved = True
                        logger.info("Review results approved by user")
                        st.success("Review approved!")

                    # Show approval status
                    if st.session_state.is_review_approved:
                        st.info("✓ Review has been approved")
                    else:
                        st.warning("Please review and approve the review before running optimization")

                    run_optimization = st.button(
                        "🚀 Run optimization",
                        type="primary",
                        disabled=not st.session_state.is_review_approved
                    )

                    if run_optimization and st.session_state.is_review_approved:
                        try:
                            with st.spinner(f"Running {selected_section} optimization..."):
                                logger.info(f"Initializing ChatOpenAI for {selected_section} optimization")
                                llm = ChatOpenAI(
                                    api_key=api_key,
                                    model=MODELS[model],
                                    temperature=temperature
                                )

                                optimization_result = run_optimization_chain(
                                    _llm=llm,
                                    review_result=st.session_state.edited_review_result,  # Use approved version
                                    executive_summary_section=st.session_state.input_text,
                                    prompts=st.session_state.prompts
                                )

                                st.session_state.optimization_result = optimization_result
                                logger.info(f"{selected_section} optimization completed and stored in session state")
                                st.success(f"{selected_section} optimization complete!")

                                st.subheader("Optimized Summary Section")

                                from bs4 import BeautifulSoup

                                def clean_html(html_content: str) -> str:
                                    """Fix and sanitize HTML content for rendering in Streamlit."""
                                    try:
                                        # Use BeautifulSoup to fix and clean up the HTML
                                        soup = BeautifulSoup(html_content, "html.parser")
                                        cleaned_html = soup.prettify()
                                        return cleaned_html
                                    except Exception as e:
                                        logger.error(f"Error cleaning HTML content: {str(e)}")
                                        return html_content  # Return original if cleaning fails

                                # Generate diff using Redlines
                                diff = Redlines(st.session_state.input_text, optimization_result)
                                diff_html = diff.output_markdown  # Redlines generates HTML-style Markdown

                                # Clean and fix the HTML output
                                cleaned_diff_html = clean_html(diff_html)

                                # Display cleaned HTML in Streamlit
                                st.markdown(cleaned_diff_html, unsafe_allow_html=True)

                                st.download_button(
                                    label=f"📥 Download {selected_section} optimization Results",
                                    data=optimization_result,
                                    file_name=f"{selected_section.lower()}_optimization_results.txt",
                                    mime="text/plain"
                                )
                                logger.debug(f"{selected_section} optimization results displayed and download button created")

                        except Exception as e:
                            logger.error(f"{selected_section} optimization error: {str(e)}\n{traceback.format_exc()}")
                            st.error(f"An error occurred: {str(e)}")


    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}\n{traceback.format_exc()}")
        st.error("A critical error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()