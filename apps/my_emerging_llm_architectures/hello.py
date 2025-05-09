from preswald import text, plotly, connect, get_df, table, selectbox, image, expander, download_button, sidebar, button
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Initialize connection
connect()

# Create dataframes for technology tables
def create_tech_tables():
    # Embedding Models
    embedding_df = pd.DataFrame({
        "Name": ["OpenAI", "Cohere", "Hugging Face", "Voyage AI", "BAAI BGE", "Jina AI"],
        "Link": ["https://platform.openai.com/docs/guides/embeddings", 
                "https://cohere.com/embeddings",
                "https://huggingface.co/sentence-transformers",
                "https://voyageai.com/",
                "https://github.com/BAAI-Lab/Embeddings",
                "https://jina.ai/"]
    })
    
    # Vector Databases
    vectordb_df = pd.DataFrame({
        "Name": ["Pinecone", "Weaviate", "ChromaDB", "pgvector", "Qdrant", "Milvus"],
        "Link": ["https://www.pinecone.io/", 
                "https://weaviate.io/",
                "https://www.trychroma.com/",
                "https://github.com/pgvector/pgvector",
                "https://qdrant.tech/",
                "https://milvus.io/"]
    })
    
    # Data Pipelines
    datapipelines_df = pd.DataFrame({
        "Name": ["Databricks", "Airflow", "Unstructured", "LlamaHub", "Airbyte", "Mage"],
        "Link": ["https://databricks.com/", 
                "https://airflow.apache.org/",
                "https://unstructured.io/",
                "https://llamahub.ai/",
                "https://airbyte.com/",
                "https://www.mage.ai/"]
    })
    
    # Prompt Engineering
    prompt_df = pd.DataFrame({
        "Name": ["Humanloop", "Promptflow", "LMQL", "OpenAI Playground", "Vercel Prompt", "Dust"],
        "Link": ["https://humanloop.com/", 
                "https://microsoft.github.io/promptflow/",
                "https://lmql.ai/",
                "https://platform.openai.com/playground",
                "https://vercel.com/templates/next.js/prompt-playground",
                "https://dust.tt/"]
    })
    
    # Orchestration
    orchestration_df = pd.DataFrame({
        "Name": ["LangChain", "LlamaIndex", "Haystack", "Semantic Kernel", "DSPy", "LiteLLM"],
        "Link": ["https://python.langchain.com/", 
                "https://www.llamaindex.ai/",
                "https://haystack.deepset.ai/",
                "https://github.com/microsoft/semantic-kernel",
                "https://github.com/stanfordnlp/dspy",
                "https://github.com/BerriAI/litellm"]
    })
    
    # Agent Frameworks
    agents_df = pd.DataFrame({
        "Name": ["CrewAI", "AutoGPT", "BabyAGI", "LangGraph", "AgentGPT", "GPT-Engineer"],
        "Link": ["https://github.com/joaomdmoura/crewai", 
                "https://github.com/Significant-Gravitas/Auto-GPT",
                "https://github.com/yoheinakajima/babyagi",
                "https://github.com/langchain-ai/langgraph",
                "https://agentgpt.reworkd.ai/",
                "https://github.com/gpt-engineer-org/gpt-engineer"]
    })
    
    # More tables can be added as needed for other categories
    
    return {
        "embedding": embedding_df,
        "vectordb": vectordb_df,
        "datapipelines": datapipelines_df,
        "prompt": prompt_df,
        "orchestration": orchestration_df,
        "agents": agents_df
    }

def create_architecture_diagram(highlight_section="all"):
    """Create a Plotly-based architecture diagram with highlighting capability"""
    
    # Define colors for different states
    active_color = "rgb(59, 130, 246)"  # Blue for highlighted sections
    inactive_color = "rgb(229, 231, 235)"  # Light gray for non-highlighted
    default_color = "rgb(147, 197, 253)"  # Default blue
    
    # Determine the color for each component based on highlighting
    def get_color(section):
        if highlight_section == "all":
            return default_color
        return active_color if highlight_section == section else inactive_color
    
    # Component definitions with positions and colors
    components = [
        # Data Sources
        {"x": [0.05, 0.25], "y": [0.05, 0.15], "name": "Data Sources", "section": "data"},
        # Embedding Models
        {"x": [0.05, 0.25], "y": [0.2, 0.3], "name": "Embedding Models", "section": "embedding"},
        # Vector Databases
        {"x": [0.05, 0.25], "y": [0.35, 0.45], "name": "Vector Databases", "section": "vectordb"},
        # Prompt Engineering
        {"x": [0.3, 0.5], "y": [0.05, 0.15], "name": "Prompt Engineering", "section": "prompt"},
        # Orchestration
        {"x": [0.3, 0.5], "y": [0.2, 0.3], "name": "Orchestration", "section": "orchestration"},
        # External APIs
        {"x": [0.3, 0.5], "y": [0.35, 0.45], "name": "External APIs/Tools", "section": "orchestration"},
        # LLM APIs
        {"x": [0.55, 0.75], "y": [0.05, 0.15], "name": "LLM APIs", "section": "llmapi"},
        # Validation
        {"x": [0.55, 0.75], "y": [0.2, 0.3], "name": "Validation & Safety", "section": "validation"},
        # Monitoring
        {"x": [0.55, 0.75], "y": [0.35, 0.45], "name": "Monitoring & LLMOps", "section": "monitoring"},
        # Multimodal
        {"x": [0.05, 0.25], "y": [0.5, 0.6], "name": "Multi-Modal Integration", "section": "multimodal"},
        # Agents
        {"x": [0.3, 0.5], "y": [0.5, 0.6], "name": "Agent Frameworks", "section": "agent"},
        # Compliance
        {"x": [0.55, 0.75], "y": [0.5, 0.6], "name": "Compliance & Governance", "section": "compliance"},
        # App Hosting
        {"x": [0.8, 0.95], "y": [0.2, 0.35], "name": "App Hosting", "section": "hosting"}
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add each component as a rectangle
    for comp in components:
        fig.add_shape(
            type="rect",
            x0=comp["x"][0], y0=comp["y"][0], 
            x1=comp["x"][1], y1=comp["y"][1],
            line=dict(color="Black", width=1),
            fillcolor=get_color(comp["section"])
        )
        
        # Add component name as text
        fig.add_annotation(
            x=(comp["x"][0] + comp["x"][1])/2,
            y=(comp["y"][0] + comp["y"][1])/2,
            text=comp["name"],
            showarrow=False,
            font=dict(color="white", size=12)
        )
    
    # Add connection lines (simplified for this version)
    # In a more complete version, we would add all the connecting arrows
    
    # Figure layout
    fig.update_layout(
        title="Emerging LLM Application Stack 2025",
        showlegend=False,
        plot_bgcolor='white',
        width=900,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_adoption_heatmap():
    """Create a heatmap showing LLM architecture component adoption by industry"""
    
    # Define industries and components
    industries = ["Finance", "Healthcare", "Legal", "Education", "Manufacturing", "Retail", "Media", "Government"]
    components = ["Vector Databases", "Embedding Models", "Orchestration", 
                 "Prompt Engineering", "LLM APIs", "Agent Frameworks", 
                 "Fine-Tuning", "Monitoring", "Compliance Tools"]
    
    # Create sample data (in real application, this would be actual research data)
    np.random.seed(42)
    z_data = np.random.randint(3, 10, size=(len(industries), len(components)))
    
    # Adjust some values to show interesting patterns
    z_data[0, 8] = 9  # Finance has high compliance tool adoption
    z_data[1, 8] = 9  # Healthcare has high compliance tool adoption
    z_data[2, 8] = 9  # Legal has high compliance tool adoption
    z_data[6, 6] = 9  # Media has high fine-tuning adoption
    z_data[3, 5] = 9  # Education has high agent framework adoption
    
    # Create heatmap
    fig = px.imshow(
        z_data,
        labels=dict(x="Architecture Components", y="Industries", color="Adoption Rate"),
        x=components,
        y=industries,
        color_continuous_scale="Blues",
        title="Adoption Rates of LLM Architecture Components Across Industries in 2025"
    )
    
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        width=900,
        height=600
    )
    
    return fig

def create_component_importance_chart():
    """Create a chart showing the relative importance of architecture components"""
    
    components = ["Vector Databases", "Embedding Models", "Orchestration", 
                 "Prompt Engineering", "LLM APIs", "Agent Frameworks", 
                 "Fine-Tuning", "Monitoring", "Compliance Tools"]
    
    # Sample data for 2023 and 2025
    importance_2023 = [7, 8, 6, 8, 9, 4, 5, 5, 3]
    importance_2025 = [8, 8, 7, 7, 8, 8, 9, 8, 7]
    
    # Create dataframe
    df = pd.DataFrame({
        'Component': components + components,
        'Year': ['2023']*len(components) + ['2025']*len(components),
        'Importance': importance_2023 + importance_2025
    })
    
    # Create grouped bar chart
    fig = px.bar(
        df, 
        x='Component', 
        y='Importance', 
        color='Year',
        barmode='group',
        title='Relative Importance of Architecture Components in LLM Applications',
        color_discrete_sequence=["lightblue", "darkblue"]
    )
    
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Importance Score (0-10)"),
        legend_title="Year",
        width=900,
        height=600
    )
    
    return fig

def display_article_section(section):
    """Display a specific section of the article"""
    
    if section == "intro":
        text("# Emerging Architectures for LLM Applications 2025")
        text("*Posted May 2, 2025*")
        
        text("""
        Large language models have evolved from a promising new primitive to the foundation of a flourishing AI ecosystem since the original publication of "Emerging Architectures for LLM Applications" in 2023. In this 2025 update, we examine how LLM application architectures have matured, providing insights into the current state of the stack and emerging patterns that define modern AI applications.
        
        The past two years have witnessed significant advancements in model capabilities, development practices, and support infrastructure. This refreshed reference architecture reflects the systems, tools, and design patterns now considered standard among AI startups and technology leaders, with a particular focus on what has changed since 2023.
        
        This work draws on extensive conversations with founders, engineers, and AI researchers building production LLM systems. We're especially grateful for input from industry leaders working on the cutting edge of LLM applications.
        """)
    
    elif section == "stack":
        text("## The LLM App Stack in 2025")
        
        text("""
        Here's our current view of the LLM application stack in 2025:
        """)
        
        # Display the full architecture diagram
        fig = create_architecture_diagram("all")
        plotly.show(fig)
        
        text("""
        The most notable changes since 2023 include:
        
        1. **Agent Frameworks** have moved from experimental to production-ready
        2. **Fine-Tuning** has become more accessible and is now a standard component
        3. **Multi-Modal Integration** across text, image, audio, and video
        4. **Compliance & Governance Tools** have emerged as a critical layer
        5. **Specialized LLM Deployment Infrastructure** has matured significantly
        
        Let's explore how each area of the stack has evolved.
        """)
        
        # Display technology tables
        text("### Links to Key Projects")
        text("Here's a list of links to each project category for quick reference:")
        
        tech_tables = create_tech_tables()
        
        with column(2):
            text("#### Embedding Models")
            table(tech_tables["embedding"])
            
            text("#### Vector Databases")
            table(tech_tables["vectordb"])
            
            text("#### Data Pipelines")
            table(tech_tables["datapipelines"])
            
            text("#### Prompt Engineering")
            table(tech_tables["prompt"])
            
            text("#### Orchestration")
            table(tech_tables["orchestration"])
            
            text("#### Agent Frameworks")
            table(tech_tables["agents"])
    
    elif section == "data-preprocessing":
        text("## Data Preprocessing / Embedding")
        
        # Show diagram with data section highlighted
        fig = create_architecture_diagram("data")
        plotly.show(fig)
        
        text("### Data Pipeline Evolution")
        text("""
        Data integration capabilities have substantially matured since 2023. While traditional ETL tools like Databricks and Airflow remain popular, we've seen specialized document processing frameworks gain significant traction:
        
        - **Unstructured.io** has established itself as the de-facto standard for document processing, handling everything from PDFs to images with text extraction
        - **LlamaIndex** has evolved from a retrieval tool to a comprehensive data framework with specialized connectors for enterprise systems
        - **Multimodal Pipelines** now commonly process images, audio, and video alongside text
        
        The biggest shift has been toward streaming architectures that continuously update vector stores as new data becomes available, moving away from batch processing. Real-time data synchronization is now considered essential for production applications.
        """)
        
        # Show diagram with embedding section highlighted
        fig = create_architecture_diagram("embedding")
        plotly.show(fig)
        
        text("### Embedding Models")
        text("""
        In 2023, OpenAI's text-embedding-ada-002 dominated the embedding landscape. By 2025, the ecosystem has diversified:
        
        - **Specialized Embedding Models** are now common, with purpose-built embeddings for different data types (code, legal documents, medical text)
        - **Open-Source Leadership** has emerged with models like Voyage AI's embeddings and BAAI's BGE models matching or exceeding proprietary options
        - **Multi-Modal Embeddings** can now represent images, audio, and text in the same vector space
        - **Hybrid Retrieval Systems** combining sparse (keyword) and dense (semantic) embeddings are now standard practice
        
        Enterprise applications increasingly maintain multiple embedding spaces optimized for different retrieval tasks, supported by more sophisticated vector databases.
        """)
        
        # Show diagram with vector db section highlighted
        fig = create_architecture_diagram("vectordb")
        plotly.show(fig)
        
        text("### Vector Databases")
        text("""
        The vector database landscape has consolidated somewhat while simultaneously growing more sophisticated:
        
        - **Enterprise Features** like ACID transactions, fine-grained access controls, and automatic backups are now standard
        - **Hybrid Search** combining lexical, vector, and metadata filtering is the default approach
        - **Vector Database Extensions** for PostgreSQL, MongoDB, and other traditional databases have matured significantly
        - **Cloud-Native Options** are available from all major providers (AWS, GCP, Azure)
        - **Specialized Query Patterns** beyond simple similarity search, including dense retrieval, hierarchical navigation, and graph-based exploration
        
        Pinecone, Weaviate, and pgvector remain market leaders, though the landscape has matured with more specialized options for different deployment scenarios.
        """)
        
        # Display technology tables
        tech_tables = create_tech_tables()
        
        with column(2):
            text("#### Data Pipelines")
            table(tech_tables["datapipelines"])
            
            text("#### Embedding Models")
            table(tech_tables["embedding"])
            
            text("#### Vector Databases")
            table(tech_tables["vectordb"])
    
    elif section == "prompt-construction":
        text("## Prompt Construction / Retrieval")
        
        # Show diagram with prompt section highlighted
        fig = create_architecture_diagram("prompt")
        plotly.show(fig)
        
        text("### Prompt Engineering")
        text("""
        Prompt engineering has evolved from an art to a more structured discipline:
        
        - **Prompt Management Platforms** now provide version control, A/B testing, and performance analytics
        - **Prompt Libraries** offer reusable, tested prompt components for common tasks
        - **Template Languages** provide structure and validation for complex prompts
        - **Dynamic Prompt Construction** techniques automatically adjust prompts based on contextual factors
        
        The heuristic-based approaches of 2023 have been largely replaced by systematic, data-driven prompt optimization.
        """)
        
        # Show diagram with orchestration section highlighted
        fig = create_architecture_diagram("orchestration")
        plotly.show(fig)
        
        text("### Orchestration")
        text("""
        The orchestration layer has seen dramatic changes since 2023:
        
        - **LangChain** has matured into a comprehensive framework with enterprise features
        - **Integration Platforms** now connect LLMs to existing enterprise systems through standardized interfaces
        - **Chain-of-Thought Pipelines** breaking complex tasks into manageable steps are now standard practice
        - **Memory Systems** for maintaining context across interactions have become more sophisticated
        - **Specialized Domain Libraries** have emerged for finance, healthcare, legal, and other regulated industries
        
        Most notably, orchestration has shifted from being primarily about chaining prompts to managing complex workflows involving multiple models, tools, and external systems.
        """)
        
        # Display technology tables
        tech_tables = create_tech_tables()
        
        with column(2):
            text("#### Prompt Engineering")
            table(tech_tables["prompt"])
            
            text("#### Orchestration")
            table(tech_tables["orchestration"])
    
    elif section == "multi-modal":
        text("## Multi-Modal Systems")
        
        # Show diagram with multi-modal section highlighted
        fig = create_architecture_diagram("multimodal")
        plotly.show(fig)
        
        text("""
        Perhaps the most significant architectural shift since 2023 has been the integration of multiple modalities:
        
        - **Text-to-Image Generation** is now a standard component in many applications
        - **Image Understanding** capabilities extract structured information from visual inputs
        - **Audio and Video Processing** enable rich media analysis and generation
        - **Cross-Modal Reasoning** combines insights across different data types
        - **Unified Representation Spaces** allow seamless transitions between modalities
        
        What began as separate, specialized systems have converged into unified architectures capable of working across text, images, audio, and video simultaneously.
        """)
    
    elif section == "agents":
        text("## Agent Frameworks")
        
        # Show diagram with agent section highlighted
        fig = create_architecture_diagram("agent")
        plotly.show(fig)
        
        text("""
        The most significant change since 2023 has been the maturation of agent frameworks:
        
        - **Tool-Using Agents** can reliably execute multi-step tasks using external APIs and systems
        - **Agent Collaboration** enables multiple specialized agents to work together on complex problems
        - **Planning and Reasoning Frameworks** help decompose complex tasks into manageable steps
        - **Safety and Control Mechanisms** ensure agents operate within defined boundaries
        - **Feedback Loops** allow agents to learn from success and failure
        
        While early agent frameworks like Auto-GPT were primarily proof-of-concept demonstrations, today's production agent systems reliably solve complex tasks with minimal human intervention.
        """)
        
        tech_tables = create_tech_tables()
        
        with column(1):
            text("#### Agent Frameworks")
            table(tech_tables["agents"])
    
    elif section == "visualizations":
        text("## LLM Architecture Visualizations")
        
        text("### Industry Adoption Heatmap")
        text("This visualization shows the adoption rates of different LLM architecture components across various industries.")
        
        fig = create_adoption_heatmap()
        plotly.show(fig)
        
        text("### Component Importance Evolution")
        text("This chart shows how the relative importance of different architecture components has changed from 2023 to 2025.")
        
        fig = create_component_importance_chart()
        plotly.show(fig)
        
        # Load and display the Superstore data
        superstore_df = pd.read_csv("data/my_sample_superstore.csv")
        text("### Sample Superstore Data")
        text("Below is a sample of the Superstore data loaded from the CSV file:")
        table(superstore_df.head(10))

def main():
    """Main function to drive the Preswald app"""
sidebar()
text("# Emerging Architectures")
text("## for LLM Applications 2025")
        
text("### Navigation")
section_selection = selectbox(
            "Choose a section:",
            options=[
                "Introduction", 
                "The LLM App Stack in 2025",
                "Data Preprocessing / Embedding",
                "Prompt Construction / Retrieval",
                "Multi-Modal Systems",
                "Agent Frameworks",
                "Visualizations & Analysis"
            ],
            default="Introduction"
        )
    
    # Display content based on selection
    if section_selection == "Introduction":
        display_article_section("intro")
    elif section_selection == "The LLM App Stack in 2025":
        display_article_section("stack")
    elif section_selection == "Data Preprocessing / Embedding":
        display_article_section("data-preprocessing")
    elif section_selection == "Prompt Construction / Retrieval":
        display_article_section("prompt-construction")
    elif section_selection == "Multi-Modal Systems":
        display_article_section("multi-modal")
    elif section_selection == "Agent Frameworks":
        display_article_section("agents")
    elif section_selection == "Visualizations & Analysis":
        display_article_section("visualizations")

# Run main if script is executed directly
if __name__ == "__main__":
    main()

# Make data and functions available in REPL
__all__ = [
    'create_tech_tables', 
    'create_architecture_diagram', 
    'create_adoption_heatmap',
    'create_component_importance_chart',
    'display_article_section',
    'main'
] 