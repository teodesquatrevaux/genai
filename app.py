import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchTool
from langchain_openai import ChatOpenAI

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="🎥 Générateur de Scripts Vidéo", layout="wide")

# --- Barre latérale pour les clés API ---
st.sidebar.title("🔑 Configuration des Clés API")
st.sidebar.markdown("Veuillez entrer vos clés API pour utiliser l'application.")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("Cette application utilise un 'Crew' d'agents IA pour générer des scripts vidéo basés sur votre sujet.")

# --- Interface Principale ---
st.title("🎥 Générateur de Scripts Vidéo (CrewAI)")
st.markdown("""
Bienvenue ! Cette application utilise une équipe d'agents IA pour créer un plan de script vidéo.
Entrez un sujet, et le 'Crew' va :
1.  **Analyser** les tendances et angles populaires.
2.  **Rechercher** des faits et des sources crédibles.
3.  **Rédiger** un plan de script complet en Markdown.
""")

sujet_video = st.text_area(
    "Quel est le sujet de votre vidéo ?",
    value="L'impact du télétravail sur la productivité et le bien-être",
    height=100
)

# --- Logique d'exécution du Crew ---
if st.button("🚀 Lancer la Génération du Script"):
    
    # 1. Valider les clés API
    if not openai_api_key or not tavily_api_key:
        st.error("❌ Veuillez entrer vos clés API OpenAI et Tavily dans la barre latérale pour continuer.")
        st.stop()

    # 2. Définir les variables d'environnement pour cette exécution
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    try:
        # 3. Initialiser les outils et le LLM
        with st.spinner("🛠️ Initialisation des outils et du LLM..."):
            web_search_tool = TavilySearchTool(max_results=3)
            # Utilise le modèle gpt-4o comme dans le notebook
            llm = ChatOpenAI(model="gpt-4o")

        st.info("🤖 Création des agents du Crew...")

        # 4. Définir les Agents (copiés de votre notebook)
        # --- Agent 1: L'Analyste des Tendances ---
        trend_analyst = Agent(
            role="Analyste de Tendances Vidéo",
            goal="Identifier les 3 angles et sous-sujets les plus populaires et les questions "
                 "que se posent les gens sur le sujet : {topic}",
            backstory="Vous êtes un expert en stratégie de contenu YouTube. Vous savez "
                      "détecter ce qui captive le public et génère de l'engagement.",
            tools=[web_search_tool],
            llm=llm,
            verbose=False,  # Mettre à False pour une UI Streamlit propre
            allow_delegation=False
        )

        # --- Agent 2: Le Chercheur (RAG) ---
        research_agent = Agent(
            role="Chercheur Web Senior",
            goal="Pour chaque angle identifié, trouver 2-3 faits marquants, statistiques, ou "
                 "exemples concrets. **Chaque fait doit être accompagné de son URL source**.",
            backstory="Vous êtes un 'fact-checker' méticuleux. Votre mission est de "
                      "fournir des informations vérifiables et sourcées pour "
                      "construire la crédibilité du script.",
            tools=[web_search_tool],
            llm=llm,
            verbose=False,
            allow_delegation=False
        )

        # --- Agent 3: Le Rédacteur de Script ---
        script_writer = Agent(
            role="Rédacteur de Scripts Vidéo",
            goal="Rédiger un plan de script vidéo (format Markdown) basé sur les tendances et "
                 "les faits bruts fournis. Le script doit être structuré (Intro, "
                 "Parties, Conclusion) et **intégrer les citations**.",
            backstory="Vous êtes un scénariste de talent, capable de transformer "
                      "des informations brutes en une histoire engageante et rythmée.",
            llm=llm,
            verbose=False,
            allow_delegation=False
        )

        st.info("📋 Définition des tâches...")

        # 5. Définir les Tâches (copiées de votre notebook)
        # Tâche 1: Trouver les tendances
        task_trends = Task(
            description="Analyser les tendances actuelles et les questions populaires pour le sujet : {topic}.",
            expected_output="Une liste de 3 angles de script pertinents et les questions clés.",
            agent=trend_analyst,
            async_execution=False # Streamlit fonctionne mieux en séquentiel
        )

        # Tâche 2: Rechercher les faits
        task_research = Task(
            description="Collecter des faits, statistiques et sources pour les angles identifiés.",
            expected_output="Un rapport structuré avec des faits et leurs URL sources pour chaque angle.",
            agent=research_agent,
            context=[task_trends],
            async_execution=False
        )

        # Tâche 3: Rédiger le script
        task_script = Task(
            description="Rédiger le plan détaillé du script vidéo en utilisant les angles et les faits sourcés.",
            expected_output="Un script vidéo complet en Markdown, incluant une intro, "
                            "plusieurs parties (une par angle) et une conclusion. "
                            "Les citations sources doivent être incluses.",
            agent=script_writer,
            context=[task_research],
            async_execution=False
        )

        st.info("🚀 Assemblage du Crew et lancement de la mission...")

        # 6. Créer et Lancer le Crew
        video_crew = Crew(
            agents=[trend_analyst, research_agent, script_writer],
            tasks=[task_trends, task_research, task_script],
            process=Process.sequential,  # Processus séquentiel comme dans le notebook
            verbose=False # Mettre à 2 pour voir les logs dans le terminal
        )

        # Lancer le kickoff dans un spinner
        with st.spinner("🤖 L'équipe est au travail ! (Cela peut prendre 1 à 2 minutes)"):
            result = video_crew.kickoff(inputs={'topic': sujet_video})

        # 7. Afficher le résultat
        st.success("✅ Mission terminée ! Voici votre script.")
        st.markdown("---")
        st.subheader("Script Vidéo Généré")
        
        # Le 'result.raw' contient le Markdown final
        if result and hasattr(result, 'raw'):
            st.markdown(result.raw)
        else:
            st.write(result) # Fallback si .raw n'existe pas

    except Exception as e:
        st.error(f"❌ Une erreur est survenue pendant l'exécution du Crew : {e}")
        st.error("Veuillez vérifier vos clés API, vos crédits OpenAI et que le modèle 'gpt-4o' est disponible.")

    # 8. Nettoyer les variables d'environnement après l'exécution
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "TAVILY_API_KEY" in os.environ:
        del os.environ["TAVILY_API_KEY"]
