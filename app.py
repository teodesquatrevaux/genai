import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchResults
from langchain_openai import ChatOpenAI

# --- Configuration des Clés API ---
# Utilisez les secrets Streamlit pour gérer vos clés API en toute sécurité.
# Vous devez configurer ces secrets dans les paramètres de votre application Streamlit.
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    api_keys_configured = True
except KeyError:
    api_keys_configured = False

# --- Configuration de l'outil de recherche ---
# (Nous le définissons ici pour vérifier la clé TAVILY)
try:
    web_search_tool = TavilySearchResults(k=3)
    tavily_ready = True
except Exception as e:
    tavily_ready = False
    tavily_error = str(e)


# --- Interface Streamlit ---
st.set_page_config(page_title="Générateur de Script Vidéo", layout="wide")
st.title("🚀 Générateur de Script Vidéo avec CrewAI")
st.markdown("Entrez un sujet et la durée souhaitée pour générer un script vidéo complet et sourcé.")

# --- Vérification des Clés API ---
if not api_keys_configured:
    st.error("⚠️ Clés API (OPENAI_API_KEY, TAVILY_API_KEY) non trouvées.")
    st.info("Veuillez configurer vos 'secrets' Streamlit pour utiliser cette application.")
    st.stop()

if not tavily_ready:
    st.error(f"⚠️ Erreur lors de l'initialisation de l'outil Tavily : {tavily_error}")
    st.stop()

# --- Widgets d'entrée ---
with st.container(border=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Quel est le sujet de votre vidéo ?", 
                            placeholder="Ex: Le télétravail (post-covid) en 2025")
    with col2:
        duration = st.slider("Durée souhaitée (en minutes) :", 
                             min_value=1, max_value=15, value=5, step=1)

# --- Bouton de Lancement ---
if st.button("Générer le Script", type="primary", use_container_width=True, disabled=(not topic)):
    
    st.markdown(f"### 📝 Script pour : {topic} ({duration} min)")
    
    with st.spinner("Génération en cours... (Cela peut prendre quelques minutes)"):
        try:
            # --- Initialisation du LLM ---
            # Nous le faisons ici pour utiliser les secrets chargés
            llm = ChatOpenAI(model="gpt-4o")

            # --- Définition des Agents ---
            # (Identique à votre notebook)
            trend_analyst = Agent(
                role="Analyste de Tendances Vidéo",
                goal="Identifier les 3 angles et sous-sujets les plus populaires et les questions que se posent les gens sur le sujet : {topic}",
                backstory="Vous êtes un expert en stratégie de contenu YouTube. Vous savez détecter ce qui captive le public et génère de l'engagement.",
                tools=[web_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            research_agent = Agent(
                role="Chercheur Web Senior",
                goal="Pour chaque angle identifié, trouver 2-3 faits marquants, statistiques, ou exemples concrets. **Chaque fait doit être accompagné de son URL source**.",
                backstory="Vous êtes un 'fact-checker' méticuleux. Votre mission est de fournir des informations vérifiables et sourcées pour enrichir le script.",
                tools=[web_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            script_writer = Agent(
                role="Rédacteur de Scripts Vidéo",
                goal=f"Rédiger un script vidéo captivant et bien structuré basé sur les angles et les faits fournis, **en visant une durée approximative de {duration} minutes**.",
                backstory="Vous êtes un scénariste créatif avec une expertise pour transformer des faits bruts en narrations engageantes. Le script doit être prêt à être enregistré.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # --- Définition des Tâches ---
            # (Nous intégrons le sujet et la durée)
            task_trends = Task(
                description=f"Analyser les tendances actuelles pour le sujet : '{topic}'. Identifier les 3 angles les plus pertinents et les questions que le public se pose.",
                expected_output="Un rapport listant 3 angles/sous-sujets populaires et les questions associées.",
                agent=trend_analyst
            )

            task_research = Task(
                description="Pour chaque angle identifié, trouver 2-3 faits marquants, statistiques, ou exemples avec leurs URL sources.",
                expected_output="Un rapport structuré avec des faits et leurs URL sources pour chaque angle.",
                agent=research_agent,
                context=[task_trends] # Dépendance
            )

            # TÂCHE MODIFIÉE : Ajout de la durée
            task_script = Task(
                description=f"Rédiger le plan détaillé du script vidéo en utilisant les angles et les faits sourcés. **Le script doit être calibré pour une vidéo d'environ {duration} minutes**.",
                expected_output=f"Un script vidéo complet en Markdown, incluant une intro, plusieurs parties (une par angle) et une conclusion. **Adapté pour une durée de {duration} minutes**. Les citations sources [Source](URL) doivent être incluses.",
                agent=script_writer,
                context=[task_research] # Dépendance
            )

            # --- Création et Exécution de la Crew ---
            crew = Crew(
                agents=[trend_analyst, research_agent, script_writer],
                tasks=[task_trends, task_research, task_script],
                process=Process.sequential,
                verbose=2
            )

            # Préparation des inputs
            inputs = {'topic': topic}

            # Lancement de la Crew
            result = crew.kickoff(inputs=inputs)
            
            # Affichage du résultat
            st.success("Script généré avec succès !")
            st.markdown(result)

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération : {e}")
            st.exception(e)

else:
    if not topic:
        st.info("Veuillez entrer un sujet pour commencer.")