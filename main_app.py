import streamlit as st

# Import des fonctions des pages
import concat_app 
import comptes_app

# streamlit run "C:\Users\Yacine AMMI\Yacine\Notebooks\Concat app\Outil Comptes\main_app.py"--server.maxUploadSize 3000


# Configuration de la navigation
pg = st.navigation([
    st.Page(concat_app.main_concat, title="Concatenation", icon="🔗"),
    st.Page(comptes_app.main_comptes, title="Comptes", icon="📊"),
])

# Exécution de la page sélectionnée
pg.run()