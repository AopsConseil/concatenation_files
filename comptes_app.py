import streamlit as st
import pandas as pd
from io import BytesIO
from utils import generate_json_c,generate_json_p, get_params_ccn, calcule_cot_prest
import streamlit.components.v1 as components
from collections import Counter
import itertools
import json
import altair as alt
import plotly.express as px
import numpy as np
import os

# streamlit run "C:\Users\Yacine AMMI\Yacine\Notebooks\Concat app\Outil Comptes\comptes_app.py"--server.maxUploadSize 3000

def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)
    
@st.cache_resource
def init_appearance(logo, title):
    
    # Separation
    st.divider()

    log, titl = st.columns([1,2])
    
    log.image(logo, width=200)
    
    # Titre de l'application
    titl.title(title)

    # Separation
    st.divider()
    
@st.cache_data
def load_file_preview(file, nrows=None, dtype=None, usecols=None):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols)
    elif file.name.endswith('.pickle') or file.name.endswith('.pkl'):
        return pd.read_pickle(file)
    else:
        raise ValueError("Unsupported file format")
    
@st.cache_data
def previw_uploaded_files(uploaded_file):
    if uploaded_file:
        try:
            uploaded_file.seek(0)
        except:
            pass
        
        df = load_file_preview(uploaded_file, nrows=None)
        #st.write(f"**{uploaded_file.name}**")
        #preview_file(df, nrows=50)
    return df

@st.cache_data
def preview_file(df, nrows=50, height=250, col=None):
    return df.head(nrows)
    
def change_init_state():
    st.session_state.init_valid = False
    
def init():
    st.session_state.new_columns = []
    st.session_state.assureurs = []
    st.session_state.ignore_errors = False
    st.session_state.validated = False
    st.session_state.errors = False
    st.session_state.init_valid = False
    st.session_state.params_json = None
    st.session_state.cot_df = None
    st.session_state.prest_df = None
    st.session_state.cot_json = None
    st.session_state.prest_json = None
    st.session_state.cot_net_json = None
    
def main_comptes():
    current_dir = os.path.dirname(__file__)
    
    page_ico = os.path.join(current_dir, 'resources', 'merge.png')
    logo = os.path.join(current_dir, 'resources', 'Logo_AOPS_conseil.png')
    title = 'Outil de calcul des :orange[Comptes]'
    
    st.set_page_config(layout="wide", page_title='Outil de concatenation', page_icon=page_ico)
    
    init_appearance(logo, title)

    # Initialiser les listes
    if 'cot_df' not in st.session_state:
        init()
    
    cot_col, prest_col = st.columns(2)
    
    # Chargement des fichiers
    cot_uploaded_files = cot_col.file_uploader("Choisir le fichier de cotisation", accept_multiple_files=False, type=['csv', 'pkl', 'pickle'], on_change=change_init_state)
    
    if cot_uploaded_files:
        st.session_state.cot_df = previw_uploaded_files(cot_uploaded_files)
        cot_col.dataframe(preview_file(st.session_state.cot_df))
    
    # Chargement des fichiers
    prest_uploaded_files = prest_col.file_uploader("Choisir le fichier de prestations", accept_multiple_files=False, type=['csv', 'pkl', 'pickle'], on_change=change_init_state)

    if prest_uploaded_files:
        st.session_state.prest_df = previw_uploaded_files(prest_uploaded_files)
        prest_col.dataframe(preview_file(st.session_state.prest_df))
        
    # Separation
    st.divider()
    
    if (st.session_state.cot_df is not None) and  (st.session_state.prest_df is not None):
        
        cols = st.columns([2,2,1])
            
        ccn = cols[0].selectbox(label="Selectionnez la CCN", options=['HPA', 'BAD', 'BJOH','SVP','ICR', 'INTERMITTENT'], placeholder=f'CCN', index=None, key=f"CCN")
        
        # st.session_state.cot_df['annee_comptable'] = st.session_state.cot_df['annee_comptable'].astype(int)
        # st.session_state.cot_df['annee_survenance'] = st.session_state.cot_df['annee_survenance'].astype(int)
        
        annee_compt_dispo = st.session_state.cot_df['annee_comptable'].unique()
        annee_compt_dispo[::-1].sort()
        
        annee_compt = cols[1].selectbox(label="Selectionnez l'année comptable", options=annee_compt_dispo, placeholder=f'Année comptable', index=None, key=f"annee_compt")
        cols[2].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        
        if cols[2].button('Générer json'):
            
            if (ccn is None) or (annee_compt is None):
                st.error('Veuillez sélectionner les paramètres de calcul')
                st.stop()
            
            st.subheader('Data')
            st.session_state.cot_json = generate_json_c(st.session_state.cot_df, ccn, annee_compt)
            st.session_state.prest_json = generate_json_p(st.session_state.prest_df, ccn, annee_compt)
            st.session_state.params_json = get_params_ccn(str(ccn), int(annee_compt))
            
            cols = st.columns(2)
        
            cols[0].subheader("Paramètres Cotisations")
            cols[0].json(st.session_state.cot_json,expanded=True)
            if st.session_state.params_json:
                cols[0].json(st.session_state.params_json, expanded=True)
            else:
                cols[0].write(ccn, annee_compt , st.session_state.params_json)
                
            cols[1].subheader("Paramètres Prestations")
            cols[1].json(st.session_state.prest_json,expanded=True)
            if st.session_state.params_json:
                cols[1].json(st.session_state.params_json, expanded=True)
            else:
                cols[1].write(ccn, annee_compt , st.session_state.params_json)   
                
        if (st.session_state.cot_json is not None) and (st.session_state.prest_json is not None) and (st.session_state.params_json is not None):
            st.header('Calcul cotisations nettes et prestations')
            
            if st.button('Calcul cotisations nettes et prestations'):
                st.session_state.cot_prest_json = calcule_cot_prest(st.session_state.cot_json, st.session_state.prest_json, st.session_state.params_json)

                st.subheader("Résultats")
                st.json(st.session_state.cot_prest_json, expanded=True)

                cotisations = [cot for assureur_data in st.session_state.cot_prest_json['data'] for cot in assureur_data['cotisations']]
                df_cotisations = pd.DataFrame(cotisations).sort_values(['survenance', 'régime'], axis=0)

                prestations = [prest for assureur_data in st.session_state.cot_prest_json['data'] for prest in assureur_data['prestations']]
                df_prestations = pd.DataFrame(prestations).sort_values(['survenance', 'régime'], axis=0)

                # Afficher les métriques et les tableaux pour chaque assureur
                assureurs = df_cotisations['assureur'].unique()
                for assureur in assureurs:
                    st.subheader(f"Assureur: {assureur}")

                    df_cot_assureur = df_cotisations[df_cotisations['assureur'] == assureur]
                    df_prest_assureur = df_prestations[df_prestations['assureur'] == assureur]

                    total_brut = df_cot_assureur["montant"].sum()
                    total_net = df_cot_assureur["cotisations_nettes"].sum()
                    total_prest = df_prest_assureur["montant"].sum()

                    cols = st.columns(3)
                    cols[0].metric("Cotisations brutes", f"{total_brut:,.2f}")
                    cols[1].metric("Cotisations nettes", f"{total_net:,.2f}")
                    cols[2].metric("Prestations", f"{total_prest:,.2f}")

                    st.write("Cotisations:")
                    st.dataframe(df_cot_assureur)
                    st.write("Prestations:")
                    st.dataframe(df_prest_assureur)
                
    
if __name__ == "__main__":
    main_comptes()