import streamlit as st
import pandas as pd
from io import BytesIO
from utils import concat_datasets, get_configurations, correction_dates_integrale, edit_mapping, check_mappings_complete, save_mappings, load_mappings, restore_editions, mise_en_forme_df
import streamlit.components.v1 as components
from collections import Counter
import itertools
import json
import os

MAPPINGS_FILE = r'C:\Users\Yacine AMMI\Yacine\Notebooks\Concat app\Outil Comptes\mappings.json'

# streamlit run "C:\Users\Yacine AMMI\Yacine\Notebooks\Concat app\Outil Comptes\concat_app.py"--server.maxUploadSize 3000

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

# Fonction de transformation personnalisée des colonnes de date
def custom_date_transformation(df, date_columns, raise_errors=True, fichier=None):
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])  # Conversion en format datetime avec gestion des erreurs

        except:
            try:
                df[col] = correction_dates_integrale(df, col)
                if fichier is not None:
                    st.warning(f"La colonnes {col} du fichiers {fichier} contien des dates en différents formats.")
                else:
                    st.warning(f"La colonnes {col} contien des dates en différents formats.")
                    
            except:
                if raise_errors:
                    if fichier is not None:
                        st.error(f"La transformation de la colonne {col} du fichier {fichier} en date n'a pas réussi, elle sera concaténé en tant que texte.")
                    else:
                        st.error(f"La transformation de la colonne {col} en date n'a pas réussi, elle sera concaténé en tant que texte.")
                    
    return df

def int_transform_columns(df, int_columns):
    for col in int_columns:
        df[col] = pd.to_numeric(df[col]).astype('Int64')
    return df

@st.cache_data
def load_file_preview(file, nrows=None, dtype=None, usecols=None):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, engine="openpyxl", usecols=usecols)
    else:
        raise ValueError("Unsupported file format")

@st.cache_data
def preview_file(df, nrows=50, height=250):
    st.dataframe(df.head(nrows), height=height)

def estimate_csv_rows(file):
    file.seek(0)
    for i, line in enumerate(file):
        pass
    file.seek(0)  # Reset file pointer to the beginning
    return i

def load_file(file, nrows=None, dtype=None, usecols=None):
    CHUNK_SIZE = 5000  # Number of rows per chunk

    if file.name.endswith('.csv'):
        total_rows = nrows if nrows is not None else estimate_csv_rows(file)
        
        st.write(f'Chargement du fichier {file.name}....')  # Display label
        progress_bar = st.progress(0)
        chunks = []
        rows_processed = 0

        # Reset file pointer to the beginning for reading
        file.seek(0)
        chunk_iter = pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols, chunksize=CHUNK_SIZE)
        
        for chunk in chunk_iter:
            chunks.append(chunk)
            rows_processed += len(chunk)
            progress_bar.progress(min(1.0, rows_processed / total_rows))
            if rows_processed >= total_rows:
                break

        progress_bar.progress(1.0)
        return pd.concat(chunks, ignore_index=True)

    elif file.name.endswith('.xlsx'):
        with st.spinner(f'Chargement du fichier {file.name}....'):
            # Streamlit file uploader provides an in-memory buffer
            return pd.read_excel(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, engine="calamine", usecols=usecols)
    else:
        raise ValueError("Unsupported file format")

def telechargement(result_df):
    col1, col2, col3 = st.columns([2,2,1])

    with col1:
        # Champ de texte pour le nom du fichier
        file_name = col1.text_input("Nom du fichier sans extension", "dataframe_concatene")
        
    with col2:
        # Sélection du format de téléchargement
        download_format = col2.selectbox("Choisir le format de téléchargement", ["CSV", "Excel", "Pickle"])
    
    with col3:
        col3.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        if st.button('Appliquer'):
            if file_name and download_format:
                with st.spinner('Chargement du fichier, \nmerci de patienter ....'):
                    buffer = BytesIO()
                    if download_format == "CSV":
                        # Téléchargement en format CSV
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Télécharger en CSV", data=csv, file_name=f'{file_name}.csv', mime='text/csv')
                    elif download_format == "Excel":
                        # Téléchargement en format Excel
                        result_df.to_excel(buffer, index=False, engine='xlsxwriter')
                        st.download_button(label="Télécharger en Excel", data=buffer, file_name=f'{file_name}.xlsx', mime='application/vnd.ms-excel')
                    elif download_format == "Pickle":
                        # Téléchargement en format Pickle
                        result_df.to_pickle(buffer)
                        st.download_button(label="Télécharger en Pickle", data=buffer, file_name=f'{file_name}.pkl', mime='application/octet-stream')
            else:
                st.error('Veuillez remplir les champs requis')

def concatener(uploaded_files):
    
    #st.write(st.session_state.new_columns)
    
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = [col["name"] for col in st.session_state.new_columns]

    if 'rename_dicts' not in st.session_state:
        rename_dicts = {}
        for col in st.session_state.new_columns:
            for old_col in col["cols"]:
                if isinstance(old_col, list):
                    for nested_col in old_col:
                        if nested_col:
                            rename_dicts[nested_col] = col["name"]
                else:
                    if old_col:
                        rename_dicts[old_col] = col["name"]
        st.session_state.rename_dicts = rename_dicts

    selected_columns = st.session_state.selected_columns
    rename_dicts = st.session_state.rename_dicts
    final_dfs = []

    date_columns = [col["name"] for col in st.session_state.new_columns if col["type"] == "Date"]
    int_columns = [col["name"] for col in st.session_state.new_columns if col["type"] == "Entier"]

    for i, uploaded_file in enumerate(uploaded_files):
        dtype_mapping = {}
        usecols = []
        additional_columns_to_sum = {}

        for col in st.session_state.new_columns:
            if col["type"] == "Texte":
                dtype_mapping[col["cols"][i]] = str
                usecols.append(col["cols"][i])
                    
            elif col["type"] == "Décimal":
                if isinstance(col["cols"][i], list):
                    for nested_col in col["cols"][i]:
                        dtype_mapping[nested_col] = float
                        usecols.append(nested_col)
                    additional_columns_to_sum[col["name"]] = col["cols"][i]
                else:
                    dtype_mapping[col["cols"][i]] = float
                    usecols.append(col["cols"][i])
                    
            elif col["type"] == "Entier" or col["type"] == "Date":
                usecols.append(col["cols"][i])

        uploaded_file.seek(0)  # Revenir au début du fichier
        df = load_file(uploaded_file, nrows=None, dtype=dtype_mapping, usecols=usecols.remove('assureur'))
        df['assureur'] = st.session_state.assureurs[i]
            
        # Sum nested columns
        for new_col_name, nested_cols in additional_columns_to_sum.items():
            df[new_col_name] = df[nested_cols].fillna(0).sum(axis=1)
            df.drop(columns=nested_cols, inplace=True)
        
        # Renommer les colonnes
        rename_dict = {}
        for col in st.session_state.new_columns:
            if isinstance(col["cols"][i], list):
                for nested_col in col["cols"][i]:
                    if nested_col:
                        rename_dict[nested_col] = col["name"]
            else:
                if col["cols"][i]:
                    rename_dict[col["cols"][i]] = col["name"]
        df.rename(columns=rename_dict, inplace=True)
        
        # st.write([df.columns.duplicated()])
        ############################################# Prob renaming two columns in the same time
        # df = df.loc[:,~df.columns.duplicated()].copy()
        
        # Apply custom transformation for date columns
        df = custom_date_transformation(df, date_columns, raise_errors=True, fichier=uploaded_file.name)

        # Apply custom transformation for integer columns
        df = int_transform_columns(df, int_columns)
        
        final_dfs.append(df)

    # Concaténer les DataFrames
    with st.spinner('Concatenation en cours, merci de patienter ....'):
        try:
            result_df = concat_datasets(final_dfs, keep_all_columns=False, raise_alerte=True, selected_columns=selected_columns)

            # Apply custom transformation for date columns
            result_df = custom_date_transformation(result_df, date_columns)

            return result_df

        except ValueError as e:
            st.error(str(e))


# Fonction pour générer un aperçu des colonnes sélectionnées avec stylisation
def generate_preview(dfs):
    preview_data = {"Colonne": [], "Type": []}
    for i, df in enumerate(dfs):
        preview_data[f"Fichier {i + 1}"] = []
        
    for col in st.session_state.new_columns:
        preview_data["Colonne"].append(col["name"])
        preview_data["Type"].append(col["type"])
        for i, df in enumerate(dfs):
            # Handle cases where col["cols"][i] can be a list
            if isinstance(col["cols"][i], list):
                if (None in col["cols"][i]) or (not col["cols"][i]):
                    preview_data[f"Fichier {i + 1}"].append("Non sélectionnée")
                else:
                    selected_cols = ", ".join([str(c) for c in col["cols"][i]])
                    preview_data[f"Fichier {i + 1}"].append(selected_cols)
            else:
                preview_data[f"Fichier {i + 1}"].append(col["cols"][i] if col["cols"][i] else "Non sélectionnée")
    
    preview_df = pd.DataFrame(preview_data)
    
    # Appliquer la stylisation pour les cases "Non sélectionnée"
    def highlight_non_selected(val):
        color = '#FD636B' if val == "Non sélectionnée" else ''
        return f'background-color: {color}'
    
    styled_preview_df = preview_df.style.applymap(highlight_non_selected, subset=[f"Fichier {i + 1}" for i in range(len(dfs))])
    
    return preview_df, styled_preview_df

def generate_dataframe_summary(result_df, config_type):
    configurations = get_configurations()

    if config_type not in configurations:
        st.error(f"Configuration type '{config_type}' not found.")
        return

    config = configurations[config_type]
    
    st.write("## Aperçu du fichier concaténé")
    st.write(f"**Type de configuration : {config_type}**")

    # Nombre de lignes et de colonnes
    st.write(f"Nombre de lignes : {len(result_df)}, Nombre de colonnes : {len(result_df.columns)}")

    # Description des colonnes
    column_descriptions = []
    for col in config:
        col_name = col["name"]
        col_type = col["type"]
        unique_values = result_df[col_name].nunique() if col_name in result_df.columns else "N/A"
        missing_values = result_df[col_name].isna().sum() if col_name in result_df.columns else "N/A"
        column_descriptions.append(
            {
                "Nom de la colonne": col_name,
                "Type": col_type,
                "Valeurs uniques": unique_values,
                "Valeurs manquantes": missing_values
            }
        )
    
    # Afficher la description des colonnes sous forme de DataFrame stylisé
    summary_df = pd.DataFrame(column_descriptions)
    st.write(summary_df)
    
    if config_type == "Prestations":
        st.write("**Resumé dates**")
        st.write(result_df.describe(include=['datetime64']))


def normalize_column_name(col_name):
    # Normaliser le nom de la colonne : splitter par espace, convertir en minuscules
    return col_name.lower().replace("_", " ").split()

def validate_files(dfs, required_columns):
    normalized_required_columns = list(itertools.chain.from_iterable([normalize_column_name(col) for col in required_columns]))
    for i, df in enumerate(dfs):
        normalized_columns = list(itertools.chain.from_iterable([normalize_column_name(col) for col in df.columns]))
        if not any(col in normalized_columns for col in normalized_required_columns):
            return False, i
    return True, None, 

def init():
    st.session_state.config = []
    st.session_state.new_columns = []
    st.session_state.include = None
    st.session_state.assureurs = []
    st.session_state.ignore_errors = False
    st.session_state.validated = False
    st.session_state.errors = False
    st.session_state.df_final = None
    st.session_state.init_valid = False
    st.session_state.df_json = None
    st.session_state.params_json = None
    st.session_state.cot_net_json = None

    
    # # Initialize session state variables
    # if 'validated' not in st.session_state:
    #     st.session_state.validated = True

    # if 'ignore_errors' not in st.session_state:
    #     st.session_state.ignore_errors = False

    # if 'df_final' not in st.session_state:
    #     st.session_state.df_final = None

def change_init_state():
    st.session_state.init_valid = False

@st.cache_resource
def init_appearance(logo, title):
    
    #logo
    # st.logo(logo, icon_image=logo)
    
    # Separation
    st.divider()

    log, titl = st.columns([1,2])
    
    # log.image(logo, width=200)
    
    # Titre de l'application
    titl.title(title)

    # Separation
    st.divider()
    
@st.cache_data
def previw_uploaded_files(uploaded_files):
    dfs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            df = load_file_preview(uploaded_file, nrows=50)
            dfs.append(df)
            st.write(f"**{uploaded_file.name}**")
            preview_file(df, nrows=50)
    return dfs

def load_config(include):
    st.session_state.new_columns = [ col for col in st.session_state.config if not include.get(col['name'], False) ]

def main_concat():
        
    # Get the current directory (where your app is running)
    current_dir = os.path.dirname(__file__)

    # Use relative paths to the JSON, images, and other resources
    MAPPINGS_FILE = os.path.join(current_dir, 'resources', 'mappings.json')
    page_ico = os.path.join(current_dir, 'resources', 'merge.png')
    logo = os.path.join(current_dir, 'resources', 'Logo_AOPS_conseil.png')
    title = 'Outil de :orange[Concatenation] des Fichiers '
    
    st.set_page_config(layout="wide", page_title='Outil de concatenation', page_icon=page_ico)
    
    init_appearance(logo, title)

    # Chargement des fichiers
    uploaded_files = st.file_uploader("Choisir des fichiers", accept_multiple_files=True, type=['csv', 'xlsx', 'xls'], on_change=change_init_state)

    if uploaded_files:
        dfs = previw_uploaded_files(uploaded_files)

    # Separation
    st.divider()

    # Initialiser les listes pour stocker les noms et types des colonnes ajoutées
    if 'new_columns' not in st.session_state:
        init()

    col1, col2 = st.columns([2,1])
    with col1:
        configurations = get_configurations()
        # Sélection de la configuration
        config_choice = st.selectbox("Choisir une configuration prédéfinie", list(configurations.keys()), index=None, on_change=change_init_state, placeholder='Choisir une option')
        
    with col2:
        col2.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        if st.button("Appliquer la configuration"):
            
            if not uploaded_files:
                st.error("Veuillez charger au moins un fichier avant de continuer.")
            elif not config_choice:
                st.error("Veuillez choisir une configuration avant de continuer.")
            else:
                if config_choice == 'Prestations':
                    required_columns = ["taux", "remboursement", "tauxro"]
                    
                    all_files_valid, invalid_file_index = validate_files(dfs, required_columns)
                    
                    if all_files_valid:
                        st.session_state.config = configurations[config_choice]
                        st.success("Configuration appliquée avec succès !")
                        st.session_state.init_valid = True
                    else:
                        st.error(f'Le fichier {invalid_file_index + 1} ne contient aucune des colonnes requises pour la configuration "{config_choice}". Veuillez vérifier vos fichiers.')
                elif config_choice == 'Cotisations':
                    required_columns = ["cot", "cotisations", "cotisation"]
                    
                    all_files_valid, invalid_file_index = validate_files(dfs, required_columns)
                    
                    if all_files_valid:
                        st.session_state.config = configurations[config_choice]
                        st.success("Configuration appliquée avec succès !")
                        st.session_state.init_valid = True
                    else:
                        st.error(f'Le fichier {invalid_file_index + 1} ne contient aucune des colonnes requises pour la configuration "{config_choice}". Veuillez vérifier vos fichiers.')

    # Separation
    st.divider()

    # Configurations prédéfinies
    if st.session_state.config and uploaded_files and st.session_state.init_valid:
    
        if not st.session_state.include:
                    st.session_state.include = {
                    "niveau_couverture_oblg": False,
                    "type_bénéf": False,
                    "remboursement_base": False,
                    "remboursement_fac": False,
                    "remboursement_TTC": True,
                    "cotisations_base": True,
                    "cotisations_fac": False,
                    "cotisations_TTC": False,
                }
        
        with st.form("my_form"):
            
            cols = st.columns(4)
            
            niv_oblg =  not cols[0].toggle('Niveau couverture obligatoire', key='niv_oblg_enabeling')
            type_benef =  not cols[1].toggle('Type bénéf', key='type_bénéf_enabeling')
            mnt_fac =  not cols[2].toggle('Montants facultatifs', key='montants_facultatives_enabeling')
            
            if mnt_fac:
                base = True
                fac = True
                ttc = False
            else:
                base = False
                fac = False
                ttc = True

            st.session_state.include = {
                "niveau_couverture_oblg": niv_oblg,
                "type_bénéf": type_benef,
                "remboursement_base": base,
                "remboursement_fac": fac,
                "remboursement_TTC": ttc,
                "cotisations_base": base,
                "cotisations_fac": fac,
                "cotisations_TTC": ttc,
            }
            
            if cols[3].form_submit_button("Charger"):
                load_config(st.session_state.include)
            
        
        if st.session_state.new_columns:
            with st.form("cols_choose_form"):     
                # Afficher les colonnes ajoutées
                st.header("Configuration des colonnes")
                
                
                st.subheader("Assureur", divider='grey')
                assr = ['AG2R', 'MACIF', 'AESIO', 'MH', 'HM', 'SMF', 'SMH', 'MUTAMI', 'MFAS']

                
                n = len(dfs)
                cols = st.columns(4)
                
                st.session_state.assureurs = []
                for i, df in enumerate(dfs):
                    selected_assr = cols[i%4].selectbox(label="", options=assr, placeholder=f'Assureur du fichier {i+1}', index=None, label_visibility="collapsed")
                    dfs[i]['assureur'] = selected_assr
                    st.session_state.assureurs.append(selected_assr)
                    
                if 'number' not in st.session_state:
                    st.session_state.number = {k: 1 for k in range(len(dfs))}
                
                # selected_columns = {k: set() for k in range(len(dfs))}

                for i, col in enumerate(st.session_state.new_columns):
                    col_name = col["name"]
                    col_type = col["type"]
                    
                    
                    if col_name == 'assureur':
                        # Mettre à jour les valeurs dans session_state
                        st.session_state.new_columns[i]["name"] = col_name
                        st.session_state.new_columns[i]["type"] = col_type
                        st.session_state.new_columns[i]["cols"] = [col_name for df in dfs]
                    
                    else:
                        
                        st.subheader(f'{col_name}', divider='grey')
                        
                        cols = st.columns(4)
                        file_cols = []
                        for j, df in enumerate(dfs):
                            columns = list(df.columns)
                            
                            # Filter out already selected columns
                            available_columns = columns.copy() #[c for c in columns if c not in selected_columns[j]]


                            if col_name not in ["remboursement_TTC", "cotisations_TTC",'remboursement_base', 'remboursement_fac', 'cotisations_base', 'cotisations_fac']:
                                selected_col = cols[j%4].selectbox(label="", options=available_columns, placeholder=f'Colonne du fichier {j+1}', index=None, key=f"cols_{i}_{j}", label_visibility="collapsed")
                                file_cols.append(selected_col) #if selected_col else None)
                                #if selected_col:
                                #    selected_columns[j].add(selected_col)
                            else:
                                additional_cols = cols[j%4].multiselect(label="", options=available_columns, placeholder=f'Colonne du fichier {j+1}', key=f"cols_{i}_{j}", label_visibility="collapsed", max_selections=3)
                                file_cols.append(additional_cols)
                        
                        # Mettre à jour les valeurs dans session_state
                        st.session_state.new_columns[i]["name"] = col_name
                        st.session_state.new_columns[i]["type"] = col_type
                        st.session_state.new_columns[i]["cols"] = file_cols

                    #st.divider()

                    # ChangeButtonColour('Supprimer', font_color='red' )  # button text to find, color to assign

                st.form_submit_button("Charger")
            
            # Aperçu des colonnes sélectionnées
            if st.session_state.new_columns:
                st.divider()
                st.header("Aperçu des colonnes sélectionnées")
                preview_df, styled_preview_df = generate_preview(dfs)
                st.dataframe(styled_preview_df, hide_index=True, use_container_width=True)
                #st.write(st.session_state.new_columns)
            
        # Bouton pour finaliser l'ajout de colonnes
        if (not st.session_state.ignore_errors) and (st.session_state.new_columns):
            if st.button('Valider les colonnes'):
                
                validation_errors = []

                if not uploaded_files:
                    validation_errors.append("Veuillez charger 2 fichiers ou plus.")
                    
                if len(uploaded_files) == 1:
                    validation_errors.append("Veuillez charger 2 fichiers ou plus.")
                    
                # Vérification des noms de colonnes uniques
                if not st.session_state.new_columns:
                    validation_errors.append("Veuillez ajouter des colonnes.")
                
                # Vérification des noms de colonnes uniques
                column_names = [col["name"] for col in st.session_state.new_columns]
                if len(column_names) != len(set(column_names)):
                    validation_errors.append("Les noms des colonnes doivent être uniques.")

                # Vérification des noms sont remplis
                if ("" in column_names) or (None in column_names):
                    validation_errors.append("Veuillez remplir tous les noms des colonnes ajoutées.")
                
                # Vérification des types de colonnes valides
                for col in st.session_state.new_columns:
                    if col["type"] not in ["Texte", "Entier", "Décimal", "Date"]:
                        validation_errors.append(f"Le type de la colonne '{col['name']}' est invalide.")
                
                # Vérification des colonnes sélectionnées dans chaque DataFrame
                for col in st.session_state.new_columns:
                    if not all(col["cols"]):
                        validation_errors.append(f"La colonne '{col['name']}' doit être sélectionnée dans tous les fichiers.")

                # Vérification de la convertibilité des types de colonnes
                for col in st.session_state.new_columns:
                    for j, df in enumerate(dfs):
                        if col["cols"][j]:
                            try:
                                if col["type"] == "Texte":
                                    df[col["cols"][j]].astype(str)
                                elif col["type"] == "Entier":
                                    df[col["cols"][j]].astype(int)
                                elif col["type"] == "Décimal":
                                    df[col["cols"][j]].astype(float)
                                elif col["type"] == "Date":
                                    try:
                                        pd.to_datetime(df[col["cols"][j]].astype(str))
                                    except ValueError:
                                        correction_dates_integrale(df, col["cols"][j])
                            except (ValueError, TypeError, KeyError):
                                validation_errors.append(f"La colonne '{col['cols'][j]}' du fichier {j+1} ne peut pas être convertie en type '{col['type']}'.")
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                    st.session_state.validated = False
                    st.session_state.ignore_errors = False
                    st.session_state.errors = True
                else:
                    st.success("Toutes les colonnes ont été validées!")
                    st.session_state.validated = True
                    st.session_state.ignore_errors = False
                    
        if st.session_state.errors and not st.session_state.validated:
            if st.button('Ignorer les erreurs'):
                st.warning("Vous avez choisi de continuer en ignorant les erreurs.")
                st.session_state.validated = False
                st.session_state.ignore_errors = True
    
        if st.session_state.validated or st.session_state.ignore_errors:
            st.divider()
            st.header('Concatenation')
            #st.write(st.session_state.new_columns)
            if st.button('Concatener'):
                st.session_state.df_final = concatener(uploaded_files)
                st.session_state.df_formatted = st.session_state.df_final.copy()
                
        if st.session_state.df_final is not None:
            st.subheader("DataFrame concaténé")
            
            st.dataframe(st.session_state.df_final.head(50), hide_index=True)

            with st.popover("Détails du fichier"):
                generate_dataframe_summary(st.session_state.df_final, config_choice)
            
            st.divider()
            
            def handle_column_mapping(col_to_map, col_index):
                
                def update_mapping(mapping_df, col):
                    for index, row in mapping_df.iterrows():
                        st.session_state.mappings[col][row['Ancienne']] = row['Nouvelle']
            
                st.session_state[f"{col_to_map}_edited_df"] = edit_mapping(
                    st.session_state.df_formatted, col_to_map, st.session_state.mappings[col_to_map]
                ).reset_index(drop=True)

                cols[col_index].write(f"**{col_to_map}**")
                
                edited_df = cols[col_index].data_editor(
                    st.session_state[f"{col_to_map}_edited_df"],
                    column_config={
                        'Ancienne': st.column_config.TextColumn(disabled=True, required=True),
                        'Nouvelle': st.column_config.SelectboxColumn(required=True, options=set(st.session_state.mappings[col_to_map].values()))
                    },
                    hide_index=True,
                    key=f'{col_to_map}_edited_dict'
                )

                if cols[col_index].button(f'Mettre en forme {col_to_map}'):
                    edited_df = restore_editions(st.session_state[f"{col_to_map}_edited_df"], st.session_state[f"{col_to_map}_edited_dict"])
                    
                    if check_mappings_complete(edited_df.replace('nan', None)):
                        update_mapping(edited_df, col_to_map)
                        save_mappings(MAPPINGS_FILE, st.session_state.mappings)
                        st.session_state.df_formatted = mise_en_forme_df(st.session_state.df_formatted, st.session_state.mappings, col_to_map)
                        cols[col_index].success(f"{col_to_map} mis en forme avec succès.")
                    else:
                        cols[col_index].error(f"Veuillez compléter tous les mappings pour {col_to_map} avant de continuer.")
                        
            st.subheader("Mise en forme")
            if 'niv_couv_edited_df' not in st.session_state:
                st.session_state.df_formatted = st.session_state.df_final.copy()
                st.session_state.niv_couv_edited_df = None
                st.session_state.niv_couv_oblg_edited_df = None
                st.session_state.cat_assr_edited_df = None
                st.session_state.type_bénéf_edited_df = None
            
            mappings = load_mappings(MAPPINGS_FILE)

            # if 'mappings' not in st.session_state:
            st.session_state.mappings = mappings

            cols = st.columns(2)
            
            cols_to_map = ['niveau_couverture', 'niveau_couverture_oblg', 'categorie_assuré', 'type_bénéf']
            cols_available = [col for col in cols_to_map if col in st.session_state.df_final]
            
            for idx, col_to_map in enumerate(cols_available):
                handle_column_mapping(col_to_map, idx % 2)

            if st.session_state.df_formatted is not None:
                st.subheader('Résultat final')
                st.dataframe(st.session_state.df_formatted.head(50), hide_index=True)
                
            # st.dataframe(st.session_state.df_final.select_dtypes(include=float).sum())
            st.header('Téléchargement')
            if st.session_state.df_formatted is not None:
                telechargement(st.session_state.df_formatted)
            else:
                telechargement(st.session_state.df_final)
            
    # if st.button('Traiter un autre cas'):
    #     if 'dfs_all' not in st.session_state:
    #         st.session_state.dfs_all = []
    #     if st.session_state.df_formatted is not None:
    #         st.session_state.dfs_all.append(st.session_state.df_format)
    #     else:
    #         st.session_state.dfs_all.append(st.session_state.df_final)
    #     if 'dfs_all' in st.session_state:
    #         # Chargement des fichiers
    #         uploaded_files = st.file_uploader("Choisir des fichiers", accept_multiple_files=True, type=['csv', 'xlsx', 'xls'], on_change=change_init_state)

    #         if uploaded_files:
    #             dfs = previw_uploaded_files(uploaded_files)

            
        
if __name__ == "__main__":
    main_concat()