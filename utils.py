# utils.py

import pandas as pd
import json
from streamlit import warning, success, selectbox
import os
from numpy import array, nan
import numpy as np
# def load_file(file, n_rows=None, dtype=None, usecols=None):
#     if file.name.endswith('.csv'):
#         return pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols)
#     elif file.name.endswith('.xlsx'):
#         return pd.read_excel(file, nrows=n_rows, dtype=dtype, na_values="@", keep_default_na=True, engine="openpyxl", usecols=usecols)
#     else:
#         raise ValueError("Unsupported file format")


def concatenate_datasets(dfs, selected_columns):
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    return concatenated_df[selected_columns]

def validate_column_types(dfs, column_types):
    for df in dfs:
        for col, dtype in column_types.items():
            if col in df.columns and not df[col].map(lambda x: isinstance(x, dtype)).all():
                raise ValueError(f"Column {col} does not match expected type {dtype}")

def concat_datasets(dfs, keep_all_columns=False, raise_alerte=True, selected_columns=None, rename_dicts=None):
    """
    Concaténer plusieurs DataFrames en vérifiant les colonnes communes et les types de données.
    
    Args:
    dfs (list): Liste des DataFrames à concaténer.
    keep_all_columns (bool): Garder toutes les colonnes ou seulement les colonnes communes.
    raise_alerte (bool): Lancer une exception en cas de types de données incohérents.
    selected_columns (list): Liste des colonnes sélectionnées pour la concaténation.
    rename_dicts (list): Liste de dictionnaires de renommage des colonnes pour chaque DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame concaténé.
    """
    
    # Renommer les colonnes des DataFrames si nécessaire
    if rename_dicts:
        expanded_rename_dict = {}
        for key, value in rename_dicts.items():
            if isinstance(key, (list, tuple)):
                for col in key:
                    expanded_rename_dict[col] = value
            else:
                expanded_rename_dict[key] = value

        # Appliquer le renommage aux DataFrames
        for i in range(len(dfs)):
            dfs[i] = dfs[i].rename(columns=expanded_rename_dict)
    
    # Obtenir les colonnes communes et non communes parmi tous les DataFrames
    common_cols = set(dfs[0].columns)
    all_cols = set(dfs[0].columns)
    
    for df in dfs[1:]:
        all_cols |= set(df.columns)
        common_cols &= set(df.columns)
    
    only_in_first_df = all_cols - common_cols
    
    warnings = "\nWARNINGS:\n"
    alerte = "\nALERTE!!!!!:\n"
    
    # Collecter les colonnes non communes et les DataFrames correspondants
    cols_non_communes = {}
    for i, df in enumerate(dfs, start=1):
        cols_non_communes[f'DF{i}'] = set(df.columns) - common_cols

    for df_name, cols in cols_non_communes.items():
        if cols:
            warnings += f"\n            - Colonnes seulement dans {df_name}: {cols}"
    
    # Concaténer les DataFrames
    if keep_all_columns:
        result_df = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        result_df = pd.concat([df[list(common_cols)] for df in dfs], ignore_index=True)
    
    # Réorganiser les colonnes pour correspondre à l'ordre du premier DataFrame
    if keep_all_columns:
        result_df = result_df[dfs[0].columns.tolist() + list(all_cols - set(dfs[0].columns))]
    else:
        result_df = result_df[dfs[0].columns.intersection(common_cols).tolist()]
    
    # Si des colonnes sélectionnées sont spécifiées, les utiliser pour la concaténation
    if selected_columns is not None:
        result_df = result_df[selected_columns]
        
    # Sortir des warnings pour les colonnes non gardées si keep_all_columns est False
    if not keep_all_columns:
        cols_supp = "\n\n      Colonnes non gardées:"
        if only_in_first_df:
            cols_supp += f"\n            - Colonnes du premier DataFrame: {only_in_first_df}"
        warnings += cols_supp
    
    print(warnings)
        
    return result_df

def correction_dates_integrale(df_raw, col):
    df = df_raw[[col]].copy()

    # Convertir toutes les valeurs en chaînes de caractères
    df[col] = df[col].astype(str)

    # Remplacer les valeurs spécifiques par des dates de référence
    df[col] = df[col].replace('2958465', '2099-12-31')

    # Identifier et convertir les valeurs numériques en dates
    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
    df.loc[numeric_mask, col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[numeric_mask, col].astype(int), 'D')
    df.loc[numeric_mask, col] = df.loc[numeric_mask, col].astype(str)
    
    # Supprimer les chaînes de temps
    df[col] = df[col].str.replace(" 00:00:00", "")
    
    # Remplacer '9999' et '2999' par '2099'
    df[col] = df[col].str.replace('9999', '2099')
    df[col] = df[col].str.replace('2999', '2099')
    
    # Identifier et convertir les dates au format avec tirets
    dash_mask = df[col].str.contains('-')
    slash_mask = df[col].str.contains('/')

    # Traiter les dates avec tirets
    if dash_mask.any():
        df.loc[dash_mask, col] = pd.to_datetime(df.loc[dash_mask, col])

    # Traiter les dates avec barres obliques
    if slash_mask.any():
        # Utiliser to_datetime avec coerce pour tenter les formats les plus courants
        try:
            df.loc[slash_mask, col] = pd.to_datetime(df.loc[slash_mask, col], format='%m/%d/%Y')
        except ValueError:
            df.loc[slash_mask, col] = pd.to_datetime(df.loc[slash_mask, col], format='%d/%m/%Y')

    return pd.to_datetime(df[col])

def get_configurations():
    return {
        "Prestations": [
            {"name": "assureur", "type": "Texte", "cols": []},
            {"name": "niveau_couverture_oblg", "type": "Texte", "cols": []},
            {"name": "niveau_couverture", "type": "Texte", "cols": []},
            {"name": "categorie_assuré", "type": "Texte", "cols": []},
            {"name": "type_bénéf", "type": "Texte", "cols": []},
            {"name": "date_survenance", "type": "Date", "cols": []},
            {"name": "date_comptable", "type": "Date", "cols": []},
            {"name": "remboursement_base", "type": "Décimal", "cols": []},
            {"name": "remboursement_fac", "type": "Décimal", "cols": []},
            {"name": "remboursement_TTC", "type": "Décimal", "cols": []},
        ],
        "Cotisations": [
            {"name": "assureur", "type": "Texte", "cols": []},
            {"name": "niveau_couverture_oblg", "type": "Texte", "cols": []},
            {"name": "niveau_couverture", "type": "Texte", "cols": []},
            {"name": "categorie_assuré", "type": "Texte", "cols": []},
            {"name": "type_bénéf", "type": "Texte", "cols": []},
            {"name": "annee_survenance", "type": "Texte", "cols": []},
            {"name": "annee_comptable", "type": "Texte", "cols": []},
            {"name": "cotisations_base", "type": "Décimal", "cols": []},
            {"name": "cotisations_fac", "type": "Décimal", "cols": []},
            {"name": "cotisations_TTC", "type": "Décimal", "cols": []}
        ],
    }

def get_params_ccn(ccn, compta):
    parametres = [
        {"ccn":"BAD","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs","type_montant":'cotisations_base'},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1","type_montant":'cotisations_fac'},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2","type_montant":'cotisations_fac'},
            {"régime": 4, "cat_assure": "accueils"}
        ],
            "régimes_prest": [
            {"régime": 1, "cat_assure": "actifs","type_montant":'remboursement_base'},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1","type_montant":'remboursement_fac'},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2","type_montant":'remboursement_fac'},
            {"régime": 4, "cat_assure": "accueils"}

        ],
            "chargements": [
                {"régime": 1, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.13}},
                {"régime": 2, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.15}},
                {"régime": 3, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}},
                {"régime": 4, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}}
            ],
            "PSAP": [
                {"régime": 1, "survenance": {2020: 0, 2021: -85, 2022: 742+205, 2023: 64653+19808}},
                {"régime": 2, "survenance": {2020: 0, 2021: 0, 2022: 0, 2023: 1277}},
                {"régime": 3, "survenance": {2020: 0, 2021: 0, 2022: 14, 2023: 7236}},
                {"régime": 4, "survenance": {2020: 64, 2021: 0, 2022: 2, 2023: 10152}}
            ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"BAD","compta":2022, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs","type_montant":'cotisations_base'},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1","type_montant":'cotisations_fac'},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2","type_montant":'cotisations_fac'},
            {"régime": 4, "cat_assure": "accueils"}
        ],
            "régimes_prest": [
                {"régime": 1, "cat_assure": "actifs", "type_montant": "remboursement_base"},
                {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1", "type_montant": "remboursement_fac"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2", "type_montant": "remboursement_fac"},
                {"régime": 4, "cat_assure": "accueils"}
            ],
            "chargements": [
                {"régime": 1, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.13}},
                {"régime": 2, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.15}},
                {"régime": 3, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}},
                {"régime": 4, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}}
            ],
            "PSAP": [
                {"régime": 1, "survenance": {2020: 0, 2021: -85, 2022: 742+205, 2023: 64653+19808}},
                {"régime": 2, "survenance": {2020: 0, 2021: 0, 2022: 0, 2023: 1277}},
                {"régime": 3, "survenance": {2020: 0, 2021: 0, 2022: 14, 2023: 7236}},
                {"régime": 4, "survenance": {2020: 64, 2021: 0, 2022: 2, 2023: 10152}}
            ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"BJOH2","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
                {"régime": 1, "cat_assure": "actifs","niveau_couv":"socle" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "socle"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle" },
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "régimes_prest": [
                {"régime": 1, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle',"type_bénéf":"assuré" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "option 1","type_bénéf":"assuré"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle","type_bénéf":"assuré"},
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle","type_bénéf":"conjoint"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "chargements": [
                {"régime": 1, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.13}},
                {"régime": 2, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.15}},
                {"régime": 3, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}},
                {"régime": 4, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}}
            ],
            "PSAP": [
                {"régime": 1, "survenance": {2020: 0, 2021: -85, 2022: 742+205, 2023: 64653+19808}},
                {"régime": 2, "survenance": {2020: 0, 2021: 0, 2022: 0, 2023: 1277}},
                {"régime": 3, "survenance": {2020: 0, 2021: 0, 2022: 14, 2023: 7236}},
                {"régime": 4, "survenance": {2020: 64, 2021: 0, 2022: 2, 2023: 10152}}
            ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"BJOH","compta":2022, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
                {"régime": 1, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle',"type_montant": "cotisations_base","type_bénéf":"assuré" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "option 1","type_bénéf":"assuré"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle","type_bénéf":"assuré", "type_montant": "cotisations_fac"},
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle","type_montant": "cotisations_base","type_bénéf":"conjoint"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "régimes_prest": [
                {"régime": 1, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle',"type_montant": "remboursement_base","type_bénéf":"assuré" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "option 1","type_bénéf":"assuré"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle","type_bénéf":"assuré", "type_montant": "remboursement_fac"},
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle","type_montant": "remboursement_base","type_bénéf":"conjoint"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "chargements": [
                {"régime": 1, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.13}},
                {"régime": 2, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.15}},
                {"régime": 3, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}},
                {"régime": 4, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}}
            ],
            "PSAP": [
                {"régime": 1, "survenance": {2020: 0, 2021: -85, 2022: 742+205, 2023: 64653+19808}},
                {"régime": 2, "survenance": {2020: 0, 2021: 0, 2022: 0, 2023: 1277}},
                {"régime": 3, "survenance": {2020: 0, 2021: 0, 2022: 14, 2023: 7236}},
                {"régime": 4, "survenance": {2020: 64, 2021: 0, 2022: 2, 2023: 10152}}
            ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    }, {"ccn":"BJOH","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
                {"régime": 1, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle',"type_montant": "cotisations_base","type_bénéf":"assuré" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "option 1","type_bénéf":"assuré"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle","type_bénéf":"assuré", "type_montant": "cotisations_fac"},
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle","type_montant": "cotisations_base","type_bénéf":"conjoint"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "régimes_prest": [
                {"régime": 1, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle',"type_montant": "remboursement_base","type_bénéf":"assuré" },
                {"régime": 2, "cat_assure": "actifs", "niveau_couverture_oblg": "option 1","type_bénéf":"assuré"},
                {"régime": 3, "cat_assure": "actifs", "niveau_couverture_oblg": "socle","type_bénéf":"assuré", "type_montant": "remboursement_fac"},
                {"régime": 4, "cat_assure": "actifs", "niveau_couverture_oblg": 'socle', "niveau_couv":"socle","type_montant": "remboursement_base","type_bénéf":"conjoint"},
                {"régime": 5, "cat_assure": "accueils", "niveau_couv":"socle"},
                {"régime": 6, "cat_assure": "accueils","niveau_couv":"option 1"}
            ],
            "chargements": [
                {"régime": 1, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.13}},
                {"régime": 2, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.15}},
                {"régime": 3, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}},
                {"régime": 4, "survenance": {2020: 0.13, 2021: 0.13, 2022: 0.13, 2023: 0.1485}}
            ],
            "PSAP": [
                {"régime": 1, "survenance": {2020: 0, 2021: -85, 2022: 742+205, 2023: 64653+19808}},
                {"régime": 2, "survenance": {2020: 0, 2021: 0, 2022: 0, 2023: 1277}},
                {"régime": 3, "survenance": {2020: 0, 2021: 0, 2022: 14, 2023: 7236}},
                {"régime": 4, "survenance": {2020: 64, 2021: 0, 2022: 2, 2023: 10152}}
            ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    }, 
    {"ccn":"HPA","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"},
            {"régime": 4, "cat_assure": "accueils", "niveau_couv": "socle"},
            {"régime": 5, "cat_assure": "accueils", "niveau_couv": "option 1"},
            {"régime": 6, "cat_assure": "accueils", "niveau_couv": "option 2"}
        ],"régimes_prest": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"},
            {"régime": 4, "cat_assure": "accueils", "niveau_couv": "socle"},
            {"régime": 5, "cat_assure": "accueils", "niveau_couv": "option 1"},
            {"régime": 6, "cat_assure": "accueils", "niveau_couv": "option 2"}
        ],
        "chargements": [
            {"régime": 1, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089396441619599, 2023: 0.13042842168979}},
            {"régime": 2, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089349891833361, 2023: 0.132454599558257}},
            {"régime": 3, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.08929411961561, 2023: 0.132454583740632}},
            {"régime": 4, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089296087204585, 2023: 0.132454328173773}},
            {"régime": 5, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089343782223139, 2023: 0.132430193657481}},
            {"régime": 6, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089292748629601, 2023: 0.132455189505034}}
        ],
        "PSAP": [
            {"régime": 1, "survenance": {2020: -8.09, 2021: -448.03, 2022: 3187.03402500001, 2023: 81786.7492499991}},
            {"régime": 2, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 76.8314450000001, 2023: 1501.6955}},
            {"régime": 3, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 504.166845, 2023: 11100.06225}},
            {"régime": 4, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 277.52858, 2023: 11602.2437499999}},
            {"régime": 5, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 4.23857, 2023: 35.04425}},
            {"régime": 6, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 61.369315, 2023: 2753.122}}
        ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 5, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 6, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"SVP","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"}
        ],
            "régimes_prest": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"}
        ],
        "chargements": [
            {"régime": 1, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089396441619599, 2023: 0.13042842168979}},
            {"régime": 2, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089349891833361, 2023: 0.132454599558257}},
            {"régime": 3, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.08929411961561, 2023: 0.132454583740632}}
        ],
        "PSAP": [
            {"régime": 1, "survenance": {2020: -8.09, 2021: -448.03, 2022: 3187.03402500001, 2023: 81786.7492499991}},
            {"régime": 2, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 76.8314450000001, 2023: 1501.6955}},
            {"régime": 3, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 504.166845, 2023: 11100.06225}}
        ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"SVP","compta":2022, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"}
        ],
            "régimes_prest": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 2"}
        ],
        "chargements": [
            {"régime": 1, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089396441619599, 2023: 0.13042842168979}},
            {"régime": 2, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089349891833361, 2023: 0.132454599558257}},
            {"régime": 3, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.08929411961561, 2023: 0.132454583740632}}
        ],
        "PSAP": [
            {"régime": 1, "survenance": {2020: -8.09, 2021: -448.03, 2022: 3187.03402500001, 2023: 81786.7492499991}},
            {"régime": 2, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 76.8314450000001, 2023: 1501.6955}},
            {"régime": 3, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 504.166845, 2023: 11100.06225}}
        ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"ICR","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "accueils", "niveau_couv": "socle"}
        ],
            "régimes_prest": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "socle"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 3, "cat_assure": "accueils", "niveau_couv": "socle"}
        ],
        "chargements": [
            {"régime": 1, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089396441619599, 2023: 0.13042842168979}},
            {"régime": 2, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089349891833361, 2023: 0.132454599558257}},
            {"régime": 3, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.08929411961561, 2023: 0.132454583740632}}
        ],
        "PSAP": [
            {"régime": 1, "survenance": {2020: -8.09, 2021: -448.03, 2022: 3187.03402500001, 2023: 81786.7492499991}},
            {"régime": 2, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 76.8314450000001, 2023: 1501.6955}},
            {"régime": 3, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 504.166845, 2023: 11100.06225}}
        ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    },{"ccn":"INTERMITTENT","compta":2023, "params":[{
            "taxes": 0.1327,
            "DES": 0.02,
            "forfaits_pat": 0.008,
            "autres_frais": 0,
            "régimes_cot": [
            {"régime": 1, "cat_assure": "actifs", "niveau_couv": "option 1"},
            {"régime": 2, "cat_assure": "actifs", "niveau_couv": "option 2"},
            {"régime": 3, "cat_assure": "actifs", "niveau_couv": "option 3"},
            {"régime": 4, "cat_assure": "accueils", "niveau_couv": "option 1"},
            {"régime": 5, "cat_assure": "accueils", "niveau_couv": "option 2"},
            {"régime": 6, "cat_assure": "accueils", "niveau_couv": "option 3"}
        ],
            "régimes_prest": [
            {"régime": 1, "niveau_couv":"actif option 1"},
            {"régime": 2,  "niveau_couv": "actif option 2"},
            {"régime": 3, "niveau_couv": "actif option 3"},
            {"régime": 4,  "niveau_couv": "inactif option 1"},
            {"régime": 5,  "niveau_couv": "inactif option 2"},
            {"régime": 6,  "niveau_couv": "inactif option 3"}
        ],
        "chargements": [
            {"régime": 1, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089396441619599, 2023: 0.13042842168979}},
            {"régime": 2, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089349891833361, 2023: 0.132454599558257}},
            {"régime": 3, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.08929411961561, 2023: 0.132454583740632}},
            {"régime": 4, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089296087204585, 2023: 0.132454328173773}},
            {"régime": 5, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089343782223139, 2023: 0.132430193657481}},
            {"régime": 6, "survenance": {2019: 0.08, 2020: 0.08, 2021: 0.08, 2022: 0.089292748629601, 2023: 0.132455189505034}}
        ],
        "PSAP": [
            {"régime": 1, "survenance": {2020: -8.09, 2021: -448.03, 2022: 3187.03402500001, 2023: 81786.7492499991}},
            {"régime": 2, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 76.8314450000001, 2023: 1501.6955}},
            {"régime": 3, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 504.166845, 2023: 11100.06225}},
            {"régime": 4, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 277.52858, 2023: 11602.2437499999}},
            {"régime": 5, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 4.23857, 2023: 35.04425}},
            {"régime": 6, "survenance": {2019: 0, 2020: 0, 2021: 0, 2022: 61.369315, 2023: 2753.122}}
        ],
            "autres_contributions": [
                {"régime": 1, "survenance": {2020: 0, 2021: 4395, 2022: 0, 2023: 0}},
                {"régime": 2, "survenance": {2020: 0, 2021: 666, 2022: 0, 2023: 0}},
                {"régime": 3, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 4, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 5, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}},
                {"régime": 6, "survenance": {2020: 0, 2021: 457, 2022: 0, 2023: 0}}
            ]
        }]
    }
]
    
    for param in parametres:
        if param["ccn"] == ccn and param["compta"] == compta:
            return param["params"]
    return None


def generate_json_c(df, ccn, compta):
    #df=df.rename(columns={'Montant de cotisations Facultative TTC':'option','Montant de cotisations Obligatoire TTC':'base'})
    df['annee_survenance'] = pd.to_numeric(df['annee_survenance'], errors='coerce')
    df['annee_comptable'] = pd.to_numeric(df['annee_comptable'], errors='coerce')
    

    df_filtered = df[df['annee_comptable'] == int(compta)]


    fac_cols = ['niveau_couverture_oblg', 'type_bénéf']
    available_cols = [x for x in fac_cols if x in df_filtered.columns]
        
    for col in ['assureur','categorie_assuré','niveau_couverture'] + available_cols:
        df_filtered[col] = df_filtered[col].str.lower()
        
    if len(available_cols) > 0 :
            cols_unpiv = ['assureur','annee_comptable','annee_survenance', 'categorie_assuré','niveau_couverture'] + available_cols
            cols_group=['assureur', 'categorie_assuré', 'niveau_couverture','annee_survenance','type_montant'] + available_cols
            cols_filtered=['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance'] + available_cols
    else:
            cols_unpiv = ['assureur','annee_comptable','annee_survenance', 'categorie_assuré','niveau_couverture']
            cols_group= ['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance','type_montant']
            cols_filtered=['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance']
                
    if 'cotisations_base' in df.columns: 
        df_filtered = pd.melt(df_filtered, 
                id_vars=cols_unpiv, 
                value_vars=['cotisations_base', 'cotisations_fac'],
                var_name='type_montant', 
                value_name='cot_TTC')
            
        grouped = df_filtered.groupby(cols_group).agg(
                montant=('cot_TTC', 'sum')
            ).reset_index()
            
        if len(available_cols) > 0:
            cotisations = grouped.rename(columns={
                'assureur': 'assureur',
                'categorie_assuré': 'cat_assure',
                'niveau_couverture': 'niveau_couv',
                'annee_survenance': 'survenance',
                'type_montant':"type_montant"
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "cotisations": x[available_cols + ['cat_assure', 'niveau_couv','type_montant', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()
                
        else:
            cotisations = grouped.rename(columns={
                'assureur': 'assureur',
                'categorie_assuré': 'cat_assure',
                'niveau_couverture': 'niveau_couv',
                'annee_survenance': 'survenance',
                'type_montant':"type_montant"
                }).groupby('assureur').apply(lambda x: {
                "assureur": x['assureur'].iloc[0],
                "cotisations": x[['cat_assure', 'niveau_couv','type_montant', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()

    else:
        grouped = df_filtered.groupby(cols_filtered).agg(
                    montant=('cotisations_TTC', 'sum')
                ).reset_index()
            
        if len(available_cols) > 0:
            cotisations = grouped.rename(columns={
                'assureur': 'assureur',
                'categorie_assuré': 'cat_assure',
                'niveau_couverture': 'niveau_couv',
                'annee_survenance': 'survenance'
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "cotisations": x[available_cols + ['cat_assure', 'niveau_couv', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()
                
        else:
            cotisations = grouped.rename(columns={
                'assureur': 'assureur',
                'categorie_assuré': 'cat_assure',
                'niveau_couverture': 'niveau_couv',
                'annee_survenance': 'survenance'
                }).groupby('assureur').apply(lambda x: {
                "assureur": x['assureur'].iloc[0],
                "cotisations": x[['cat_assure', 'niveau_couv', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()

        
    json_result = {
            "ccn": ccn,
            "compta": compta,
            "data": cotisations
        }
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float64):
            return float(o)
        raise TypeError
        
    return json.loads(json.dumps(json_result, default=convert, indent=4))

def generate_json_p(df, ccn, compta):
    if not pd.api.types.is_integer_dtype(df['date_survenance']):
        df['annee_survenance'] = pd.to_datetime(df['date_survenance'], errors='coerce').dt.year
    if not pd.api.types.is_integer_dtype(df['date_comptable']):
        df['annee_comptable'] = pd.to_datetime(df['date_comptable'], errors='coerce').dt.year


    df_filtered = df[df['annee_comptable'] == int(compta)]

    fac_cols = ['niveau_couverture_oblg', 'type_bénéf']
    available_cols = [x for x in fac_cols if x in df_filtered.columns]
        
    for col in ['assureur', 'categorie_assuré', 'niveau_couverture'] + available_cols:
        df_filtered[col] = df_filtered[col].str.lower()
        
    if len(available_cols) > 0 :
        cols_unpiv = ['assureur', 'annee_comptable', 'annee_survenance', 'categorie_assuré', 'niveau_couverture'] + available_cols
        cols_group = ['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance', 'type_montant'] + available_cols
        cols_filtered = ['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance'] + available_cols
    else:
        cols_unpiv = ['assureur', 'annee_comptable', 'annee_survenance', 'categorie_assuré', 'niveau_couverture']
        cols_group = ['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance', 'type_montant']
        cols_filtered = ['assureur', 'categorie_assuré', 'niveau_couverture', 'annee_survenance']
        
    if 'remboursement_base' in df.columns: 
        df_filtered = pd.melt(df_filtered, 
                id_vars=cols_unpiv, 
                value_vars=['remboursement_base', 'remboursement_fac'],
                var_name='type_montant', 
                value_name='remboursement_TTC')
            
        grouped = df_filtered.groupby(cols_group).agg(
                montant=('remboursement_TTC', 'sum')
            ).reset_index()
            
        if len(available_cols) > 0:
            prestations = grouped.rename(columns={
                    'assureur': 'assureur',
                    'categorie_assuré': 'cat_assure',
                    'niveau_couverture': 'niveau_couv',
                    'annee_survenance': 'survenance',
                    'type_montant': 'type_montant'
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "prestations": x[available_cols + ['cat_assure', 'niveau_couv', 'type_montant', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()
        else:
            prestations = grouped.rename(columns={
                    'assureur': 'assureur',
                    'categorie_assuré': 'cat_assure',
                    'niveau_couverture': 'niveau_couv',
                    'annee_survenance': 'survenance',
                    'type_montant': 'type_montant'
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "prestations": x[['cat_assure', 'niveau_couv', 'type_montant', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()
    else:
        grouped = df_filtered.groupby(cols_filtered).agg(
                montant=('remboursement_TTC', 'sum')
            ).reset_index()
            
        if len(available_cols) > 0:
            prestations = grouped.rename(columns={
                    'assureur': 'assureur',
                    'categorie_assuré': 'cat_assure',
                    'niveau_couverture': 'niveau_couv',
                    'annee_survenance': 'survenance'
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "prestations": x[available_cols + ['cat_assure', 'niveau_couv', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()
        else:
            prestations = grouped.rename(columns={
                    'assureur': 'assureur',
                    'categorie_assuré': 'cat_assure',
                    'niveau_couverture': 'niveau_couv',
                    'annee_survenance': 'survenance'
                }).groupby('assureur').apply(lambda x: {
                    "assureur": x['assureur'].iloc[0],
                    "prestations": x[['cat_assure', 'niveau_couv', 'survenance', 'montant']].to_dict(orient='records')
                }).tolist()

    json_result = {
        "ccn": ccn,
        "compta": compta,
        "data": prestations
    }

    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float64):
            return float(o)
        raise TypeError

    return json.loads(json.dumps(json_result, default=convert, indent=4))

def calcule_cot_prest(cot_data, prest_data, params):
    taxes_v = params[0]["taxes"]
    DES_v = params[0]["DES"]
    autres_frais = params[0]["autres_frais"]
    forfaits_pat = params[0]["forfaits_pat"]

    df_cot = pd.json_normalize(cot_data["data"], "cotisations", ["assureur"])
    df_prest = pd.json_normalize(prest_data["data"], "prestations", ["assureur"])
    resultats_par_assureur = []

    cot_assureurs = set(df_cot["assureur"].unique())
    prest_assureurs = set(df_prest["assureur"].unique())
    all_assureurs = cot_assureurs.union(prest_assureurs)
    missing_in_cot = prest_assureurs - cot_assureurs
    missing_in_prest = cot_assureurs - prest_assureurs
    
    if missing_in_cot:
        print(f"Assureurs manquants dans les cotisations: {missing_in_cot}")
    if missing_in_prest:
        print(f"Assureurs manquants dans les prestations: {missing_in_prest}")
    

    missing_years = {
        "chargements": set(),
        "PSAP": set(),
        "autres_contributions": set(),
    }
    
    for assureur in all_assureurs:
        cotisations = []
        prestations = []
        
        df_cot_assureur = df_cot[df_cot["assureur"] == assureur]
        df_prest_assureur = df_prest[df_prest["assureur"] == assureur]
        
        regimes_cot = params[0]["régimes_cot"]
        regimes_prest = params[0]["régimes_prest"]
        
        # Calcul des cotisations
        for reg in regimes_cot:
            conditions = []
            for col, mod in reg.items():
                if col != "régime":
                    conditions.append(f'({col} == "{mod.lower()}")')
            condition_str = " & ".join(conditions)
            
            filtered_df_cot = df_cot_assureur.query(condition_str)
            
            df_cot_assr = (filtered_df_cot.groupby("survenance")["montant"].sum().reset_index())
            
            for row in df_cot_assr.itertuples(index=False):
                survenance = int(row.survenance)
                regime = reg.copy()
                taxes = row.montant - row.montant / (1 + taxes_v)
                cot_nettes_taxes = row.montant - taxes
                
                chargement = 0
                chargement_param = next(
                    (
                        c
                        for c in params[0]["chargements"]
                        if c["régime"] == regime["régime"]
                    ),
                    None,
                )
                if chargement_param and survenance in chargement_param["survenance"]:
                    chargement = chargement_param["survenance"][survenance]
                else:
                    missing_years["chargements"].add(survenance)
                
                DES = cot_nettes_taxes * DES_v
                chargement = cot_nettes_taxes * chargement
                cotisations_nettes = cot_nettes_taxes - DES - autres_frais - chargement
                
                cotisation_dict = {
                    "survenance": survenance,
                    "montant": row.montant,
                    "taxes": taxes,
                    "cot_nettes_taxes": cot_nettes_taxes,
                    "chargements": chargement,
                    "DES": DES,
                    "autres_frais": autres_frais,
                    "cotisations_nettes": cotisations_nettes,
                    "assureur": assureur,
                }
                cotisation_dict.update(regime)
                cotisations.append(cotisation_dict)
        
        # Calcul des prestations
        for reg in regimes_prest:
            conditions = []
            for col, mod in reg.items():
                if col != "régime":
                    conditions.append(f'({col} == "{mod.lower()}")')
            condition_str = " & ".join(conditions)
            
            filtered_df_prest = df_prest_assureur.query(condition_str)
            
            df_prest_assr = (filtered_df_prest.groupby("survenance")["montant"].sum().reset_index())
            
            for row in df_prest_assr.itertuples(index=False):
                survenance = int(row.survenance)
                regime = reg.copy()
                psap = 0
                psap_param = next(
                    (
                        p
                        for p in params[0]["PSAP"]
                        if p["régime"] == regime["régime"]
                    ),
                    None,
                )
                if psap_param and survenance in psap_param["survenance"]:
                    psap = psap_param["survenance"][survenance]
                else:
                    missing_years["PSAP"].add(survenance)
                
                autres_contributions = 0
                autres_contributions_param = next(
                    (
                        ac
                        for ac in params[0]["autres_contributions"]
                        if ac["survenance"] == survenance and ac["régime"] == regime["régime"]
                    ),
                    None,
                )
                if autres_contributions_param:
                    autres_contributions = autres_contributions_param["value"]
                else:
                    missing_years["autres_contributions"].add(survenance)
                    
                cot_nettes_taxes_prest = next(
                    (c["cot_nettes_taxes"] for c in cotisations if c["survenance"] == survenance and c["régime"] == regime["régime"]),
                    0
                )
                forfait_patientele = cot_nettes_taxes_prest * forfaits_pat
                prestation_dict = {
                    "survenance": survenance,
                    "montant": row.montant,
                    "PSAP": psap,
                    "forfait_patientele": forfait_patientele,
                    "autres_contributions": autres_contributions,
                    "assureur": assureur,
                }
                prestation_dict.update(regime)
                prestations.append(prestation_dict)
        
        for key, years in missing_years.items():
            if years:
                print(f"Il manque des informations de {key} pour les années : {sorted(years)}")
        
        resultats_par_assureur.append({
            "assureur": assureur,
            "cotisations": cotisations,
            "prestations": prestations
        })
    
    result = {
        "ccn": cot_data["ccn"],
        "compta": int(cot_data["compta"]),
        "data": resultats_par_assureur
    }
    
    return result


#MAPPINGS_FILE = r'C:\Users\Yacine AMMI\Yacine\Notebooks\Concat app\Outil Comptes\mappings.json'

def load_mappings(file_path):
    if not os.path.exists(file_path):
        return {
            'niveau_couverture': {},
            'categorie_assuré': {},
            'type_bénéf': {}
        }
    
    with open(file_path, 'r', encoding='utf-8') as file:
        mappings = json.load(file)
    return mappings

def save_mappings(file_path, mappings):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(mappings, file, ensure_ascii=False, indent=4)
        
def restore_editions(session_state_df, session_state_dict):
    for index, updates in session_state_dict["edited_rows"].items():
        for key, value in updates.items():
            session_state_df.loc[session_state_df.index == index, key] = value
    return pd.DataFrame(session_state_df)

def mise_en_forme_df(df, mappings, col):
    
    if col in df.columns:
        
        df_formatted = df.copy()
        mapper = mappings[col]
        
        maps_not_in_mapper = df.loc[
            df[col].notna() & 
            (~df[col].str.lower().isin(mapper.keys()))
        , col].str.lower().unique()
        
        if len(maps_not_in_mapper) > 0:
            warning(f"""{col} mapping interompu! Les valeurs {maps_not_in_mapper} ne sont pas disponibles!
            Veuillez modifier le dictionnaire des valeurs ou faire la transcodification manuellement.""")
        else:
            df_formatted[col] = df[col].str.lower().map(mapper)
            #success(f"{col} mapping avec succès!")
    
    return df_formatted

def edit_mapping(df, col, mapper):
    # Create a categorical series from the lowercase values
    cat_series = pd.Categorical(df[col].str.lower())

    # Create a mapping array
    unique_cats = cat_series.categories
    mapping_array = array([mapper.get(cat, nan) for cat in unique_cats])

    # Apply the mapping
    mapped_values = mapping_array[cat_series.codes]

    # Create the result DataFrame
    return pd.DataFrame({ 'Ancienne': cat_series, 'Nouvelle': mapped_values }).drop_duplicates()

# Function to check if all mappings are complete
def check_mappings_complete(mapping_df):
    return all(mapping_df['Nouvelle'].notna())

# def mise_en_forme_df(df):
    
#     df_formatted = df.copy()
    
#     niv_couv = {
#         'MH': {
#             "socle": "Socle", 
#             "socle seul": 'Socle',
#             "option 1 oblig": "Option 1" , 
#             "option 1 oblig sans opt2 fac": "Option 1",
#             "option 1 fac": "Option 1", 
#             "option 2 fac": "Option 2", 
#             "option 2 oblig": "Option 2",
#             "option 2 fac en cplt opt1 oblig": "Option 2"
#         }
#     }
    
#     cat_assuré = {
#         'MH': {
#             "actifs": "Actifs", 
#             "accueils": "Accueils",
#             'portabilité': "Actifs",
#             'loi evin': "Actifs"
#         }
#     }
    
#     type_bénéf = {
#         'MH': {
#             "assuré": "Assuré",
#             "conjoint": "Conjoint",
#             "enfant": "Enfant",
#             }
#     }
    
#     mapper = {
#         'niveau_couverture': niv_couv,
#         'categorie_assuré': cat_assuré,
#         'type_bénéf': type_bénéf
#     }
    
#     for assureur in df['assureur'].unique():
        
#         for col, map in mapper.items():
#             if col in df.columns:
#                 maps_not_in_mapper = df.loc[
#                     (df['assureur'] == assureur) & 
#                     (df[col].notna()) & 
#                     (~df[col].str.lower().isin(map[assureur].keys()))
#                 , col].str.lower().unique()
                
#                 if len(maps_not_in_mapper) > 0:
#                     warning(f"""{col} mapping interompu!, les valeurs {maps_not_in_mapper} de l'assureur {assureur} ne sont pas disponibles!
#                 Veuillez modifier le dictionnaire des valeurs ou faire la Transcodification manuellement""")
#                 else:
#                     df_formatted.loc[df_formatted['assureur'] == assureur, col] = df.loc[df['assureur'] == assureur, col].str.lower().map(map[assureur])
#                     success(f"{col} mapping de l'assureur {assureur} avec succès!")
    
#     return df_formatted