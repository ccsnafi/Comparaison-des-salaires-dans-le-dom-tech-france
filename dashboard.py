import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Configuration de la page Streamlit
st.set_page_config(page_title="Dashboard Salaires Data France vs USA", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("salaries.csv")

df = load_data()
df_fr_us = df[df['employee_residence'].isin(['FR', 'US'])].copy()

# Sidebar : Filtres
st.sidebar.header("Filtres")
jobs = st.sidebar.multiselect(
    "Choisissez un ou plusieurs métiers :",
    sorted(df_fr_us['job_title'].unique()),
    default=sorted(df_fr_us['job_title'].unique())[:2]
)
country = st.sidebar.multiselect("Pays :", ['FR', 'US'], default=['FR', 'US'])
year = st.sidebar.selectbox("Année :", sorted(df_fr_us['work_year'].unique()), index=0)

# Filtrage des données
filtered = df_fr_us[
    (df_fr_us['job_title'].isin(jobs)) &
    (df_fr_us['employee_residence'].isin(country)) &
    (df_fr_us['work_year'] == year)
]

# Choix de la coloration des courbes
with st.sidebar:
    st.markdown("---")
    color_by = st.radio(
        "Différencier les couleurs par :",
        options=["Pays", "Métier", "Pays + Métier"],
        index=0,
        help="Affiche les courbes colorées soit par pays, soit par métier, soit par les deux combinés."
    )

if color_by == "Pays":
    hue_col = "employee_residence"
elif color_by == "Métier":
    hue_col = "job_title"
else:
    filtered['pays_metier'] = filtered['employee_residence'] + " - " + filtered['job_title']
    hue_col = "pays_metier"

# Création des onglets
tab1, tab2, tab3 = st.tabs(["📊 Visualisations", "🔮 Projection salariale", "📝 Insights & Conclusion"])

with tab1:
    st.title("📊 Analyse des salaires data : France vs USA")
    st.write(f"**Métiers :** {', '.join(jobs) if jobs else '---'} | **Année :** {year} | **Pays :** {', '.join(country) if country else '---'}")
    st.markdown(f"**Nombre d'observations :** {len(filtered)}")

    if not jobs or len(filtered) == 0:
        st.warning("Sélectionnez au moins un métier et assurez-vous d'avoir des données pour votre filtre.")
    else:
        # Boxplot salaires par métier/pays/métier+pays
        st.subheader(f"Distribution des salaires (USD) par métier{' et pays' if color_by != 'Métier' else ''}")
        fig1, ax1 = plt.subplots(figsize=(10,5))
        sns.boxplot(
            data=filtered,
            x='job_title', y='salary_in_usd', hue=hue_col, ax=ax1
        )
        ax1.set_title(f"Salaires par métier ({year})")
        ax1.set_xlabel("Métier")
        ax1.set_ylabel("Salaire (USD)")
        ax1.legend(title=hue_col.replace("_", " ").capitalize(), loc='best')
        st.pyplot(fig1)

        # Impact du télétravail
        st.subheader("Impact du télétravail (%)")
        if filtered['remote_ratio'].nunique() > 0:
            fig2, ax2 = plt.subplots(figsize=(7,4))
            sns.boxplot(
                data=filtered,
                x='remote_ratio', y='salary_in_usd', hue=hue_col, ax=ax2
            )
            ax2.set_title("Salaire selon le télétravail")
            ax2.set_xlabel("Remote ratio (%)")
            ax2.set_ylabel("Salaire (USD)")
            ax2.legend(title=hue_col.replace("_", " ").capitalize(), loc='best')
            st.pyplot(fig2)
        else:
            st.info("Pas assez de données pour afficher le télétravail pour cette sélection.")

        # Niveau d'expérience sur tous les métiers sélectionnés
        st.subheader("Salaire selon le niveau d'expérience (tous métiers sélectionnés)")
        if filtered['experience_level'].nunique() > 0:
            fig_exp, ax_exp = plt.subplots(figsize=(9, 5))
            sns.boxplot(
                data=filtered,
                x='experience_level',
                y='salary_in_usd',
                hue=hue_col,
                ax=ax_exp
            )
            ax_exp.set_title("Salaire selon le niveau d'expérience")
            ax_exp.set_xlabel("Niveau d'expérience")
            ax_exp.set_ylabel("Salaire (USD)")
            ax_exp.legend(title=hue_col.replace("_", " ").capitalize(), loc='best')
            st.pyplot(fig_exp)
        else:
            st.info("Pas assez de données pour afficher le salaire par niveau d'expérience pour cette sélection.")

        # Taille d'entreprise sur tous les métiers sélectionnés
        st.subheader("Salaire selon la taille d'entreprise (tous métiers sélectionnés)")
        if filtered['company_size'].nunique() > 0:
            fig3, ax3 = plt.subplots(figsize=(9,5))
            sns.boxplot(
                data=filtered,
                x='company_size', y='salary_in_usd', hue=hue_col, ax=ax3
            )
            ax3.set_title("Salaire selon la taille d'entreprise")
            ax3.set_xlabel("Taille d'entreprise")
            ax3.set_ylabel("Salaire (USD)")
            ax3.legend(title=hue_col.replace("_", " ").capitalize(), loc='best')
            st.pyplot(fig3)
        else:
            st.info("Pas assez de données pour afficher la taille d'entreprise pour cette sélection.")

with tab2:
    st.title("🔮 Projection des salaires à venir (2026-2027)")

    if not jobs:
        st.warning("Sélectionnez au moins un métier.")
    else:
        for job in jobs:
            for c in country:
                sub = df_fr_us[(df_fr_us['job_title'] == job) & (df_fr_us['employee_residence'] == c)]
                trend = sub.groupby('work_year')['salary_in_usd'].mean().reset_index()
                if len(trend) < 2:
                    st.info(f"Pas assez de données pour projeter ({job}, {c})")
                    continue
                X = trend['work_year'].values.reshape(-1, 1)
                y = trend['salary_in_usd'].values
                model = LinearRegression().fit(X, y)
                future_years = np.array([[2026], [2027]])
                future_preds = model.predict(future_years)

                # Affichage de la courbe
                fig4, ax4 = plt.subplots(figsize=(7,4))
                ax4.plot(X, y, 'o-', label='Historique')
                ax4.plot(future_years, future_preds, 'r^--', label='Prévision')
                ax4.set_title(f"{job} ({c}) : évolution & projections")
                ax4.set_xlabel("Année")
                ax4.set_ylabel("Salaire moyen (USD)")
                ax4.legend()
                st.pyplot(fig4)
                for year_pred, pred in zip(future_years.flatten(), future_preds):
                    st.write(f"**Prévision {job} {c} {year_pred} : {int(pred):,} USD**")

        st.info("Projection basée sur une tendance linéaire des années précédentes. À utiliser comme tendance, pas comme vérité absolue.")

with tab3:
    st.title("📝 Insights clés & Conclusion")

    st.markdown("""
    ### Insights clés

    - Les salaires data sont nettement plus élevés aux USA qu’en France, en particulier pour les métiers techniques.
    - Le télétravail est plus répandu et mieux rémunéré aux États-Unis.
    - En France, l’écart de salaire entre juniors et seniors existe, mais il est moins extrême qu’aux USA.
    - La taille de l’entreprise influence les rémunérations dans les deux pays, mais le différentiel France/USA reste marqué à tous les niveaux.
    - La France affiche une progression des salaires dans la data, mais l’écart avec les USA demeure significatif.

    ### Conclusion

    Cette analyse montre la puissance de l’approche combinée Data Analyst (exploration, visualisation) et Data Scientist (projection) :
    - Les tendances haussières devraient se poursuivre, mais l’écart France/USA reste important à horizon 2027.
    - Les prévisions sont indicatives : elles reposent sur les évolutions passées et n’intègrent pas les changements économiques ou technologiques majeurs.

    > Ce projet illustre la valeur d’un Data Analyst moderne, capable d’apporter des insights business solides tout en s’ouvrant à la Data Science pour anticiper les évolutions du marché.
    """)
