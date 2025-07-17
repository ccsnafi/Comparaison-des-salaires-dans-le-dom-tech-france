import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Configuration (optionnel)
st.set_page_config(page_title="Dashboard Salaires Data France vs USA", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("salaries.csv")

df = load_data()
df_fr_us = df[df['employee_residence'].isin(['FR', 'US'])].copy()

# Sidebar : Filtres
st.sidebar.header("Filtres")
job = st.sidebar.selectbox("Choisissez un métier :", sorted(df_fr_us['job_title'].unique()), index=0)
country = st.sidebar.multiselect("Pays :", ['FR', 'US'], default=['FR', 'US'])
year = st.sidebar.selectbox("Année :", sorted(df_fr_us['work_year'].unique()), index=0)

# Filtrage principal
filtered = df_fr_us[
    (df_fr_us['job_title'] == job) &
    (df_fr_us['employee_residence'].isin(country)) &
    (df_fr_us['work_year'] == year)
]

# Tabs principaux
tab1, tab2, tab3 = st.tabs(["📊 Visualisations", "🔮 Projection salariale", "📝 Insights & Conclusion"])

with tab1:
    st.title("📊 Analyse des salaires data : France vs USA")
    st.write(f"**Métier :** {job} | **Année :** {year} | **Pays :** {', '.join(country)}")
    st.markdown(f"**Nombre d'observations** : {len(filtered)}")
    
    # Boxplot salaires
    st.subheader("Distribution des salaires (USD)")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.boxplot(data=filtered, x='employee_residence', y='salary_in_usd', ax=ax1)
    ax1.set_title(f"Salaire {job} : France vs USA ({year})")
    ax1.set_xlabel("Pays")
    ax1.set_ylabel("Salaire (USD)")
    st.pyplot(fig1)

    # Barplot remote ratio
    st.subheader("Impact du télétravail (%)")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.boxplot(data=filtered, x='remote_ratio', y='salary_in_usd', hue='employee_residence', ax=ax2)
    ax2.set_title("Salaire selon le télétravail")
    ax2.set_xlabel("Remote ratio (%)")
    ax2.set_ylabel("Salaire (USD)")
    st.pyplot(fig2)

    # Taille entreprise
    st.subheader("Taille d'entreprise et salaires")
    fig3, ax3 = plt.subplots(figsize=(7,4))
    sns.boxplot(data=filtered, x='company_size', y='salary_in_usd', hue='employee_residence', ax=ax3)
    ax3.set_title("Salaire selon la taille d'entreprise")
    ax3.set_xlabel("Taille d'entreprise")
    ax3.set_ylabel("Salaire (USD)")
    st.pyplot(fig3)

with tab2:
    st.title("🔮 Projection des salaires à venir (2026-2027)")

    for c in country:
        # Filtrer par métier/pays
        sub = df_fr_us[(df_fr_us['job_title'] == job) & (df_fr_us['employee_residence'] == c)]
        if sub['work_year'].nunique() < 2:
            st.info(f"Pas assez de données pour projeter ({job}, {c})")
            continue
        trend = sub.groupby('work_year')['salary_in_usd'].mean().reset_index()
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
            st.write(f"**Prévision {c} {year_pred} : {int(pred):,} USD**")

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
