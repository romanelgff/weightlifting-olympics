#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dans cette application Bokeh on s'intéressera aux compétitions d'haltérophilie de 1960 à 2016. Les données ont été sélectionnées de façon à ce que l'on ne se retrouve qu'avec les
athlètes médaillés (Bronze, Argent, Or). Deux onglets sont définis dans l'application. Le premier présente un graphique de l'évolution du poids des athlètes en haltérophilie, en 
fonction du type de médaille gagnée, de 1960 à 2016, groupés selon qu'ils soient hommes ou femmes. On remarque que les femmes ont intégré très tard les JO dans ce domaine, à l'an 
2000 seulement ! Nous avons donc malheureusement peu de données pour elles. Il est possible de choisir à la droite du graphe le type de médaille souhaité pour la représentation,
ainsi que de sélectionner la représentation pour les hommes, ou que pour les femmes en cliquant sur les cases à gauche de 'F' (femme) et 'H' (homme). Lorsqu'un seul genre est
sélectionné, une droite de régression linéaire apparaîtra pour chaque nouveau type de médaille choisi, ainsi que l'équation associée dans la légende. Finalement, en passant la 
souris sur les différents cercles du graphique, les informations suivantes s'affichent à l'écran: l'année, les pays participant cette année-là, la taille moyenne des athlètes et 
leur poids moyen.

Ce premier onglet sous-entend une corrélation positive entre le poids des athlètes masculins et l'année des jeux. A l'inverse, la régression pour le poids des femmes semble être 
corrélée négativement à l'année des JO, à l'exception des médaillées d'or. Nous n'avons néanmoins pas encore assez de données les concernant pour conclure.

Le deuxième onglet affiche une cartographie des différents pays participants à la catégorie d'haltérophilie. La taille des diamants représentant les pays est directement liée au
nombre de médailles gagnées par le pays. De plus, on peut sélectionner à droite de la carte un type de médaille pour connaître quel pays a déjà gagné une médaille d'or, d'argent 
ou de bronze. En passant la souris sur les pays, on peut en plus accéder à certaines informations: le pays, sa capitale, le nombre de médailles gagnées au total et le nombre de
médailles d'or, d'argent ou de bronze remportées selon le type sélectionné.

Cet onglet met en avant les pays dominants en haltérophilie, dont l'Union Soviétique a fait partie pendant des années avec un total de 47 médailles remportées dont 32 d'or ! La
Russie a pris le relais depuis et s'en sort plutôt bien, avec un total de 26 médailles dont 3 d'or depuis la dissolution de l'URSS en 1991. La Chine est également très proche
des résultats russes avec 55 médailles gagnées depuis la création des JO, dont 34 médailles d'or. Les pays européens sont de leur côté à la traîne.

Vous avez désormais toutes les cartes en main pour utiliser cette application Bokeh ! Bon visionnage.
"""

# Importing required packages
import pandas as pd
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import row, column
from bokeh.io import curdoc, show
from bokeh.models import  LinearInterpolator, CheckboxGroup, RadioGroup
from bokeh.models import ImageURL
from bokeh.models import HoverTool
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models.widgets import Dropdown
import numpy as np
import json


# Definition des fonctions
def coor_wgs84_to_web_mercator(lon, lat): #Converts decimal longitude/latitude to Web Mercator format
    k = 6378137
    x = lon * (k * np.pi/180.0)
    y = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
    return (x,y)

def newCountries(countries_list):
    myCountries = []
    index = 0
    for countries in countries_list:
        for country in countries:
            if country not in myCountries:
                myCountries.append(country)
        index += 1
    return myCountries

## Partie Line Chart #####################################

# Manipulation du data frame
df = pd.read_csv("athlete_events.csv", encoding = 'utf-8')

# Analyse du fichier JSON pour les regions du monde
fp = open("capitals.geojson", "r", encoding = 'utf-8')
dico = json.load(fp)

df_lift = df.loc[df.Sport=='Weightlifting'].loc[df.Year >= 1960].copy().dropna()
print(df_lift.iloc[0])

df_lift = df_lift.groupby(["Year", "Sex", "Medal"]).agg(Weight=('Weight','mean'), Height=('Height', 'mean'), Countries=('Team',list)).reset_index()
df_lift.Countries = [list(set(liste)) for liste in df_lift.Countries]

source = ColumnDataSource(df_lift.loc[df_lift.Medal == 'Bronze'])

# Regression lineaire pour le graphe
x=df_lift.loc[df_lift.Medal == 'Bronze'].loc[df_lift.Sex == 'M']['Year']
y=df_lift.loc[df_lift.Medal == 'Bronze'].loc[df_lift.Sex == 'M']['Weight']

# Determination du meilleur ajustement
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x] 
y_predicted = y_predicted + [y_predicted[-1]]*5

predict = ColumnDataSource({'Prediction':y_predicted, 'Year':x})

# Definir la figure
fig1 = figure(title = "Weight of weightlifters in the Olympic Games since the 1960s", width = 1000)

# Pour la checkbox plus tard !
url='http://img.over-blog-kiwi.com/0/66/87/83/obpicCYiRZL.jpeg'

image3 = ImageURL(url=dict(value=url), x=1980, y=70, anchor="bottom_left")
game = fig1.add_glyph(image3)
game.visible = False

# Taille des cercles fonction de la taille des athletes
size_mapper = LinearInterpolator(
    x=[df_lift.Height.min(),df_lift.Height.max()],
    y=[5,60]
)

# Affichage des donnees
fig1.circle_y(x = 'Year', y = 'Weight', source = source, legend = 'Sex', size = {'field':'Height', 'transform':size_mapper}, fill_alpha = 0.4, fill_color = factor_cmap('Sex', palette=['#DD1C77',"navy"], factors=['F','M']))

# Details et annotations pour la figure
fig1.legend.title = "Information"
fig1.legend.location = "bottom_left"
fig1.xaxis.axis_label = 'Year'
fig1.yaxis.axis_label = 'Weight'

# HoverTool
hover_tool = HoverTool(tooltips=[("Countries", '@Countries'), ("Height (cm)", '@Height'), ("Weight (kg)", '@Weight'), ("Year", '@Year')])
fig1.add_tools(hover_tool)

#show(fig1)

# Ajout d'un bouton radio et d'une check box
bouton_radio = RadioGroup(labels=["Gold", "Silver", "Bronze"], active=2)
sex_selection = CheckboxGroup(labels=['F','M'], active=[0,1])

# Definition des callback functions
def callback_radio(new):
    
    valeurs = ['Gold','Silver','Bronze']
    val = valeurs[bouton_radio.active]
    game.visible = False
    
    if sex_selection.active == [1]:
        df_tokeep = df_lift.loc[df_lift.Medal == val].loc[df_lift.Sex == 'M']
        x = df_tokeep['Year']
        y = df_tokeep['Weight']

        par = np.polyfit(x, y, 1, full=True)
        slope=par[0][0]
        intercept=par[0][1]
        y_predicted = [slope*i + intercept  for i in x]
        
        leg = 'y='+str(round(slope,2))+'x'+str(round(intercept,2))
        if val == 'Silver':
            col = 'pink'
        elif val == 'Bronze':
            col = 'chocolate'
        else:
            col = 'red'
            
        predict.data.update(dict(Prediction=y_predicted, Year=x))
        fig1.line(x='Year', y='Prediction', color=col, legend_label=leg, source = predict)
    
    elif sex_selection.active == [0]:
        df_tokeep = df_lift.loc[df_lift.Medal == val].loc[df_lift.Sex == 'F']
        x = df_tokeep['Year']
        y = df_tokeep['Weight']
        
        par = np.polyfit(x, y, 1, full=True)
        slope=par[0][0]
        intercept=par[0][1]
        y_predicted = [slope*i + intercept  for i in x]
        
        leg = 'y='+str(round(slope,2))+'x'+str(round(intercept,2))
        if val == 'Silver':
            col = 'purple'
        elif val == 'Bronze':
            col = 'silver'
        else:
            col = 'green'
            
        predict.data.update(dict(Prediction=y_predicted, Year=x))
        fig1.line(x='Year', y='Prediction', color=col, legend_label=leg, source = predict)
    
    elif sex_selection.active == [0,1]:
        df_tokeep = df_lift.loc[df_lift.Medal == val]

    medal = df_tokeep.loc[df_lift.Medal == val]['Medal']
    weight = df_tokeep.loc[df_lift.Medal == val]['Weight']
    height = df_tokeep.loc[df_lift.Medal == val]['Height']
    sex = df_tokeep.loc[df_lift.Medal == val]['Sex']
    countries = df_tokeep.loc[df_lift.Medal == val]['Countries']
    year = df_tokeep.loc[df_lift.Medal == val]['Year']
    
    new_data = dict(Year=year, Sex=sex, Medal=medal, Weight=weight, Height=height, Countries=countries)
    source.data.update(new_data)

bouton_radio.on_click(callback_radio)

def update(attr, old, new):
    
    sex_to_plot = [sex_selection.labels[i] for i in sex_selection.active]
    game.visible = False    
    
    if bouton_radio.active == 0:
        val = 'Gold'
    elif bouton_radio.active == 1:
        val = 'Silver'
    else:
        val = 'Bronze'
    
    if sex_selection.active == []:
        game.visible = True
    
    df = df_lift.loc[df_lift.Medal == val]
    df_tokeep = df[df.Sex.isin(sex_to_plot)]
    donnees2 = ColumnDataSource(df_tokeep)
    source.data.update(donnees2.data)
    
sex_selection.on_change('active', update)

## Partie Cartographie #############################

# Liste de pays participant aux competitions d'halterophilie
myCountries = newCountries(df_lift['Countries'])

countries = []
capitals = []
coordx = []
coordy=[]

# On prepare la construction d’un ColumnDataFrame, en convertissant les coordonnees
for p in dico["features"]:
    print(p["properties"]["country"])
    countries.append(p["properties"]["country"])
    capitals.append(p["properties"].get("city",None))
    X,Y = coor_wgs84_to_web_mercator(p["geometry"]["coordinates"][0],p["geometry"]["coordinates"][1])
    coordx.append(X)
    coordy.append(Y)

# Correction de certains pays avant mise dans le dictionnaire
countries[42] = 'Great Britain'
countries[167] = 'South Korea'
countries[169] = 'North Korea'

DicoCountries = pd.DataFrame({'countries':countries, 'capitals':capitals, 'coordx': coordx, 'coordy':coordy})

# Ajout de l'union soviétique (car tres importante en halterophilie)
coordxRussia = float(DicoCountries[DicoCountries.countries.isin(['Russia'])]['coordx'])+300000
coordyRussia = float(DicoCountries[DicoCountries.countries.isin(['Russia'])]['coordy'])+300000
DicoCountries = DicoCountries.append({'countries':'Soviet Union', 'capitals':'Moscow', 'coordx': coordxRussia, 'coordy':coordyRussia}, ignore_index=True)

# Creation d'une liste de pays participant aux jeux olympiques d'halterophilie avec leur coordonnees
olymp_countries = DicoCountries[DicoCountries.countries.isin(myCountries)]

# Nouvelle base de donnees a manipuler !
df_liftbis = df.loc[df.Sport=='Weightlifting'].loc[df.Year >= 1960].copy().dropna()

# Colonnes indiquant si oui ou non le pays a eu une (ou plus) medaille d'or, d'argent ou de bronze
df_liftbis['gold'] = df_liftbis['Medal'].apply(lambda x: 1 if x == 'Gold' else 0)
df_liftbis['silver'] = df_liftbis['Medal'].apply(lambda x: 1 if x == 'Silver' else 0)
df_liftbis['bronze'] = df_liftbis['Medal'].apply(lambda x: 1 if x == 'Bronze' else 0)

df_liftbis = df_liftbis.groupby(['Team']).agg(Medal=('Medal','count'), Gold=('gold', 'sum'), Silver=('silver', 'sum'), Bronze=('bronze', 'sum')).reset_index()
df_liftbis = df_liftbis.rename(columns = {'Team': 'countries'}, inplace = False)

# Dataframe final sur lequel on s'appuiera par la suite
result = df_liftbis.merge(olymp_countries, how = 'left', on = ['countries'])
source2 = ColumnDataSource(result)

# Liste de couleurs (si le pays a deja eu une medaille d'or = 'gold', sinon 'navy')
color = list(result['Gold'].apply(lambda x: 'gold' if x > 0 else 'navy'))
source2.add(color, "color")

# On definit la figure de cartographie
fig2 = figure(x_axis_type="mercator", y_axis_type="mercator", active_scroll="wheel_zoom", title="Medals won in weightlifting by participating countries", width=1000)
tile_provider = get_provider(Vendors.CARTODBPOSITRON)
fig2.add_tile(tile_provider)

# HoverTool
hover_tool2 = HoverTool(tooltips=[('Country','@countries'), 
                                  ('Capital','@capitals'), 
                                  ('Medals total number','@Medal'),
                                  ('Gold medals', '@Gold')])

# Colonne 'taille' donnant la taille des points sur la carte
taille = df_liftbis['Medal'].apply(lambda x: x*0.5 + 20 if x > 0 else 0)
source2.add(taille, "taille")

# Affichage des donnees
fig2.diamond(x = "coordx", y = "coordy", size = 'taille', source = source2, fill_color = 'color')
fig2.add_tools(hover_tool2)

#show(fig2)

#Creation du widgets menu
menu = Dropdown(label = "Choix des médailles", menu = ['Gold', 'Silver', 'Bronze'])

#Definition des callback functions
def callback_menu(new): # menu deroulant
    print(new.item)
    
    if new.item == 'Silver':
        col = 'silver'
    elif new.item == 'Bronze':
        col = 'chocolate'
    else:
        col = 'gold'
    
    hover_tool2.update(tooltips=[('Country','@countries'), 
                                  ('Capital','@capitals'), 
                                  ('Medals total number','@Medal'),
                                  (new.item+' medals', '@'+new.item)])
    
    new_color = list(result[new.item].apply(lambda x: col if x > 0 else 'navy'))
    
    source2.data['color'] = new_color # remplace le label de l'axe d'ordonnee

menu.on_click(callback_menu) 

#Preparation des onglets avec les widgets
layout = row(fig1, column(bouton_radio, sex_selection))
layout2 = row(fig2, menu)

# Mise en onglet
tab1 = Panel(child = layout, title = "Weight evolution of athletes in weightlifting") 
tab2 = Panel(child = layout2, title = "Olympic weightlifters map")
tabs = Tabs(tabs = [tab1, tab2])

curdoc().add_root(tabs)