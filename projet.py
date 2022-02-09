import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate



#DATASET
df = pd.read_csv('ecoclean2.csv', sep =',', error_bad_lines=False) 
df.set_index('Date', inplace = True)
df.index = pd.to_datetime(df.index)
df['Année'] = df['Année'].astype(str)

df_r = pd.read_csv('regionclean2.csv', sep =',', error_bad_lines=False) 
df_r.drop('Unnamed: 0', axis=1, inplace=True)
df_r['Année']=df_r['Année'].astype(str)

datas=pd.read_csv('datasdjuclean.csv', sep=',')
datas.set_index('Periode', inplace = True)
datas.index = pd.to_datetime(datas.index)

colors = ['#19D3F3','#FECB52','#636EFA','#AB63FA','#00CC96','#EF553B' ]
###MAIN###
st.set_page_config(page_title='Projet',
                   layout="wide")

st.sidebar.image('logo.png', width=200)
st.sidebar.header('Projet Eco2mix')
st.sidebar.markdown('Sommaire')


menu = st.sidebar.radio(
    "Ou voulez vous allez? ",
    ("Accueil","L'électricité : les enjeux", "À l'échelle nationale", "À l'échelle régionale","Prédiction de consommation")
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Tom Morel - Florian Maury | Projet Eco2mix Datascientest')






###INTRO###
if menu == 'Accueil':
  st.image("bdd.png")
  st.header('Projet Energie') #ou title ?
  st.write("Ce streamlit a pour intérêt de présenter notre projet concernant la production, la consommation, et l'évolution du mix énergétique francais depuis 2013.")
  st.title("Objectifs :")
  st.write("1) Constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional.")
  st.write("2) Analyse au niveau national afin d'en déduire une prévision de consommation.")
  st.write("3) Analyse par filière de production : énergies renouvelables / nucléaire.")
  st.write("4) Focus sur les énergies renouvelables, à l’échelle nationale et régionale.")
  
  st.title("Problématique :")
  st.write("Observation entre consommation et production, en distinguant la part du nucléaire et la part des ENR dans le mix énergétique, au niveau régional et national, dans le temps depuis 2013.")
  st.subheader("Les bases de données exploitées :")


  st.write("Données éco2mix RTE")
  st.write(df.head(12))

  st.write("Données régions")
  st.write(df_r.head(5))

  st.write("Données DJU")
  st.write(datas.head(5))

### L'enjeu électrique###

elif menu == "L'électricité : les enjeux" : 
    st.header("L'électricité : les enjeux")
    question = st.selectbox("Que voulez-vous savoir ?",['Comment produit-on de l’électricité ?','Comment achemine t-on l’électricité ?','Qu’est ce que le Blackout ?','Qu’est ce qu’une énergie fossile ?','Qu’est ce qu’une énergie renouvelable ?','Quels sont les intérêts du mix électrique ?'])
    
    if question == "Comment produit-on de l’électricité ?": 
        st.title("Comment produit-on de l’électricité ?")
        
        question_type_nrj = st.selectbox("Quel moyen de production d'électricité vous intéresse ?",["Les centrales thermiques","à partir des mouvements de l'eau","Les éoliennes","à partir de la biomasse","Les panneaux photovoltaïque"])
        
        
        
        if question_type_nrj =="Les centrales thermiques":
            col1, col2 = st.columns(2)
            with col1:
              st.write("Par la combustion de matière fossile comme le charbon, le pétrole ou le gaz (ou l’uranium pour les centrales nucléaires) de la chaleur se dégage. Les centrales thermiques sont équipées de chaudières dans lesquelles est réchauffée l’eau froide. En brûlant, les combustibles dégagent de la chaleur. Celle-ci permet de chauffer l’eau et de la transformer en vapeur, à l’image de l’eau mise dans une casserole réchauffée, sous pression, la vapeur met en mouvement une turbine qui à son tour entraîne un alternateur (c’est un convertisseur d’énergie synergique en énergie électrique) producteur d’électricité. Dans les centrales nucléaires, il existe plusieurs circuits indépendants pour produire l’électricité. Ce courant est ensuite dirigé vers un transformateur qui va élever la tension du courant produit et lui permettre d’être diffusé sur les lignes à très haute tension du réseau de transport électrique.")
            with col2:
              st.image('thermique2.png', width=300)
              st.image('thermique.png', width=300)
              
        elif question_type_nrj =="à partir des mouvements de l'eau":
            col1, col2 = st.columns(2)
            with col1:
              st.write("Dans une centrale hydraulique")
              st.write("La production d’électricité grâce à l’énergie hydraulique dépend du mouvement de l’eau. Par exemple, la force de l’eau qui coule dans un fleuve crée du courant et active la rotation d’une turbine. La force d’une chute d’eau qui coule entre deux niveaux de hauteur provoque également la rotation d’une turbine. L’eau qui se déplace dans un fleuve peut être retenue en grande quantité dans un barrage. Lors de l’ouverture des vannes, l’eau s’écoule dans des tuyaux pour rejoindre la centrale. Ainsi, le mouvement de l’eau fait tourner une turbine, qui entraîne à son tour un alternateur qui produit de l’électricité. Le courant est ensuite dirigé vers un transformateur, qui augmente la tension de l’électricité pour lui permettre d’être diffusée dans les lignes à haute tension du réseau de distribution d’électricité.")

              st.write("Dans une centrale d’énergie marine")
              st.write("Cette énergie est en cours de développement et nécessite encore des avancées technologiques. La mer est riche en énergies exploitables sous diverses formes : énergie des courants marins, énergie des vagues, énergie des marées, énergie exploitant la différence de température entre surface et grande profondeur, biomasse marine et énergie osmotique (différence de salinité). Ces flux énergétiques permettent de produire de l’électricité à partir de la force de l’eau. Le développement des énergies marines permet de réduire la part d’énergies fossiles dans la consommation d’énergie globale, et donc de diminuer nos émissions de gaz à effet de serre. L’électricité produite est ensuite diffusée sur les lignes à haute tension du réseau de distribution d’électricité.")

            with col2:
              #st.image('hydraulique.png', width=300)
              st.image('hydraulique2.jpg', width=300)
              
        elif question_type_nrj =="Les éoliennes":
            col1, col2 = st.columns(2)
            with col1:
              st.write("Les éoliennes installées sur terre ou en mer transforment l’énergie du vent en électricité. Une éolienne est constituée d’un mat et d’une hélice (ou pales du rotor). Lorsque le vent souffle, l’hélice tourne. Comme les moulins à vent, l’énergie éolienne dépend de la force du vent. L’énergie produite passe alors par un multiplicateur, une sorte de boite de vitesse, qui accélère la vitesse du rotor. Dans son mouvement il entraîne une génératrice (une grosse dynamo), qui, en tournant, convertit le vent et produit de l’électricité. Le courant produit descend ensuite le long de la tour et rejoint le réseau de distribution d’électricité.")
            with col2:
              st.image('eolienne.jpg', width=300)  
                
        elif question_type_nrj =="à partir de la biomasse":
            col1, col2 = st.columns(2)
            with col1:
              st.write("La plus ancienne énergie utilisée par l’homme est la combustion de matières organiques comme le bois, les végétaux, les déchets agricoles ... Ces matières, une fois brûlées, dégagent de la chaleur. Cette chaleur permet de faire chauffer de l’eau dans une chaudière qui produit de la vapeur. En suivant le même principe que les centrales thermiques, la vapeur dégagée va permettre la rotation d’une turbine, qui va entrainer un alternateur producteur d’électricité. L’électricité ne pouvant être contenue va être envoyée vers les lignes à haute tension du réseau de distribution électrique.")
            with col2:
              st.image('biomasse.jpg', width=300)    
              
        elif question_type_nrj =="Les panneaux photovoltaïque":
            col1, col2 = st.columns(2)
            with col1:
              st.write("La production d’électricité à partir de l’énergie solaire se fait au moyen du procédé photovoltaïque utilisé sur les panneaux solaires. Le phénomène de transformation de la lumière en électricité est ce que l’on appelle « l’effet photovoltaïque ». Un panneau solaire est composé de plusieurs cellules. Ces cellules sont composées de silicium, un matériau semi-conducteur contenu dans le sable. Les rayons du soleil viennent heurter la surface des cellules et mettre en mouvement les électrons. Ces électrons vont se déplacer vers d’autres noyaux en créant du courant électrique. La production du courant continu est relayée vers un onduleur (convertisseur) puis convertie en courant alternatif pour être compatible avec le réseau de distribution d’électricité.")
            with col2:
              st.image('photovoltaique.jpg', width=300) 
            
            
            
            
            
            
    elif question == "Comment achemine t-on l’électricité ?":
        st.title("Comment achemine t-on l’électricité ?")
        col1, col2 = st.columns(2)
        with col1:
            st.write("L’électricité circule depuis l’endroit où elle est produite jusqu’à l’endroit où elle est consommée. Le transport de l’électricité se fait grâce à un réseau de grand transport et d’interconnexion et à un réseau de distribution :")
            st.write("Le réseau de grand transport achemine l’électricité produite à la sortie des centrales sur de longues distances grâce à des lignes à Très Haute Tension (entre 225 000 et 400 000 volts).")
            st.write("L’électricité est ensuite dirigée vers un poste de transformation dit « poste source », qui transforme la Très Haute Tension en Haute Tension (environ 90 000 volts) et en Moyenne Tension (20 000 volts). On compte environ 2 200 postes sources en France.")
            st.write("L’électricité transformée à Moyenne Tension est ensuite acheminée sur le réseau de distribution et peut être à son tour transformée en Basse Tension (entre 230 et 400 volts) grâce aux 750 000 postes de transformation dits « postes de distribution » présents sur le réseau français.")
            st.write("L’électricité Basse Tension est ensuite acheminée jusque vers les 35 millions de foyers français desservis.")
            st.write("Pour transporter l’électricité, il faut bien évidemment des câbles conducteurs. Ceux-ci sont généralement composés d’un conducteur en métal (cuivre ou, pour les nouvelles générations, alliage d’aluminium), d’une couche d’isolation et d’une gaine de protection.")
            
            st.write("L'un des enjeux du transport de l'électricité est de limiter le plus que possible la distance à partir entre le lieu de production et le lieu de consommation, car il y a des pertes par rapport à la distance parcourue.")
        with col2:
            st.image('transport.jpg', width=300)
         
        
    elif question == "Qu’est ce que le Blackout ?":
          st.title("Qu’est ce que le Blackout ?")
          col1, col2 = st.columns(2)
          with col1:
              st.write("Le « blackout » est le terme anglo-saxon désignant une coupure généralisée de l’approvisionnement en électricité sur toute ou partie d’un territoire. Cette coupure est due à un déséquilibre sur le réseau où la demande est fortement supérieure à la capacité de production.")
              st.write("Des mesures existent en France pour compenser ce manque. Tout d’abord, la plus connue, est l’importation d’électricité auprès des pays frontaliers dont l’Allemagne et l’Espagne (entre 5 et 7 GW sur les 12GW théoriques, soit l’équivalent des réacteurs nucléaires à l’arrêt).") 
              st.write("De plus, RTE peut réduire de 5% la tension fournie sur le réseau. Cette action quasiment imperceptible et sans réel danger pour nos équipements domestiques permettra une réduction d’environ 4 000 MW (4 GW) soit la consommation de Paris intra-muros et de Marseille.")
              st.write("Enfin, en cas de dernier recours, RTE peut demander aux différents gestionnaires du réseau de distribution de réaliser des coupures localisées tournantes de maximum 2h. La Bretagne et le Sud Est de la France seraient particulièrement touchés par le délestage du fait d’un faible niveau de production d’électricité dans ces zones.")
          with col2:
              st.image('blackout.jpg', width=300)
          st.title("La problématique du stockage de l'électricité")
          st.image('hydrogene.jpg')
          st.write("L’électricité est un vecteur très pratique pour le transport de l’énergie, mais difficile à stocker sous sa forme propre. Elle est donc généralement transformée pour être stockée sous une autre forme : énergie mécanique, thermique ou chimique par exemple. Au contraire, l’énergie thermique est généralement stockée sous sa forme originale (chaleur).")

          st.write("Exemples de systèmes pour l’énergie électrique :")

          st.write("Stockage gravitaire de masse d’eau avec les stations de transfert d’énergie par pompage (STEP) ;")
          st.write("Stockage thermodynamique avec les systèmes de stockage par air comprimé (CAES) ;")
          st.write("Stockage d’énergie cinétique avec les volants d’inertie ;")
          st.write("Stockage électrochimique avec les batteries (au plomb, sodium-soufre, lithium-ion,etc.).")
          st.write("Stockage 'power to gas' consiste à transformer de l'électricité en hydrogène par électrolyse de l'eau afin de la stocker à un moment où elle est excédentaire sur le réseau ")
    
              
    elif question == "Qu’est ce qu’une énergie fossile ?":
          st.title("Qu’est ce qu’une énergie fossile ?")
          col1, col2 = st.columns(2)
          with col1:
              st.write("On appelle « énergie fossile » l’énergie produite par la combustion du charbon, du pétrole ou du gaz naturel. Ces combustibles, riches en carbone et hydrogène, sont issus de la transformation de matières organiques enfouies dans le sol pendant des millions d’années (d'où le terme 'fossiles'). Ce sont des énergies non renouvelables puisqu’une fois utilisées, elles ne peuvent être reconstituées qu'à l'échelle des temps géologiques.")
              st.write("On appelle hydrocarbures des composés chimiques dont les molécules sont constituées d'atomes de carbone et d'hydrogène. Ce sont les principaux constituants du pétrole brut et du gaz naturel, ainsi que des produits pétroliers issus des raffineries.")
              st.write("La combustion d’énergie fossile est responsable de plus de 80 % des émissions de CO2 dans le monde.")
          with col2: 
              
              st.image("fossile2.jpg", width=300)
          st.image('partfossile.png',width=700)    
              
    elif question == "Qu’est ce qu’une énergie renouvelable ?":
        st.title("Qu’est ce qu’une énergie renouvelable ?")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Une énergie est dite renouvelable lorsqu'elle est produite par une source que la nature renouvelle en permanence, contrairement à une énergie dépendant de sources qui s’épuisent. Les énergies renouvelables sont très diverses mais elles proviennent toutes de deux sources naturelles principales : La Terre et le Soleil")
            st.write("Le caractère renouvelable de ces énergies, leur faible émission de déchets, de rejets polluants et de gaz à effet de serre sont des avantages. Mais leur pouvoir énergétique, relativement disséminé, est beaucoup plus faible que celui des énergies non renouvelables fortement concentrées.")
        with col2:
            st.image('ENR.jpg', width=300)
            
            
    elif question == "Quels sont les intérêts du mix électrique ?":
        st.title("Quels sont les intérêts du mix électrique ?")
        st.write("Le mix électrique est un concept qui peut s’appliquer à un pays, une zone géographique ou à une entreprise.")
        st.write("Pour un pays ou une zone géographique, le mix électrique représente la répartition de la production d’électricité selon les modes de production présents sur le territoire.")
        st.image("mix.png")



###VISUALISATION###

elif menu == "À l'échelle nationale":

  st.title("À l'échelle nationale:")
  st.subheader("La consommation et la production d'énergie électrique en France.")
  #COURBE
  r = df.groupby(['id_mois','Mois'], as_index = False).agg({'Consommation (MW)' : 'mean',
                                                            'Production Total (MW)' : 'mean',
                                                            'Production Total ENR (MW)' : 'mean',
                                                            'Production Total ENNR (MW)' : 'mean'})
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Consommation (MW)'],
                           mode='lines+markers',
                           name='Consommation (MW)'))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Production Total (MW)'],
                           mode='lines+markers',
                           name='Production Total (MW)'))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Production Total ENR (MW)'],
                           mode='lines+markers',
                           name='Production Total ENR (MW)'))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Production Total ENNR (MW)'],
                           mode='lines+markers',
                           name='Production Total ENNR (MW)'))
  fig.update_layout(
      title_text="Courbe de la consommation et de la production d'electricité en France depuis 2013")
  fig.update_xaxes(rangeslider_visible=True)
  st.plotly_chart(fig, use_container_width=True)


  #BOXPLOT
  r = df.groupby(['Date','Année'], as_index = False).agg({'Consommation (MW)' : 'mean',
                                                  'Production Total (MW)' : 'mean',
                                                  'Ech. physiques (MW)' : 'mean'})
  fig = go.Figure()
  fig.add_trace(go.Box(
    y=r['Consommation (MW)'],
    x=r['Année'],
    name='Consommation',
    marker_color='#636EFA'
  ))
  fig.add_trace(go.Box(
    y=r['Production Total (MW)'],
    x=r['Année'],
    name='Prod',
    marker_color='#FF4136'
  ))
  fig.update_layout( 
    title_text="Représentation statistique de la Consommation et Production",
    yaxis_title='MW',
    boxmode='group' # group together boxes of the different traces for each value of x
  )
  st.plotly_chart(fig, use_container_width=True)

  st.subheader("La production des différentes énergies en France.")
  
  #PIECHARTS
  col1, col2 = st.columns([3, 1])

  with col2:
    result1 = st.selectbox("Choisir une année?",['2013','2014','2015','2016','2017','2018','2019','2020','2021'])
    result2 = st.selectbox("Choisir l'année à comparer?",['Cliquez ici','2013','2014','2015','2016','2017','2018','2019','2020','2021'])

  if result2 == 'Cliquez ici':
    with col1:
      r = df.groupby(['Année']).agg({'Eolien (MW)':'mean',
                                     'Solaire (MW)' : 'mean',
                                     'Hydraulique (MW)' : 'mean',
                                     'Bioénergies (MW)' : 'mean', 
                                     'Nucléaire (MW)' : 'mean',
                                     'Thermique (MW)' : 'mean' 
                                     })
      labels = ["Eolien", "Solaire", "Hydraulique", "Bioénergie", "Nucléaire", "Thermique" ]
      fig5 = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
      fig5.add_trace(go.Pie(labels=labels, values=r[r.index == result1].values.tolist()[0], name=result1, marker_colors=colors),
                    1, 1)
      fig5.update_traces(hole=.25, hoverinfo="label+percent+name")
      fig5.update_layout(
          title_text="Repartion de la production d'electricité des énergies en France en " + result1 ,
          # Add annotations in the center of the donut pies.
          annotations=[dict(text=result1, x=0.5, y=0.5, font_size=20, showarrow=False)],
                       )

      st.plotly_chart(fig5, use_container_width=True)

  else:

    with col1:
      
      r = df.groupby(['Année']).agg({'Eolien (MW)':'mean',
                                     'Solaire (MW)' : 'mean',
                                     'Hydraulique (MW)' : 'mean',
                                     'Bioénergies (MW)' : 'mean', 
                                     'Nucléaire (MW)' : 'mean',
                                     'Thermique (MW)' : 'mean' 
                                     })
      labels = ["Eolien", "Solaire", "Hydraulique", "Bioénergie", "Nucléaire", "Thermique" ]

      # Create subplots: use 'domain' type for Pie subplot
      fig5 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
      fig5.add_trace(go.Pie(labels=labels, values=r[r.index == result1].values.tolist()[0], name=result1, marker_colors=colors),
                    1, 1)
      fig5.add_trace(go.Pie(labels=labels, values=r[r.index == result2].values.tolist()[0], name=result2, marker_colors=colors),
                    1, 2)

      # Use `hole` to create a donut-like pie chart
      fig5.update_traces(hole=.25, hoverinfo="label+percent+name")

      fig5.update_layout(
          title_text="Repartion de la production d'electricité des énergies en France en " + result1 + " et " + result2,
          # Add annotations in the center of the donut pies.
          annotations=[dict(text=result1, x=0.18, y=0.5, font_size=20, showarrow=False),
                       dict(text=result2, x=0.82, y=0.5, font_size=20, showarrow=False)])

      st.plotly_chart(fig5, use_container_width=True)


  #BARRES
  r = df.groupby(['Année'], as_index = False).agg({'Eolien (MW)':'mean',
                                                 'Solaire (MW)' : 'mean',
                                                 'Hydraulique (MW)' : 'mean',
                                                 'Bioénergies (MW)' : 'mean',
                                                 'Nucléaire (MW)' : 'mean',
                                                 'Thermique (MW)' : 'mean',
                                                 'Nucléaire (MW)' : 'mean'
                                                 })

  
  fig = go.Figure()
  fig.add_trace(go.Bar(x=r['Année'], y=r['Nucléaire (MW)'], name='Nucléaire', marker_color=colors[4]))
  fig.add_trace(go.Bar(x=r['Année'], y=r['Thermique (MW)'], name='Thermique', marker_color=colors[5]))
  fig.add_trace(go.Bar(x=r['Année'], y=r['Hydraulique (MW)'], name='Hydraulique', marker_color=colors[2]))
  fig.add_trace(go.Bar(x=r['Année'], y=r['Eolien (MW)'], name='Eolien', marker_color=colors[0]))
  fig.add_trace(go.Bar(x=r['Année'], y=r['Solaire (MW)'], name='Solaire', marker_color=colors[1]))
  fig.add_trace(go.Bar(x=r['Année'], y=r['Bioénergies (MW)'], name='Bioénergies', marker_color=colors[3]))



  fig.update_layout(barmode='stack', title_text="Production en MW des différentes énergies par année depuis 2013" )
  st.plotly_chart(fig, use_container_width=True)

  #Courbes enr
  #COURBE
  r = df.groupby(['id_mois','Mois'], as_index = False).agg({'Eolien (MW)' : 'mean',
                                                            'Hydraulique (MW)' : 'mean',
                                                            'Solaire (MW)' : 'mean',
                                                            'Bioénergies (MW)' : 'mean',
                                                            'Nucléaire (MW)' : 'mean',
                                                            'Thermique (MW)' : 'mean'})
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Eolien (MW)'],
                           mode='lines+markers',
                           name='Eolien (MW)', marker_color=colors[0]))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Hydraulique (MW)'],
                           mode='lines+markers',
                           name='Hydraulique (MW)', marker_color=colors[2]))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Solaire (MW)'],
                           mode='lines+markers',
                           name='Solaire (MW)', marker_color=colors[1]))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Bioénergies (MW)'],
                           mode='lines+markers',
                           name='Bioénergies (MW)', marker_color=colors[3]))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Nucléaire (MW)'],
                           mode='lines+markers',
                           name='Nucléaire (MW)', marker_color=colors[4]))
  fig.add_trace(go.Scatter(x=r['Mois'], y=r['Thermique (MW)'],
                           mode='lines+markers',
                           name='Thermique (MW)', marker_color=colors[5]))
  fig.update_layout(
      title_text="Courbe de la production d'electricité des différentes énergies en France depuis 2013")
  fig.update_xaxes(rangeslider_visible=True)
  st.plotly_chart(fig, use_container_width=True)


elif menu == "À l'échelle régionale":


  st.title("À l'échelle régionale")
  
  #BARRes
  col1, col2 = st.columns((4,1))
  with col2:
    result = st.selectbox("Choisir  l'année?",['2013','2014','2015','2016','2017','2018','2019','2020','2021'])
  with col1:
    r = df.groupby(['Année','Région'], as_index = False).agg({'Consommation (MW)' : 'mean',
                                                              'Production Total (MW)' : 'mean',
                                                              'Ech. physiques (MW)' : 'mean'})
    rf = r[r['Année'] == result] 
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
    x=rf['Région'],
    y=rf['Consommation (MW)'],
    name='Conso. (MW',
    marker_color='indianred'
    ))
    fig2.add_trace(go.Bar(
    x=rf['Région'],
    y=rf['Production Total (MW)'],
    name='Prod. Total (MW)',
    marker_color='lightsalmon'
    ))
    fig2.add_trace(go.Bar(
    x=rf['Région'],
    y=rf['Ech. physiques (MW)'],
    name='Ech. (MW)',
    marker_color='rgb(55, 83, 109)'
    ))
    fig2.update_layout(title_text='Consommation, Production et Echanges d energie dans chaque régions en ' + result)
    fig2.update_layout(barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
   
  #SCATTER
  r = df_r.groupby(['Région'], as_index = False).agg({'Superficie (Km2)' : 'mean',
                                                  'Densité (pop/Km2)' : 'mean',
                                                  'Ech. physiques (MW)' : 'mean',
                                                  'Production Total (MW)' : 'mean',
                                                  'Consommation (MW)' : 'mean',
                                                  'Population' : 'mean'})
  fig = px.scatter(r, x="Consommation (MW)", y= "Production Total (MW)",
                 size="Population", color="Région",
                 hover_name="Région", log_x=True, size_max=60)
  st.plotly_chart(fig, use_container_width=True)
  



  #COMPARATIF
  r = df.groupby(['Région', 'Année'], as_index = False).agg({'Eolien (MW)':'mean',
                                                               'Solaire (MW)' : 'mean',
                                                               'Hydraulique (MW)' : 'mean',
                                                               'Bioénergies (MW)' : 'mean',
                                                               'Nucléaire (MW)' : 'mean',
                                                               'Thermique (MW)' : 'mean'})
  r2 = df.groupby(['Région','Année','id_mois','Mois'], as_index = False).agg({'Consommation (MW)' : 'mean',
                                                                              'Production Total (MW)' : 'mean',
                                                                              'Production Total ENR (MW)' : 'mean'})
  
  
  st.title('Comparatif entre régions')
  st.write('Ici, vous pouvez comparer le mix énergétique de 2 régions différentes.')
   
  col1, col2, col3 = st.columns((1,3,3))
  with col1:
    result_region1 = st.selectbox("Choisir une région?",['Auvergne-Rhône-Alpes','Nouvelle-Aquitaine','Bourgogne-Franche-Comté','Pays de la Loire','Centre-Val de Loire',"Provence-Alpes-Côte d'Azur",'Occitanie','Hauts-de-France','Bretagne','Île-de-France','Normandie','Grand Est'])
    result_region2 = st.selectbox("A comparer avec?",['Cliquez ici','Auvergne-Rhône-Alpes','Nouvelle-Aquitaine','Bourgogne-Franche-Comté','Pays de la Loire','Centre-Val de Loire',"Provence-Alpes-Côte d'Azur",'Occitanie','Hauts-de-France','Bretagne','Île-de-France','Normandie','Grand Est'])
    result_année1 = st.selectbox("Choisir l'année?",['2013','2014','2015','2016','2017','2018','2019','2020','2021'])
  
  if result_region2 == 'Cliquez ici':
  

    rf = r2[(r2['Région'] == result_region1) & (r2['Année'] == result_année1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Consommation (MW)'],
                             mode='lines+markers',
                             name='Conso (MW)'))
    fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Production Total (MW)'],
                             mode='lines+markers',
                             name='Prod (MW)'))

    fig.update_layout(title_text='Consommation et Production en ' + result_region1+ ", " + result_année1)
    st.plotly_chart(fig, use_container_width=True)

    r_region= r.loc[(r['Région'] == result_region1) & (r['Année'] == result_année1)]
    liste = r_region.values.tolist()[0]
    labels = ["Eolien", "Solaire", "Hydraulique", "Bioénergie","Nucléaire","Thermique" ]
    fig6 = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
    fig6.add_trace(go.Pie(labels=labels, values=liste[2:],marker_colors=colors), 1, 1)
    fig6.update_traces(hoverinfo="label+percent+name")
    fig6.update_layout(
          title_text=result_region1 + ", " + result_année1 + ", proportion des énergies")

    st.plotly_chart(fig6, use_container_width=True)

    r_reg = r[r['Région'] == result_region1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Nucléaire (MW)'], name='Nucléaire', marker_color=colors[4]))
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Thermique (MW)'], name='Thermique', marker_color=colors[5]))
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Hydraulique (MW)'], name='Hydraulique', marker_color=colors[2]))
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Eolien (MW)'], name='Eolien', marker_color=colors[0]))
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Solaire (MW)'], name='Solaire', marker_color=colors[1]))
    fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Bioénergies (MW)'], name='Bioénergies', marker_color=colors[3]))
    fig.update_layout(barmode='stack', title_text="Production des enrgies  en " + result_region1 + " depuis 2013")
    st.plotly_chart(fig, use_container_width=True)

  else: 
    with col2:

      rf = r2[(r2['Région'] == result_region1) & (r2['Année'] == result_année1)]

      fig = go.Figure()
      fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Consommation (MW)'],
                             mode='lines+markers',
                             name='Conso (MW)'))
      fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Production Total (MW)'],
                             mode='lines+markers',
                             name='Prod (MW)'))

      fig.update_layout(title_text='Consommation et Production en ' + result_region1+ ", " + result_année1)
      st.plotly_chart(fig, use_container_width=True)
 
    with col3:
      rf = r2[(r2['Région'] == result_region2) & (r2['Année'] == result_année1)]

      fig = go.Figure()
      fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Consommation (MW)'],
                             mode='lines+markers',
                             name='Conso (MW)'))
      fig.add_trace(go.Scatter(x=rf['Mois'], y=rf['Production Total (MW)'],
                             mode='lines+markers',
                             name='Prod (MW)'))

      fig.update_layout(title_text='Consommation et Production dans la région ' + result_region2 + " en " + result_année1)
      st.plotly_chart(fig, use_container_width=True) 

        
    col1, col2 = st.columns((2))
    with col1:
      r_region= r.loc[(r['Région'] == result_region1) & (r['Année'] == result_année1)]
      liste = r_region.values.tolist()[0]
      labels = ["Eolien", "Solaire", "Hydraulique", "Bioénergie","Nucléaire","Thermique" ]
      fig6 = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
      fig6.add_trace(go.Pie(labels=labels, values=liste[2:],marker_colors=colors), 1, 1)
      fig6.update_traces(hoverinfo="label+percent+name")
      fig6.update_layout(
            title_text=result_region1 + ", " + result_année1 + ", proportion des énergies")

      st.plotly_chart(fig6, use_container_width=True)
    with col2:
        r_region= r.loc[(r['Région'] == result_region2) & (r['Année'] == result_année1)]
        liste = r_region.values.tolist()[0]
        labels = ["Eolien", "Solaire", "Hydraulique", "Bioénergie","Nucléaire", "Thermique"]
        fig6 = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
        fig6.add_trace(go.Pie(labels=labels, values=liste[2:],marker_colors=colors),1, 1)
        fig6.update_traces(hoverinfo="label+percent+name")
        fig6.update_layout(
            title_text=result_region2 + ", " + result_année1 + ", proportion des énergies")

        st.plotly_chart(fig6, use_container_width=True)

    col1, col2= st.columns((1,3))
    with col1:
      energie = st.selectbox("Choisir l'énergie?",['Eolien (MW)','Solaire (MW)','Hydraulique (MW)','Bioénergies (MW)','Nucléaire (MW)','Thermique (MW)'])
    with col2:

      r_1= r.loc[(r['Région'] == result_region1) ]
      r_2= r.loc[(r['Région'] == result_region2) ]

      fig = go.Figure()
      fig.add_trace(go.Bar(
          x=r_1['Année'],
          y=r_1[energie],
          name=result_region1,
          marker_color='#F29F58'
      ))
      fig.add_trace(go.Bar(
          x=r_2['Année'],
          y=r_2[energie],
          name=result_region2,
          marker_color='#4F51F8'
      ))

      fig.update_layout(barmode='group', xaxis_tickangle=-45,title_text = "Production d'energie "+ energie )
      st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns((2))
    with col1:
      r_reg = r[r['Région'] == result_region1]
      fig = go.Figure()
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Nucléaire (MW)'], name='Nucléaire', marker_color=colors[4]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Thermique (MW)'], name='Thermique', marker_color=colors[5]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Hydraulique (MW)'], name='Hydraulique', marker_color=colors[2]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Eolien (MW)'], name='Eolien', marker_color=colors[0]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Solaire (MW)'], name='Solaire', marker_color=colors[1]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Bioénergies (MW)'], name='Bioénergies', marker_color=colors[3]))
      fig.update_layout(barmode='stack', title_text="Production des enrgies  en " + result_region1 + " depuis 2013")
      st.plotly_chart(fig, use_container_width=True)
    with col2:
      r_reg = r[r['Région'] == result_region2]
      fig = go.Figure()
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Nucléaire (MW)'], name='Nucléaire', marker_color=colors[4]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Thermique (MW)'], name='Thermique', marker_color=colors[5]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Hydraulique (MW)'], name='Hydraulique', marker_color=colors[2]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Eolien (MW)'], name='Eolien', marker_color=colors[0]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Solaire (MW)'], name='Solaire', marker_color=colors[1]))
      fig.add_trace(go.Bar(x=r_reg['Année'], y=r_reg['Bioénergies (MW)'], name='Bioénergies', marker_color=colors[3]))
      fig.update_layout(barmode='stack', title_text="Production des energies en " + result_region2 + " depuis 2013")
      st.plotly_chart(fig, use_container_width=True)




elif menu == 'Prédiction de consommation':
  st.title('Modélisation et prédiction de consommation')
  col1, col2 = st.columns((1,3))
  with col1:
    resultat = st.selectbox("A travers quel paramètre voulez faire votre modélisation?",['DJU','Population'])

  if resultat == "DJU":
    with col2:
      fig = plt.figure(figsize=(16,8))
      ax1 = sns.lineplot(data=datas, x=datas.index,  y="Conso", color="orange", alpha=0.7, legend='brief', label="Conso")
      ax2 = plt.twinx()
      ax2 = sns.lineplot(data=datas, x=datas.index,  y="dju", color="purple", ax=ax2, legend='brief', label="DJU France")
      plt.legend()
      plt.title("Evolution du DJU et consommation en énergie France", fontsize=22)
      st.pyplot(fig)
    
    st.write('Voici la modélisation de la régression linéaire et de la distribution des résidus')     
    col1, col2 = st.columns(2)
    with col1:
      fig = plt.figure(figsize=(12,8))
      ax = sns.regplot(data=datas, x="dju", y="Conso", robust=True, ci=None, line_kws={"color":"orange"})
      plt.title("Régression linéaire : Consommation en fonction de DJU", fontsize=22)
      st.pyplot(fig)
    with col2:
      from scipy.stats import norm, shapiro
      import statsmodels.formula.api as smf
      reg_conso = smf.ols('Conso ~ dju', data=datas).fit()
      fig = plt.figure(figsize=(12,8))
      ax = sns.distplot(reg_conso.resid, fit=norm)
      plt.xlabel('Résidus')
      plt.title("Distribution des résidus du modèle de régression linéaire", fontsize=22)
      st.pyplot(fig)
    st.subheader('Prédiction de consommation')

    x = datas[["dju"]]
    y = datas[["Conso"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    
    col1,col2 = st.columns((1,3))
    with col1:

      valeur = st.text_input("Choisissez une température moyenne (en degré) :")
      mois = st.selectbox('Choisissez le mois :', ['Janvier','Fevrier','Mars','Avril','Mai','Juin','Juillet','Aout','Septempbre','Octobre','Novembre','Decembre'])
    
    if valeur == '':
      st.write('...')

    else:
      liste31 = ['Janvier','Mars','Mai','Juillet','Aout','Octobre','Decembre']
      liste30 = ['Avril','Juin','Septempbre','Novembre']

      def degres_à_dju(degrés, mois):
        n = 18 - degrés
        for i in liste31:
            if mois == i :
                dju = n * 31
        for i in liste30:
            if mois == i:
                dju = n * 30
        if mois == 'Fevrier':
            dju = n * 28
        
        return dju

      reg = reg.fit(x,y)
      a = reg.predict(np.array(degres_à_dju(float(valeur), mois)).reshape(1,-1)).round(3)
      a = str(a[0][0])
      with col2:

        st.write("La prédiction de consommation moyenne est de " + a + " MW, pour une température moyenne de "  +  valeur + " celsius, en " + mois)

        b = str(reg.score(x_train, y_train).round(3))
        c = str(cross_val_score(reg,x_train,y_train).mean().round(3))

        st.write("Coefficient de détermination R2 : " + b)
        st.write("Coefficient de détermination obtenu par Cv : "+ c)


    st.subheader('Prédiction 2020 avec les modeles de séries temporelles:')
    st.image('df_ajuste.png')

  else:
    df_r.rename(columns={'Consommation (MW)': 'Consommation'}, inplace=True)
         
    col1, col2 = st.columns(2)
    with col1:
      fig = plt.figure(figsize=(12,8))
      ax = sns.regplot(data=df_r, x="Population", y="Consommation", robust=True, ci=None, line_kws={"color":"orange"})
      plt.title("Régression linéaire : Consommation en fonction de DJU", fontsize=22)
      st.pyplot(fig)
    with col2:
      from scipy.stats import norm, shapiro
      import statsmodels.formula.api as smf
      reg_conso = smf.ols('Consommation ~ Population', data=df_r).fit()
      fig = plt.figure(figsize=(12,8))
      ax = sns.distplot(reg_conso.resid, fit=norm)
      plt.xlabel('Résidus')
      plt.title("Distribution des résidus du modèle de régression linéaire", fontsize=22)
      st.pyplot(fig)
    st.subheader('Prédiction de consommation')

    col1,col2 = st.columns((1,3))
    with col1:
      df_region_ml = df_r
      df_region_ml.drop(["Région","Année","Production Total (MW)","Superficie (Km2)","Densité (pop/Km2)"], axis = 1, inplace = True) 
      x = df_region_ml[["Population"]]
      y = df_region_ml[["Consommation"]]
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)
      reg = LinearRegression()
      reg.fit(x_train, y_train)
      y_pred = reg.predict(x_test)

    
      habitants = st.text_input("Choisissez un nombre d'habitants:")
    
    if habitants == '':
      st.write('...')

    else:
      with col2:
        reg = reg.fit(x,y)
        a = reg.predict(np.array(int(habitants)).reshape(1,-1)).round(3)
        a = str(a[0][0])
        
        st.write("La prédiction de consommation moyenne est de " + a + " MW, pour "  +  habitants + " habitants ")

        b = str(reg.score(x_train, y_train).round(3))
        c = str(cross_val_score(reg,x_train,y_train).mean().round(3))
        st.write("Coefficient de détermination R2 : " + b)
        st.write("Coefficient de détermination obtenu par Cv : "+ c)

    


























