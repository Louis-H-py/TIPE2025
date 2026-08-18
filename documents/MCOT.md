# MCOT

###### Transition, transformation, conversion

## 1.0.0

### Titre :

Étude Théorique de la localisation et de la séparation informatique de sources sonores.

## Ancrage dans le thème de l’année :

Ce TIPE s'intéresse aux méthodes informatiques de localisation sonore et d'isolation du signal.
Le projet **transforme** des signaux sonores et des décalages temporels en une information de localisation
et **convertit** des enregistrements en signaux séparés, il s'inscrit donc dans le thème "transition, transformation,
conversion".

_47 mots_

## Motivations :

Ce TIPE est un mélange traitement du signal et théorie de l'information. Les solutions utilisent des réseaux de neurones
récurrents, des outils de corrélation et la transformée de Fourrier. Il s'inscrit donc dans mon souhait
d'orientation en statistiques, probabilités et sciences des données.

_46 mots_

## Positionnement thématique :

Informatique (Pratique) — Informatique (Théorique) — Physique (Ondulatoire)

## Mots clés :

| Français                        | Anglais                     |
|---------------------------------|-----------------------------|
| Signaux sonore                  | _Audio signal_              |
| Localiser                       | _Locate_                    |
| Séparation aveugle de signaux   | _Blind signal separation_   |
| Réseaux de neurones artificiels | _Artificial neural network_ |
| Théorie de l'information        | _Information theory_        |

## Bibliographie commentée :

Au milieu d’un endroit bruyant, l’humain est capable de reconnaître un son familier, d’en estimer la direction, la
distance par rapport à lui, et de faire abstraction du bruit ambiant et des autres sons pour n’écouter que ce son en
particulier, en profitant de l’effet cocktail party. Reproduire cette prouesse informatiquement consiste à résoudre à la
fois un problème de localisation acoustique [1] et un problème de séparation aveugle de sources sonores [2]. Cela fait
donc appel à la fois au traitement du signal, à la science des données et à la théorie de l'information.

Pour parvenir à ce résultat, dans un premier temps, il faut pouvoir trouver la position dans l’espace de la source
sonore que l’on souhaite ensuite isoler [1]. Pour cela, on étudie le décalage temporel entre les signaux reçus par
plusieurs microphones, en utilisant par exemple une corrélation croisée entre les signaux [3]. Ensuite, on essaye de
remonter aux coordonnées de la source. Cela se traduit par un problème d’optimisation où l'on cherche la position pour
laquelle la différence entre les décalages temporels théoriques et pratiques est la plus faible. On peut également
réaliser une étude statistique en analysant plusieurs pics de corrélation croisée afin de déterminer plusieurs sources
sonores avec une plus grande précision.

Une fois qu’une source a pu être localisée, il faut pouvoir isoler le signal qu’elle émet [2]. De nombreuses méthodes
existent. L’une des plus communes est l’analyse en composantes indépendantes (ICA) [4]. Cette méthode est par exemple
utilisée en électroencéphalographie pour séparer les différents signaux nerveux électriques émanant de plusieurs organes
afin d’extraire uniquement celui d’un organe en particulier que l’on souhaite étudier [5]. Cependant, l’analyse en
composantes indépendantes peut rencontrer de nombreuses difficultés en présence de bruit et de décalages temporels.

La séparation aveugle de sources [2] est en réalité un problème qui peut s’avérer complexe et qui nécessite des méthodes
mathématiques d’interpolation et d’extrapolation exigeantes pour pouvoir traiter les cas les plus difficiles (présence
d’un bruit important, de nombreuses sources ou de décalages temporels entre les enregistrements). Pour résoudre ce
problème, les réseaux de neurones artificiels [6] s’avèrent être excellents, notamment les réseaux de neurones
récurrents comme les LSTM (Long Short-Term Memory, réseau récurrent à mémoire court et long terme) [7], qui peuvent
traiter de multiples séries temporelles pour en extraire les informations recherchées, dans notre cas les signaux
émanant des sources sonores. En isolant très efficacement les sources sonores, ces réseaux de neurones ont maintenant
une place importante dans le domaine de la séparation aveugle de sources.

De plus, pour améliorer les résultats, on peut transformer les signaux sonores pour les rendre plus faciles à analyser,
par exemple en réalisant l’analyse sur un spectrogramme ou une transformée de Fourier. Cela peut potentiellement
permettre une meilleure localisation et une meilleure séparation des sources sonores.

Ces avancées ouvrent la voie à de nombreuses applications concrètes, comme l’imagerie sonore en trois dimensions à
l’aide de matrices de microphones [8], qui étudient la manière dont le son rebondit sur des surfaces pour les imager
avec une très grande précision, ou encore l’amélioration des aides auditives, notamment au travers de dispositifs qui
permettent de diminuer le bruit ambiant tout en conservant le son d’une personne qui nous parle.

## Problématique retenue :

Comment convertir des signaux sonores en informations de localisation et les transformer informatiquement pour isoler
les sources sonores ?

_18 mots_

## Objectif du TIPE :

Le TIPE se décompose en plusieurs objectifs :

- Conception d'un environment de simulation des signaux sonores.
- Concevoir des séries de tests pour réaliser une comparaison des différentes approches qui seront abordées.
- concevoir des algorithmes de localisation utilisation l'auto-corrélation avec différents signaux.
- Concevoir des algorithmes d'isolation du signal sonore 'naïf'.
- Utiliser des réseaux de neurones pour améliorer l'isolation des sources sonores.
- Assembler les meilleurs algorithmes de localisation et d'isolation.

_74 mots_

## Références bibliographiques :

[1] Vera-Diaz, Juan & Pizarro, Daniel & Macias-Guarasa, Javier. (2018). Towards End-to-End Acoustic Localization Using Deep Learning: From Audio Signals to Source Position Coordinates. Sensors. 18. 3418. 10.3390/s18103418. 
[2] J. . -F. Cardoso, "Blind signal separation: statistical principles," in Proceedings of the IEEE, vol. 86, no. 10, pp. 2009-2025, Oct. 1998, doi: 10.1109/5.720250.
[3] Wikipédia Corrélation croisée https://fr.wikipedia.org/wiki/Corr%C3%A9lation_crois%C3%A9e (7 Juillet 2024)
[4] Naik, Ganesh & Kumar, Dinesh. (2011). An Overview of Independent Component Analysis and Its Applications. Informatica. 35. 63-81. 
[5] Klug, M., Berg, T. & Gramann, K. Optimizing EEG ICA decomposition with data cleaning in stationary and mobile experiments. Sci Rep 14, 14119 (2024).
[6] LeCun, Y., Bengio, Y. & Hinton, G. “Deep learning.” Nature 521, 436–444 (2015).
[7] Hochreiter S, Schmidhuber J. Long short-term memory. Neural Comput. 1997 Nov 15;9(8):1735-80. doi: 10.1162/neco.1997.9.8.1735. PMID: 9377276.
[8]  Legg, Mathew & Bradley, Stuart. (2014). Automatic 3D scanning surface generation for microphone array acoustic imaging. Applied Acoustics. 76. 230–237. 10.1016/j.apacoust.2013.08.008.