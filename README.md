# <img src="https://github.com/hampusborgos/country-flags/raw/main/svg/fr.svg" width="40"> TIPE 2025 - Transition, transformation, conversion

Travail d'initiative personnelle encadré (TIPE) réalisé en classe préparatoire, filière MP, en 2025, sur le thème **« Transition, transformation, conversion »**.

Ce projet se situe à la croisée de l'informatique pratique, de l'informatique théorique et de la physique ondulatoire.

**Auteur :** Louis H.

**Stack :** 🐍 Python | 🧮 NumPy | 🔮 Scikit-learn | 🧠 Keras | 📊 Plotly

## Étude théorique de la localisation et de la séparation informatique de sources sonores

> Comment convertir des signaux sonores en informations de localisation, puis les traiter informatiquement afin d'isoler les différentes sources sonores ?

## Description du problème

On considère plusieurs sources sonores fixes, indépendantes et idéales, situées dans un espace homogène, isotrope et sans réflexions. Ces sources émettent de la musique.

On dispose également de microphones idéaux et fixes.

L'objectif est, dans un premier temps, de localiser les sources sonores dans l'espace, puis d'exploiter ces informations afin d'isoler le signal émis par chacune d'elles de ceux des autres sources.

## Résolution

La localisation des sources est réalisée à l'aide d'une étude de **corrélation croisée**, suivie d'un partitionnement des résultats à l'aide de l'algorithme **Mean Shift**.

La séparation des sources sonores est abordée à travers plusieurs méthodes :
- **PCA** (Analyse en composantes principales) ;
- **ICA** (Analyse en composantes indépendantes) ;
- **Réseaux de neurones récurrents**, notamment les architectures LSTM, GRU et leurs variantes bidirectionnelles.

## Résultats

- **Localisation :** jusqu'à 5 sources sonores peuvent être localisées, selon le nombre de sources présentes.
- **Séparation :** une source sonore peut être isolée parmi un ensemble pouvant comporter jusqu'à 7 sources.

### Présentation

> [**Présentation.pdf**](https://github.com/Louis-H-py/TIPE2025/blob/main/documents/Pr%C3%A9sentation.pdf)

**Note :** 20/20 (classé dans le top 1 % parmi environ 7 000 candidats).

---
# <img src="https://github.com/hampusborgos/country-flags/raw/main/svg/gb.svg" width="40"> TIPE 2025 - Transition, Transformation, Conversion

Year-long Research Project completed in 2025 as part of the French preparatory program (MP track), on the theme **"Transition, Transformation, Conversion"**.

This project combines **Applied Computer Science**, **Theoretical Computer Science**, and **Wave Physics**.

**Author:** Louis H.

**Stack :** 🐍 Python | 🧮 NumPy | 🔮 Scikit-learn | 🧠 Keras | 📊 Plotly

## Theoretical Study of Sound Source Localization and Separation

> How can sound signals be transformed into location information and processed to isolate individual sound sources?

## Problem Description

We consider several fixed, independent and ideal sound sources in a homogeneous, isotropic environment without reflections. These sources emit music.

We also consider a set of fixed and ideal microphones.

The objective is first to locate the sound sources in space, then use this information to isolate the signal emitted by each source from the signals emitted by the others.

## Approach

Sound source localization is performed using **cross-correlation analysis**, followed by **clustering** using the **Mean Shift** algorithm.

Sound source separation is explored using several approaches:
- **PCA** (Principal Component Analysis);
- **ICA** (Independent Component Analysis);
- **Recurrent Neural Networks (RNNs)**, including LSTM, GRU and their bidirectional variants.

## Results

- **Localization:** up to 5 sound sources can be located, depending on the number of sources present.
- **Separation:** a single sound source can be isolated from a mixture containing up to 7 sources.


### Presentation

> [**Presentation.pdf**](https://github.com/Louis-H-py/TIPE2025/blob/main/documents/Pr%C3%A9sentation.pdf)

**Grade:** 20/20 (ranked in the top 1% among approximately 7,000 candidates).
