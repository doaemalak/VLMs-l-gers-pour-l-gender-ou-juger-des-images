# Réflexion éthique

## 1. Erreurs et éléments inventés (hallucinations)

Un VLM peut décrire des choses absentes de l'image. Nous l'avons observé : le VLM 1
décrit un enfant « déguisé en Père Noël » ou « en super-héros », ce qui est faux.
**Risque pour l'option A :** une LoRA entraînée sur des légendes fausses apprend de
fausses associations entre mots et images, et ces erreurs se propagent ensuite dans
toutes les images générées.
**Ce que nous avons fait :** choisir le VLM qui invente le moins dans nos exemples
(SmolVLM), comparer avec des légendes humaines, et recommander une **relecture
humaine** d'un échantillon de légendes avant tout entraînement.

## 2. Biais des légendes et des modèles

Les VLM et Stable Diffusion ont appris sur des données d'Internet. Ils peuvent
reproduire des stéréotypes (genre, âge, origine), par exemple dans la façon de
décrire les personnes (« a woman » / « a girl »). Les légendes humaines de Flickr8k
ont aussi leurs propres biais.
**Ce que nous avons fait :** choisir des **photos de chiens** pour la LoRA, ce qui
limite les biais liés aux personnes. Ces biais restent présents dans l'onglet
« Légende » de la démo pour les images de personnes.

## 3. Personnes réelles et vie privée

Les photos Flickr8k montrent souvent des personnes qui n'ont pas consenti à ce
qu'on entraîne un générateur d'images sur leur visage.
**Ce que nous avons fait :** ne garder que les images où au moins 3 légendes sur 5
parlent d'un chien ; ne pas entraîner de LoRA sur des personnes ; ne déposer
aucune image dans Git.

## 4. Filtre de sécurité désactivé

Pour l'évaluation (étape 9) et la démo, le filtre de sécurité de Stable Diffusion
est désactivé, car il remplace parfois à tort une image par un carré noir, ce qui
fausse les scores.
**Pourquoi c'est acceptable ici :** les phrases d'évaluation sont des légendes de
chiens contrôlées. **Limite :** dans la démo, l'utilisateur peut écrire n'importe
quelle phrase ; la démo doit donc rester une démonstration en classe, pas un
service ouvert au public. La licence CreativeML OpenRAIL-M interdit de toute façon
les usages nuisibles.

## 5. Confiance dans un juge automatique

Nous évaluons les images avec CLIP. C'est pratique et reproductible, mais CLIP voit
mal les détails (couleurs, nombre d'objets) et a ses propres biais. Nos conclusions
sont donc présentées avec prudence (tests statistiques, tendances signalées comme
telles) et devraient être confirmées par une évaluation humaine.

## 6. Licences et droits

Chaque modèle a sa licence (voir `docs/LICENCES.md`) : Qwen2.5-VL-3B interdit
l'usage commercial ; Stable Diffusion 1.5 interdit certains usages. Les photos
Flickr8k appartiennent à leurs auteurs et sont réservées à la recherche et à
l'enseignement.

## 7. Impact environnemental

La quantification réduit la mémoire nécessaire (−37 % à −56 % en 4 bits), ce qui
permet d'utiliser des GPU plus modestes. La LoRA n'entraîne que 0,19 % des
paramètres (13 minutes par LoRA sur un T4) au lieu de réentraîner tout le modèle.
Nous avons aussi évité de relancer des calculs inutiles (reprise après coupure).
