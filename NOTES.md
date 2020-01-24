
# NueDeck

make the release in the nuedeck
then `ln -s ~/projects/deck-js-stuffs/nuedeck/release/nuedeck`
(check that the starting values are 0,0 (currentSlide etc)

to live reload the simple pres

~~~
with_node
#npm install -g simple-hot-reload-server
hrs .
~~~

`firefox http://localhost:8082/2019-06-13-naver-labs.html`



# While making it

https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/



# WIP

Ex of imbalanced task (BROAD)
- say that we call + the positive (minority, anomaly, rare) class
(disclaimer about multiclass not well defined)

what classical classification performance you get

Explain with, for instance, gradient using xentropy as surrogate (or for mle)
- all ok create a gradient on ~ 1 (= ∇ln (1) = 1/1)
- the few negative, with proba 0.4 give 2.5 only (1/0.4) or with proba 0.1 only 10
- so with let's say 1 + for 10 -, we get a stable gradient if all + have 0.1 pred

For models that learn, can reweigh but not perfect (ideal ratio depends on Bayesian error and régularisation etc) can x valid?
Example of reweighting resampling strategies

For 1nn, screwed even though at the limit there is not much pb (illustrate with a region of no Bayesian error and one with it)
importance of kNN in many situations (cf ecml etc)

... Gammann

... Also mention cone and deep versions of., with emphasis on ratio losses and the v and the fact that there is also deep thing like this (and our with Kevin)

Insist on guarantees that we are good in f and that we are also in generalization, modulo the 01 eps1Partie interpretable.
Explain context : timeseries with patterns both unsupervised and supervised, mixed or not, etc

In supervised show FS are bad etc

Then follow cikm, especially explain the adversarial and the gradient penalty

Then back to unsup vite fait, Prob mod, autoenc, mention neural expectation?






# For announcement

Title:

Dealing with Imbalanced Data, and, Interpretability via Adversarial Regularization

Abstract:

This talk will cover two topics: imbalanced data and interpretability of convolutional models.

First, we consider how to deal with imbalanced classification where one (positive, anomaly) class is rare, and where measures such as the F1 score are often of interest.
Most methods that consider classification accuracy fail in such situations and re-balancing the dataset or using weighted-accuracy are imperfect solutions.
We first show how the k-NN classifier can be adapted to correct for class imbalance.
We also explain how we can derive guarantees both in terms of learning and generalization for models learned to optimize a weighted-accuracy.

In a second part, we use adversarial approaches in a novel way to have more interpretable convolutional models.
By minimizing the Wasserstein distance between the (learned) convolutional filters and teh sub-series from the dataset, filters become more interpretable as they will look like sub-series.
We show that this approach is promising, showing good behavior in terms of scalability, interpretability and accuracy on 85 time series classification datasets.


## reflown

Title:

Dealing with Imbalanced Data, and, Interpretability via Adversarial Regularization

Abstract:

This talk will cover two topics: imbalanced data and interpretability
of convolutional models.

First, we consider how to deal with imbalanced classification where
one (positive, anomaly) class is rare, and where measures such as the
F1 score are often of interest.  Most methods that consider
classification accuracy fail in such situations and re-balancing the
dataset or using weighted-accuracy are imperfect solutions.  We first
show how the k-NN classifier can be adapted to correct for class
imbalance.  We also explain how we can derive guarantees both in terms
of learning and generalization for models learned to optimize a
weighted-accuracy.

In a second part, we use adversarial approaches in a novel way to have
more interpretable convolutional models.  By minimizing the
Wasserstein distance between the (learned) convolutional filters and
teh sub-series from the dataset, filters become more interpretable as
they will look like sub-series.  We show that this approach is
promising, showing good behavior in terms of scalability,
interpretability and accuracy on 85 time series classification
datasets.



## tmp d2fr d2en

This presentation will focus on two topics: unbalanced data and the interpretability of convolutional models.

First, we examine how to deal with cases of unbalanced classification where a class (positive, anomaly) is rare, and where measures such as the F1 score are interesting.
Most methods that take into account classification accuracy fail in such situations and rebalancing the data set or using weighted accuracy is an imperfect solution.
We first show how the k-NN classifier can be adapted to correct the class imbalance.
We also explain how we can obtain guarantees both in terms of learning and generalization of the models learned to optimize weighted accuracy.

In a second part, we use antagonistic approaches in an innovative way to have more interpretable convolutional models.
By minimizing the Wasserstein distance between the convolutional filters (learned) and the subseries of the data set, the filters become more interpretable because they will look like subseries.
We show that this approach is promising, showing good behaviour in terms of scalability, interpretability and accuracy over 85 time series of classification data.




Translated with www.DeepL.com/Translator








## extrait de mail avec Diane

> 1. γNN, une modif toute simple de k-NN pour les cas déséquilibrés (F-mesure par exemple) (soumis ECML 2019)
> 2. CONE, un truc pas deep avec des bornes et tout, pour optimiser la F-mesure (présenté à AISTATS 2019)
> 3. AI<->PR, interprétabilité dans les time-series avec une régularisation adversariale (rejeté mais on le re-soumets demain...)
> 4. un truc sur le « transfert négatif » en deep domain adaptation (Kevin Bascol, ICIP 2019)
> 5. détection de micro expression avec Pyramide de Riesz (thèse finie, pas de learning)

> 6. des travaux en cours sur du DA avec wasserstein ou d'autres du PAC-bayesien pour du deep learning, mais c'est en cours et pas très au point

​

A première vu, tout est assez différent de ce qu'on fait ici (il n'y a pas de sujet ou je me suis dit: "c'est fou on a un papier exactement sur ce sujet"), mais tout est quand même pertinant pour le centre. Donc il n'y a pas de mauvais choix et c'est vraiment ce dont toi tu as le plus envie de parler.

​C'est vrai que le troisième à l'air sympa. Mais encore une fois, pas de mauvais choix, c'est comme tu le sens.

Ca te va qu'on invite les gens de l'inria ? Après je ne sais pas qui viendrait, mais on fait en général de la pub dans tous les labos autour.

​
