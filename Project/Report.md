## Report di Progetto: Analisi dei Classificatori Generativi Gaussiani (LAB 05)

### 1. Prestazioni Baseline e Ruolo dei Priors
In questa prima fase, abbiamo addestrato i modelli Gaussiani sul dataset originale a 6 feature, effettuando uno split fisso tra Training e Validation set per garantire una comparazione coerente. Essendo un problema di classificazione binaria, la regola di decisione è basata sul Log-Likelihood Ratio (LLR). 

Le probabilità a priori (priors) fornite, $P(C=1) = P(C=0) = 0.5$, definiscono matematicamente la soglia di decisione ottimale, che risulta pari a **0** ($-\log(0.5/0.5)$).

I tassi di errore sul Validation Set sono i seguenti:
*   **MVG (Multivariate Gaussian):** 7.00%
*   **NBG (Naive Bayes Gaussian):** 7.20%
*   **LDA (Linear Discriminant Analysis):** 9.25%
*   **TCG (Tied Covariance Gaussian):** 9.30%

Il modello MVG risulta essere il più accurato. Si nota inoltre una forte convergenza empirica tra le prestazioni della LDA e del TCG (~9.3%): entrambi i modelli, avendo assunto implicitamente (o esplicitamente) matrici di covarianza identiche per le due classi, producono la medesima frontiera di decisione lineare.

### 2. Analisi di Covarianza, Correlazione e Giustificazione del Naive Bayes
Per comprendere le ottime performance del modello Naive Bayes, abbiamo analizzato la matrice di covarianza delle singole classi, estraendo successivamente la Matrice di Correlazione di Pearson tramite la formula:
$$Corr(i, j) = \frac{Cov(i, j)}{\sqrt{Var(i)}\sqrt{Var(j)}}$$

L'analisi ha evidenziato che i coefficienti di correlazione al di fuori della diagonale principale sono prossimi allo zero (valori nell'ordine di $10^{-2}$ o inferiori). 
*   **Conclusione:** Le feature del dataset sono intrinsecamente e debolmente correlate (weakly correlated). 
Questo dato giustifica appieno i risultati di classificazione: il modello Naive Bayes opera forzando a zero i valori esterni alla diagonale della matrice di covarianza. Poiché nel nostro dataset reale tali valori sono già trascurabili, l'assunzione di indipendenza dell'NBG non causa una perdita significativa di informazione, portando a un peggioramento puramente marginale dello 0.20% rispetto al modello MVG completo, a fronte di un notevole risparmio computazionale.

### 3. L'Assunzione Gaussiana e il Taglio delle Feature 5 e 6
L'analisi visiva delle singole feature ha mostrato che l'assunzione Gaussiana (che modella i dati come campane simmetriche) non è rispettata su tutto il dataset. In particolare, le feature 5 e 6 mostrano distribuzioni anomale (es. bimodali) per la Classe 1, violando palesemente l'assunto teorico.

Per testare l'impatto di questo errore di modellazione, abbiamo ripetuto la classificazione scartando le feature 5 e 6 e utilizzando solo le prime quattro. I risultati (es. MVG al 7.95%) mostrano un **peggioramento generalizzato** delle prestazioni per tutti i modelli.
*   **Conclusione:** Pur non essendo geometricamente rappresentabili da una densità Gaussiana perfetta, le feature 5 e 6 possiedono un forte potere discriminante. I classificatori, nonostante l'inesattezza dell'assunzione di base, riescono comunque a estrarre varianze e medie utili a separare lo spazio, dimostrando una notevole robustezza.

### 4. Impatto Geometrico dei Dati: Feature 1-2 vs Feature 3-4
Per testare i limiti dei modelli MVG e TCG, abbiamo isolato due coppie specifiche di feature che presentano distribuzioni spaziali diametralmente opposte.

*   **Caso A: Solo Feature 1 e 2 (Medie simili, Varianze diverse)**
    *   *Risultati:* MVG (36.50%) vs TCG (49.45%)
    *   *Analisi:* Avendo i centri spaziali quasi sovrapposti, l'unico modo per separare le classi è modellare la differenza di "apertura" delle loro distribuzioni. Il TCG fallisce totalmente (prestazioni assimilabili a una scelta casuale) a causa del vincolo della covarianza condivisa, che non gli permette di distinguere due classi centrate nello stesso punto. L'MVG vince nettamente grazie alla sua natura quadratica.
*   **Caso B: Solo Feature 3 e 4 (Medie diverse, Varianze simili)**
    *   *Risultati:* MVG (9.45%) vs TCG (9.40%)
    *   *Analisi:* In questo scenario, le distribuzioni reali dei dati rispecchiano fedelmente l'assunzione teorica del TCG (stessa forma, centri diversi). Di conseguenza, il TCG eguaglia e supera di una frazione il modello MVG, beneficiando di una minore complessità parametrica che lo protegge dal rischio di overfitting.

### 5. Pre-processing con PCA e Verdetto Finale
Infine, è stato applicato l'algoritmo PCA per la riduzione della dimensionalità prima dell'addestramento. I test con dimensioni ridotte (es. $m=4$) hanno prodotto errori superiori alla baseline (MVG: 8.05%, NBG: 8.85%, TCG: 9.25%).
*   **Conclusione:** La PCA non si rivela un pre-processing efficace per incrementare l'accuratezza su questo dataset. Scartare le dimensioni a minore varianza comporta la perdita di informazioni utili alla separazione geometrica operata dall'MVG.

**Verdetto Finale:** 
Alla luce dell'intera analisi, il modello che ha fornito la migliore accuratezza in assoluto sul Validation Set è il **Multivariate Gaussian (MVG) applicato alle 6 feature originali (7.00%)**. La sua flessibilità nel calcolare matrici di covarianza dedicate ha permesso di sfruttare le marcate differenze di varianza tra la Classe 0 e la Classe 1, compensando brillantemente la debole correlazione tra le variabili.
