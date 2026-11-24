# Paper Soccer — sześć algorytmów Reinforcement Learning

> **Notatka techniczna do projektu** · Tabular Q-learning · Linear Q-learning · Double DQN · Maskable PPO · A2C · Dyna-style Double DQN  
> **Cel:** zrozumieć metody, ich zalety i ograniczenia oraz rzeczywiste wyzwania implementacyjne w grze Paper Soccer.

---

## Spis treści

1. [Wspólne założenia i specyfika gry](#1-wspólne-założenia-i-specyfika-gry)
2. [Tabular Q-learning](#2-tabular-q-learning)
3. [Linear Q-learning](#3-linear-q-learning)
4. [Double DQN](#4-double-dqn)
5. [Maskable PPO](#5-maskable-ppo)
6. [A2C (Advantage Actor–Critic)](#6-a2c-advantage-actorcritic)
7. [Dyna-style Double DQN (planowanie z modelem gry)](#7-dyna-style-double-dqn-planowanie-z-modelem-gry)
8. [Zestawienie i rekomendacje implementacyjne](#8-zestawienie-i-rekomendacje-implementacyjne)
9. [Źródła i materiały](#9-źródła-i-materiały)

## 1. Wspólne założenia i specyfika gry

### 1.1. Mechanika Paper Soccer

Dla porównania sześciu metod implementujemy **jedno wspólne środowisko** z ustalonymi regułami:

- Plansza to siatka punktów, domyślnie **8 × 10 pól**, z bramkami po krótszych bokach.
- Piłka przemieszcza się o jeden odcinek w jednym z **8 kierunków** (pion, poziom, przekątne).
- Raz użytej krawędzi nie można użyć ponownie.
- Ruch na uprzednio odwiedzony punkt lub punkt graniczny wywołuje **odbicie**: aktualny gracz nadal wykonuje ruch.
- Gdy gracz zdobędzie właściwą bramkę, wygrywa; możliwe są też samobóje. Brak legalnego ruchu kończy grę przegraną gracza, który nie może się ruszyć.

To typowe zasady; przed implementacją trzeba doprecyzować **geometrię bramek, początkowe zaznaczenie granic i ewentualny zakaz przecinania odcinków**. Wszyscy agenci korzystają z identycznego wariantu. Przykładowy opis zasad: [Paper Football — Game Rules](https://www.paper-football.com/).

### 1.2. Formalizacja środowiska

Reprezentujemy grę jako graf $G=(V,E)$, a stan jako:

$$
s_t = (p_t, U_t, i_t),
$$

gdzie:

- $p_t \in V$ — aktualna pozycja piłki;
- $U_t \subseteq E$ — zbiór już wykorzystanych krawędzi;
- $i_t \in \{1,2\}$ — gracz wykonujący następny ruch.

Zbiór dozwolonych akcji (kierunków) zależy od stanu:

$$
\mathcal{A}(s) = \{a\in\{0,\ldots,7\}:\text{odpowiednia krawędź jest legalna}\}.
$$

Wygodna maska akcji ma postać:

$$
m(s,a)=\begin{cases}1,&a\in\mathcal{A}(s),\\0,&\text{w przeciwnym razie}.\end{cases}
$$

Można zacząć od nagrody **terminalnej** $+1$ za zwycięstwo, $-1$ za porażkę i $0$ w pozostałych stanach. Nagrodę zawsze interpretujemy **z perspektywy gracza, którego decyzję aktualizujemy**.

> [!IMPORTANT]
> **Odbicia zmieniają matematykę gry dwuosobowej.** Po ruchu sterowanie nie zawsze przechodzi na przeciwnika. Standardowe aktualizacje Q-learningu z automatyczną zamianą znaku po *każdym* odcinku byłyby błędne; w metodach actor–critic należy równie starannie zdefiniować perspektywę nagrody i trajektorie.

Dla wartości stanów zapisanych z perspektywy **gracza będącego na ruchu** definiujemy:

$$
\eta_t=\begin{cases}
+1,& i_{t+1}=i_t,\\
-1,& i_{t+1}\neq i_t.
\end{cases}
$$

Wtedy dla przejścia nieterminalnego jeden z naturalnych targetów zero-sum ma postać:

$$
y_t=r_t+\gamma\eta_t\max_{a'\in\mathcal A(s_{t+1})}Q(s_{t+1},a').
$$

Alternatywnie można definiować przejście **od decyzji jednego gracza do jego kolejnej decyzji**: środowisko rozgrywa w międzyczasie ruchy przeciwnika. Jest to szczególnie wygodne w gotowych bibliotekach jednagentowego RL. Trzeba jednak zadbać, by polityka przeciwnika była wyraźnie określona i nie zmieniała się w trakcie pojedynczego epizodu.

### 1.3. Dlaczego 8 × 10 jest trudne?

Dla prostokątnej siatki **8 × 10 komórek** (bez dodatkowych krawędzi bramkowych) jest łącznie $338$ potencjalnych połączeń poziomych, pionowych i diagonalnych:

$$
|E| = 8\cdot11+9\cdot10+2\cdot8\cdot10=338.
$$

Każda z tych krawędzi może być użyta lub nie. Liczba *formalnych* konfiguracji wynosi więc $2^{338}$, choć ogromna większość jest nieosiągalna. To wyjaśnia, dlaczego **mała przestrzeń akcji nie oznacza małej przestrzeni stanów**.

---

## 2. Tabular Q-learning

**Rodzina:** model-free, value-based, off-policy  
**Aproksymacja:** brak — osobna wartość dla każdej pary $(s,a)$  
**Trudność implementacji:** ★☆☆☆☆ (niska)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★★★ (bardzo wysoka)

### Idea

Utrzymujemy tablicę wartości:

$$
Q(s,a)\approx\mathbb{E}[\text{zdyskontowany wynik}\mid s,a].
$$

Wybór akcji może być $\varepsilon$-greedy: eksploracja z prawdopodobieństwem $\varepsilon$, w przeciwnym razie najlepsza **legalna** akcja. Podstawowy schemat aktualizacji to:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha\bigl[y_t-Q(s_t,a_t)\bigr],
$$

$$
y_t=\begin{cases}
r_t,&\text{stan końcowy},\\
r_t+\gamma\eta_t\displaystyle\max_{a'\in\mathcal A(s_{t+1})}Q(s_{t+1},a'),&\text{w przeciwnym razie}.
\end{cases}
$$

Wariant z $\eta_t$ dotyczy **self-play z wartościami z perspektywy aktywnego gracza**. Przy opakowaniu środowiska w jednagentowy MDP przeciwko ustalonej polityce przeciwnika używa się zwykłego targetu Q-learningu na przejściach agenta.

### Plusy

- Bardzo łatwo wdrożyć i przetestować; doskonały **punkt odniesienia**.
- Przejrzysta interpretacja oszacowanych wartości $Q$.
- Nie wymaga frameworka deep learningowego ani GPU.
- Dobrze pokazuje wpływ rozmiaru planszy na pokrycie przestrzeni stanów.

### Minusy

- **Nie generalizuje** pomiędzy podobnymi stanami.
- Słaba efektywność próbkowa przy dużej liczbie konfiguracji krawędzi.
- Duża tablica lub słownik, często pełen stanów widzianych tylko raz.
- W self-play polityka przeciwnika ewoluuje, co dodatkowo utrudnia ocenę postępu.

### Wyzwania implementacyjne w Paper Soccer

1. **Jednoznaczny klucz stanu:** `(pozycja, bitset użytych krawędzi, gracz/perspektywa)`; rekomendowany słownik `dict` z domyślnymi zerowymi wartościami, a nie gęsta tablica.
2. **Maskowanie akcji:** maksimum tylko po akcjach legalnych.
3. **Odbicia i terminale:** brak automatycznego przełączania gracza; poprawne przypisanie zwycięzcy przy zablokowaniu.
4. **Eksploracja:** przeglądanie wszystkich możliwych stanów jest nierealistyczne na 8 × 10.
5. **Baseline wobec silniejszych metod:** warto raportować również pokrycie stanów, nie tylko wynik meczów.

**Rola w projekcie:** **minimalny, interpretowalny baseline**. Prawdopodobnie dużo lepiej poradzi sobie na małych planszach niż na standardowej — jest to hipoteza do zmierzenia, a nie założony wynik.

---

## 3. Linear Q-learning

**Rodzina:** model-free, value-based, off-policy z aproksymacją  
**Aproksymacja:** liniowa funkcja cech  
**Trudność implementacji:** ★★☆☆☆ (niska–średnia)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★★☆ (wysoka)

### Idea

Zamiast zapamiętywać każdą konfigurację gry, uczymy współczynniki cech:

$$
Q_\theta(s,a)=\theta^\top\phi(s,a),
$$

gdzie $\phi(s,a)\in\mathbb R^d$ reprezentuje **informację o stanie po wykonaniu ruchu** (lub o stanie i akcji łącznie). Przybliżony krok semi-gradient Q-learningu to:

$$
\delta_t = y_t-Q_\theta(s_t,a_t),
$$

$$
\theta\leftarrow\theta+\alpha\delta_t\,\phi(s_t,a_t).
$$

Dla self-play ze zmianą perspektywy używamy takiego samego znaku $\eta_t$ w $y_t$ jak w algorytmie tablicowym.

### Przykładowe cechy $\phi(s,a)$

| Cecha | Intuicja |
|---|---|
| Zmiana odległości do bramki rywala | Czy ruch przybliża do celu? |
| Liczba legalnych wyjść z nowego wierzchołka | Czy agent ryzykuje pułapkę? |
| Czy ruch prowadzi do odbicia | Czy agent zachowa inicjatywę? |
| Odległość do własnej bramki | Czy ruch zwiększa zagrożenie? |
| Lokalna liczba wykorzystanych krawędzi | Jak bardzo zatłoczony jest obszar? |
| Występowanie natychmiastowej wygranej/przegranej | Czy ruch ma natychmiastowy skutek terminalny? |

Cechy należy standaryzować lub normalizować, a także testować, czy nie wykorzystują informacji niedostępnej pozostałym agentom (np. spojrzenia w przyszłość poza regułami gry).

### Plusy

- Potrafi **generalizować** pomiędzy różnymi planszami i konfiguracjami krawędzi.
- Bardzo szybka predykcja i mała pamięć.
- Interpretowalne współczynniki; łatwo badać znaczenie poszczególnych cech.
- Stanowi naturalny pomost między tablicowym Q-learningiem a siecią neuronową.

### Minusy

- Silnie zależy od **ręcznie przygotowanych cech**.
- Liniowość może nie uchwycić wieloetapowych pułapek i kombinacji odbić.
- Off-policy + bootstrapping + aproksymacja funkcji może prowadzić do **rozbieżności** (tzw. *deadly triad*).
- Dobór skali cech, kroku uczenia i eksploracji może wyraźnie zmieniać wynik.

### Wyzwania implementacyjne w Paper Soccer

1. Przygotować cechy bez kosztownego przeszukiwania całego drzewa, aby porównanie pozostało przejrzyste.
2. Utrzymywać wspólną orientację cech dla obydwu zawodników.
3. Monitorować normę $\|\theta\|_2$, wartości $Q_\theta$ i ewentualne eksplozje aktualizacji.
4. Zastosować ostrożne skalowanie cech, mniejszy learning rate i clipping gradientów; **nie uznawać** ich za gwarancję zbieżności.
5. Rozdzielić wpływ aproksymacji od wpływu własnych heurystyk: opisać komplet cech i ewentualnie zrobić krótką ablację.

**Rola w projekcie:** sprawdzić, **czy ogólne cechy stanu pomagają przezwyciężyć ograniczenia tablicy**, zanim użyjemy sieci neuronowej.

---

## 4. Double DQN

**Rodzina:** model-free, value-based, off-policy, deep RL  
**Aproksymacja:** sieć neuronowa $Q_\theta(s,a)$  
**Trudność implementacji:** ★★★☆☆ (średnia)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★★☆ (wysoka)

### Idea

Sieć przewiduje do 8 wartości akcji:

$$
Q_\theta(s)=\bigl(Q_\theta(s,0),\ldots,Q_\theta(s,7)\bigr).
$$

Zwykły DQN używa tej samej maksymalizacji do wybrania i oceny kolejnej akcji, co może zawyżać oszacowania. **Double DQN** rozdziela te dwie czynności:

$$
a^*=\underset{a'\in\mathcal A(s_{t+1})}{\operatorname{argmax}}\;Q_\theta(s_{t+1},a'),
$$

$$
y_t=\begin{cases}
r_t,&\text{stan końcowy},\\
r_t+\gamma\eta_t Q_{\theta^-}(s_{t+1},a^*),&\text{w przeciwnym razie}.
\end{cases}
$$

Tu $\theta$ to parametry bieżącej sieci, a $\theta^-$ — okresowo aktualizowanej **sieci docelowej**. Minimalizujemy błąd:

$$
\mathcal L(\theta)=\mathbb E_{(s,a,r,s')\sim\mathcal D}
\left[\left(Q_\theta(s,a)-y\right)^2\right].
$$

$\mathcal D$ oznacza **experience replay buffer**. Znak $\eta_t$ znów jest potrzebny wyłącznie w określonej wyżej konwencji dwóch graczy.

### Jaką sieć zastosować?

- **Wersja podstawowa:** MLP na wektorze zawierającym pozycję piłki, mapę krawędzi i informacje o graczu.
- **Wersja opcjonalna:** CNN na wielokanałowej reprezentacji siatki (np. osobne kanały dla wykorzystanych poziomych, pionowych i ukośnych krawędzi).
- Wejście powinno zawierać **pełną informację o stanie**, a nie tylko położenie piłki.

### Plusy

- Generalizacja bez konieczności projektowania wszystkich cech ręcznie.
- Mała przestrzeń akcji (8 kierunków) bardzo dobrze pasuje do DQN.
- Double DQN ogranicza jeden z problemów zwykłego DQN — zawyżanie ocen akcji.
- Replay buffer umożliwia wielokrotne wykorzystanie doświadczeń.

### Minusy

- Więcej hiperparametrów oraz trudniejszy debugging niż w klasycznym Q-learningu.
- Replay miesza doświadczenia z różnych etapów self-play — rozkład przeciwników nie jest stacjonarny.
- Rzadkie nagrody terminalne utrudniają propagowanie sygnału uczenia.
- Brak jawnego wieloruchowego planowania może utrudniać długie kombinacje.

### Wyzwania implementacyjne w Paper Soccer

1. **Dokładny tensor obserwacji:** wykorzystane krawędzie, granice, bramki, pozycja piłki i to, kto gra.
2. **Maska akcji w eksploracji i targetach:** agent nie może wybierać nielegalnego ruchu, także w `argmax` sieci docelowej.
3. **Replay buffer:** przechowywać również maskę następnego stanu, `terminated`, bieżącego gracza oraz informację, czy po ruchu nastąpiła zmiana gracza.
4. **Self-play:** trenować z pulą zamrożonych checkpointów, zamiast jednocześnie zmieniać przeciwnika przy każdym kroku.
5. **Kontrola treningu:** logować TD loss, rozkład oszacowanych $Q$, częstotliwość zwycięstw i jakość przeciwko stałym przeciwnikom.

**Rola w projekcie:** odpowiedź na pytanie, **czy nieliniowa aproksymacja przewyższa ręcznie zaprojektowaną reprezentację liniową**.

---

## 5. Maskable PPO

**Rodzina:** model-free, on-policy, policy-gradient / actor–critic  
**Aproksymacja:** sieci polityki $\pi_\theta$ i wartości $V_\psi$  
**Trudność implementacji:** ★★☆☆☆ (z biblioteką) / ★★★★☆ (od zera)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★☆☆ (średnia–wysoka)

### Idea

Uczenie nie opiera się na ocenie każdej akcji przez funkcję $Q$, lecz bezpośrednio na polityce:

$$
\pi_\theta(a\mid s)=\Pr_\theta(A_t=a\mid S_t=s).
$$

Przed normalizacją logits wykluczamy ruchy nielegalne. W uproszczeniu:

$$
\pi_\theta^{\mathrm{mask}}(a\mid s)=
\frac{m(s,a)\exp(\ell_\theta(s,a))}
{\sum_{b\in\mathcal A(s)}\exp(\ell_\theta(s,b))},
$$

gdzie $\ell_\theta$ oznacza logit akcji; wzór zakłada niepusty zbiór legalnych ruchów. Jeśli nie ma ruchów, **gra musi zostać zakończona przed wywołaniem polityki**.

PPO aktualizuje politykę z ograniczeniem skali pojedynczego kroku:

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)\right],
$$

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

$\hat A_t$ to oszacowanie przewagi akcji, zwykle przez GAE.

### Plusy

- Uczy **bezpośrednio rozkładu akcji**, a nie tylko ich wartości.
- Maskowanie jest naturalnym rozwiązaniem przy zmieniającym się zbiorze legalnych ruchów.
- Istnieje gotowa implementacja `sb3-contrib/MaskablePPO`.
- Stosunkowo prosty pipeline treningowy po poprawnym zdefiniowaniu środowiska.

### Minusy

- On-policy — nie wykorzystuje doświadczeń tak wielokrotnie jak DQN.
- Wrażliwy na nagrody rzadkie i niestabilność treningu przeciwko ewoluującym przeciwnikom.
- Sam PPO nie planuje wieloetapowo podczas podejmowania decyzji.
- Nie rozwiązuje automatycznie problemu gry dwuosobowej — potrzebuje właściwego opakowania środowiska.

### Wyzwania implementacyjne w Paper Soccer

1. **Maskowanie:** metoda `action_masks()` musi zwracać dokładnie 8 wartości `bool`, gdzie `True` oznacza akcję legalną.
2. **Gra przeciwko przeciwnikowi:** typowe Gymnasium + SB3 to interfejsy jednagentowe. Najprościej opakować środowisko tak, aby uczył się jeden gracz, a drugi wykonywał ruchy automatycznie według zamrożonej polityki; w kolejnych epizodach zamieniamy strony.
3. **Odbicia:** gdy ten sam gracz zyskuje dodatkowy ruch, obsłużyć kolejną decyzję bez błędnej zmiany właściciela nagrody.
4. **Ewaluacja:** używać mask-aware `evaluate_policy`/`MaskableEvalCallback` z `sb3-contrib` — standardowe odpowiedniki SB3 mogą oceniać model niepoprawnie.
5. **Niestacjonarny przeciwnik:** harmonogram snapshotów i przeciwników musi być utrzymywany poza aktualną trajektorią rollout.

Dokumentacja: [Maskable PPO — SB3 Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

**Rola w projekcie:** reprezentant **bezpośredniego uczenia polityki**, umożliwiający porównanie z metodami Q-learningowymi.

---

## 6. A2C (Advantage Actor–Critic)

**Rodzina:** model-free, on-policy, policy-gradient / actor–critic  
**Aproksymacja:** sieci polityki $\pi_\theta(a\mid s)$ (*actor*) i wartości $V_\psi(s)$ (*critic*)  
**Trudność implementacji:** ★★★☆☆ (średnia z maskowaniem; ★★☆☆☆ bez maskowania)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★★☆ (niepewna — zależy od reprezentacji i treningu)

### Idea

A2C łączy dwa modele:

- **Actor** określa, którą legalną akcję wybrać: $\pi_\theta(a\mid s)$.
- **Critic** ocenia, jak dobry jest aktualny stan: $V_\psi(s)$.

Actor nie uczy się bezpośrednio z nagrody, ale z **przewagi** (*advantage*) akcji nad typowym wynikiem w danym stanie:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

W praktyce zamiast nieznanego $Q^\pi$ stosujemy np. **$n$-krokowy target**:

$$
\widehat G_t^{(n)}=
\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n(1-d_{t+n})V_\psi(s_{t+n}),
$$

$$
\widehat A_t=\widehat G_t^{(n)}-V_\psi(s_t),
$$

gdzie $d_{t+n}=1$ oznacza, że na końcu rozpatrywanego fragmentu występuje stan terminalny. Dla przejrzystości $t$ oznacza tu **kolejne decyzje uczącego się gracza** w jednagentowym adapterze środowiska (ruchy przeciwnika rozgrywane są wewnątrz adaptera). Otrzymujemy więc zwykłe dodatnie bootstrapowanie wartości z perspektywy tego samego gracza.

Przykładowe straty (minimalizowane) to:

$$
L_{\mathrm{actor}}(\theta)
=-\mathbb E_t\left[\log\pi_\theta(a_t\mid s_t)\,\operatorname{stopgrad}(\widehat A_t)\right],
$$

$$
L_{\mathrm{critic}}(\psi)
=\frac12\mathbb E_t\left[\left(V_\psi(s_t)-\widehat G_t^{(n)}\right)^2\right],
$$

$$
L_{\mathrm{total}}=
L_{\mathrm{actor}}+c_v L_{\mathrm{critic}}-c_e\,\mathbb E_t\left[H(\pi_\theta(\cdot\mid s_t))\right].
$$

$H$ to entropia polityki: składnik entropijny zachęca do eksploracji.

### Maskowanie akcji — konieczne w Paper Soccer

Tak samo jak w PPO, rozkład polityki musi być normalizowany **wyłącznie po legalnych akcjach**:

$$
\pi^{\mathrm{mask}}_\theta(a\mid s)=
\begin{cases}
\dfrac{\exp(\ell_\theta(s,a))}
{\sum_{b\in\mathcal A(s)}\exp(\ell_\theta(s,b))}, & a\in\mathcal A(s),\\[6pt]
0, & a\notin\mathcal A(s).
\end{cases}
$$

Dla pustego zbioru $\mathcal A(s)$ środowisko **kończy partię**, zanim wywoła politykę.

> [!IMPORTANT]
> **Standardowe `stable_baselines3.A2C` nie udostępnia gotowej obsługi masek akcji** porównywalnej z `sb3_contrib.MaskablePPO`. Nie należy zakładać, że samo udostępnienie `env.action_masks()` automatycznie włączy masking w A2C. Najczyściej zaimplementować niedużą wersję **Masked A2C w PyTorch**, wykorzystując wspólny kod środowiska i rozkładu `Categorical` z maskowanymi logitami. Inne podejście wymaga modyfikacji mechanizmu polityki i rolloutu w SB3. Maskowanie powinno działać **zarówno podczas zbierania trajektorii, jak i obliczania log-probability oraz entropii**.

### Czym różni się od Maskable PPO?

Oba algorytmy mogą mieć tę samą architekturę actor–critic i taką samą maskę. **A2C** wykonuje klasyczną aktualizację gradientu polityki na zebranych trajektoriach, a **PPO** ogranicza wielkość zmian polityki przez clipped objective i zwykle wielokrotnie wykorzystuje tę samą partię rolloutów w kolejnych epokach optymalizacji.

PPO używa m.in. ilorazu prawdopodobieństw starej i nowej polityki; prosty A2C nie potrzebuje go do pojedynczej aktualizacji on-policy. Dzięki temu porównanie **A2C vs PPO** jest ciekawą analizą stabilizacji uczenia polityki, a nie porównaniem dwóch odległych paradygmatów.

### Plusy

- Zrozumiała konstrukcja **actor–critic** i prostszy cel optymalizacyjny niż PPO.
- Brak replay buffera, target network i kosztownego przeszukiwania drzewa.
- Dobrze nadaje się do porównania z PPO przy **tej samej reprezentacji stanu i podobnej architekturze sieci**.
- Można łatwo wizualizować rozkłady prawdopodobieństw legalnych ruchów i wartości stanów.

### Minusy

- **On-policy**: doświadczenia nie są ponownie wykorzystywane tak intensywnie jak w metodach off-policy.
- Aktualizacje mogą mieć **większą wariancję** i być mniej stabilne niż w PPO (zależnie od hiperparametrów).
- Przy rzadkich nagrodach terminalnych potrzebuje ostrożnego doboru horyzontu $n$, $\gamma$, współczynnika entropii i wartości początkowych.
- Brak gotowego maskowania w podstawowym A2C SB3 oznacza dodatkową pracę implementacyjną.

### Wyzwania implementacyjne w Paper Soccer

1. **Masked Categorical:** nielegalne kierunki mają prawdopodobieństwo zero przy zbieraniu doświadczeń, uczeniu i ewaluacji. Należy unikać `log(0)` oraz prób normalizacji pustej maski.
2. **Jednagentowy adapter i odbicia:** jeden krok RL to decyzja uczącego się gracza, po której adapter może automatycznie wykonać akcje przeciwnika do momentu ponownego oddania sterowania agentowi lub końca gry. Jeśli nastąpi odbicie, agent ma kolejną decyzję bez przełączenia strony.
3. **Nagrody i bootstrapping:** nagroda oraz wartość końcowa muszą być mierzone **z perspektywy uczącego się gracza**, by nie stosować błędnej aktualizacji po ruchach przeciwnika.
4. **On-policy rollouts:** nie zmieniać przeciwnika w środku pojedynczej trajektorii; checkpointy dobieramy pomiędzy epizodami.
5. **Zmienność wyników:** raportować wiele seedów, entropy, actor loss, critic loss i win rate wobec stałych przeciwników.
6. **Rzetelne zestawienie z PPO:** uzgodnić architekturę, obserwację, maskę, pulę przeciwników i budżety; różnice hiperparametrów optymalizować osobno, ale w porównywalnym zakresie.

**Rola w projekcie:** ocenić, **czy mechanizm PPO (clipping i powtarzane aktualizacje) daje mierzalną przewagę nad prostszym A2C** w grze o rzadkich nagrodach i zmiennych legalnych akcjach.

Dokumentacja: [A2C — Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html); artykuł źródłowy dla rodziny advantage actor–critic: [Mnih et al. (2016)](https://arxiv.org/abs/1602.01783).

---

## 7. Dyna-style Double DQN — model-based RL z symulowanymi doświadczeniami

**Rodzina:** model-based reinforcement learning, podejście Dyna + off-policy, deep value-based RL  
**Model środowiska:** **dokładny, znany symulator reguł Paper Soccer** — nie trenujemy osobnego predyktora dynamiki  
**Aproksymacja:** ta sama sieć $Q_\theta(s,a)$ i Double DQN co w sekcji 4  
**Trudność implementacji:** ★★★☆☆ (średnia; rozszerzenie już działającego Double DQN)  
**Trudność uzyskania silnego agenta 8 × 10:** ★★★★☆ (wysoka; dodatkowe symulacje nie gwarantują sukcesu)

> [!NOTE]
> **„Dyna-style Double DQN” to nazwa projektowego wariantu, a nie jeden standardowy algorytm o kanonicznej implementacji.** Bierzemy schemat Dyna (uczenie na doświadczeniach rzeczywistych i generowanych przez model) i stosujemy aktualizacje Double DQN. Od początku dokumentujemy dokładną procedurę generowania symulacji.

### Idea: czym różni się od Double DQN?

W zwykłym Double DQN aktualizujemy sieć na podstawie przejść *zaobserwowanych podczas gry*:

$$
(s_t,a_t,r_t,s_{t+1},d_t)\in\mathcal D_{\mathrm{real}}.
$$

W wariancie Dyna wykorzystujemy znaną funkcję przejścia $f$ do wygenerowania **dodatkowych, poprawnych przejść**, także dla legalnych akcji, których w danym stanie agent wcześniej nie wybrał:

$$
(s',r,d,i')=f(s,a,i),
\qquad a\in\mathcal A(s).
$$

Dzięki temu tworzymy drugi zbiór doświadczeń $\mathcal D_{\mathrm{sim}}$ i trenujemy na mieszaninie:

$$
\mathcal D_{\mathrm{train}}=
\mathcal D_{\mathrm{real}}\cup\mathcal D_{\mathrm{sim}}.
$$

**Uwaga:** ponowne pobieranie *tych samych* rzeczywistych przejść z replay buffera byłoby zwykłym replay, nie dodatkowym planowaniem. W Dyna ważne jest **rozgałęzianie symulacji od znanych stanów** (np. sprawdzanie innych legalnych ruchów) lub wykonywanie krótkich nowych rolloutów.

### Aktualizacja Double DQN nie zmienia się

Wyznaczamy najlepszą legalną akcję za pomocą sieci online:

$$
a^*=\underset{b\in\mathcal A(s')}{\operatorname{argmax}}\;Q_\theta(s',b).
$$

Dla wartości z perspektywy gracza będącego na ruchu, przy przejściu nieterminalnym:

$$
y=r+\gamma\eta\,Q_{\theta^-}(s',a^*),
\qquad
\eta=\begin{cases}
+1,&\text{ten sam gracz po odbiciu},\\
-1,&\text{ruch przechodzi na przeciwnika}.
\end{cases}
$$

Przy stanie terminalnym $y=r$. W tym wariancie znak $\eta$ obowiązuje **również dla przejść symulowanych**. Jeśli cała implementacja korzysta zamiast tego z adaptera jednego uczącego się gracza, symulowane przejścia muszą mieć **taką samą semantykę**, łącznie z automatycznymi ruchami zamrożonego przeciwnika. Nie należy mieszać tych dwóch typów przejść w jednym replay bufferze bez rozróżnienia.

### Minimalna procedura w Paper Soccer

1. Trenujemy klasycznego Double DQN i przechowujemy pełne, możliwe do odtworzenia stany gry (w tym wykorzystane krawędzie i aktywnego gracza).
2. Po każdym rzeczywistym przejściu losujemy **$K$ stanów startowych** z odwiedzonych pozycji.
3. W każdym stanie wybieramy legalną akcję do symulacji — np. losową lub $\varepsilon$-greedy; warto dopuszczać **alternatywne, kontrfaktyczne akcje**.
4. Uruchamiamy `simulate(state, action)` na **kopii stanu**, bez ingerowania w rzeczywistą partię. Zapisujemy poprawny następny stan, nagrodę, terminal i maskę legalnych akcji.
5. Dodajemy symulowane próbki do oddzielnego bufora (albo oznaczamy ich pochodzenie) i uczymy Double DQN na ustalonej mieszaninie przejść realnych i symulowanych.
6. W wariancie podstawowym symulujemy **jedno przejście**. Opcjonalny wariant rozwinięty używa rolloutów długości $H>1$ (z prawidłowym modelem zachowania przeciwnika, jeśli jest on rozgrywany wewnątrz adaptera).

Schemat przepływu:

```text
       rzeczywista gra
             |
       real replay buffer
             |                  zapisane stany
             |                        |
             |                  dokładny model gry
             |                        |
             |                  simulated replay
             |                        |
             +--------- mieszanie ----+
                         |
                Double DQN update
```

### Najważniejsze parametry eksperymentalne

- $K$ — **liczba symulowanych przejść** na jeden rzeczywisty ruch, np. $K\in\{0,5,10,20\}$;
- $H$ — długość symulowanego rolloutu; na początek $H=1$;
- $\rho$ — udział próbek symulowanych w minibatchu;
- liczba kroków optymalizacji na rzeczywisty ruch: **powinna być kontrolowana**, aby odróżnić korzyść ze *specjalnie wygenerowanych doświadczeń* od samego zwiększenia liczby aktualizacji.

Przy $K=0$ i tych samych pozostałych ustawieniach otrzymujemy kontrolny wariant Double DQN **bez dodatkowego planowania**.

### Plusy

- Korzysta z **dokładnego modelu reguł**, więc nie wprowadza błędu wyuczonego modelu dynamiki.
- Większość kodu (sieć, target network, TD loss, legal action masking) można współdzielić z Double DQN.
- Pozwala analizować **sample efficiency**: jakość przy danej liczbie rzeczywistych interakcji.
- Daje osobne, bardzo czytelne pytanie badawcze: *czy dodatkowe planowanie poprawia grę?*
- Nie wymaga MCTS, policy/value network AlphaZero ani uczenia modelu świata.

### Minusy

- Symulacje **też zużywają zasoby**. Przewaga przy tej samej liczbie rzeczywistych ruchów nie musi oznaczać przewagi przy tym samym czasie treningu.
- Przy rzadkich nagrodach terminalnych jednoetapowe symulacje mogą generować głównie zerowe nagrody; poprawa nie jest pewna.
- Wynik mocno zależy od sposobu wyboru stanów, akcji i proporcji real/simulated.
- Same losowe symulacje z niewłaściwych pozycji mogą pogorszyć trening przez zły rozkład danych.
- Niestacjonarność self-play i używanie różnych checkpointów przeciwnika mogą wprowadzać dodatkowe niespójności.

### Wyzwania implementacyjne w Paper Soccer

1. **Wierne odtwarzanie stanu:** kopiowanie bitsetu krawędzi, odwiedzonych wierzchołków, pozycji, aktywnego gracza, statusu odbicia i geometrii bramek.
2. **Model ≡ silnik gry:** testować, że `simulate(copy(s), a)` daje *identyczny* wynik jak rzeczywiste `step(s, a)` dla tych samych legalnych par $(s,a)$.
3. **Legalność:** symulować wyłącznie akcje $a\in\mathcal A(s)$ i zachowywać maski stanów następnych.
4. **Perspektywa nagród:** odbicia, zmiana gracza, samobój i blokada muszą działać tak samo w real i sim replay.
5. **Porównanie z bazowym Double DQN:** te same wejścia, sieć, optimizer, replay, przeciwnicy, seedy i procedura ewaluacji. Osobno raportować real steps, simulated steps, gradient updates i wall-clock.
6. **Nie zakładać z góry przewagi Dyna:** testować krzywe uczenia i wrażliwość na $K$, a nie jedynie końcowy wynik z jednym budżetem.

**Rola w projekcie:** kontrolowane porównanie **model-free Double DQN** z **model-based rozszerzeniem tej samej sieci**, przy minimalnych dodatkowych kosztach implementacji względem AlphaZero.

Punkt wyjścia: [Sutton & Barto, rozdział *Dyna: Integrated Planning, Acting, and Learning*](http://incompleteideas.net/book/the-book-2nd.html) oraz [Double DQN — van Hasselt et al. (2016)](https://arxiv.org/abs/1509.06461).

---

## 8. Zestawienie i rekomendacje implementacyjne

| Metoda | Generalizuje między stanami? | Uczy polityki czy wartości? | Planning przy decyzji? | Nakład wdrożeniowy | Główne ryzyko |
|---|---|---|---|---|---|
| **Tabular Q-learning** | Nie | $Q(s,a)$ | Nie | **1/5** | Eksplozja stanów |
| **Linear Q-learning** | Tak, przez cechy | $Q_\theta(s,a)$ | Nie | **2/5** | Słabe cechy, niestabilność |
| **Double DQN** | Tak, przez NN | $Q_\theta(s,a)$ | Nie | **3/5** | Stabilizacja i replay |
| **Maskable PPO** | Tak, przez NN | $\pi_\theta, V_\psi$ | Nie | **2/5** z biblioteką | Adapter self-play |
| **A2C (masked)** | Tak, przez NN | $\pi_\theta,V_\psi$ | Nie | **3/5** | Maskowanie, wariancja gradientów i reward |
| **Dyna-style Double DQN** | Tak, przez NN | $Q_\theta(s,a)$ + znany model gry | **Tak, w treningu** | **3/5** po Double DQN | Koszt symulacji i dobór stanów startowych |

### Proponowany wspólny interfejs

```python
class PaperSoccerAgent:
    def act(self, observation, legal_actions, deterministic=True) -> int:
        """Zwraca indeks legalnej akcji od 0 do 7."""

    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
```

Warto osobno zdefiniować warstwę `GameState` (reguły, terminale, maska), warstwę `Env` (trening) i `TournamentRunner` (ewaluacja). Dzięki temu **każdy algorytm gra w tę samą grę**.

### Zalecana kolejność

1. Deterministyczny silnik Paper Soccer + testy własności reguł.
2. Random agent i agent heurystyczny jako wspólni przeciwnicy.
3. Tabular Q-learning, a następnie Linear Q-learning.
4. Double DQN.
5. Maskable PPO na adapterze jednagentowym.
6. A2C z maskowaniem akcji; najlepiej wydzielić wspólny kod polityki actor–critic dla A2C i PPO.
7. Dyna-style Double DQN — rozszerzenie działającego Double DQN o symulator i replay symulowanych przejść.

### Krytyczne testy przed treningiem

- Żadna akcja nie może używać wykorzystanej krawędzi.
- Odbicie zachowuje gracza; brak odbicia zmienia gracza.
- Wynik terminalny ma poprawny znak dla każdego zawodnika (także po samobóju).
- Każda partia kończy się w skończonej liczbie ruchów (krawędzie nie są używane ponownie).
- Zapis i odtworzenie stanu daje tę samą maskę i zwycięzcę.
- A2C i PPO nigdy nie wybierają nielegalnych akcji, a aktualizacje korzystają z prawidłowych `log_prob` dla maskowanej polityki.
- Double DQN i Dyna-style Double DQN używają tej samej architektury Q-network; dodatkowe przejścia w Dyna są poprawne i oznaczone jako symulowane.
- `simulate(copy(s), a)` i zwykłe `step(s,a)` zwracają identyczne przejścia na legalnych stanach; symulacja nie zmienia oryginalnego stanu.
- Każda metoda otrzymuje tę samą informację o stanie; różnić się może sposób aproksymacji i wykorzystanie znanego modelu gry.

## 9. Źródła i materiały

- [Paper Football / Paper Soccer — zasady](https://www.paper-football.com/).
- [Watkins & Dayan (1992): *Q-learning*](https://link.springer.com/article/10.1007/BF00992698).
- [van Hasselt, Guez & Silver (2016): *Deep Reinforcement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461).
- [Schulman et al. (2017): *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347).
- [SB3-Contrib: *Maskable PPO* — dokumentacja](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
- [Mnih et al. (2016): *Asynchronous Methods for Deep Reinforcement Learning* — A3C/A2C](https://arxiv.org/abs/1602.01783).
- [Stable-Baselines3: *A2C* — dokumentacja](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html).
- [Huang & Ontañón (2020): *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171).
- [Sutton & Barto: *Reinforcement Learning: An Introduction* (2nd ed.), rozdział o Dyna](http://incompleteideas.net/book/the-book-2nd.html).
- [Sutton (1990): *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming*](https://doi.org/10.1016/S0020-7373(08)80020-1).

---

*Dokument koncepcyjny — trudności i przewidywana skuteczność są oceną projektową, a nie empirycznymi wynikami na Paper Soccer.*
