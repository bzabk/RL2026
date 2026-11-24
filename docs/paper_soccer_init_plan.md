# Paper Soccer — plan projektu i porównania agentów RL

> **Proponowany tytuł:** *Learning to Play Paper Soccer: Comparing Model-Free and Model-Based Reinforcement Learning*  
> **Badane metody:** Tabular Q-learning · Linear Q-learning · Double DQN · Maskable PPO · A2C · **Dyna-style Double DQN**  
> **Cel końcowy:** sześć wytrenowanych agentów, odtwarzalny turniej, analiza jakości i kosztu oraz zestaw wykresów do prezentacji.

---

## 1. Krótki opis projektu

Paper Soccer jest deterministyczną grą dwuosobową z pełną informacją, rozgrywaną na grafie, którego krawędzie są stopniowo zużywane. Liczba kierunków ruchu jest mała, lecz liczba konfiguracji planszy szybko rośnie. Nietypową cechą są **odbicia**, które pozwalają temu samemu graczowi kontynuować sekwencję ruchów.

Chcemy sprawdzić, **jak zmienia się siła gry i koszt uczenia**, gdy przechodzimy od tablicowego Q-learningu przez aproksymację liniową i głęboką do dwóch metod actor–critic (A2C i PPO). **Szósty agent — Dyna-style Double DQN — bada wpływ planowania przez symulowanie dodatkowych przejść w dokładnie znanym modelu gry.** Dzięki temu zestaw obejmuje zarówno metody *model-free*, jak i jedną metodę *model-based*.

**Główny rezultat:** nie tylko ranking agentów, ale również odpowiedź na pytanie, *dlaczego* określone podejście działa lepiej lub gorzej w tej grze.

### Czego nie zakładamy z góry?

- Że PPO zawsze wygra z A2C; oba mogą być wrażliwe na rzadkie nagrody i dobór hiperparametrów.
- Że Linear Q-learning będzie lepszy od Tabular Q-learning na każdej planszy.
- Że PPO będzie mocniejsze od Double DQN lub odwrotnie.
- Że ranking z jednego turnieju wystarczy do wykazania ogólnej przewagi algorytmu.
- Że Dyna na pewno pokona Double DQN: dodatkowe symulacje mogą pomóc w sample efficiency, ale także zwiększyć koszt i wprowadzić niekorzystny rozkład próbek.

## 2. Pytania badawcze (RQ)

| Kod | Pytanie |
|---|---|
| **RQ1** | Jaką siłę gry osiągają poszczególne algorytmy na standardowej planszy 8 × 10? |
| **RQ2** | Czy liniowa aproksymacja poprawia skalowalność tablicowego Q-learningu? |
| **RQ3** | Jak na tej samej grze wypadają metody value-based (Q-learning i Double DQN) w porównaniu z actor–critic (A2C i PPO)? |
| **RQ4** | Czy PPO jest stabilniejsze lub efektywniejsze próbkowo od prostszego A2C przy takim samym budżecie interakcji i podobnej architekturze? |
| **RQ5** | Jak wraz z rozmiarem planszy zmieniają się wyniki i efektywność uczenia? |
| **RQ6** | Czy dodatkowe, model-based symulacje w Dyna-style Double DQN poprawiają sample efficiency i siłę gry względem tego samego Double DQN bez planowania? |

## 3. Hipotezy do sprawdzenia

Poniższe hipotezy to **przewidywania przed eksperymentem**, nie wnioski.

**H1 — Generalizacja.** Na 8 × 10 Linear Q-learning osiągnie wyższy średni wynik przeciwko ustalonym przeciwnikom niż Tabular Q-learning przy takim samym budżecie interakcji treningowych.

**H2 — Reprezentacja nieliniowa.** Double DQN osiągnie lepsze wyniki niż Linear Q-learning na 8 × 10, przy tej samej liczbie rzeczywistych interakcji treningowych.

**H3 — Metody polityki i wartości.** Wyniki Double DQN i Maskable PPO **będą się różnić** pod względem jakości i/lub efektywności uczenia. Celowo nie zakładamy kierunku różnicy.

**H4 — PPO vs A2C.** Maskable PPO osiągnie bardziej stabilne wyniki pomiędzy niezależnymi uruchomieniami niż A2C z maskowaniem, przy zbliżonej architekturze, informacji o stanie i budżecie rzeczywistych interakcji. Porównamy również sample efficiency; przewagi PPO nie zakładamy z góry.

**H5 — Skala planszy.** Różnica między wynikami metod aproksymujących a Tabular Q-learningiem wzrośnie wraz z rozmiarem planszy.

**H6 — Model-based planowanie.** Przy tej samej liczbie **rzeczywistych** interakcji Dyna-style Double DQN z odpowiednio dobraną liczbą symulacji osiągnie lepszy wynik niż Double DQN bez dodatkowych przejść. Przewaga może zniknąć przy tym samym budżecie **wall-clock**, co mierzymy osobno.

Każdą hipotezę oceniamy zarówno na podstawie wielkości efektu, jak i niepewności estymacji; nie sprowadzamy analizy wyłącznie do stwierdzenia „potwierdzona/odrzucona”.

---

## 4. Wspólne środowisko i dane wejściowe

### 4.1. Reguły

Budujemy jedno deterministyczne środowisko `PaperSoccerEnv`, używane przez wszystkie metody:

- Standardowa plansza **8 × 10 pól**, plus opcjonalnie mniejsze **4 × 4** i **6 × 8**.
- Piłka startuje w środku; legalne są lokalne przesunięcia do 8 sąsiadów.
- Wykorzystane krawędzie są niedostępne.
- Odbicie po wejściu na odwiedzony punkt / granicę zachowuje gracza.
- Gol, samobój i zablokowanie rozstrzygają wynik według **jednej spisanej specyfikacji**.

Przed treningiem należy ustalić m.in. **dokładne krawędzie bramek, oznaczanie linii brzegowych i regułę przecinania przekątnych**. Wszystkie algorytmy muszą działać na identycznych zasadach. Dobrym punktem wyjścia są [zasady Paper Soccer](https://www.paper-football.com/).

### 4.2. Stan i akcje

$$
s_t=(p_t,U_t,i_t),\qquad\mathcal A(s_t)\subseteq\{0,\ldots,7\}.
$$

- $p_t$: położenie piłki;
- $U_t$: użyte krawędzie (bitset / kanały siatki);
- $i_t$: aktywny gracz;
- $\mathcal A(s_t)$: legalne kierunki.

**Każdy agent widzi tę samą informację**. Model liniowy dostaje jej jawnie wyliczone cechy; sieci mogą korzystać z tensorów lub wektorów, ale żadna metoda nie może otrzymywać ukrytych informacji o przyszłym wyniku.

Dla sieci warto zastosować **kanoniczną orientację** (perspektywa gracza na ruchu), by wejście nie zależało od arbitralnego koloru zawodnika. Przekształcenia orientacji muszą również odpowiednio przekształcać akcje.

### 4.3. Nagroda i perspektywa

Proponujemy przede wszystkim nagrodę terminalną:

$$
r_T=\begin{cases}
+1,&\text{zwycięstwo rozpatrywanego gracza},\\
-1,&\text{jego porażka}.
\end{cases}
$$

W stanach nieterminalnych $r_t=0$ (wariant podstawowy). **Nie dodajemy na początku nagrody za zbliżanie się do bramki** — mogłaby wprowadzać preferencje heurystyczne niezależne od faktycznej wygranej. Reward shaping zostawiamy jako ewentualną ablację.

**Uwaga krytyczna o grze dwuosobowej:** zwykłe jednagentowe targety TD nie mogą być bezrefleksyjnie stosowane przy treningu obu stron. Jeśli $Q(s,a)$ i $V(s)$ są zapisane z perspektywy aktywnego gracza, przejście z oddaniem ruchu przeciwnikowi wymaga zmiany znaku, a przejście po odbiciu — nie. Dla PPO **i A2C** użyjemy tego samego jednagentowego adaptera: automatycznie rozgrywa on ruchy zamrożonego przeciwnika, aż nadejdzie kolejna decyzja agenta. Standardowe A2C z SB3 nie obsługuje natywnie masek akcji; planujemy więc A2C z maskowanym rozkładem akcji (np. lekka implementacja PyTorch), a nie karanie za nielegalne ruchy. Kroki RL w tym adapterze mogą obejmować kilka elementarnych odcinków — obie jednostki raportujemy osobno.

### 4.4. Testy poprawności (blokujące dalszy trening)

- Inwariant: wykorzystana krawędź nigdy nie wraca do zbioru legalnych.
- Każdy ruch jest legalny i przesuwa piłkę o jeden prawidłowy odcinek.
- Odbicia zachowują tego samego gracza, pozostałe ruchy go zmieniają.
- Poprawne zwycięstwo, samobój, blokada i stany terminalne.
- Kompletna informacja o stanie pozwala odtworzyć tę samą partię.
- Specjalny test A2C/PPO: maskowane polityki przydzielają zero prawdopodobieństwa nielegalnym akcjom podczas treningu i ewaluacji; odbicia nie zmieniają gracza.
- Specjalny test Dyna: przejście symulowane z kopii $(s,a)$ daje dokładnie taki sam stan, nagrodę, terminal, gracza i maskę jak prawdziwy krok z $(s,a)$, nie modyfikując oryginału.

---

## 5. Sześciu agentów — co dokładnie porównujemy?

| Agent | Funkcja / polityka | Sposób treningu | Koszt decyzji |
|---|---|---|---|
| **Tabular Q-learning** | Słownik $Q(s,a)$ | Interakcje + TD + self-play | Odczyt tablicy |
| **Linear Q-learning** | $Q_\theta(s,a)=\theta^\top\phi(s,a)$ | Interakcje + TD + self-play | Iloczyn skalarny |
| **Double DQN** | Sieć $Q_\theta(s,a)$ | Replay + sieć docelowa + self-play | Jedna predykcja NN |
| **Maskable PPO** | $\pi_\theta(a\mid s)$ i $V_\psi(s)$ | Rollouty on-policy z pulą przeciwników | Jedna predykcja NN |
| **A2C (masked)** | $\pi_\theta(a\mid s)$ i $V_\psi(s)$ | Krótkie rollouty on-policy + actor–critic z pulą przeciwników | Jedna predykcja NN |
| **Dyna-style Double DQN** | Ten sam $Q_\theta(s,a)$ co Double DQN + dokładny model gry | Real replay + kontrfaktyczne lub krótkie symulowane przejścia + Double DQN | **Jedna predykcja NN; planowanie tylko podczas treningu** |

### Przeciwnicy treningowi

Aby ograniczyć niestacjonarność dla metod model-free:

1. **Faza wstępna:** część partii przeciwko polityce losowej i prostej polityce heurystycznej.
2. **Faza self-play:** przeciwnik jest losowany z puli **zamrożonych wcześniejszych checkpointów** tej samej rodziny agenta.
3. **Równowaga stron:** w zbliżonej liczbie epizodów agent gra z obu stron planszy.
4. **Ewaluacja:** zawsze przeciwko **niezmiennym** przeciwnikom testowym, niewłączanym do procesu dopasowywania hiperparametrów.

Dla A2C i PPO używamy tego samego adaptera przeciwnika, podobnej sieci actor–critic, tej samej reprezentacji obserwacji i identycznej obsługi maski. A2C i PPO różnią się mechanizmem aktualizacji polityki. Pula checkpointów jest aktualizowana według ustalonego harmonogramu pomiędzy epizodami.

**Dla Double DQN i Dyna-style Double DQN** wykorzystujemy **tę samą sieć, optymalizator, eksplorację, pulę przeciwników i protokół przejść**. Różnica polega wyłącznie na użyciu znanego modelu reguł do generowania dodatkowych przykładów treningowych. Jeśli symulacja obejmuje akcje przeciwnika, musi stosować zamrożony checkpoint określony dla danego epizodu; nie może zakładać dowolnie przewidzianych ruchów przeciwnika.

### Jak liczyć budżet treningowy?

Najważniejsza jednostka: **pojedynczy narysowany odcinek / przejście elementarnego środowiska**, niezależnie od tego, czy ruch wykonał agent, czy jego przeciwnik.

Dodatkowo raportujemy:

- liczbę partii treningowych;
- liczbę **decyzji uczącego się agenta**;
- rzeczywisty czas treningu i maksymalne zużycie pamięci;
- dla A2C i PPO **liczbę rolloutów, aktualizacji gradientowych i epok optymalizacji**;
- liczbę wywołań sieci oraz liczbę aktualizacji parametrów (także dla Double DQN);
- dla Dyna: **real environment steps, simulated transitions, liczba symulacji na krok $K$, udział próbek symulowanych $\rho$**, całkowita liczba aktualizacji oraz czas symulowania gry.

**Ważne rozróżnienie:** przejścia **symulowane** w Dyna liczymy osobno — nie są dodatkowymi rzeczywistymi interakcjami, lecz wymagają czasu obliczeniowego. Ocenimy jakość przy (a) stałej liczbie real steps, (b) stałej liczbie aktualizacji sieci i (c) stałym czasie treningu. Adaptery on-policy mogą wykonać kilka elementarnych odcinków pomiędzy kolejnymi decyzjami uczącego się agenta. Raportujemy zatem zarówno **elementarne ruchy środowiska (obu graczy)**, jak i **liczbę decyzji i aktualizacji uczącego się agenta**. Ten sam budżet ruchów nie oznacza takiego samego czasu treningu ani liczby aktualizacji — pokazujemy oba wymiary.

---

## 6. Projekt eksperymentów

### E1. Porównanie siły gry na 8 × 10 — eksperyment główny

**Procedura:** po treningu wykonujemy turniej **round-robin** — każdy z sześciu agentów gra z każdym. To **15 unikalnych par**.

- Każda para gra z zamianą stron, w jednakowej liczbie spotkań.
- Wszystkie modele otrzymują wspólny limit treningowy według kilku budżetów.
- W głównym turnieju stosujemy deterministyczny wybór najlepszego legalnego ruchu (poza kontrolowanymi tie-breakami). Dodatkowo możemy oceniać polityki stochastyczne A2C/PPO w osobnym eksperymencie.
- Główna metryka: **win rate**; dodatkowo ranking punktowy / Elo jako opis pomocniczy.
- Ocenę uzupełniają mecze ze **wspólną pulą** agentów: random legal i prosta heurystyka — sam turniej między sześcioma wytrenowanymi modelami może zawierać nieprzechodnie zależności.

**Główne wyniki:** macierz zwycięstw, tabela siły gry, przedziały ufności.

### E2. Efektywność uczenia (*sample efficiency*)

Zapisujemy checkpointy modelu w kilku ustalonych punktach (proponowane do pilotażu):

$$
N_{\mathrm{env}}\in\{10^4,5\cdot10^4,10^5,5\cdot10^5,10^6\}.
$$

Każdy checkpoint oceniamy przeciwko **tej samej puli** przeciwników. Dla Dyna podajemy w szczególności **dwie osie X**: rzeczywiste interakcje i łączny koszt treningu; symulowanych przejść nie dopisujemy do rzeczywistych. W razie potrzeby modyfikujemy budżet **przed testem końcowym**, na podstawie pilotażu, nie wybierając punktów tylko pod korzystny wykres.

**Ważne:** wyznaczone budżety obejmują ruchy obu graczy. Osobno logujemy decyzje uczącego się agenta, aby uniknąć mylącego porównania algorytmów korzystających z odmiennych adapterów i harmonogramów aktualizacji.

**Główne wyniki:** win rate vs liczba ruchów; win rate vs czas treningu.

### E3. Skalowanie z rozmiarem planszy

Osobno trenujemy każdego agenta na:

- **4 × 4:** kontrola poprawności / łatwiejsza eksploracja;
- **6 × 8:** poziom pośredni;
- **8 × 10:** konfiguracja docelowa.

W pierwszej wersji **nie testujemy transferu między rozmiarami**, lecz wykonujemy niezależny trening na każdej planszy. Transfer wymagałby osobnej architektury wspierającej zmienne rozmiary i byłby kolejnym eksperymentem.

**Główne wyniki:** jakość gry vs rozmiar planszy; liczba unikalnych stanów odwiedzonych przez wariant tablicowy; czas treningu.

### E4. A2C vs Maskable PPO — czy clipping poprawia stabilność?

To **centralny eksperyment porównujący dwa blisko spokrewnione algorytmy actor–critic**. Dla A2C i PPO ustalamy:

- tę samą definicję stanu, identyczne maskowanie oraz jednakową sieć actor–critic (lub jak najbardziej podobną architekturę i liczbę parametrów);
- identyczną pulę przeciwników, zamianę stron i schemat nagród;
- z góry ustalone punkty pomiarowe według liczby elementarnych ruchów;
- osobne, ale porównywalne strojenie learning rate, liczby kroków rolloutu, entropii i parametru wartości.

Badamy trzy kwestie:

1. **Jakość gry:** win rate A2C vs PPO przeciwko wspólnym stałym przeciwnikom i bezpośrednio w turnieju.
2. **Stabilność:** rozrzut wyników pomiędzy seedami, gwałtowne spadki w learning curves, zmienność lossów polityki i krytyka.
3. **Efektywność:** jakość po tej samej liczbie elementarnych ruchów i po tym samym czasie treningu.

**Opcjonalna ablacja:** dla A2C i PPO ustawiamy identyczny współczynnik entropii lub taką samą pojemność sieci, aby ocenić wpływ samej procedury optymalizacji. Nie traktujemy różnych liczb epok uczenia PPO jako „darmowych” aktualizacji.

> [!IMPORTANT]
> **Nie używamy zwykłego SB3 A2C bez maski, jeśli PPO ma maskowanie.** A2C wymaga maskowanej polityki również na etapie treningu. Jeśli implementacja niestandardowa okaże się problematyczna, można współdzielić z PPO kod maskowanego rozkładu `Categorical`, ale nadal odrębnie implementować klasyczny cel A2C.

**Główne wyniki:** dwie learning curves A2C/PPO z pasmami niepewności, tabela różnic po ustalonych budżetach oraz wykres stabilności między seedami.

### E5. Dyna-style Double DQN vs Double DQN — wartość dodatkowego planowania

To **główny eksperyment model-based vs model-free**. Wariant Dyna ma tę samą sieć, reprezentację stanu, maskowanie, target network, eksplorację i przeciwników co podstawowy Double DQN. Różni się generowaniem **dodatkowych przejść przez dokładny model gry**.

**Warianty i kontrola:**

- $K\in\{0,5,10,20\}$ symulowanych przejść na **jeden rzeczywisty ruch**. $K=0$ to kontrola bez model-based planning.
- Podstawowy wariant: symulacja pojedynczych przejść ($H=1$) z **zapisanych wcześniej stanów**, ze sprawdzaniem alternatywnych legalnych akcji; opcjonalnie krótkie rollouty z $H>1$.
- Te same seedy, reguły i przeciwnicy oraz ten sam limit **real environment steps**.
- Kontrolny wariant **compute-matched**: Double DQN bez symulacji, ale z taką samą liczbą aktualizacji gradientowych jak Dyna. Dzięki temu sprawdzamy, czy zysk wynika z **nowych przejść**, a nie tylko z większej liczby treningowych update'ów.
- Zapisujemy osobno liczbę symulacji, kroków realnych, aktualizacji, wall-clock i udział przejść symulowanych w minibatchach.

**Pytania pomocnicze:**

1. Czy jakość gry wzrasta wraz z $K$, czy obserwujemy nasycenie albo pogorszenie?
2. Czy Dyna osiąga dany win rate po mniejszej liczbie **real steps**?
3. Czy zysk przy ograniczeniu real steps utrzymuje się przy **tym samym czasie treningu**?
4. Czy krótkie symulacje przy rzadkich nagrodach terminalnych rzeczywiście poprawiają propagację wartości?

**Uwaga metodologiczna:** symulacje muszą uwzględniać zużyte krawędzie, legalne akcje, odbicia, samobóje i zmianę gracza. W realnym self-play z dwiema perspektywami znak wartości w targetach jest taki sam dla przejść realnych i symulowanych. Nie wolno przypisywać sztucznych nagród za nieodbyte zwycięstwa.

**Główne wyniki:** wykres win rate vs real steps dla $K=0,5,10,20$; wykres win rate vs wall-clock; porównanie Dyna, bazowego Double DQN oraz compute-matched Double DQN.

### E6. Stabilność i powtarzalność

Każdy główny wariant uruchamiamy dla kilku seedów, docelowo **5 niezależnych treningów**. Jako minimum pilotażowe można wykonać 3 seedy i jawnie zaznaczyć zwiększoną niepewność wniosków.

Kontrolujemy:

- ziarna losowe środowiska, inicjalizacji i treningu;
- jednolite warianty zasad i przeciwników testowych;
- dobór hiperparametrów wyłącznie na walidacji / w pilotażu;
- stałą konfigurację sprzętową i wersje zależności.

Dla porównań par agentów korzystamy z symetrycznych meczów oraz — gdzie to możliwe — dopasowanych seedów i rozkładów startowych. Raportujemy **przedziały ufności**; uwzględniamy różnice pomiędzy niezależnymi treningami, a nie traktujemy wszystkich meczów jako całkowicie niezależnych obserwacji.

---

## 7. Metryki i sposób analizy

### 7.1. Siła gry

Dla agenta $A$ w $n$ meczach (bez remisów):

$$
\widehat{\mathrm{WinRate}}(A)=\frac{W_A}{n}.
$$

Jeśli przyjęta wersja gry dopuszcza remisy, wynik punktowy definiujemy jako $(W_A+\tfrac12 D_A)/n$; liczbę remisów należy raportować osobno. Dla pary $(A,B)$ zapisujemy również wyniki po zamianie stron.

### 7.2. Koszt i efektywność

- **Sample efficiency:** wynik wobec stałych przeciwników po $N_{\mathrm{env}}$ elementarnych ruchach treningu.
- **Wall-clock efficiency:** wynik po $t$ minutach/godzinach rzeczywistego treningu.
- **Inference latency:** mediana, p95 czasu decyzji (również przy odbiciach).
- **Compute budget:** aktualizacje gradientu, epoki optymalizacji, wywołania sieci, pamięć i czas CPU/GPU.
- **Stabilność:** średnia i rozrzut wyników dla seedów.

### 7.3. Dodatkowa analiza strategii

- Średnia długość wygranych/przegranych partii.
- Liczba odbić na partię i długość sekwencji odbić.
- Częstość przegranej przez zablokowanie vs gol/samobój.
- Dla Tabular Q-learningu: liczba odwiedzonych stanów i wizyt na parę $(s,a)$.
- Dla A2C i PPO: entropia polityki, wartości $V_\psi(s)$, długość rolloutów i rozkład prawdopodobieństw legalnych ruchów.
- Dla Dyna: liczba unikalnych symulowanych par $(s,a)$, liczba kontrfaktycznych akcji, udział real/sim w bufferze, koszt generowania symulacji oraz jakość względem $K$.

Są to metryki **opisowe**: nie dowodzą same w sobie lepszej strategii.

---

## 8. Jakie wykresy chcemy pokazać na prezentacji?

Poniższe wykresy są **planowanymi wizualizacjami**, a nie przykładowymi wynikami.

| Priorytet | Wykres | Osie / zawartość | Co pozwoli odpowiedzieć? |
|---|---|---|---|
| **MUST 1** | **Heatmapa wyników turnieju 6 × 6** | Wiersz = agent A, kolumna = B, kolor = win rate A vs B | Kto z kim wygrywa? Czy występują nieprzechodnie przewagi? |
| **MUST 2** | **Learning curves** | X = ruchy treningowe, Y = win rate vs stała pula; linie = 6 algorytmów | Kto uczy się najszybciej? |
| **MUST 3** | **Jakość vs koszt treningu** | X = godziny treningu, Y = win rate; punkty/checkpointy | Czy lepszy wynik wymaga większych zasobów? |
| **MUST 4** | **Ranking ze zmiennością** | Agent na osi Y, średnia wyników i przedział ufności na X | Czy różnice są wyraźne i stabilne? |
| **MUST 5** | **Wpływ planowania Dyna ($K$)** | X = real steps lub wall-clock; Y = win rate; linie = Double DQN, Dyna ($K=5,10,20$) i kontrola compute-matched | Czy nowe symulowane doświadczenia pomagają ponad zwykłe Double DQN? |
| **SHOULD 6** | **Skalowanie planszy** | X = 4 × 4, 6 × 8, 8 × 10; Y = win rate / score | Jak rośnie trudność zadania dla każdej metody? |
| **SHOULD 7** | **Jakość vs latencja decyzji** | X = mediana/p95 czasu wybrania ruchu, Y = win rate | Ile kosztuje wybór akcji dla każdego agenta? |
| **OPTIONAL 8** | **A2C vs PPO — stabilność** | X = budżet ruchów, Y = win rate; pasma = zmienność między seedami | Czy PPO stabilizuje uczenie względem A2C? |
| **OPTIONAL 9** | **Wizualizacja pojedynczej pozycji** | Plansza z wybranym ruchem i wartościami/probability akcji | Jak algorytmy różnią się w strategicznie trudnym stanie? |

### Sugerowana kolejność slajdów

1. **Motywacja** — zasady Paper Soccer, dlaczego gra jest trudna dla RL.
2. **Sześć metod** — diagram: tabular Q → linear Q → Double DQN → A2C/PPO (actor–critic) + Dyna-style Double DQN (model-based).
3. **Środowisko i fairness** — stan, maska akcji, self-play, odbicia.
4. **Wykres MUST 1** — heatmapa turnieju.
5. **Wykres MUST 2** — learning curves.
6. **Wykres MUST 3** — jakość vs czas treningu.
7. **Wykres MUST 4** — ranking i przedziały ufności.
8. **Wykres MUST 5** — Dyna vs Double DQN: wpływ liczby symulowanych przejść $K$.
9. **Wykres SHOULD 6 / 7** — skalowanie lub koszt inferencji.
10. **A2C vs PPO** — porównanie stabilności, sample efficiency i kosztu aktualizacji.
11. **Wnioski, ograniczenia i dalsza praca** — co faktycznie wynika z eksperymentów.

> [!TIP]
> Na prezentacji pokazujmy **zapis prawdziwej partii** lub animację sekwencji odbić. Jeden zrzut ze zwycięską pułapką może lepiej wyjaśnić zachowanie agentów niż dodatkowa tabela.

---

## 9. Uczciwość porównania — zasady eksperymentu

### Warunki wspólne

1. **Ten sam silnik gry i reguły** dla sześciu agentów.
2. **Te same informacje o stanie**, chociaż inne reprezentacje (bitset, cechy, tensor).
3. **Te same protokoły testowe**: strony, przeciwnicy, liczba partii i sposób raportowania.
4. **Porównywalne budżety interakcji treningowych**; osobno pełen koszt obliczeń.
5. **Kilka losowych inicjalizacji** dla głównych porównań, jeśli pozwala budżet.
6. **Nie dobieramy hiperparametrów na końcowym turnieju**.
7. **Uwzględniamy koszt optymalizacji** — PPO zwykle wykonuje więcej epok na tym samym rolloutcie niż A2C; raportujemy liczbę aktualizacji i wall-clock.
8. **Dyna korzysta z tych samych reguł, ale dodatkowo symuluje.** Nie traktujemy symulowanych kroków jako darmowych real steps; pokazujemy ich liczbę i koszt. W eksperymencie porównawczym kontrolujemy także liczbę aktualizacji gradientowych.

### Trzy potencjalne błędy, których unikamy

- **Błąd #1: agent gra przeciwko sobie, ale reward jest źle podpisany.** Wartości muszą być liczone z jednoznacznej perspektywy gracza; zmiana właściciela ruchu po odbiciu nie zachodzi automatycznie.
- **Błąd #2: wygrywa algorytm, który dostał więcej obliczeń.** Raportujemy zarówno budżet próbek, jak i czas CPU/GPU oraz koszt inferencji.
- **Błąd #3: sześć modeli porównujemy tylko między sobą.** Wspólny random i heurystyczny benchmark są potrzebne, aby wyniki miały znaczenie także poza małym turniejem.

## 10. Wykonalny harmonogram i zakres MVP

| Etap | Co powstaje? | Kryterium ukończenia |
|---|---|---|
| **0. Specyfikacja reguł** | Opis zasad + przypadki brzegowe | Brak niejednoznaczności w zasadach goli, odbić i blokady |
| **1. Silnik i testy** | `GameState`, `step`, maski, terminale, testy jednostkowe | Powtarzalne, legalne partie bez błędów |
| **2. Baselines** | Random, heurystyka, Tabular Q-learning | Pierwszy działający turniej i learning curve |
| **3. Aproksymacja** | Linear Q-learning, Double DQN | Logowane checkpointy i ewaluacja na 8 × 10 |
| **4. Policy gradient** | Maskable PPO + adapter przeciwnika | Stabilne treningi z maskami i prawidłową nagrodą |
| **5. Actor–critic baseline** | A2C z maskowanym rozkładem akcji i tym samym adapterem co PPO | Legalne akcje, poprawne rollouty oraz porównanie z PPO |
| **6. Planowanie Dyna** | Dyna-style Double DQN, odtwarzanie stanów i symulacje $K=0,5,10,20$ | Zgodność real/sim transitions, odtwarzalny test Dyna vs bazowy Double DQN |
| **7. Ewaluacja** | Turniej, seedy, metryki, wykresy | Dane umożliwiają porównanie wszystkich sześciu metod |
| **8. Prezentacja** | Slajdy, przykłady partii, ograniczenia | Odpowiedź na RQ1–RQ6, bez nadinterpretacji |

### Minimum viable project (MVP)

> [!NOTE]
> Dyna-style Double DQN nie wymaga drugiej niezależnej sieci: reużywa implementację Double DQN. Mimo tego **szósty agent to dodatkowy zakres**. Jeśli projekt okaże się zbyt duży, redukujemy liczbę wariantów $K$ i eksperymentów pobocznych, **nie** rezygnujemy z kontroli poprawności symulacji ani uczciwego porównania czasu obliczeniowego.


- Wszystkie sześć algorytmów działa, wybiera wyłącznie legalne akcje i zapisuje checkpointy.
- Dla sześciu metod przeprowadzamy test na 8 × 10 przeciwko stałej puli przeciwników i pełny turniej.
- Zapisujemy learning curves, czas treningu i koszt pojedynczej decyzji.
- A2C i PPO otrzymują wspólną maskę, porównywalną architekturę i identyczny adapter środowiska.
- Dyna-style Double DQN otrzymuje tę samą sieć co Double DQN oraz **minimalny wariant $K=5$**; porównujemy go z $K=0$ i raportujemy koszt symulacji.
- Powtarzamy główne eksperymenty przynajmniej dla kilku seedów, jawnie raportując ograniczenia budżetowe.

### Wersja rozszerzona

- Dodatkowe seedy i dłuższe treningi.
- Ablacja samych cech w Linear Q-learningu.
- Ablacja hiperparametrów A2C/PPO (np. współczynnik entropii, liczba kroków rolloutu, clipping w PPO).
- Pełna ablacja Dyna: $K=0,5,10,20$, długość rolloutu $H$, udział symulacji w replay, kontrola compute-matched oraz wpływ sposobu doboru stanów do symulacji.
- Transfer między rozmiarami planszy (oddzielny eksperyment z architekturą obsługującą rozmiary).

---

## 11. Wstępne decyzje techniczne

**Proponowany stack:** Python, NumPy, PyTorch, Gymnasium, `stable-baselines3` + `sb3-contrib`, Matplotlib/Seaborn lub Plotly, pytest. Główną logikę gry utrzymujemy niezależną od bibliotek RL.

**Przykładowa struktura repozytorium:**

```text
paper-soccer-rl/
├── src/paper_soccer/
│   ├── game.py               # czysty silnik reguł
│   ├── env.py                # adapter Gymnasium
│   ├── opponents.py          # losowy, heurystyczny, snapshoty
│   ├── simulator.py          # odtwarzanie i symulacja z kopii stanu
│   ├── agents/
│   │   ├── tabular_q.py
│   │   ├── linear_q.py
│   │   ├── double_dqn.py
│   │   ├── dyna_double_dqn.py  # rozszerza Double DQN o planowanie
│   │   ├── maskable_ppo.py
│   │   └── a2c.py
│   ├── evaluation/
│   │   ├── tournament.py
│   │   ├── metrics.py
│   │   └── plots.py
│   └── utils/
├── configs/
├── experiments/
├── tests/
├── checkpoints/
└── README.md
```

**Sprzęt:** projekt można rozwijać na MacBooku Pro M4 z 36 GB RAM. Największym kosztem prawdopodobnie będzie generowanie wielu epizodów i trening sieci, a nie sam wybór akcji. Najpierw testujemy małe plansze i małe budżety, a następnie 8 × 10.

### Minimalny zestaw zapisywanych danych

Każda ocena powinna zapisać: `model_id`, `algorithm`, `seed`, `board_size`, `training_env_steps`, `training_wall_time`, `gradient_updates`, `rollout_steps`, `opponent_id`, `player_side`, `winner`, `termination_reason`, `number_of_moves`, `decision_latency_ms`, `real_steps`, `simulated_steps`, `planning_K`, `planning_H`, `simulated_replay_fraction`, `model_simulation_time_ms`.

Dzięki temu wszystkie wykresy można odtworzyć z wyników surowych, bez ręcznego przepisywania danych.

## 12. Co uznamy za sukces projektu?

- Implementacje są poprawne — każdy agent podejmuje wyłącznie legalne decyzje.
- Każdy z sześciu algorytmów ma odtwarzalny zapis modelu i działający evaluator.
- Istnieje **uczciwy protokół porównania** siły gry, prób i kosztu.
- Mamy co najmniej pięć kluczowych wykresów z sekcji 8, w tym porównanie Dyna z Double DQN.
- Potrafimy powiedzieć, **które wnioski wynikają z danych**, a które pozostają hipotezami ze względu na małą liczbę seedów lub ograniczony budżet.

---

## Materiały referencyjne

- [Paper Soccer — przykładowe zasady](https://www.paper-football.com/).
- [Gymnasium — dokumentacja środowisk](https://gymnasium.farama.org/).
- [SB3-Contrib — MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
- [Double DQN — van Hasselt et al. (2016)](https://arxiv.org/abs/1509.06461).
- [PPO — Schulman et al. (2017)](https://arxiv.org/abs/1707.06347).
- [A2C — Stable-Baselines3, dokumentacja](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html).
- [Advantage actor–critic — Mnih et al. (2016)](https://arxiv.org/abs/1602.01783).
- [Sutton & Barto — *Reinforcement Learning: An Introduction*, rozdział Dyna](http://incompleteideas.net/book/the-book-2nd.html).
- [Sutton (1990) — *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming*](https://doi.org/10.1016/S0020-7373(08)80020-1).
- [Maskowanie nielegalnych akcji — Huang & Ontañón (2020)](https://arxiv.org/abs/2006.14171).

*To propozycja projektu badawczego, nie raport z wykonanych eksperymentów. Wszystkie wykresy opisano jako planowane, a hipotezy jako wymagające weryfikacji.*
