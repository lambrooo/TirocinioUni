
# Spiegazione della Simulazione in Java

Questo documento analizza la versione Java della simulazione, creata come strumento di confronto per chi conosce Java e vuole capire la logica del progetto Python originale.

---

## 1. Concetti di Base del Progetto

Prima di analizzare il codice, è importante capire l'idea di fondo. Stiamo creando un "Digital Twin" (un gemello digitale) estremamente semplificato di un sistema fisico, come potrebbe essere un macchinario industriale, un motore o una stanza.

### a. I Componenti Fondamentali

- **Environment (Ambiente)**
  - **Cosa rappresenta?** È il mondo fisico o il sistema che vogliamo simulare. Ha uno stato interno (es. la sua temperatura attuale, il carico su un motore) e delle "leggi fisiche" che lo governano (es. un oggetto caldo tende a raffreddarsi, un motore sotto sforzo si scalda).
  - **Nel codice:** La classe `Environment` mantiene lo stato corrente delle variabili e il suo metodo `step()` calcola come questo stato cambia nel tempo, anche in base a influenze esterne.

- **Sensor (Sensore)**
  - **Cosa rappresenta?** Un dispositivo di misurazione del mondo reale (un termometro, un tachimetro). I sensori non sono perfetti: hanno sempre un margine di errore (il **rumore**) e a volte possono guastarsi o dare letture completamente sballate (le **anomalie**).
  - **Nel codice:** La classe `Sensor` non conosce il valore "vero" dell'ambiente. Lo legge e ci aggiunge del rumore casuale (gaussiano) per simulare l'imprecisione. Può anche introdurre anomalie come `bias` (un errore costante) o `outlier` (un valore singolo e assurdo).

- **Actuator (Attuatore)**
  - **Cosa rappresenta?** Un dispositivo che può *agire* sull'ambiente per modificarlo. Esempi sono un condizionatore che raffredda una stanza, un acceleratore che aumenta il carico su un motore, una valvola che regola un flusso.
  - **Nel codice:** La classe `Actuator` riceve un comando generico e lo traduce in un'azione concreta che viene passata all'ambiente.

- **Agent (Agente)**
  - **Cosa rappresenta?** È il "cervello" del sistema, il controllore. Il suo compito è leggere i dati (imperfetti) provenienti dai sensori e decidere quali comandi inviare agli attuatori per far sì che l'ambiente raggiunga o mantenga uno stato desiderato (chiamato **setpoint**).
  - **Nel codice:** La classe `Agent` implementa la logica di controllo. In questo caso, un semplice controllo proporzionale: più il valore misurato è lontano dall'obiettivo, più forte sarà il comando per correggerlo.

### b. Il Concetto Chiave: Il "Feedback Loop" (Ciclo di Retroazione)

Questi componenti non agiscono in modo isolato, ma creano un ciclo continuo, che è il cuore di ogni sistema di controllo:

1.  Il **Sensore** misura lo stato dell'**Ambiente** -> `(lettura rumorosa)`
2.  L'**Agente** riceve la lettura, la confronta con il suo obiettivo (`setpoint`) e calcola un'azione correttiva -> `(comando)`
3.  L'**Attuatore** riceve il comando e lo traduce in un'azione fisica -> `(azione)`
4.  L'azione modifica lo stato dell'**Ambiente**.
5.  ...e il ciclo ricomincia dal punto 1.

Questo ciclo `Ambiente -> Sensore -> Agente -> Attuatore -> Ambiente` è la base della simulazione. Il nostro codice non fa altro che ripetere questo ciclo per `N` passi, registrando tutto ciò che accade.

---

## 2. Obiettivo del Progetto

Questo progetto traduce la logica e la struttura della simulazione Python in un codice Java idiomatico. L'obiettivo è puramente didattico, per mostrare le somiglianze e le differenze sintattiche tra i due linguaggi.

**Semplificazioni effettuate**: 
- **Niente PyTorch**: Invece di usare una libreria di calcolo scientifico, usiamo semplici `double` per le variabili.
- **Niente Grafici/WandB**: Per mantenere il codice snello e senza dipendenze esterne complesse, la parte di visualizzazione è stata rimossa. Il focus è sulla logica della simulazione e sul salvataggio dei dati in CSV.

---

## 3. Come Compilare ed Eseguire

1.  **Prerequisiti**: Assicurati di avere un JDK (Java Development Kit) installato.
2.  **Compilazione**: Apri un terminale e posizionati nella directory `src/main/java`.
    ```bash
    cd java_simulation/src/main/java
    ```
    Esegui il comando di compilazione:
    ```bash
    javac com/simulation/*.java
    ```
3.  **Esecuzione**: Dalla stessa directory (`src/main/java`), esegui la classe principale:
    ```bash
    java com.simulation.Simulation
    ```
4.  **Output**: Verrà creato un file `simulation_log_java.csv` nella directory radice del progetto (`Tirocinio`) con i risultati della simulazione.

---

## 3. Analisi del Codice

Il progetto è diviso in classi che rispecchiano quelle della versione Python.

### a. `Environment.java`

```java
public class Environment {
    private double temperature;
    // ...
    private final Random random = new Random();

    public Environment(double initialTemp, ...) {
        this.temperature = initialTemp;
        // ...
    }

    public void step(double tempAction, double loadAction) {
        // ...
    }
}
```
- Le variabili d'istanza Python (`self.temperature`) diventano campi privati (`private double temperature`).
- Il costruttore `__init__` diventa un costruttore pubblico `public Environment(...)`.
- Il rumore gaussiano (`torch.randn()`) viene generato con `random.nextGaussian()` dalla classe `java.util.Random`.

### b. `Sensor.java`

```java
public class Sensor {
    private Map<String, Object> anomaly = null;
    // ...
    public double read(double trueValue) {
        // ...
        if (this.anomaly != null) {
            // ...
        }
        return reading;
    }
    public void introduceAnomaly(String anomalyType, double value) {
        this.anomaly = new HashMap<>();
        this.anomaly.put("type", anomalyType);
        this.anomaly.put("value", value);
    }
}
```
- Il dizionario Python per le anomalie è stato tradotto con una `HashMap<String, Object>`.
- Il controllo `if self.anomaly:` diventa il classico `if (this.anomaly != null)`.

### c. `Agent.java` e `AgentCommands` Record

```java
record AgentCommands(double tempCommand, double loadCommand) {}

public class Agent {
    // ...
    public AgentCommands getCommands(double tempReading, double loadReading) {
        // ...
        return new AgentCommands(tempCommand, loadCommand);
    }
}
```
- Per restituire i due valori di comando (che in Python era una tupla implicita), abbiamo usato un `record` Java. Un `record` è una classe speciale, concisa, pensata per contenere dati immutabili. È l'equivalente moderno di un semplice oggetto DTO (Data Transfer Object).

### d. `Simulation.java` (Classe Principale)

```java
public class Simulation {
    public static void main(String[] args) {
        // 1. Inizializzazione
        Map<String, Sensor> sensors = new HashMap<>();
        sensors.put("temperature", new Sensor(...));
        List<String[]> logData = new ArrayList<>();

        // 2. Loop di simulazione
        for (int step = 0; step < nSteps; step++) {
            // ...
        }

        // 3. Salvataggio in CSV
        saveToCsv(logData);
    }

    private static void saveToCsv(List<String[]> logData) {
        // ... logica con PrintWriter ...
    }
}
```
- Il punto di ingresso è il classico `public static void main(String[] args)`.
- I dizionari e le liste Python sono stati tradotti con `HashMap` e `ArrayList`.
- Il loop `for` è sintatticamente identico a quello che `range()` produce in Python.
- Il salvataggio in CSV, gestito da `pandas` in Python, qui viene fatto manualmente tramite un `PrintWriter` che scrive su un file, riga per riga.
