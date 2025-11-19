import gymnasium as gym
import numpy as np
import random
import time # Lo useremo per monitorare i progressi

print("--- Avvio Progetto 1: Il Tassista (Addestramento) ---")

# 1. Creare l'ambiente
# Nota: l'addestramento è MOLTO più veloce senza rendering
env = gym.make("Taxi-v3") 

# 2. Ispezionare gli spazi
num_stati = env.observation_space.n
num_azioni = env.action_space.n

# 3. Inizializzare la Q-Table
q_table = np.zeros((num_stati, num_azioni))
print(f"Q-Table (cervello) creata con dimensioni: {q_table.shape}")

# 4. Definire gli Iperparametri
num_episodi = 10000  # Partite totali da giocare per imparare
learning_rate = 0.1   # Alpha (α): Tasso di apprendimento
discount_factor = 0.99  # Gamma (γ): Importanza delle ricompense future

# Iperparametri per Epsilon-Greedy (Esplorazione vs. Sfruttamento)
max_epsilon = 1.0       # Tasso di esplorazione INIZIALE (100% casuale)
min_epsilon = 0.01      # Tasso di esplorazione MINIMO (1% casuale)
# Calcoliamo il valore di "decadimento" lineare
# Vogliamo che epsilon passi da 1.0 a 0.01 in 10.000 passi
epsilon_decay_value = (max_epsilon - min_epsilon) / num_episodi
epsilon = max_epsilon   # Epsilon corrente parte dal massimo

print(f"Iperparametri impostati per {num_episodi} episodi.")

# 5. --- IL CICLO DI ADDESTRAMENTO ---
print("\n--- Inizio Addestramento ---")
start_time = time.time() # Memorizziamo l'ora di inizio

for episodio in range(num_episodi):
    
    # 5.1 Resettare l'ambiente per un nuovo episodio
    (stato, info) = env.reset()
    
    terminato = False
    troncato = False
    
    # 5.2 Ciclo interno (una singola partita, passo dopo passo)
    while not (terminato or troncato):
        
        # 5.3 Scelta dell'Azione (Epsilon-Greedy)
        # Generiamo un numero casuale tra 0 e 1
        random_tradeoff = random.uniform(0, 1)
        
        if random_tradeoff > epsilon:
            # === EXPLOITATION (SFRUTTAMENTO) ===
            # Scegliamo l'azione migliore (con il Q-value più alto) 
            # per lo stato attuale dalla Q-Table
            azione = np.argmax(q_table[stato, :])
        else:
            # === EXPLORATION (ESPLORAZIONE) ===
            # Scegliamo un'azione casuale
            azione = env.action_space.sample()

        # 5.4 Eseguire l'azione e osservare il risultato dall'ambiente
        (nuovo_stato, ricompensa, terminato, troncato, info) = env.step(azione)
        
        # 5.5 Aggiornamento della Q-Table (La formula di Bellman!)
        
        # Troviamo il Q-value massimo per il *nuovo_stato* (il termine: max_a' Q(s', a'))
        max_q_futuro = np.max(q_table[nuovo_stato, :])
        
        # Il nostro Q-value attuale (vecchio)
        q_vecchio = q_table[stato, azione]
        
        # LA FORMULA COMPLETA:
        # Q_nuovo = Q_vecchio + alpha * (Ricompensa + gamma * max_Q_futuro - Q_vecchio)
        q_nuovo = q_vecchio + learning_rate * (ricompensa + discount_factor * max_q_futuro - q_vecchio)
        
        # Aggiorniamo la tabella con il nuovo valore calcolato
        q_table[stato, azione] = q_nuovo
        
        # 5.6 Aggiornare lo stato per il prossimo ciclo
        stato = nuovo_stato
    
    # 5.7 Fine Episodio: Decadimento di Epsilon
    # Riduciamo epsilon, ma ci assicuriamo che non scenda mai sotto il minimo
    epsilon = max(min_epsilon, epsilon - epsilon_decay_value)

    # Log di progresso (ogni 1000 episodi)
    if (episodio + 1) % 1000 == 0:
        print(f"Episodio {episodio + 1} / {num_episodi} completato. Epsilon attuale: {epsilon:.4f}")

# 6. Fine addestramento
env.close()
end_time = time.time()

print("\n--- Addestramento Terminato ---")
print(f"Tempo totale di addestramento: {end_time - start_time:.2f} secondi")

# Diamo un'occhiata al nostro "cervello" addestrato
print("\nEsempio di valori dalla Q-Table (prime 5 righe):")
print(q_table[:5])

# --- SEZIONE 7: VALUTAZIONE DELL'AGENTE ADDETRATO ---

print("\n--- Inizio Valutazione ---")
print("Avvio di 5 episodi con l'agente addestrato...")

# 7.1 Creare un nuovo ambiente per la visualizzazione
# Questa volta, attiviamo il rendering "human"
env_visual = gym.make("Taxi-v3", render_mode="human")

num_episodi_test = 5

for episodio in range(num_episodi_test):
    
    (stato, info) = env_visual.reset()
    
    terminato = False
    troncato = False
    
    print(f"\n--- Inizio Episodio Test {episodio + 1} ---")
    
    while not (terminato or troncato):
        
        # 7.2 Scelta dell'Azione (SOLO EXPLOITATION)
        # Scegliamo l'azione migliore dalla nostra Q-Table addestrata
        # Non c'è più Epsilon, l'agente è "sicuro di sé"
        azione = np.argmax(q_table[stato, :])

        # 7.3 Eseguire l'azione e osservare
        (nuovo_stato, ricompensa, terminato, troncato, info) = env_visual.step(azione)
        
        # 7.4 Rallentare per permetterci di vedere
        # La finestra si aggiornerà automaticamente dopo env.step()
        time.sleep(0.25) # Pausa di 0.25 secondi
        
        # Aggiornare lo stato
        stato = nuovo_stato

    print(f"--- Fine Episodio Test {episodio + 1} ---")

# 7.5 Chiudere l'ambiente di visualizzazione
env_visual.close()
print("\n--- Valutazione Terminata ---")