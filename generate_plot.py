import matplotlib.pyplot as plt
import numpy as np

# Creazione della figura e dell'asse con margini più ampi
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# Rimozione dei bordi superiore e destro
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Configurazione delle frecce per gli assi
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Generazione dati per la Frontiera Efficiente (iperbole)
# r > 0, sigma > 0. Curva iperbolica per la frontiera (bullet-shape)
r_frontier = np.linspace(2, 12, 500)
sigma_frontier = 2 + 0.1 * (r_frontier - 6)**2

# Selezioniamo la parte efficiente (rendimento > rendimento a varianza minima, che è a R=5.5)
efficient_mask = r_frontier >= 5.5
r_eff = r_frontier[efficient_mask]
sigma_eff = sigma_frontier[efficient_mask]

# Parte inefficiente (tratteggiata)
inefficient_mask = (r_frontier < 5.5) & (r_frontier > 3.0) # troncata per non affollare il basso
r_ineff = r_frontier[inefficient_mask]
sigma_ineff = sigma_frontier[inefficient_mask]

# Tracciamento della Frontiera Efficiente
ax.plot(sigma_eff, r_eff, color='#1f77b4', linewidth=3, label='Frontiera Efficiente')
ax.plot(sigma_ineff, r_ineff, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.6)

# Definizione del punto di tangenza (Portafoglio Ottimo)
# Scegliamo un punto sulla frontiera
r_opt = 8.0
sigma_opt = 2 + 0.1 * (r_opt - 6)**2  # circa 2.4

# Generazione dati per le Curve di Indifferenza
A = 1.04
sigma_range = np.linspace(1.5, 5, 200)

# Curva di tangenza U2 (passa per r_opt, sigma_opt)
u2 = r_opt - 0.5 * A * sigma_opt**2
r_u2 = u2 + 0.5 * A * sigma_range**2

# Curva di utilità inferiore U1 (meno preferita, più a destra)
u1 = u2 - 1.5
r_u1 = u1 + 0.5 * A * sigma_range**2

# Curva di utilità superiore U3 (preferita ma non raggiungibile)
u3 = u2 + 1.5
r_u3 = u3 + 0.5 * A * sigma_range**2

# Tracciamento delle Curve di Indifferenza
ax.plot(sigma_range, r_u1, color='#d62728', linestyle='-', linewidth=1.8, label='Curva di Indifferenza $U_1$')
ax.plot(sigma_range, r_u2, color='#2ca02c', linestyle='-', linewidth=2.5, label='Curva di Indifferenza $U_2$ (Ottimo)')
ax.plot(sigma_range, r_u3, color='#ff7f0e', linestyle='-', linewidth=1.8, label='Curva di Indifferenza $U_3$')

# Punto di tangenza (Portafoglio Ottimo P*)
ax.scatter(sigma_opt, r_opt, color='black', s=100, zorder=5)

# Annotazione Portafoglio Ottimo: posizionata in alto a sinistra del punto per evitare sovrapposizioni
ax.annotate('Portafoglio\nOttimo ($P^*$)', 
            xy=(sigma_opt, r_opt), 
            xytext=(sigma_opt - 0.7, r_opt + 1.6),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2, connectionstyle="arc3,rad=0.1"),
            fontsize=11, fontweight='bold', ha='center', va='bottom')

# Linee tratteggiate verso gli assi
ax.plot([0, sigma_opt], [r_opt, r_opt], color='gray', linestyle=':', linewidth=1.5)
ax.plot([sigma_opt, sigma_opt], [0, r_opt], color='gray', linestyle=':', linewidth=1.5)

# Etichette sui valori ottimi negli assi (con offset corretti per evitare sovrapposizioni)
ax.text(-0.15, r_opt, r'$E(R^*)$', ha='right', va='center', fontsize=12, fontweight='bold')
ax.text(sigma_opt, -0.4, r'$\sigma^*$', ha='center', va='top', fontsize=12, fontweight='bold')

# Etichette delle curve di utilità (allineate alla fine di ciascun tracciato)
ax.text(sigma_range[-1], r_u1[-1] + 0.3, r'$U_1$', fontsize=12, color='#d62728', fontweight='bold', ha='center', va='bottom')
ax.text(sigma_range[-1], r_u2[-1] + 0.3, r'$U_2$', fontsize=12, color='#2ca02c', fontweight='bold', ha='center', va='bottom')
ax.text(sigma_range[-1], r_u3[-1] + 0.3, r'$U_3$', fontsize=12, color='#ff7f0e', fontweight='bold', ha='center', va='bottom')

# Etichette informative sul grafico (riposizionate per massima leggibilità)
# La dicitura della frontiera è spostata a destra e in alto, del tutto separata dal tracciato
ax.text(4.7, 12.5, 'Frontiera dei\nPortafogli Efficienti', color='#1f77b4', fontsize=11, fontweight='bold', ha='center')
ax.text(2.6, 3.4, 'Portafogli\nInefficienti', color='gray', fontsize=9.5, style='italic', ha='center')

# Titoli degli assi con freccia (labelpad aumentato per evitare sovrapposizioni)
ax.set_xlabel(r'Rischio (Deviazione Standard, $\sigma$)', fontsize=13, labelpad=20)
ax.set_ylabel(r'Rendimento Atteso, $E(R)$', fontsize=13, labelpad=35)

# Aggiunta delle punte di freccia agli assi
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Limiti del grafico (estesi per dare più aria e non tagliare le diciture)
ax.set_xlim(0, 5.8)
ax.set_ylim(0, 14.5)

# Rimozione dei numeri degli assi (asse qualitativo/teorico)
ax.set_xticks([])
ax.set_yticks([])

# Legenda e Titolo
ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.title('La scelta del Portafoglio Ottimo', fontsize=15, fontweight='bold', pad=20)

# Mostra e salva il grafico
plt.tight_layout()
plt.savefig('scelta_portafoglio_ottimo_corretto_v2.png', dpi=300, bbox_inches='tight')
print("Grafico salvato con successo!")
