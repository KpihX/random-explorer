def valid_input() : 
    with open('scenario2.txt', 'r', encoding = 'utf-8') as f:
        count =1
        xmax=0
        ymax=0
        u_s1 = []
        u_d1 = []
        u_s2 = []
        u_d2 = []
        R = 0
        obs =[]
        for ligne in f: 
            s = ligne.strip()
            p = s.split()
            if (count <=11):
                if len(p)!=1:
                    return []
                else:
                    if count ==1:
                        xmax=int(float(p[0]))
                    if count ==2:
                        ymax=int(float(p[0]))
                    if count == 3 or count ==4:
                        u_s1.append(int(float(p[0])))
                    if count == 5 or count ==6:
                        u_d1.append(int(float(p[0])))
                    if count == 7 or count ==8:
                        u_s2.append(int(float(p[0])))
                    if count == 9 or count ==10:
                        u_d2.append(int(float(p[0])))
                    if count == 11:
                        R = int(float(p[0]))
            if count>11:
                if len(s.split())!=4:
                    return []
                else:
                    nombres = [int(float(x)) for x in p]
                    obs.append(nombres)
            count +=1
        return [xmax,ymax,u_s1,u_d1, u_s2, u_d2, R, obs]

print(valid_input())

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_environment():
    xmax, ymax, u_s1, u_d1, u_s2, u_d2, r, R = valid_input()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Tracer les limites de l'environnement
    ax.set_xlim(-10, xmax + 10)
    ax.set_ylim(-10, ymax + 10)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Environnement de Path Planning', fontsize=14)
    
    # 2. Tracer une grille
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Tracer les obstacles (rectangles noirs)
    for i, (xo, yo, lx, ly) in enumerate(R):
        rect = patches.Rectangle(
            (xo, yo), lx, ly, 
            linewidth=2, 
            edgecolor='black', 
            facecolor='black',
            alpha=0.7,
            label='Obstacles' if i == 0 else None
        )
        ax.add_patch(rect)

    # 4. Tracer le point de départ (u_s1)
    ax.scatter(u_s1[0], u_s1[1], 
               color='red', 
               s=200, 
               marker='o', 
               edgecolor='black',
               linewidth=2,
               label=f'Départ ({u_s1[0]}, {u_s1[1]})',
               zorder=5)
    
    # 5. Tracer le point d'arrivée (u_d1)
    ax.scatter(u_d1[0], u_d1[1], 
               color='red', 
               s=200, 
               marker='s',  # carré pour l'arrivée
               edgecolor='black',
               linewidth=2,
               label=f'Arrivée ({u_d1[0]}, {u_d1[1]})',
               zorder=5)
    
 
    # 7. Ajouter les limites de l'environnement
    ax.plot([0, xmax, xmax, 0, 0], [0, 0, ymax, ymax, 0], 
            'k-', linewidth=3, alpha=0.5)
    
    # 8. Ajouter une légende
    ax.legend(loc='upper right', fontsize=10)
    
    # 9. Égaliser les échelles
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax

fig, ax = plot_environment()
plt.show()
