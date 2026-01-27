import sys
from .utils import Console

from .config import load_config

console = Console()

def valid_input(file_path:str) : 
    with open(file_path, 'r', encoding = 'utf-8') as f:
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
    
def main(file_path:str=""):
    if file_path is None or file_path == "":
        config = load_config()
        if config is None:
            sys.exit(1)

        file_path = config.get("file_path", "data/scenario0.txt")

    console.display(valid_input(file_path), "valid-input", border_style="green")

