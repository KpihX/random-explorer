def valid_input() : 
    with open('data/scenario2.txt', 'r', encoding = 'utf-8') as f:
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
    
def main():
    print(valid_input())
