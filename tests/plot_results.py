import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from game_motor.experiment import process_game_statistics

# parse log file
def parse_log(address, nb_action):
    with open(address, 'rb') as f:
        f = f.readlines()
    list_act = [] 
    episode = 0
    act_count = np.zeros(nb_action)
    # parse file
    for line in f:
        st = line.decode("utf-8").strip()
        if ('episode' in st) and ('features' not in st) and ('eps' not in st):
            eps = st[43:]
            if eps != episode :
                list_act.append(act_count)
                act_count = np.zeros(nb_action)
                episode = eps
        if 'opt action' in st :
            act = int(st[48:])
            act_count[act] +=1
        if 'random action' in st :
            act = int(st[51:])
            act_count[act] +=1
    # create dataframe
    tab = np.stack(list_act,axis=0)
    df = pd.DataFrame(tab, columns = [str(i) for i in range(nb_action)])
    
    return df


def parse_stats(address):
    with open(address, 'rb') as fp:
        dic = pickle.load(fp)
    return process_game_statistics(dic)
    

# main         
if __name__ == "__main__":
    map_id = 1
    
    with open('DFP_basic_1_stats','rb') as fp:
        dic_tot = pickle.load(fp)
    
    # plot health end of the episode
    plt.figure()
    plt.plot(dic_tot['health'])
    plt.title('D1 scenario : Health')
    plt.xlabel('# of episodes')
    plt.ylabel('Health')
    
    
    # plot medikits end of the episode    
    plt.figure()
    plt.plot(dic_tot['medikit'])
    plt.title('D1 scenario : # medikits collected')
    plt.xlabel('# of episodes')
    plt.ylabel('# medikits')
     
           
    # for the first states
    with open('DFP_basic_1_actions','rb') as fp:
        df = pickle.load(fp)
    
    plt.figure()
    df.iloc[:1000].sum(axis=0).plot.bar()
    plt.xlabel('Actions')
    plt.ylabel('# chosen')
    
    # for the middle states
    plt.figure()
    df.iloc[9000:10000].sum(axis=0).plot.bar()
    plt.xlabel('Actions')
    plt.ylabel('# chosen')
    
    # for end states
    plt.figure()
    df.iloc[33575:34575].sum(axis=0).plot.bar()
    plt.xlabel('Actions')
    plt.ylabel('# chosen')          
               
               
               
    # Compare naive/trained bot
    address1 = 'compare_bots_naive'
    address2 = 'compare_bots_trained_10000'
    address3 = 'compare_bots_trained_38000'

    stat_naive = parse_stats(address1)[map_id]
    stat_trained_10000 = parse_stats(address2)[map_id]
    stat_trained_38000 = parse_stats(address3)[map_id]
    
    l1 = [np.mean(stat_naive['health']), np.mean(stat_trained_10000['health']), np.mean(stat_trained_38000['health'])]
    l2 = [np.mean(stat_naive['medikit']), np.mean(stat_trained_10000['medikit']), np.mean(stat_trained_38000['medikit'])]
    
    
    fig, ax = plt.subplots()
    plt.bar(['naive', 'trained_10000', 'trained_38000'], l1)
    plt.title('D1 scenario : Average Health over 100 episodes')
    plt.ylabel('Health')
    plt.show()

    fig, ax = plt.subplots()
    plt.bar(['naive', 'trained_10000', 'trained_38000'], l2)
    plt.title('D1 scenario : Average # medikits over 100 episodes')
    plt.ylabel('# medikits')
    plt.show()
